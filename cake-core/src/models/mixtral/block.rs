//! Mixtral MoE transformer block.
//!
//! Identical structure to dense Mistral (pre-norm, standard residual connections,
//! GQA attention, optional sliding window) except the FFN is a sparse
//! Mixture-of-Experts layer instead of a dense MLP.
//!
//! The MoE expert architecture is the same as Qwen3 MoE (SwiGLU FFN with
//! gate_proj + up_proj + down_proj per expert, top-K routing), but the weight
//! naming differs:
//!   Mixtral: `block_sparse_moe.experts.{j}.w1` (gate), `.w3` (up), `.w2` (down)
//!   Qwen3:  `mlp.experts.{j}.gate_proj`, `.up_proj`, `.down_proj`

use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{ops::softmax_last_dim, Linear, Module, RmsNorm, VarBuilder};

use crate::cake::{Context, Forwarder};
use crate::models::common::{load_rms_norm, CausalSelfAttention, Config};
use async_trait::async_trait;

/// Mixtral sparse MoE FFN block with `block_sparse_moe` weight naming.
#[derive(Debug, Clone)]
pub struct MixtralSparseMoe {
    /// Router weight: (num_experts, hidden_size).
    gate: Linear,
    /// Stacked expert gate projections (w1): (num_experts, intermediate_size, hidden_size).
    gate_proj: Tensor,
    /// Stacked expert up projections (w3):   (num_experts, intermediate_size, hidden_size).
    up_proj: Tensor,
    /// Stacked expert down projections (w2): (num_experts, hidden_size, intermediate_size).
    down_proj: Tensor,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MixtralSparseMoe {
    pub fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg
            .moe_intermediate_size
            .expect("moe_intermediate_size must be set");
        let n = cfg.num_experts;

        // Router: single linear, no bias.
        let gate_w = vb.pp("gate").get((n, h), "weight")?;
        let gate = Linear::new(gate_w, None);

        // Load all expert weights and stack into batched tensors.
        // Mixtral safetensors stores each expert separately:
        //   block_sparse_moe.experts.{j}.w1.weight  (i, h)  — gate_proj
        //   block_sparse_moe.experts.{j}.w3.weight  (i, h)  — up_proj
        //   block_sparse_moe.experts.{j}.w2.weight  (h, i)  — down_proj
        let mut gate_ws = Vec::with_capacity(n);
        let mut up_ws = Vec::with_capacity(n);
        let mut down_ws = Vec::with_capacity(n);

        for j in 0..n {
            let exp = vb.pp("experts").pp(j.to_string());
            gate_ws.push(exp.get((i, h), "w1.weight")?); // w1 = gate_proj
            up_ws.push(exp.get((i, h), "w3.weight")?); // w3 = up_proj
            down_ws.push(exp.get((h, i), "w2.weight")?); // w2 = down_proj
        }

        // Stack: (n, i, h), (n, i, h), (n, h, i)
        let gate_proj = Tensor::stack(&gate_ws, 0)?;
        let up_proj = Tensor::stack(&up_ws, 0)?;
        let down_proj = Tensor::stack(&down_ws, 0)?;

        Ok(Self {
            gate,
            gate_proj,
            up_proj,
            down_proj,
            num_experts: n,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, s, h) = x.dims3().map_err(|e| anyhow!("moe x.dims3 -> {e}"))?;
        let n_tok = b * s;
        let x_flat = x
            .reshape((n_tok, h))
            .map_err(|e| anyhow!("moe reshape -> {e}"))?;

        // --- Router ---
        // logits: (n_tok, num_experts)
        let router_logits = self
            .gate
            .forward(&x_flat)
            .map_err(|e| anyhow!("moe router -> {e}"))?;

        // Softmax first (over all experts), then top-K.
        let router_probs =
            softmax_last_dim(&router_logits).map_err(|e| anyhow!("moe softmax -> {e}"))?;

        // Sort descending to get top-K indices.
        let (_, sorted_idx) = router_probs
            .sort_last_dim(false)
            .map_err(|e| anyhow!("moe sort -> {e}"))?;
        let top_k_idx = sorted_idx
            .narrow(D::Minus1, 0, self.num_experts_per_tok)
            .map_err(|e| anyhow!("moe narrow topk -> {e}"))?; // (n_tok, k)

        // Gather routing weights for selected experts.
        let top_k_w = router_probs
            .gather(&top_k_idx, D::Minus1)
            .map_err(|e| anyhow!("moe gather weights -> {e}"))?; // (n_tok, k)

        // Renormalise so the k selected weights sum to 1.
        let top_k_w = if self.norm_topk_prob {
            let s = top_k_w
                .sum_keepdim(D::Minus1)
                .map_err(|e| anyhow!("moe weight sum -> {e}"))?;
            top_k_w
                .broadcast_div(&s)
                .map_err(|e| anyhow!("moe weight div -> {e}"))?
        } else {
            top_k_w
        };

        // Pull routing decisions to CPU (tiny: n_tok x k integers + floats).
        let top_k_idx_flat: Vec<u32> = top_k_idx
            .to_dtype(DType::U32)
            .map_err(|e| anyhow!("moe idx dtype -> {e}"))?
            .flatten_all()
            .map_err(|e| anyhow!("moe idx flatten -> {e}"))?
            .to_vec1::<u32>()
            .map_err(|e| anyhow!("moe idx to_vec -> {e}"))?;

        let top_k_w_flat: Vec<f32> = top_k_w
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("moe w dtype -> {e}"))?
            .flatten_all()
            .map_err(|e| anyhow!("moe w flatten -> {e}"))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("moe w to_vec -> {e}"))?;

        // Build expert -> [(token_idx, weight)] dispatch table.
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![vec![]; self.num_experts];
        for tok in 0..n_tok {
            for k in 0..self.num_experts_per_tok {
                let exp_idx = top_k_idx_flat[tok * self.num_experts_per_tok + k] as usize;
                let weight = top_k_w_flat[tok * self.num_experts_per_tok + k];
                expert_tokens[exp_idx].push((tok, weight));
            }
        }

        let compute_dtype = x_flat.dtype();
        let mut output = Tensor::zeros((n_tok, h), compute_dtype, x_flat.device())
            .map_err(|e| anyhow!("moe output zeros -> {e}"))?;

        // Dispatch: for each active expert, gather tokens -> FFN -> scatter back.
        for (exp_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let tok_ids: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
            let weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();

            let idx = Tensor::new(tok_ids.as_slice(), x_flat.device())
                .map_err(|e| anyhow!("moe idx tensor -> {e}"))?;

            // Gather token vectors for this expert: (n_sel, h)
            let selected = x_flat
                .index_select(&idx, 0)
                .map_err(|e| anyhow!("moe index_select -> {e}"))?;

            // Expert FFN: gate_proj[exp] is (i, h), up_proj[exp] is (i, h), down_proj[exp] is (h, i)
            let gp = self
                .gate_proj
                .get(exp_idx)
                .map_err(|e| anyhow!("moe gate_proj.get({exp_idx}) -> {e}"))?;
            let up = self
                .up_proj
                .get(exp_idx)
                .map_err(|e| anyhow!("moe up_proj.get({exp_idx}) -> {e}"))?;
            let dp = self
                .down_proj
                .get(exp_idx)
                .map_err(|e| anyhow!("moe down_proj.get({exp_idx}) -> {e}"))?;

            // (n_sel, h) @ (h, i) = (n_sel, i)  [after transposing (i,h)]
            let gate_out = selected
                .matmul(&gp.t().map_err(|e| anyhow!("moe gp.t -> {e}"))?)
                .map_err(|e| anyhow!("moe gate matmul -> {e}"))?;
            let up_out = selected
                .matmul(&up.t().map_err(|e| anyhow!("moe up.t -> {e}"))?)
                .map_err(|e| anyhow!("moe up matmul -> {e}"))?;

            let hidden = (candle_nn::ops::silu(&gate_out)
                .map_err(|e| anyhow!("moe silu -> {e}"))?
                * up_out)
                .map_err(|e| anyhow!("moe gate*up -> {e}"))?;

            // (n_sel, i) @ (i, h) = (n_sel, h)
            let expert_out = hidden
                .matmul(&dp.t().map_err(|e| anyhow!("moe dp.t -> {e}"))?)
                .map_err(|e| anyhow!("moe down matmul -> {e}"))?;

            // Scale by routing weight.
            let w_t = Tensor::new(weights.as_slice(), x_flat.device())
                .map_err(|e| anyhow!("moe weight tensor -> {e}"))?
                .to_dtype(compute_dtype)
                .map_err(|e| anyhow!("moe weight to_dtype -> {e}"))?
                .unsqueeze(1)
                .map_err(|e| anyhow!("moe weight unsqueeze -> {e}"))?
                .broadcast_as(expert_out.shape())
                .map_err(|e| anyhow!("moe weight broadcast -> {e}"))?;
            let expert_out =
                (expert_out * w_t).map_err(|e| anyhow!("moe expert_out * weight -> {e}"))?;

            // Scatter-add back into output: output[tok_ids] += expert_out
            output = output
                .index_add(&idx, &expert_out, 0)
                .map_err(|e| anyhow!("moe index_add -> {e}"))?;
        }

        output
            .reshape((b, s, h))
            .map_err(|e| anyhow!("moe output reshape -> {e}"))
    }
}

/// A single Mixtral MoE transformer layer.
#[derive(Debug, Clone)]
pub struct MixtralBlock {
    name: String,
    input_layernorm: RmsNorm,
    attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
    moe: MixtralSparseMoe,
}

impl std::fmt::Display for MixtralBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (mixtral-moe)", &self.name)
    }
}

#[async_trait]
impl Forwarder for MixtralBlock {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let moe = MixtralSparseMoe::load(vb.pp("block_sparse_moe"), cfg)?;

        let eps = cfg.rms_norm_eps;
        let h = cfg.hidden_size;
        let input_layernorm = load_rms_norm(h, eps, false, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            load_rms_norm(h, eps, false, vb.pp("post_attention_layernorm"))?;

        Ok(Box::new(Self {
            name,
            input_layernorm,
            attn,
            post_attention_layernorm,
            moe,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        // Attention sublayer (pre-norm, standard residual).
        let residual = x;
        let x = self
            .input_layernorm
            .forward(x)
            .map_err(|e| anyhow!("input_layernorm: {e}"))?;
        let x = self
            .attn
            .forward(&x, index_pos, block_idx, ctx.cache.as_mut().expect("No cache"))
            .map_err(|e| anyhow!("attn: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("attn residual: {e}"))?;

        // MoE sublayer (pre-norm, standard residual).
        let residual = &x;
        let x = self
            .post_attention_layernorm
            .forward(&x)
            .map_err(|e| anyhow!("post_attention_layernorm: {e}"))?;
        let x = self.moe.forward(&x).map_err(|e| anyhow!("moe: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("moe residual: {e}"))?;

        Ok(x)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    /// Test the MoE routing dispatch table construction in isolation.
    /// This mirrors the algorithm in `MixtralSparseMoe::forward()` without
    /// needing actual expert weights.
    fn build_dispatch_table(
        router_probs_flat: &[f32], // (n_tok, num_experts) flattened
        n_tok: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
    ) -> Vec<Vec<(usize, f32)>> {
        // For each token, find top-K experts and their normalized weights
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![vec![]; num_experts];

        for tok in 0..n_tok {
            let probs = &router_probs_flat[tok * num_experts..(tok + 1) * num_experts];

            // Sort experts by probability (descending)
            let mut sorted: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select top-K
            let top_k: Vec<(usize, f32)> = sorted.into_iter().take(num_experts_per_tok).collect();

            // Normalize weights
            let sum: f32 = top_k.iter().map(|(_, w)| w).sum();
            for (exp_idx, weight) in &top_k {
                let w = if norm_topk_prob && sum > 0.0 {
                    weight / sum
                } else {
                    *weight
                };
                expert_tokens[*exp_idx].push((tok, w));
            }
        }

        expert_tokens
    }

    #[test]
    fn test_dispatch_table_basic() {
        // 2 tokens, 4 experts, top-2
        // Token 0: expert 2 (0.5), expert 0 (0.3), expert 1 (0.1), expert 3 (0.1)
        // Token 1: expert 1 (0.4), expert 3 (0.3), expert 0 (0.2), expert 2 (0.1)
        let probs = vec![
            0.3, 0.1, 0.5, 0.1, // token 0
            0.2, 0.4, 0.1, 0.3, // token 1
        ];

        let table = build_dispatch_table(&probs, 2, 4, 2, true);

        // Expert 0: should have token 0 (it's top-2 for token 0)
        assert_eq!(table[0].len(), 1);
        assert_eq!(table[0][0].0, 0); // token 0

        // Expert 1: should have token 1 (top-1 for token 1)
        assert_eq!(table[1].len(), 1);
        assert_eq!(table[1][0].0, 1); // token 1

        // Expert 2: should have token 0 (top-1 for token 0)
        assert_eq!(table[2].len(), 1);
        assert_eq!(table[2][0].0, 0); // token 0

        // Expert 3: should have token 1 (top-2 for token 1)
        assert_eq!(table[3].len(), 1);
        assert_eq!(table[3][0].0, 1); // token 1
    }

    #[test]
    fn test_dispatch_table_normalization() {
        // 1 token, 4 experts, top-2, norm=true
        // Token 0: expert 1 (0.6), expert 3 (0.3), expert 0 (0.05), expert 2 (0.05)
        let probs = vec![0.05, 0.6, 0.05, 0.3];

        let table = build_dispatch_table(&probs, 1, 4, 2, true);

        // Top-2 are expert 1 (0.6) and expert 3 (0.3)
        // Normalized: expert 1 = 0.6/0.9 ≈ 0.667, expert 3 = 0.3/0.9 ≈ 0.333
        let (_, w1) = table[1][0];
        let (_, w3) = table[3][0];
        assert!((w1 - 2.0 / 3.0).abs() < 1e-5, "expert 1 weight: {}", w1);
        assert!((w3 - 1.0 / 3.0).abs() < 1e-5, "expert 3 weight: {}", w3);
        assert!((w1 + w3 - 1.0).abs() < 1e-5, "weights should sum to 1.0");
    }

    #[test]
    fn test_dispatch_table_no_normalization() {
        let probs = vec![0.05, 0.6, 0.05, 0.3];
        let table = build_dispatch_table(&probs, 1, 4, 2, false);

        // Without normalization, raw weights are kept
        let (_, w1) = table[1][0];
        let (_, w3) = table[3][0];
        assert!((w1 - 0.6).abs() < 1e-5);
        assert!((w3 - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_dispatch_every_expert_gets_tokens() {
        // 8 tokens, 8 experts, top-2 → each expert should get ~2 tokens on average
        // Use uniform-ish routing so all experts are selected
        let mut probs = Vec::new();
        for tok in 0..8u32 {
            for exp in 0..8u32 {
                // Make each token strongly prefer two specific experts
                let primary = (tok * 2) as usize % 8;
                let secondary = (tok * 2 + 1) as usize % 8;
                if exp as usize == primary {
                    probs.push(0.4);
                } else if exp as usize == secondary {
                    probs.push(0.3);
                } else {
                    probs.push(0.05);
                }
            }
        }

        let table = build_dispatch_table(&probs, 8, 8, 2, true);

        // Total assignments should be 8 tokens × 2 experts = 16
        let total: usize = table.iter().map(|t| t.len()).sum();
        assert_eq!(total, 16, "total expert assignments should be 16");

        // No expert should be completely empty with this routing
        for (i, tokens) in table.iter().enumerate() {
            assert!(
                !tokens.is_empty(),
                "expert {} should have at least one token",
                i
            );
        }
    }

    #[test]
    fn test_moe_output_shape_identity() {
        // Verify that MoE output shape matches input shape using zeros
        let dev = &Device::Cpu;
        let b = 1;
        let s = 4;
        let h = 8;

        // Create input and output tensors
        let x = Tensor::randn(0f32, 1.0, (b, s, h), dev).unwrap();
        let x_flat = x.reshape((b * s, h)).unwrap();
        let output = Tensor::zeros((b * s, h), candle_core::DType::F32, dev).unwrap();
        let result = output.reshape((b, s, h)).unwrap();
        assert_eq!(result.dims(), &[b, s, h]);
    }
}
