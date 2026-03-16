//! Sparse Mixture-of-Experts FFN for Qwen3 MoE.
//!
//! Each layer has `num_experts` experts (SwiGLU FFNs with `moe_intermediate_size`
//! hidden dim) and a router that selects the top-`num_experts_per_tok` experts per
//! token. Expert weights are stacked into 3-D tensors for efficient indexed access.
//!
//! Router forward (matches HuggingFace Qwen3MoeTopKRouter exactly):
//!   1. logits  = x @ gate.T             — (n_tok, num_experts)
//!   2. probs   = softmax(logits)         — normalise across all experts first
//!   3. top_k   = argsort(probs, desc)[:k] — select k best
//!   4. weights = probs[top_k] / sum(probs[top_k])   — renormalise (norm_topk_prob=true)

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{ops::softmax_last_dim, Linear, Module, VarBuilder};

use crate::models::common::Config;

/// Sparse MoE FFN block.
#[derive(Debug, Clone)]
pub struct SparseMoeMlp {
    /// Router weight: (num_experts, hidden_size).
    gate: Linear,
    /// Stacked expert gate projections: (num_experts, moe_intermediate_size, hidden_size).
    gate_proj: Tensor,
    /// Stacked expert up projections:   (num_experts, moe_intermediate_size, hidden_size).
    up_proj: Tensor,
    /// Stacked expert down projections: (num_experts, hidden_size, moe_intermediate_size).
    down_proj: Tensor,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeMlp {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.moe_intermediate_size.expect("moe_intermediate_size must be set");
        let n = cfg.num_experts;

        // Router: single linear, no bias.
        let gate_w = vb.pp("gate").get((n, h), "weight")?;
        let gate = Linear::new(gate_w, None);

        // Load all expert weights and stack into batched tensors.
        // Safetensors stores each expert separately:
        //   mlp.experts.{j}.gate_proj.weight  (i, h)
        //   mlp.experts.{j}.up_proj.weight    (i, h)
        //   mlp.experts.{j}.down_proj.weight  (h, i)
        let mut gate_ws = Vec::with_capacity(n);
        let mut up_ws = Vec::with_capacity(n);
        let mut down_ws = Vec::with_capacity(n);

        for j in 0..n {
            let exp = vb.pp("experts").pp(j.to_string());
            gate_ws.push(exp.pp("gate_proj").get((i, h), "weight")?);
            up_ws.push(exp.pp("up_proj").get((i, h), "weight")?);
            down_ws.push(exp.pp("down_proj").get((h, i), "weight")?);
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
        let router_probs = softmax_last_dim(&router_logits)
            .map_err(|e| anyhow!("moe softmax -> {e}"))?;

        // Sort descending to get top-K indices.
        // sort_last_dim(false) = descending; returns (sorted_values, sorted_indices).
        let (_, sorted_idx) = router_probs
            .sort_last_dim(false)
            .map_err(|e| anyhow!("moe sort -> {e}"))?;
        let top_k_idx = sorted_idx
            .narrow(D::Minus1, 0, self.num_experts_per_tok)
            .map_err(|e| anyhow!("moe narrow topk -> {e}"))?; // (n_tok, k)

        // Gather routing weights for selected experts.
        let top_k_w = router_probs
            .contiguous()
            .map_err(|e| anyhow!("moe probs contiguous -> {e}"))?
            .gather(&top_k_idx.contiguous().map_err(|e| anyhow!("moe idx contiguous -> {e}"))?, D::Minus1)
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

        // Fast path for single-token generation: batch all k experts into
        // 3 batched matmuls instead of 3k individual matmuls + CPU dispatch.
        if n_tok == 1 {
            let k = self.num_experts_per_tok;
            let expert_indices = top_k_idx
                .squeeze(0)
                .map_err(|e| anyhow!("moe squeeze idx -> {e}"))?;

            // Gather expert weights: (k, i, h), (k, i, h), (k, h, i)
            let sel_gate = self
                .gate_proj
                .index_select(&expert_indices, 0)
                .map_err(|e| anyhow!("moe sel gate_proj -> {e}"))?;
            let sel_up = self
                .up_proj
                .index_select(&expert_indices, 0)
                .map_err(|e| anyhow!("moe sel up_proj -> {e}"))?;
            let sel_down = self
                .down_proj
                .index_select(&expert_indices, 0)
                .map_err(|e| anyhow!("moe sel down_proj -> {e}"))?;

            // x_flat: (1, h) → broadcast to (k, 1, h)
            let x_exp = x_flat
                .unsqueeze(0)
                .map_err(|e| anyhow!("moe x unsqueeze -> {e}"))?
                .expand((k, 1, h))
                .map_err(|e| anyhow!("moe x expand -> {e}"))?;

            // Batched matmul: (k, 1, h) @ (k, h, i) = (k, 1, i)
            let gate_out = x_exp
                .matmul(&sel_gate.t().map_err(|e| anyhow!("moe sel gp.t -> {e}"))?)
                .map_err(|e| anyhow!("moe batched gate matmul -> {e}"))?;
            let up_out = x_exp
                .matmul(&sel_up.t().map_err(|e| anyhow!("moe sel up.t -> {e}"))?)
                .map_err(|e| anyhow!("moe batched up matmul -> {e}"))?;

            let hidden = crate::utils::fused_ops::silu_mul(
                &gate_out.contiguous().map_err(|e| anyhow!("moe gate contig -> {e}"))?,
                &up_out.contiguous().map_err(|e| anyhow!("moe up contig -> {e}"))?,
            )
            .map_err(|e| anyhow!("moe silu_mul -> {e}"))?;

            // Down proj: (k, 1, i) @ (k, i, h) = (k, 1, h)
            let expert_outs = hidden
                .matmul(&sel_down.t().map_err(|e| anyhow!("moe sel dp.t -> {e}"))?)
                .map_err(|e| anyhow!("moe batched down matmul -> {e}"))?;

            // Weight by routing probabilities and sum: (k, 1, h) → (1, h)
            let weights = top_k_w
                .squeeze(0)
                .map_err(|e| anyhow!("moe squeeze weights -> {e}"))?
                .to_dtype(x_flat.dtype())
                .map_err(|e| anyhow!("moe weights dtype -> {e}"))?
                .reshape((k, 1, 1))
                .map_err(|e| anyhow!("moe weights reshape -> {e}"))?;
            let output = expert_outs
                .broadcast_mul(&weights)
                .map_err(|e| anyhow!("moe weighted -> {e}"))?
                .sum(0)
                .map_err(|e| anyhow!("moe sum experts -> {e}"))?;

            return output
                .reshape((b, s, h))
                .map_err(|e| anyhow!("moe output reshape -> {e}"));
        }

        // Pull routing decisions to CPU (tiny: n_tok × k integers + floats).
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

        // Build expert → [(token_idx, weight)] dispatch table.
        let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![vec![]; self.num_experts];
        for tok in 0..n_tok {
            for k in 0..self.num_experts_per_tok {
                let exp_idx = top_k_idx_flat[tok * self.num_experts_per_tok + k] as usize;
                let weight = top_k_w_flat[tok * self.num_experts_per_tok + k];
                expert_tokens[exp_idx].push((tok, weight));
            }
        }

        let compute_dtype = x_flat.dtype();
        let mut output =
            Tensor::zeros((n_tok, h), compute_dtype, x_flat.device())
                .map_err(|e| anyhow!("moe output zeros -> {e}"))?;

        // Dispatch: for each active expert, gather tokens → FFN → scatter back.
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

            let hidden = crate::utils::fused_ops::silu_mul(
                &gate_out.contiguous().map_err(|e| anyhow!("moe gate contig -> {e}"))?,
                &up_out.contiguous().map_err(|e| anyhow!("moe up contig -> {e}"))?,
            )
            .map_err(|e| anyhow!("moe silu_mul -> {e}"))?;

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
            let expert_out = (expert_out * w_t)
                .map_err(|e| anyhow!("moe expert_out * weight -> {e}"))?;

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
