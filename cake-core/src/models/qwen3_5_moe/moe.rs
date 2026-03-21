//! Sparse MoE FFN for Qwen3.5 MoE.
//!
//! Per-expert weights are stored as individual linear layers:
//!   mlp.experts.{j}.gate_proj.weight  (moe_intermediate_size, hidden_size)
//!   mlp.experts.{j}.up_proj.weight    (moe_intermediate_size, hidden_size)
//!   mlp.experts.{j}.down_proj.weight  (hidden_size, moe_intermediate_size)
//!
//! When loaded from a GPTQ-Int4 model, the GPTQ backend transparently
//! dequantizes each weight matrix (intercepting `*.weight` → `*.qweight`).
//!
//! A shared (always-active) expert is added to the routed output:
//!   output = routed_output + sigmoid(shared_expert_gate(x)) * shared_expert(x)
//!
//! Router (same as Qwen3 MoE — matches HF Qwen3_5MoeTopKRouter):
//!   logits  = x @ gate.T
//!   probs   = softmax(logits)
//!   top_k   = argsort(probs, desc)[:k]
//!   weights = probs[top_k] / sum(probs[top_k])   (norm_topk_prob=true)

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias as linear, ops::softmax_last_dim, Linear, Module, VarBuilder};

use crate::backends::ComputeBackend;
use crate::models::common::Config;

/// Sparse MoE FFN block with shared expert (Qwen3.5 MoE).
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeSparseMlp {
    /// Router: (num_experts, hidden_size)
    gate: Linear,
    /// Per-expert gate projections (hidden_size → moe_intermediate_size)
    experts_gate: Vec<Linear>,
    /// Per-expert up projections (hidden_size → moe_intermediate_size)
    experts_up: Vec<Linear>,
    /// Per-expert down projections (moe_intermediate_size → hidden_size)
    experts_down: Vec<Linear>,
    /// Shared expert gate projection (hidden_size → shared_intermediate_size)
    shared_gate_proj: Linear,
    /// Shared expert up projection (hidden_size → shared_intermediate_size)
    shared_up_proj: Linear,
    /// Shared expert down projection (shared_intermediate_size → hidden_size)
    shared_down_proj: Linear,
    /// Scalar sigmoid gate for the shared expert output: (1, hidden_size)
    shared_expert_gate: Linear,

    num_experts: usize,
    num_experts_per_tok: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl Qwen3_5MoeSparseMlp {
    pub fn load(vb: VarBuilder, cfg: &Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.moe_intermediate_size.expect("moe_intermediate_size required");
        let si = cfg.shared_expert_intermediate_size.expect("shared_expert_intermediate_size required");
        let n = cfg.num_experts;

        // Router
        let gate_w = vb.pp("gate").get((n, h), "weight")?;
        let gate = Linear::new(gate_w, None);

        // Per-expert projections — GPTQ backend transparently dequantizes
        // each *.weight request by looking for *.qweight + *.scales + *.qzeros.
        let mut experts_gate = Vec::with_capacity(n);
        let mut experts_up = Vec::with_capacity(n);
        let mut experts_down = Vec::with_capacity(n);
        let experts_vb = vb.pp("experts");
        for j in 0..n {
            let evb = experts_vb.pp(j.to_string());
            experts_gate.push(linear(h, i, evb.pp("gate_proj"))?);
            experts_up.push(linear(h, i, evb.pp("up_proj"))?);
            experts_down.push(linear(i, h, evb.pp("down_proj"))?);
        }

        // Shared expert (standard SwiGLU MLP, not quantized)
        let se = vb.pp("shared_expert");
        let shared_gate_proj = linear(h, si, se.pp("gate_proj"))?;
        let shared_up_proj = linear(h, si, se.pp("up_proj"))?;
        let shared_down_proj = linear(si, h, se.pp("down_proj"))?;

        // Scalar sigmoid gate for the shared expert contribution
        let shared_expert_gate_w = vb.pp("shared_expert_gate").get((1, h), "weight")?;
        let shared_expert_gate = Linear::new(shared_expert_gate_w, None);

        Ok(Self {
            gate,
            experts_gate,
            experts_up,
            experts_down,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            shared_expert_gate,
            num_experts: n,
            num_experts_per_tok: cfg.num_experts_per_tok,
            backend,
        })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, s, h) = x.dims3().map_err(|e| anyhow!("moe dims3: {e}"))?;
        let n_tok = b * s;
        let x_flat = x
            .reshape((n_tok, h))
            .map_err(|e| anyhow!("moe reshape: {e}"))?;

        // --- Shared expert (always computed) ---
        let shared_out = {
            let gate = self
                .shared_gate_proj
                .forward(&x_flat)
                .map_err(|e| anyhow!("shared gate_proj: {e}"))?;
            let up = self
                .shared_up_proj
                .forward(&x_flat)
                .map_err(|e| anyhow!("shared up_proj: {e}"))?;
            let hidden = self.backend.silu_mul(
                &gate.contiguous().map_err(|e| anyhow!("shared gate contig: {e}"))?,
                &up.contiguous().map_err(|e| anyhow!("shared up contig: {e}"))?,
            )
            .map_err(|e| anyhow!("shared silu_mul: {e}"))?;
            self.shared_down_proj
                .forward(&hidden)
                .map_err(|e| anyhow!("shared down_proj: {e}"))?
        };

        // sigmoid gate on shared expert output: (n_tok, 1) broadcast to (n_tok, h)
        let gate_scalar = self
            .shared_expert_gate
            .forward(&x_flat)
            .map_err(|e| anyhow!("shared_expert_gate: {e}"))?; // (n_tok, 1)
        let gate_scalar = candle_nn::ops::sigmoid(&gate_scalar)
            .map_err(|e| anyhow!("shared gate sigmoid: {e}"))?;
        let shared_out = shared_out
            .broadcast_mul(&gate_scalar)
            .map_err(|e| anyhow!("shared gate mul: {e}"))?;

        // --- Router ---
        let router_logits = self
            .gate
            .forward(&x_flat)
            .map_err(|e| anyhow!("router: {e}"))?;
        let router_probs = softmax_last_dim(&router_logits)
            .map_err(|e| anyhow!("router softmax: {e}"))?;

        let (_, sorted_idx) = router_probs
            .sort_last_dim(false)
            .map_err(|e| anyhow!("router sort: {e}"))?;
        let top_k_idx = sorted_idx
            .narrow(D::Minus1, 0, self.num_experts_per_tok)
            .map_err(|e| anyhow!("router topk narrow: {e}"))?; // (n_tok, k)

        let top_k_w = router_probs
            .contiguous()
            .map_err(|e| anyhow!("router probs contiguous: {e}"))?
            .gather(
                &top_k_idx
                    .contiguous()
                    .map_err(|e| anyhow!("router idx contiguous: {e}"))?,
                D::Minus1,
            )
            .map_err(|e| anyhow!("router gather: {e}"))?;

        // Renormalise
        let top_k_w = {
            let s = top_k_w
                .sum_keepdim(D::Minus1)
                .map_err(|e| anyhow!("router sum: {e}"))?;
            top_k_w
                .broadcast_div(&s)
                .map_err(|e| anyhow!("router div: {e}"))?
        };

        // Fast path for single-token generation: iterate directly over k
        // selected experts instead of building a dispatch table for all experts.
        if n_tok == 1 {
            let k = self.num_experts_per_tok;
            let compute_dtype = x_flat.dtype();

            let expert_indices: Vec<u32> = top_k_idx
                .squeeze(0)
                .map_err(|e| anyhow!("idx squeeze: {e}"))?
                .to_dtype(DType::U32)
                .map_err(|e| anyhow!("idx u32: {e}"))?
                .to_vec1::<u32>()
                .map_err(|e| anyhow!("idx to_vec: {e}"))?;

            let expert_weights: Vec<f32> = top_k_w
                .squeeze(0)
                .map_err(|e| anyhow!("w squeeze: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("w f32: {e}"))?
                .to_vec1::<f32>()
                .map_err(|e| anyhow!("w to_vec: {e}"))?;

            let mut output = Tensor::zeros((1, h), compute_dtype, x_flat.device())
                .map_err(|e| anyhow!("output zeros: {e}"))?;

            for idx in 0..k {
                let exp = expert_indices[idx] as usize;
                let w = expert_weights[idx];

                let gate_out = self.experts_gate[exp]
                    .forward(&x_flat)
                    .map_err(|e| anyhow!("expert gate: {e}"))?;
                let up_out = self.experts_up[exp]
                    .forward(&x_flat)
                    .map_err(|e| anyhow!("expert up: {e}"))?;
                let hidden = self.backend.silu_mul(
                    &gate_out.contiguous().map_err(|e| anyhow!("gate contig: {e}"))?,
                    &up_out.contiguous().map_err(|e| anyhow!("up contig: {e}"))?,
                )
                .map_err(|e| anyhow!("silu_mul: {e}"))?;
                let expert_out = self.experts_down[exp]
                    .forward(&hidden)
                    .map_err(|e| anyhow!("expert down: {e}"))?;

                output = (output + (expert_out * w as f64)
                    .map_err(|e| anyhow!("scale: {e}"))?)
                    .map_err(|e| anyhow!("accumulate: {e}"))?;
            }

            let output = (output + &shared_out)
                .map_err(|e| anyhow!("add shared: {e}"))?;
            return output
                .reshape((b, s, h))
                .map_err(|e| anyhow!("output reshape: {e}"));
        }

        // CPU dispatch table
        let top_k_idx_flat: Vec<u32> = top_k_idx
            .to_dtype(DType::U32)
            .map_err(|e| anyhow!("idx u32: {e}"))?
            .flatten_all()
            .map_err(|e| anyhow!("idx flatten: {e}"))?
            .to_vec1::<u32>()
            .map_err(|e| anyhow!("idx to_vec: {e}"))?;

        let top_k_w_flat: Vec<f32> = top_k_w
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("w f32: {e}"))?
            .flatten_all()
            .map_err(|e| anyhow!("w flatten: {e}"))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("w to_vec: {e}"))?;

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
            .map_err(|e| anyhow!("output zeros: {e}"))?;

        // Dispatch: gather → expert FFN → weighted scatter-add
        for (exp_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let tok_ids: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
            let weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();

            let idx = Tensor::new(tok_ids.as_slice(), x_flat.device())
                .map_err(|e| anyhow!("idx tensor: {e}"))?;

            let selected = x_flat
                .index_select(&idx, 0)
                .map_err(|e| anyhow!("index_select: {e}"))?; // (n_sel, h)

            let gate_out = self.experts_gate[exp_idx]
                .forward(&selected)
                .map_err(|e| anyhow!("expert gate: {e}"))?; // (n_sel, i)
            let up_out = self.experts_up[exp_idx]
                .forward(&selected)
                .map_err(|e| anyhow!("expert up: {e}"))?; // (n_sel, i)

            let hidden = self.backend.silu_mul(
                &gate_out.contiguous().map_err(|e| anyhow!("gate contig: {e}"))?,
                &up_out.contiguous().map_err(|e| anyhow!("up contig: {e}"))?,
            )
            .map_err(|e| anyhow!("silu_mul: {e}"))?;

            let expert_out = self.experts_down[exp_idx]
                .forward(&hidden)
                .map_err(|e| anyhow!("expert down: {e}"))?; // (n_sel, h)

            // Scale by routing weight
            let w_t = Tensor::new(weights.as_slice(), x_flat.device())
                .map_err(|e| anyhow!("w tensor: {e}"))?
                .to_dtype(compute_dtype)
                .map_err(|e| anyhow!("w to_dtype: {e}"))?
                .unsqueeze(1)
                .map_err(|e| anyhow!("w unsqueeze: {e}"))?
                .broadcast_as(expert_out.shape())
                .map_err(|e| anyhow!("w broadcast: {e}"))?;
            let expert_out = (expert_out * w_t)
                .map_err(|e| anyhow!("expert_out * w: {e}"))?;

            output = output
                .index_add(&idx, &expert_out, 0)
                .map_err(|e| anyhow!("index_add: {e}"))?;
        }

        // Add shared expert output
        let output = (output + shared_out)
            .map_err(|e| anyhow!("add shared: {e}"))?;

        output
            .reshape((b, s, h))
            .map_err(|e| anyhow!("output reshape: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let numel: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn test_config() -> Config {
        Config {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            tie_word_embeddings: false,
            max_seq_len: 64,
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: None,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
            use_qk_norm: false,
            pre_reshape_qk_norm: false,
            sliding_window: None,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            global_layers: vec![],
            use_gelu_mlp: false,
            embed_scale: None,
            moe_intermediate_size: Some(32),
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            shared_expert_intermediate_size: Some(48),
            attn_output_gate: false,
        }
    }

    fn make_vb() -> VarBuilder<'static> {
        let cfg = test_config();
        let h = cfg.hidden_size;
        let i = cfg.moe_intermediate_size.unwrap();
        let si = cfg.shared_expert_intermediate_size.unwrap();
        let n = cfg.num_experts;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("gate.weight".into(), make_tensor(&[n, h], 40));
        for j in 0..n {
            map.insert(format!("experts.{j}.gate_proj.weight"), make_tensor(&[i, h], 41 + j as u64 * 3));
            map.insert(format!("experts.{j}.up_proj.weight"), make_tensor(&[i, h], 42 + j as u64 * 3));
            map.insert(format!("experts.{j}.down_proj.weight"), make_tensor(&[h, i], 43 + j as u64 * 3));
        }
        map.insert("shared_expert.gate_proj.weight".into(), make_tensor(&[si, h], 60));
        map.insert("shared_expert.up_proj.weight".into(), make_tensor(&[si, h], 61));
        map.insert("shared_expert.down_proj.weight".into(), make_tensor(&[h, si], 62));
        map.insert("shared_expert_gate.weight".into(), make_tensor(&[1, h], 63));

        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn test_forward_shape() {
        let cfg = test_config();
        let vb = make_vb();
        let moe = Qwen3_5MoeSparseMlp::load(vb, &cfg, Arc::new(crate::backends::CpuBackend::new())).unwrap();
        let x = make_tensor(&[1, 4, 64], 80);
        let y = moe.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_forward_single_token() {
        let cfg = test_config();
        let vb = make_vb();
        let moe = Qwen3_5MoeSparseMlp::load(vb, &cfg, Arc::new(crate::backends::CpuBackend::new())).unwrap();
        let x = make_tensor(&[1, 1, 64], 81);
        let y = moe.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_forward_nonzero() {
        let cfg = test_config();
        let vb = make_vb();
        let moe = Qwen3_5MoeSparseMlp::load(vb, &cfg, Arc::new(crate::backends::CpuBackend::new())).unwrap();
        let x = make_tensor(&[1, 4, 64], 82);
        let y = moe.forward(&x).unwrap();
        let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().any(|v| v.abs() > 1e-10), "output should not be all zeros");
    }

    #[test]
    fn test_deterministic() {
        let cfg = test_config();
        let vb1 = make_vb();
        let vb2 = make_vb();
        let moe1 = Qwen3_5MoeSparseMlp::load(vb1, &cfg, Arc::new(crate::backends::CpuBackend::new())).unwrap();
        let moe2 = Qwen3_5MoeSparseMlp::load(vb2, &cfg, Arc::new(crate::backends::CpuBackend::new())).unwrap();
        let x = make_tensor(&[1, 4, 64], 83);
        let y1: Vec<f32> = moe1.forward(&x).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let y2: Vec<f32> = moe2.forward(&x).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(y1, y2);
    }
}
