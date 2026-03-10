//! Sparse MoE FFN for Qwen3.5 MoE.
//!
//! Expert weights are stored as batched 3D tensors (not per-expert):
//!   mlp.experts.gate_up_proj  (num_experts, 2*moe_intermediate_size, hidden_size)
//!   mlp.experts.down_proj     (num_experts, hidden_size, moe_intermediate_size)
//!
//! A shared (always-active) expert is added to the routed output:
//!   output = routed_output + sigmoid(shared_expert_gate(x)) * shared_expert(x)
//!
//! Router (same as Qwen3 MoE — matches HF Qwen3_5MoeTopKRouter):
//!   logits  = x @ gate.T
//!   probs   = softmax(logits)
//!   top_k   = argsort(probs, desc)[:k]
//!   weights = probs[top_k] / sum(probs[top_k])   (norm_topk_prob=true)

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias as linear, ops::softmax_last_dim, Linear, Module, VarBuilder};

use crate::models::common::Config;

/// Sparse MoE FFN block with shared expert (Qwen3.5 MoE).
#[derive(Debug, Clone)]
pub struct Qwen3_5MoeSparseMlp {
    /// Router: (num_experts, hidden_size)
    gate: Linear,
    /// Batched fused gate+up expert projections: (num_experts, 2*moe_intermediate_size, hidden_size)
    experts_gate_up: Tensor,
    /// Batched expert down projections: (num_experts, hidden_size, moe_intermediate_size)
    experts_down: Tensor,
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
    moe_intermediate_size: usize,
}

impl Qwen3_5MoeSparseMlp {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.moe_intermediate_size.expect("moe_intermediate_size required");
        let si = cfg.shared_expert_intermediate_size.expect("shared_expert_intermediate_size required");
        let n = cfg.num_experts;

        // Router
        let gate_w = vb.pp("gate").get((n, h), "weight")?;
        let gate = Linear::new(gate_w, None);

        // Batched expert projections stored as 3-D tensors
        let experts_gate_up = vb.pp("experts").get((n, 2 * i, h), "gate_up_proj")?;
        let experts_down = vb.pp("experts").get((n, h, i), "down_proj")?;

        // Shared expert (standard SwiGLU MLP)
        let se = vb.pp("shared_expert");
        let shared_gate_proj = linear(h, si, se.pp("gate_proj"))?;
        let shared_up_proj = linear(h, si, se.pp("up_proj"))?;
        let shared_down_proj = linear(si, h, se.pp("down_proj"))?;

        // Scalar sigmoid gate for the shared expert contribution
        let shared_expert_gate_w = vb.pp("shared_expert_gate").get((1, h), "weight")?;
        let shared_expert_gate = Linear::new(shared_expert_gate_w, None);

        Ok(Self {
            gate,
            experts_gate_up,
            experts_down,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            shared_expert_gate,
            num_experts: n,
            num_experts_per_tok: cfg.num_experts_per_tok,
            moe_intermediate_size: i,
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
            let hidden = (candle_nn::ops::silu(&gate)
                .map_err(|e| anyhow!("shared silu: {e}"))?
                * up)
                .map_err(|e| anyhow!("shared gate*up: {e}"))?;
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
            .gather(&top_k_idx, D::Minus1)
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

        let i = self.moe_intermediate_size;

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

            // Fetch this expert's fused gate+up: (2*i, h) → split into gate (i,h) and up (i,h)
            let gate_up = self
                .experts_gate_up
                .get(exp_idx)
                .map_err(|e| anyhow!("gate_up.get({exp_idx}): {e}"))?; // (2*i, h)
            let gate_w = gate_up
                .narrow(0, 0, i)
                .map_err(|e| anyhow!("gate narrow: {e}"))?; // (i, h)
            let up_w = gate_up
                .narrow(0, i, i)
                .map_err(|e| anyhow!("up narrow: {e}"))?; // (i, h)
            let down_w = self
                .experts_down
                .get(exp_idx)
                .map_err(|e| anyhow!("down.get({exp_idx}): {e}"))?; // (h, i)

            let gate_out = selected
                .matmul(&gate_w.t().map_err(|e| anyhow!("gate_w.t: {e}"))?)
                .map_err(|e| anyhow!("gate matmul: {e}"))?;
            let up_out = selected
                .matmul(&up_w.t().map_err(|e| anyhow!("up_w.t: {e}"))?)
                .map_err(|e| anyhow!("up matmul: {e}"))?;

            let hidden = (candle_nn::ops::silu(&gate_out)
                .map_err(|e| anyhow!("silu: {e}"))?
                * up_out)
                .map_err(|e| anyhow!("gate*up: {e}"))?;

            let expert_out = hidden
                .matmul(&down_w.t().map_err(|e| anyhow!("down_w.t: {e}"))?)
                .map_err(|e| anyhow!("down matmul: {e}"))?; // (n_sel, h)

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
