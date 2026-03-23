//! Sparse Mixture-of-Experts FFN for Qwen3 MoE.
//!
//! Each layer has `num_experts` experts (SwiGLU FFNs with `moe_intermediate_size`
//! hidden dim) and a router that selects the top-`num_experts_per_tok` experts per
//! token. Expert weights are accessed via an [`ExpertProvider`] trait for flexible
//! storage (in-memory, disk-streamed, etc.).
//!
//! Router forward (matches HuggingFace Qwen3MoeTopKRouter exactly):
//!   1. logits  = x @ gate.T             — (n_tok, num_experts)
//!   2. probs   = softmax(logits)         — normalise across all experts first
//!   3. top_k   = argsort(probs, desc)[:k] — select k best
//!   4. weights = probs[top_k] / sum(probs[top_k])   — renormalise (norm_topk_prob=true)

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;
use crate::models::common::expert_provider::{SharedExpertProvider, StackedResidentProvider};
use crate::models::common::Config;

/// Sparse MoE FFN block.
#[derive(Debug)]
pub struct SparseMoeMlp {
    /// Router weight: (num_experts, hidden_size).
    gate_weight: Tensor,
    /// Expert weight provider (resident or disk-backed).
    expert_provider: SharedExpertProvider,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    backend: Arc<dyn ComputeBackend>,
}

impl SparseMoeMlp {
    pub fn load(vb: VarBuilder, cfg: &Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.moe_intermediate_size.expect("moe_intermediate_size must be set");
        let n = cfg.num_experts;

        // Router: single linear, no bias.
        let gate_weight = vb.pp("gate").get((n, h), "weight")?;

        // Load all expert weights and stack into batched tensors.
        let mut gate_ws = Vec::with_capacity(n);
        let mut up_ws = Vec::with_capacity(n);
        let mut down_ws = Vec::with_capacity(n);

        for j in 0..n {
            let exp = vb.pp("experts").pp(j.to_string());
            gate_ws.push(exp.pp("gate_proj").get((i, h), "weight")?);
            up_ws.push(exp.pp("up_proj").get((i, h), "weight")?);
            down_ws.push(exp.pp("down_proj").get((h, i), "weight")?);
        }

        let gate_proj = Tensor::stack(&gate_ws, 0)?;
        let up_proj = Tensor::stack(&up_ws, 0)?;
        let down_proj = Tensor::stack(&down_ws, 0)?;

        let expert_provider: SharedExpertProvider =
            Arc::new(StackedResidentProvider::new(gate_proj, up_proj, down_proj, n));

        Ok(Self {
            gate_weight,
            expert_provider,
            num_experts: n,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            backend,
        })
    }

    /// Construct with a pre-built expert provider (for disk offloading).
    #[allow(clippy::too_many_arguments)]
    pub fn with_provider(
        gate_weight: Tensor,
        expert_provider: SharedExpertProvider,
        num_experts: usize,
        num_experts_per_tok: usize,
        norm_topk_prob: bool,
        backend: Arc<dyn ComputeBackend>,
    ) -> Self {
        Self {
            gate_weight,
            expert_provider,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
            backend,
        }
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, s, h) = x.dims3().map_err(|e| anyhow!("moe x.dims3 -> {e}"))?;
        let n_tok = b * s;
        let x_flat = x
            .reshape((n_tok, h))
            .map_err(|e| anyhow!("moe reshape -> {e}"))?;

        // --- Router ---
        let router_logits = self
            .backend
            .linear_forward(&x_flat, &self.gate_weight, None)
            .map_err(|e| anyhow!("moe router -> {e}"))?;

        let last_dim = router_logits.rank() - 1;
        let router_probs = self.backend.softmax(&router_logits, last_dim)
            .map_err(|e| anyhow!("moe softmax -> {e}"))?;

        let (_, top_k_idx) = self.backend.topk(&router_probs, self.num_experts_per_tok)
            .map_err(|e| anyhow!("moe topk -> {e}"))?;

        let top_k_w = router_probs
            .contiguous()
            .map_err(|e| anyhow!("moe probs contiguous -> {e}"))?
            .gather(&top_k_idx.contiguous().map_err(|e| anyhow!("moe idx contiguous -> {e}"))?, D::Minus1)
            .map_err(|e| anyhow!("moe gather weights -> {e}"))?;

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

        // --- Fast path for single-token generation ---
        // If provider is StackedResidentProvider, use batched index_select for efficiency.
        // Otherwise, fall through to the per-expert path.
        if n_tok == 1 {
            if let Some(stacked) = self
                .expert_provider
                .as_any()
                .and_then(|a| a.downcast_ref::<StackedResidentProvider>())
            {
                return self.forward_batched_fast_path(
                    &x_flat, &top_k_idx, &top_k_w, stacked, b, s, h,
                );
            }
            // Non-stacked provider: use per-expert path below
        }

        // --- Per-expert dispatch (works with any ExpertProvider) ---
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

        for (exp_idx, tokens) in expert_tokens.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            let tok_ids: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
            let weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();

            let idx = Tensor::new(tok_ids.as_slice(), x_flat.device())
                .map_err(|e| anyhow!("moe idx tensor -> {e}"))?;

            let selected = x_flat
                .index_select(&idx, 0)
                .map_err(|e| anyhow!("moe index_select -> {e}"))?;

            // Get expert weights from provider
            let ew = self
                .expert_provider
                .get_expert(exp_idx)
                .map_err(|e| anyhow!("moe get_expert({exp_idx}) -> {e}"))?;

            let gate_out = selected
                .matmul(&ew.gate_proj.t().map_err(|e| anyhow!("moe gp.t -> {e}"))?)
                .map_err(|e| anyhow!("moe gate matmul -> {e}"))?;
            let up_out = selected
                .matmul(&ew.up_proj.t().map_err(|e| anyhow!("moe up.t -> {e}"))?)
                .map_err(|e| anyhow!("moe up matmul -> {e}"))?;

            let hidden = self.backend.silu_mul(
                &gate_out,
                &up_out,
            )
            .map_err(|e| anyhow!("moe silu_mul -> {e}"))?;

            let expert_out = hidden
                .matmul(&ew.down_proj.t().map_err(|e| anyhow!("moe dp.t -> {e}"))?)
                .map_err(|e| anyhow!("moe down matmul -> {e}"))?;

            let w_t = Tensor::new(weights.as_slice(), x_flat.device())
                .map_err(|e| anyhow!("moe weight tensor -> {e}"))?
                .to_dtype(compute_dtype)
                .map_err(|e| anyhow!("moe weight to_dtype -> {e}"))?
                .unsqueeze(1)
                .map_err(|e| anyhow!("moe weight unsqueeze -> {e}"))?;
            let expert_out = expert_out.broadcast_mul(&w_t)
                .map_err(|e| anyhow!("moe expert_out * weight -> {e}"))?;

            output = output
                .index_add(&idx, &expert_out, 0)
                .map_err(|e| anyhow!("moe index_add -> {e}"))?;
        }

        output
            .reshape((b, s, h))
            .map_err(|e| anyhow!("moe output reshape -> {e}"))
    }

    /// Batched fast path using stacked tensors + index_select.
    #[allow(clippy::too_many_arguments)]
    /// Only used when the provider is StackedResidentProvider (all experts in RAM).
    fn forward_batched_fast_path(
        &self,
        x_flat: &Tensor,
        top_k_idx: &Tensor,
        top_k_w: &Tensor,
        stacked: &StackedResidentProvider,
        b: usize,
        s: usize,
        h: usize,
    ) -> anyhow::Result<Tensor> {
        let k = self.num_experts_per_tok;
        let expert_indices = top_k_idx
            .squeeze(0)
            .map_err(|e| anyhow!("moe squeeze idx -> {e}"))?;

        let sel_gate = stacked
            .gate_proj()
            .index_select(&expert_indices, 0)
            .map_err(|e| anyhow!("moe sel gate_proj -> {e}"))?;
        let sel_up = stacked
            .up_proj()
            .index_select(&expert_indices, 0)
            .map_err(|e| anyhow!("moe sel up_proj -> {e}"))?;
        let sel_down = stacked
            .down_proj()
            .index_select(&expert_indices, 0)
            .map_err(|e| anyhow!("moe sel down_proj -> {e}"))?;

        let x_exp = x_flat
            .unsqueeze(0)
            .map_err(|e| anyhow!("moe x unsqueeze -> {e}"))?
            .expand((k, 1, h))
            .map_err(|e| anyhow!("moe x expand -> {e}"))?;

        let gate_out = x_exp
            .matmul(&sel_gate.t().map_err(|e| anyhow!("moe sel gp.t -> {e}"))?)
            .map_err(|e| anyhow!("moe batched gate matmul -> {e}"))?;
        let up_out = x_exp
            .matmul(&sel_up.t().map_err(|e| anyhow!("moe sel up.t -> {e}"))?)
            .map_err(|e| anyhow!("moe batched up matmul -> {e}"))?;

        let hidden = self.backend.silu_mul(
            &gate_out,
            &up_out,
        )
        .map_err(|e| anyhow!("moe silu_mul -> {e}"))?;

        let expert_outs = hidden
            .matmul(&sel_down.t().map_err(|e| anyhow!("moe sel dp.t -> {e}"))?)
            .map_err(|e| anyhow!("moe batched down matmul -> {e}"))?;

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

        output
            .reshape((b, s, h))
            .map_err(|e| anyhow!("moe output reshape -> {e}"))
    }
}
