//! Qwen3 MoE transformer block.
//!
//! Identical structure to dense Qwen3 (pre-norm, standard residual connections,
//! QK-norm, full-context GQA attention) except the FFN is a sparse
//! Mixture-of-Experts layer instead of a dense MLP.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm};

use crate::cake::{Context, Forwarder};
use crate::models::common::{load_rms_norm, CausalSelfAttention};
use async_trait::async_trait;

use super::moe::SparseMoeMlp;

/// A single Qwen3 MoE transformer layer.
#[derive(Debug)]
pub struct Qwen3MoeBlock {
    name: String,
    input_layernorm: RmsNorm,
    attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
    moe: SparseMoeMlp,
}

impl std::fmt::Display for Qwen3MoeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (qwen3-moe)", &self.name)
    }
}

#[async_trait]
impl Forwarder for Qwen3MoeBlock {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, ctx.backend.clone())?;

        let moe = if let Some(storage) = &ctx.tensor_storage {
            // Expert offload: stream weights from disk via DiskProvider
            let layer_prefix = format!("{name}.mlp");
            let provider: std::sync::Arc<dyn crate::models::common::expert_provider::ExpertProvider> =
                std::sync::Arc::new(crate::models::common::disk_expert_provider::DiskExpertProvider::new(
                    storage.clone(),
                    layer_prefix,
                    cfg.num_experts,
                    ctx.device.clone(),
                    ctx.dtype,
                ));
            // Load router gate from VarBuilder (it's small, stays in RAM)
            let gate_w = vb.pp("mlp").pp("gate").get((cfg.num_experts, cfg.hidden_size), "weight")?;
            let gate = candle_nn::Linear::new(gate_w, None);
            SparseMoeMlp::with_provider(
                gate, provider, cfg.num_experts, cfg.num_experts_per_tok,
                cfg.norm_topk_prob, ctx.backend.clone(),
            )
        } else {
            SparseMoeMlp::load(vb.pp("mlp"), cfg, ctx.backend.clone())?
        };

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
