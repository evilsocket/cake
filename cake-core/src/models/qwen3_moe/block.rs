//! Qwen3 MoE transformer block.
//!
//! Identical structure to dense Qwen3 (pre-norm, standard residual connections,
//! QK-norm, full-context GQA attention) except the FFN is a sparse
//! Mixture-of-Experts layer instead of a dense MLP.

use anyhow::Result;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use crate::models::common::CausalSelfAttention;
use async_trait::async_trait;

use super::moe::SparseMoeMlp;

/// A single Qwen3 MoE transformer layer.
#[derive(Debug)]
pub struct Qwen3MoeBlock {
    name: String,
    input_layernorm_weight: Tensor,
    post_attention_layernorm_weight: Tensor,
    rms_eps: f32,
    attn: CausalSelfAttention,
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
                    ctx.quant.gptq_group_size(),
                ));
            // Load router gate from VarBuilder (it's small, stays in RAM)
            let gate_weight = vb.pp("mlp").pp("gate").get((cfg.num_experts, cfg.hidden_size), "weight")?;
            SparseMoeMlp::with_provider(
                gate_weight, provider, cfg.num_experts, cfg.num_experts_per_tok,
                cfg.norm_topk_prob, ctx.backend.clone(),
            )
        } else {
            SparseMoeMlp::load(vb.pp("mlp"), cfg, ctx.backend.clone())?
        };

        let h = cfg.hidden_size;
        let input_layernorm_weight = vb.pp("input_layernorm").get(h, "weight")?;
        let post_attention_layernorm_weight =
            vb.pp("post_attention_layernorm").get(h, "weight")?;
        let rms_eps = cfg.rms_norm_eps as f32;

        Ok(Box::new(Self {
            name,
            input_layernorm_weight,
            attn,
            post_attention_layernorm_weight,
            rms_eps,
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
        let x = ctx.backend.rms_norm(x, &self.input_layernorm_weight, self.rms_eps)
            .map_err(|e| anyhow!("input_layernorm: {e}"))?;
        let x = self
            .attn
            .forward(&x, index_pos, block_idx, ctx.cache.as_mut().expect("No cache"))
            .map_err(|e| anyhow!("attn: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("attn residual: {e}"))?;

        // MoE sublayer (pre-norm, standard residual).
        let residual = &x;
        let x = ctx.backend.rms_norm(&x, &self.post_attention_layernorm_weight, self.rms_eps)
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
