//! Qwen3.5 MoE transformer block.
//!
//! Identical structure to dense Qwen3.5 (hybrid linear/full attention, pre-norm,
//! residual-RMSNorm) except the FFN is a sparse Mixture-of-Experts layer with
//! a shared always-active expert.

use anyhow::Result;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use crate::models::common::load_rms_norm_weight;
use async_trait::async_trait;

use super::moe::Qwen3_5MoeSparseMlp;

// Reuse the dense Qwen3.5 attention implementations
use crate::models::qwen3_5::{
    full_attention::Qwen3_5FullAttention, linear_attention::GatedDeltaNet,
};

#[derive(Debug, Clone)]
pub enum Qwen3_5MoeBlock {
    Linear {
        name: String,
        rms_1_weight: Tensor,
        rms_2_weight: Tensor,
        rms_eps: f32,
        attn: GatedDeltaNet,
        moe: Qwen3_5MoeSparseMlp,
    },
    Full {
        name: String,
        rms_1_weight: Tensor,
        rms_2_weight: Tensor,
        rms_eps: f32,
        attn: Qwen3_5FullAttention,
        moe: Qwen3_5MoeSparseMlp,
    },
}

impl std::fmt::Display for Qwen3_5MoeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen3_5MoeBlock::Linear { name, .. } => write!(f, "{name} (qwen3.5-moe, linear_attn)"),
            Qwen3_5MoeBlock::Full { name, .. } => write!(f, "{name} (qwen3.5-moe, full_attn)"),
        }
    }
}

impl Qwen3_5MoeBlock {
    fn layer_index(name: &str) -> usize {
        name.rsplit('.')
            .next()
            .and_then(|s| s.parse().ok())
            .expect("invalid layer name — no trailing index")
    }
}

#[async_trait]
impl Forwarder for Qwen3_5MoeBlock {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config");

        let layer_idx = Self::layer_index(&name);
        let la = cfg
            .linear_attn
            .as_ref()
            .expect("no linear_attn config for Qwen3.5-MoE");

        let layer_type = la
            .layer_types
            .get(layer_idx)
            .map(|s| s.as_str())
            .unwrap_or("linear_attention");

        let h = cfg.hidden_size;
        let rms_1_weight = load_rms_norm_weight(h, cfg.residual_rms_norm, vb.pp("input_layernorm"))?;
        let rms_2_weight =
            load_rms_norm_weight(h, cfg.residual_rms_norm, vb.pp("post_attention_layernorm"))?;
        let rms_eps = cfg.rms_norm_eps as f32;
        let moe = if let Some(storage) = &ctx.tensor_storage {
            // Expert offload: stream routed expert weights from disk
            let layer_prefix = format!("{name}.mlp");
            let provider: std::sync::Arc<dyn crate::models::common::expert_provider::ExpertProvider> =
                std::sync::Arc::new(crate::models::common::disk_expert_provider::DiskExpertProvider::new(
                    storage.clone(), layer_prefix, cfg.num_experts, ctx.device.clone(), ctx.dtype,
                ));
            let mlp_vb = vb.pp("mlp");
            let gate_weight = mlp_vb.pp("gate").get((cfg.num_experts, h), "weight")?;
            let si = cfg.shared_expert_intermediate_size.expect("shared_expert_intermediate_size");
            let se = mlp_vb.pp("shared_expert");
            let shared_gate_proj_weight = se.pp("gate_proj").get((si, h), "weight")?;
            let shared_up_proj_weight = se.pp("up_proj").get((si, h), "weight")?;
            let shared_down_proj_weight = se.pp("down_proj").get((h, si), "weight")?;
            let shared_expert_gate_weight = mlp_vb.pp("shared_expert_gate").get((1, h), "weight")?;
            Qwen3_5MoeSparseMlp::with_provider(
                gate_weight, provider, shared_gate_proj_weight, shared_up_proj_weight, shared_down_proj_weight,
                shared_expert_gate_weight, cfg.num_experts, cfg.num_experts_per_tok, ctx.backend.clone(),
            )
        } else {
            Qwen3_5MoeSparseMlp::load(vb.pp("mlp"), cfg, ctx.backend.clone())?
        };

        if layer_type == "full_attention" {
            let attn = Qwen3_5FullAttention::load(vb.pp("self_attn"), cfg, ctx.backend.clone())?;
            Ok(Box::new(Qwen3_5MoeBlock::Full {
                name,
                rms_1_weight,
                rms_2_weight,
                rms_eps,
                attn,
                moe,
            }))
        } else {
            let attn = GatedDeltaNet::load(vb.pp("linear_attn"), cfg, ctx.backend.clone())?;
            Ok(Box::new(Qwen3_5MoeBlock::Linear {
                name,
                rms_1_weight,
                rms_2_weight,
                rms_eps,
                attn,
                moe,
            }))
        }
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        match self {
            Qwen3_5MoeBlock::Linear {
                rms_1_weight, rms_2_weight, rms_eps, attn, moe, ..
            } => {
                let residual = x;
                let x = ctx.backend.rms_norm(x, rms_1_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_1: {e}"))?;
                let x = (attn.forward(
                    &x,
                    block_idx,
                    ctx.cache.as_mut().expect("No cache"),
                )? + residual)
                    .map_err(|e| anyhow!("attn residual: {e}"))?;
                let residual = &x;
                let x = ctx.backend.rms_norm(&x, rms_2_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_2: {e}"))?;
                let moe_out = moe.forward(&x)?;
                let x = (moe_out + residual).map_err(|e| anyhow!("moe residual: {e}"))?;
                Ok(x)
            }
            Qwen3_5MoeBlock::Full {
                rms_1_weight, rms_2_weight, rms_eps, attn, moe, ..
            } => {
                let residual = x;
                let x = ctx.backend.rms_norm(x, rms_1_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_1: {e}"))?;
                let x = (attn.forward(
                    &x,
                    index_pos,
                    block_idx,
                    ctx.cache.as_mut().expect("No cache"),
                )? + residual)
                    .map_err(|e| anyhow!("attn residual: {e}"))?;
                let residual = &x;
                let x = ctx.backend.rms_norm(&x, rms_2_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_2: {e}"))?;
                let moe_out = moe.forward(&x)?;
                let x = (moe_out + residual).map_err(|e| anyhow!("moe residual: {e}"))?;
                Ok(x)
            }
        }
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
        match self {
            Qwen3_5MoeBlock::Linear { name, .. } => name,
            Qwen3_5MoeBlock::Full { name, .. } => name,
        }
    }
}
