//! Qwen3.5 MoE transformer block.
//!
//! Identical structure to dense Qwen3.5 (hybrid linear/full attention, pre-norm,
//! residual-RMSNorm) except the FFN is a sparse Mixture-of-Experts layer with
//! a shared always-active expert.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm};

use crate::cake::{Context, Forwarder};
use crate::models::common::load_rms_norm;
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
        rms_1: RmsNorm,
        attn: GatedDeltaNet,
        rms_2: RmsNorm,
        moe: Qwen3_5MoeSparseMlp,
    },
    Full {
        name: String,
        rms_1: RmsNorm,
        attn: Qwen3_5FullAttention,
        rms_2: RmsNorm,
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

        let eps = cfg.rms_norm_eps;
        let h = cfg.hidden_size;
        let rms_1 = load_rms_norm(h, eps, cfg.residual_rms_norm, vb.pp("input_layernorm"))?;
        let rms_2 =
            load_rms_norm(h, eps, cfg.residual_rms_norm, vb.pp("post_attention_layernorm"))?;
        let moe = Qwen3_5MoeSparseMlp::load(vb.pp("mlp"), cfg)?;

        if layer_type == "full_attention" {
            let attn = Qwen3_5FullAttention::load(vb.pp("self_attn"), cfg)?;
            Ok(Box::new(Qwen3_5MoeBlock::Full {
                name,
                rms_1,
                attn,
                rms_2,
                moe,
            }))
        } else {
            let attn = GatedDeltaNet::load(vb.pp("linear_attn"), cfg)?;
            Ok(Box::new(Qwen3_5MoeBlock::Linear {
                name,
                rms_1,
                attn,
                rms_2,
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
                rms_1, attn, rms_2, moe, ..
            } => {
                let residual = x;
                let x = rms_1.forward(x).map_err(|e| anyhow!("rms_1: {e}"))?;
                let x = (attn.forward(
                    &x,
                    block_idx,
                    ctx.cache.as_mut().expect("No cache"),
                )? + residual)
                    .map_err(|e| anyhow!("attn residual: {e}"))?;
                let residual = &x;
                let x = rms_2.forward(&x).map_err(|e| anyhow!("rms_2: {e}"))?;
                let moe_out = moe.forward(&x)?;
                let x = (moe_out + residual).map_err(|e| anyhow!("moe residual: {e}"))?;
                Ok(x)
            }
            Qwen3_5MoeBlock::Full {
                rms_1, attn, rms_2, moe, ..
            } => {
                let residual = x;
                let x = rms_1.forward(x).map_err(|e| anyhow!("rms_1: {e}"))?;
                let x = (attn.forward(
                    &x,
                    index_pos,
                    block_idx,
                    ctx.cache.as_mut().expect("No cache"),
                )? + residual)
                    .map_err(|e| anyhow!("attn residual: {e}"))?;
                let residual = &x;
                let x = rms_2.forward(&x).map_err(|e| anyhow!("rms_2: {e}"))?;
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
