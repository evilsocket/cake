//! Qwen3.5 transformer block — dispatches to linear or full attention based on layer_types.

use anyhow::Result;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use crate::models::common::{load_rms_norm_weight, MLP};
use async_trait::async_trait;

use super::full_attention::Qwen3_5FullAttention;
use super::linear_attention::GatedDeltaNet;

/// A Qwen3.5 block is either a linear attention (DeltaNet) or full attention layer.
#[derive(Debug, Clone)]
pub enum Qwen3_5Block {
    Linear {
        name: String,
        rms_1_weight: Tensor,
        rms_2_weight: Tensor,
        rms_eps: f32,
        attn: GatedDeltaNet,
        mlp: MLP,
    },
    Full {
        name: String,
        rms_1_weight: Tensor,
        rms_2_weight: Tensor,
        rms_eps: f32,
        attn: Qwen3_5FullAttention,
        mlp: MLP,
    },
}

impl std::fmt::Display for Qwen3_5Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen3_5Block::Linear { name, .. } => write!(f, "{} (local, linear_attn)", name),
            Qwen3_5Block::Full { name, .. } => write!(f, "{} (local, full_attn)", name),
        }
    }
}

impl Qwen3_5Block {
    /// Extract the layer index from a block name like "model.language_model.layers.3".
    fn layer_index(name: &str) -> usize {
        name.rsplit('.')
            .next()
            .and_then(|s| s.parse().ok())
            .expect("invalid layer name — no trailing index")
    }
}

#[async_trait]
impl Forwarder for Qwen3_5Block {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let layer_idx = Self::layer_index(&name);
        let la = cfg.linear_attn.as_ref().expect("no linear_attn config for Qwen3.5");

        let layer_type = la.layer_types.get(layer_idx)
            .map(|s| s.as_str())
            .unwrap_or("linear_attention");

        let rms_1_weight = load_rms_norm_weight(
            cfg.hidden_size, cfg.residual_rms_norm,
            vb.pp("input_layernorm"),
        )?;
        let rms_2_weight = load_rms_norm_weight(
            cfg.hidden_size, cfg.residual_rms_norm,
            vb.pp("post_attention_layernorm"),
        )?;
        let rms_eps = cfg.rms_norm_eps as f32;
        let mlp = MLP::load(vb.pp("mlp"), cfg, ctx.backend.clone())?;

        if layer_type == "full_attention" {
            let attn = Qwen3_5FullAttention::load(vb.pp("self_attn"), cfg, ctx.backend.clone())?;
            Ok(Box::new(Qwen3_5Block::Full {
                name,
                rms_1_weight,
                rms_2_weight,
                rms_eps,
                attn,
                mlp,
            }))
        } else {
            let attn = GatedDeltaNet::load(vb.pp("linear_attn"), cfg, ctx.backend.clone())?;
            Ok(Box::new(Qwen3_5Block::Linear {
                name,
                rms_1_weight,
                rms_2_weight,
                rms_eps,
                attn,
                mlp,
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
            Qwen3_5Block::Linear {
                rms_1_weight, rms_2_weight, rms_eps, attn, mlp, ..
            } => {
                let residual = x;
                let x = ctx.backend.rms_norm(x, rms_1_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_1: {e}"))?;
                let x = (attn.forward(
                    &x,
                    block_idx,
                    ctx.cache.as_mut().expect("No cache"),
                )? + residual)
                    .map_err(|e| anyhow!("residual: {e}"))?;
                let residual = &x;
                let x = ctx.backend.rms_norm(&x, rms_2_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_2: {e}"))?;
                let x = (mlp.forward(&x).map_err(|e| anyhow!("mlp: {e}"))? + residual)
                    .map_err(|e| anyhow!("mlp residual: {e}"))?;
                Ok(x)
            }
            Qwen3_5Block::Full {
                rms_1_weight, rms_2_weight, rms_eps, attn, mlp, ..
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
                    .map_err(|e| anyhow!("residual: {e}"))?;
                let residual = &x;
                let x = ctx.backend.rms_norm(&x, rms_2_weight, *rms_eps)
                    .map_err(|e| anyhow!("rms_2: {e}"))?;
                let x = (mlp.forward(&x).map_err(|e| anyhow!("mlp: {e}"))? + residual)
                    .map_err(|e| anyhow!("mlp residual: {e}"))?;
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
            Qwen3_5Block::Linear { name, .. } => name,
            Qwen3_5Block::Full { name, .. } => name,
        }
    }
}
