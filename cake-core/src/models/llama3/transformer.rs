use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm};

use crate::cake::{Context, Forwarder};
use async_trait::async_trait;

use super::{CausalSelfAttention, MLP};

/// Transformer block with causal self attention and several caching strategies.
#[derive(Debug, Clone)]
pub struct Transformer {
    name: String,
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: MLP,
}

impl std::fmt::Display for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

#[async_trait]
impl Forwarder for Transformer {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let attn = super::CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = super::MLP::load(vb.pp("mlp"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Box::new(Self {
            name,
            rms_1,
            attn,
            rms_2,
            mlp,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let residual = x;

        let x = self.rms_1.forward(x).map_err(|e| anyhow!("rms_1: {e}"))?;
        let x = (self
            .attn
            .forward(
                &x,
                index_pos,
                block_idx,
                ctx.cache.as_mut().expect("No cache specified"),
            )
            .map_err(|e| anyhow!("attention: {e}"))?
            + residual)
            .map_err(|e| anyhow!("residual: {e}"))?;
        let residual = &x;
        let x = self.rms_2.forward(&x).map_err(|e| anyhow!("rms_2: {e}"))?;
        let x = (self.mlp.forward(&x).map_err(|e| anyhow!("mlp: {e}"))? + residual)
            .map_err(|e| anyhow!("mlp residual: {e}"))?;

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
