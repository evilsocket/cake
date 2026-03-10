//! OLMo 2 transformer block with post-norm architecture.
//!
//! OLMo 2 uses a "post-norm" pattern (norm BEFORE residual add):
//!   residual = x
//!   x = post_attention_layernorm(attn(x))
//!   x = residual + x
//!   residual = x
//!   x = post_feedforward_layernorm(mlp(x))
//!   x = residual + x
//!
//! The QK-norm lives inside `CausalSelfAttention` (via `use_qk_norm=true` in Config).

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm};

use crate::cake::{Context, Forwarder};
use crate::models::common::{CausalSelfAttention, MLP};
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct OLMo2Block {
    name: String,
    attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
    mlp: MLP,
    post_feedforward_layernorm: RmsNorm,
}

impl std::fmt::Display for OLMo2Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local, olmo2)", &self.name)
    }
}

#[async_trait]
impl Forwarder for OLMo2Block {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = MLP::load(vb.pp("mlp"), cfg)?;

        let eps = cfg.rms_norm_eps;
        let h = cfg.hidden_size;

        let post_attention_layernorm =
            candle_nn::rms_norm(h, eps, vb.pp("post_attention_layernorm"))?;
        let post_feedforward_layernorm =
            candle_nn::rms_norm(h, eps, vb.pp("post_feedforward_layernorm"))?;

        Ok(Box::new(Self {
            name,
            attn,
            post_attention_layernorm,
            mlp,
            post_feedforward_layernorm,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        // Attention sublayer: norm first, then residual add.
        let residual = x;
        let attn_out = self.attn.forward(
            x,
            index_pos,
            block_idx,
            ctx.cache.as_mut().expect("No cache"),
        ).map_err(|e| anyhow!("attn: {e}"))?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)
            .map_err(|e| anyhow!("post_attention_layernorm: {e}"))?;
        let x = (residual + attn_out).map_err(|e| anyhow!("attn residual: {e}"))?;

        // MLP sublayer: norm first, then residual add.
        let residual = &x;
        let mlp_out = self.mlp.forward(&x)
            .map_err(|e| anyhow!("mlp: {e}"))?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)
            .map_err(|e| anyhow!("post_feedforward_layernorm: {e}"))?;
        let x = (residual + mlp_out).map_err(|e| anyhow!("mlp residual: {e}"))?;

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
