//! EXAONE 4.0 transformer block with 3:1 local/global hybrid attention.
//!
//! Local layers (3 out of 4): sliding window attention + RoPE + QK-norm.
//! Global layers (every 4th): full context attention + **no RoPE** + QK-norm.
//!
//! Layer norm pattern: standard pre-norm (same as LLaMA).

use anyhow::Result;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use crate::models::common::{CausalSelfAttention, MLP};
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct EXAONE4Block {
    name: String,
    input_layernorm_weight: Tensor,
    post_attention_layernorm_weight: Tensor,
    rms_eps: f32,
    attn: CausalSelfAttention,
    mlp: MLP,
}

impl std::fmt::Display for EXAONE4Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local, exaone4)", &self.name)
    }
}

impl EXAONE4Block {
    fn layer_index(name: &str) -> usize {
        name.rsplit('.')
            .next()
            .and_then(|s| s.parse().ok())
            .expect("invalid layer name — no trailing index")
    }
}

#[async_trait]
impl Forwarder for EXAONE4Block {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let layer_idx = Self::layer_index(&name);

        let is_global = cfg.global_layers.get(layer_idx).copied().unwrap_or(false);

        // Global layers: no RoPE, full context
        // Local layers: RoPE + sliding window
        let (sliding_window, use_rope) = if is_global {
            (None, false)
        } else {
            (cfg.sliding_window, true)
        };

        let attn = CausalSelfAttention::load_custom(
            vb.pp("self_attn"),
            cfg,
            cfg.use_qk_norm,
            sliding_window,
            use_rope,
            ctx.backend.clone(),
        )?;
        let mlp = MLP::load(vb.pp("mlp"), cfg, ctx.backend.clone())?;

        let h = cfg.hidden_size;

        let input_layernorm_weight =
            vb.pp("input_layernorm").get(h, "weight")?;
        let post_attention_layernorm_weight =
            vb.pp("post_attention_layernorm").get(h, "weight")?;
        let rms_eps = cfg.rms_norm_eps as f32;

        Ok(Box::new(Self {
            name,
            input_layernorm_weight,
            attn,
            post_attention_layernorm_weight,
            rms_eps,
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
        let x = ctx.backend.rms_norm(x, &self.input_layernorm_weight, self.rms_eps)
            .map_err(|e| anyhow!("input_layernorm: {e}"))?;
        let x = (self.attn.forward(
            &x,
            index_pos,
            block_idx,
            ctx.cache.as_mut().expect("No cache"),
        ).map_err(|e| anyhow!("attn: {e}"))? + residual)
            .map_err(|e| anyhow!("attn residual: {e}"))?;

        let residual = &x;
        let x = ctx.backend.rms_norm(&x, &self.post_attention_layernorm_weight, self.rms_eps)
            .map_err(|e| anyhow!("post_attention_layernorm: {e}"))?;
        let x = (self.mlp.forward(&x)
            .map_err(|e| anyhow!("mlp: {e}"))? + residual)
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
