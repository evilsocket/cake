//! Gemma 3 transformer block with interleaved local (sliding-window, no RoPE) and
//! global (full context, RoPE) attention layers.
//!
//! Uses a 4-norm "sandwich" pattern per layer:
//!   residual = x
//!   x = input_layernorm(x) → attn(x) → post_attention_layernorm(x)
//!   x = residual + x
//!   residual = x
//!   x = pre_feedforward_layernorm(x) → mlp(x) → post_feedforward_layernorm(x)
//!   x = residual + x

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm};

use crate::cake::{Context, Forwarder};
use crate::models::common::{load_rms_norm, CausalSelfAttention, MLP};
use async_trait::async_trait;

/// A Gemma 3 transformer block.
#[derive(Debug, Clone)]
pub struct Gemma3Block {
    name: String,
    input_layernorm: RmsNorm,
    attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: MLP,
    post_feedforward_layernorm: RmsNorm,
}

impl std::fmt::Display for Gemma3Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local, gemma3)", &self.name)
    }
}

impl Gemma3Block {
    fn layer_index(name: &str) -> usize {
        name.rsplit('.')
            .next()
            .and_then(|s| s.parse().ok())
            .expect("invalid layer name — no trailing index")
    }
}

#[async_trait]
impl Forwarder for Gemma3Block {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);
        let cfg = ctx.config.as_ref().expect("No config specified");

        let layer_idx = Self::layer_index(&name);

        // Determine local vs global from per-layer schedule
        let is_global = cfg.global_layers.get(layer_idx).copied().unwrap_or(false);

        let (sliding_window, use_rope) = if is_global {
            (None, true)  // global: full context + RoPE
        } else {
            (cfg.sliding_window, false)  // local: sliding window + no RoPE
        };

        let attn = CausalSelfAttention::load_custom(
            vb.pp("self_attn"),
            cfg,
            cfg.use_qk_norm,
            sliding_window,
            use_rope,
        )?;
        let mlp = MLP::load(vb.pp("mlp"), cfg)?;

        let eps = cfg.rms_norm_eps;
        let h = cfg.hidden_size;
        let residual = cfg.residual_rms_norm; // Gemma3 uses (1+weight)*norm(x)

        let input_layernorm =
            load_rms_norm(h, eps, residual, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            load_rms_norm(h, eps, residual, vb.pp("post_attention_layernorm"))?;
        let pre_feedforward_layernorm =
            load_rms_norm(h, eps, residual, vb.pp("pre_feedforward_layernorm"))?;
        let post_feedforward_layernorm =
            load_rms_norm(h, eps, residual, vb.pp("post_feedforward_layernorm"))?;

        Ok(Box::new(Self {
            name,
            input_layernorm,
            attn,
            post_attention_layernorm,
            pre_feedforward_layernorm,
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
        // --- Attention sublayer (sandwich norm) ---
        let residual = x;
        let x = self.input_layernorm.forward(x)
            .map_err(|e| anyhow!("input_layernorm: {e}"))?;
        let x = self.attn.forward(
            &x,
            index_pos,
            block_idx,
            ctx.cache.as_mut().expect("No cache"),
        ).map_err(|e| anyhow!("attn: {e}"))?;
        let x = self.post_attention_layernorm.forward(&x)
            .map_err(|e| anyhow!("post_attention_layernorm: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("attn residual: {e}"))?;

        // --- MLP sublayer (sandwich norm) ---
        let residual = &x;
        let x = self.pre_feedforward_layernorm.forward(&x)
            .map_err(|e| anyhow!("pre_feedforward_layernorm: {e}"))?;
        let x = self.mlp.forward(&x)
            .map_err(|e| anyhow!("mlp: {e}"))?;
        let x = self.post_feedforward_layernorm.forward(&x)
            .map_err(|e| anyhow!("post_feedforward_layernorm: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("mlp residual: {e}"))?;

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
