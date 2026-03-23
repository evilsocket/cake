//! Gemma 3 transformer block with interleaved local (sliding-window, no RoPE) and
//! global (full context, RoPE) attention layers.
//!
//! Uses a 4-norm "sandwich" pattern per layer:
//!   residual = x
//!   x = input_layernorm(x) -> attn(x) -> post_attention_layernorm(x)
//!   x = residual + x
//!   residual = x
//!   x = pre_feedforward_layernorm(x) -> mlp(x) -> post_feedforward_layernorm(x)
//!   x = residual + x

use anyhow::Result;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use crate::models::common::{load_rms_norm_weight, CausalSelfAttention, MLP};
use async_trait::async_trait;

/// A Gemma 3 transformer block.
#[derive(Debug, Clone)]
pub struct Gemma3Block {
    name: String,
    input_layernorm_weight: Tensor,
    post_attention_layernorm_weight: Tensor,
    pre_feedforward_layernorm_weight: Tensor,
    post_feedforward_layernorm_weight: Tensor,
    rms_eps: f32,
    attn: CausalSelfAttention,
    mlp: MLP,
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
            ctx.backend.clone(),
        )?;
        let mlp = MLP::load(vb.pp("mlp"), cfg, ctx.backend.clone())?;

        let h = cfg.hidden_size;
        let residual = cfg.residual_rms_norm; // Gemma3 uses (1+weight)*norm(x)

        let input_layernorm_weight =
            load_rms_norm_weight(h, residual, vb.pp("input_layernorm"))?;
        let post_attention_layernorm_weight =
            load_rms_norm_weight(h, residual, vb.pp("post_attention_layernorm"))?;
        let pre_feedforward_layernorm_weight =
            load_rms_norm_weight(h, residual, vb.pp("pre_feedforward_layernorm"))?;
        let post_feedforward_layernorm_weight =
            load_rms_norm_weight(h, residual, vb.pp("post_feedforward_layernorm"))?;
        let rms_eps = cfg.rms_norm_eps as f32;

        Ok(Box::new(Self {
            name,
            input_layernorm_weight,
            post_attention_layernorm_weight,
            pre_feedforward_layernorm_weight,
            post_feedforward_layernorm_weight,
            rms_eps,
            attn,
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
        // --- Attention sublayer (sandwich norm) ---
        let residual = x;
        let x = ctx.backend.rms_norm(x, &self.input_layernorm_weight, self.rms_eps)
            .map_err(|e| anyhow!("input_layernorm: {e}"))?;
        let x = self.attn.forward(
            &x,
            index_pos,
            block_idx,
            ctx.cache.as_mut().expect("No cache"),
        ).map_err(|e| anyhow!("attn: {e}"))?;
        let x = ctx.backend.rms_norm(&x, &self.post_attention_layernorm_weight, self.rms_eps)
            .map_err(|e| anyhow!("post_attention_layernorm: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("attn residual: {e}"))?;

        // --- MLP sublayer (sandwich norm) ---
        let residual = &x;
        let x = ctx.backend.rms_norm(&x, &self.pre_feedforward_layernorm_weight, self.rms_eps)
            .map_err(|e| anyhow!("pre_feedforward_layernorm: {e}"))?;
        let x = self.mlp.forward(&x)
            .map_err(|e| anyhow!("mlp: {e}"))?;
        let x = ctx.backend.rms_norm(&x, &self.post_feedforward_layernorm_weight, self.rms_eps)
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
