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

impl Transformer {
    /// Load a transformer block directly from a VarBuilder (no Context needed).
    /// Used by VibeVoice TTS where blocks are loaded outside the Forwarder pattern.
    pub fn load_for_vibevoice(
        vb: candle_nn::VarBuilder,
        cfg: &super::Config,
        backend: std::sync::Arc<dyn crate::backends::ComputeBackend>,
    ) -> candle_core::Result<Self> {
        let attn = super::CausalSelfAttention::load(vb.pp("self_attn"), cfg, backend.clone())?;
        let mlp = super::MLP::load(vb.pp("mlp"), cfg, backend)?;
        let rms_1 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            name: String::new(),
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    /// Forward pass with external cache (for VibeVoice TTS).
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut super::Cache,
    ) -> anyhow::Result<Tensor> {
        let residual = x;
        let h = self.rms_1.forward(x).map_err(|e| anyhow!("rms_1: {e}"))?;
        let h = (self.attn.forward(&h, index_pos, block_idx, cache)
            .map_err(|e| anyhow!("attn: {e}"))? + residual)
            .map_err(|e| anyhow!("residual: {e}"))?;
        let residual = &h;
        let h = self.rms_2.forward(&h).map_err(|e| anyhow!("rms_2: {e}"))?;
        let h = (self.mlp.forward(&h).map_err(|e| anyhow!("mlp: {e}"))? + residual)
            .map_err(|e| anyhow!("mlp_residual: {e}"))?;
        Ok(h)
    }
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

        let attn = super::CausalSelfAttention::load(vb.pp("self_attn"), cfg, ctx.backend.clone())?;
        let mlp = super::MLP::load(vb.pp("mlp"), cfg, ctx.backend.clone())?;
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
        // Flush Metal command buffer between attention and MLP to prevent
        // >25 command accumulation (no-op on CPU/CUDA)
        let _ = ctx.backend.synchronize();
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
