//! Qwen3 MoE transformer block.
//!
//! Identical structure to dense Qwen3 (pre-norm, standard residual connections,
//! QK-norm, full-context GQA attention) except the FFN is a sparse
//! Mixture-of-Experts layer instead of a dense MLP.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm};

use crate::cake::{Context, Forwarder};
use crate::models::common::{load_rms_norm, CausalSelfAttention};
use async_trait::async_trait;

use super::moe::SparseMoeMlp;

/// A single Qwen3 MoE transformer layer.
#[derive(Debug, Clone)]
pub struct Qwen3MoeBlock {
    name: String,
    input_layernorm: RmsNorm,
    attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
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

        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let moe = SparseMoeMlp::load(vb.pp("mlp"), cfg, ctx.backend.clone())?;

        let eps = cfg.rms_norm_eps;
        let h = cfg.hidden_size;
        let input_layernorm = load_rms_norm(h, eps, false, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            load_rms_norm(h, eps, false, vb.pp("post_attention_layernorm"))?;

        Ok(Box::new(Self {
            name,
            input_layernorm,
            attn,
            post_attention_layernorm,
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
        let x = self
            .input_layernorm
            .forward(x)
            .map_err(|e| anyhow!("input_layernorm: {e}"))?;
        let x = self
            .attn
            .forward(&x, index_pos, block_idx, ctx.cache.as_mut().expect("No cache"))
            .map_err(|e| anyhow!("attn: {e}"))?;
        let x = (x + residual).map_err(|e| anyhow!("attn residual: {e}"))?;

        // MoE sublayer (pre-norm, standard residual).
        let residual = &x;
        let x = self
            .post_attention_layernorm
            .forward(&x)
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
