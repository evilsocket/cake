use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};

use super::adaln::modulate;
use super::attention::WanAttention;
use super::config::WanTransformerConfig;
use super::feed_forward::WanFeedForward;

const EPS: f64 = 1e-6;

/// A single Wan transformer block.
///
/// Each block contains:
/// - Self-attention with 3D RoPE + AdaLN modulation
/// - Cross-attention (Q from video, K/V from text)
/// - Feed-forward network with AdaLN modulation
/// - Learnable `modulation` parameter [1, 6, dim] added to timestep embedding
#[derive(Debug, Clone)]
pub struct WanTransformerBlock {
    /// Self-attention
    self_attn: WanAttention,
    /// Cross-attention
    cross_attn: WanAttention,
    /// Feed-forward network
    ffn: WanFeedForward,
    /// Cross-attention context norm (FP32 LayerNorm with affine)
    norm3: LayerNorm,
    /// Learnable per-block modulation bias [1, 6, dim]
    modulation: Tensor,
}

impl WanTransformerBlock {
    pub fn load(vb: VarBuilder, cfg: &WanTransformerConfig) -> Result<Self> {
        let dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;

        let self_attn = WanAttention::load(vb.pp("self_attn"), dim, num_heads, EPS)?;
        let cross_attn = WanAttention::load(vb.pp("cross_attn"), dim, num_heads, EPS)?;
        let ffn = WanFeedForward::load(vb.pp("ffn"), dim, cfg.ffn_dim)?;

        // norm3: LayerNorm with elementwise_affine=True (for cross_attn_norm=True)
        let norm3 = candle_nn::layer_norm(dim, EPS, vb.pp("norm3"))?;

        // modulation: single parameter tensor [1, 6, dim]
        let modulation = vb.get((1, 6, dim), "modulation")?;

        Ok(Self {
            self_attn,
            cross_attn,
            ffn,
            norm3,
            modulation,
        })
    }

    /// Forward pass for one block.
    ///
    /// Args:
    ///   x: hidden states [B, S, dim]
    ///   context: text embeddings [B, L, dim] (already projected)
    ///   timestep_proj: per-step modulation [B, 6, dim]
    ///   rope_cos, rope_sin: precomputed 3D RoPE [1, S, 1, head_dim/2]
    pub fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        timestep_proj: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        // Add per-block modulation bias to timestep projection
        let mod_params = timestep_proj.broadcast_add(&self.modulation)?; // [B, 6, dim]

        // Split into 6 modulation vectors
        let shift_sa = mod_params.narrow(1, 0, 1)?.squeeze(1)?;  // [B, dim]
        let scale_sa = mod_params.narrow(1, 1, 1)?.squeeze(1)?;
        let gate_sa = mod_params.narrow(1, 2, 1)?.squeeze(1)?;
        let shift_ff = mod_params.narrow(1, 3, 1)?.squeeze(1)?;
        let scale_ff = mod_params.narrow(1, 4, 1)?.squeeze(1)?;
        let gate_ff = mod_params.narrow(1, 5, 1)?.squeeze(1)?;

        // 1. Self-attention with AdaLN
        let x_mod = modulate(x, &shift_sa.unsqueeze(1)?, &scale_sa.unsqueeze(1)?, EPS)?;
        let attn_out = self.self_attn.forward_self(&x_mod, rope_cos, rope_sin)?;
        let x = (x + attn_out.broadcast_mul(&gate_sa.unsqueeze(1)?)?)?;

        // 2. Cross-attention (no modulation on output, simple residual)
        let x_normed = self.norm3.forward(&x)?;
        let cross_out = self.cross_attn.forward_cross(&x_normed, context)?;
        let x = (x + cross_out)?;

        // 3. FFN with AdaLN
        let x_mod = modulate(&x, &shift_ff.unsqueeze(1)?, &scale_ff.unsqueeze(1)?, EPS)?;
        let ff_out = self.ffn.forward(&x_mod)?;
        let x = (x + ff_out.broadcast_mul(&gate_ff.unsqueeze(1)?)?)?;

        Ok(x)
    }
}
