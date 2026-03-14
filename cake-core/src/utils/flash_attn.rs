//! Flash Attention 2 wrapper for CUDA (via candle-flash-attn).
//! Requires SM 75+ (Turing and newer).

use candle_core::{Result, Tensor};

/// Compute scaled dot-product attention using Flash Attention 2.
///
/// Input/output tensors are in (batch, heads, seq, head_dim) layout.
/// Handles GQA natively — no repeat_kv needed.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    // Flash attention expects (batch, seq, heads, head_dim)
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;

    let y = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, causal)?;

    // Back to (batch, heads, seq, head_dim)
    y.transpose(1, 2)
}
