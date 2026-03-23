//! NonlinAttention -- tanh-gated attention.
//!
//! Python forward:
//!   s, x, y = in_proj(x).chunk(3)
//!   s = tanh(s)
//!   x = x * s  (tanh gating)
//!   x = reshape to multi-head, matmul(attn_weights, x), reshape back
//!   x = x * y  (output gating)
//!   x = out_proj(x)

use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

#[derive(Debug, Clone)]
pub struct NonlinAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    out_proj_weight: Tensor,
    out_proj_bias: Option<Tensor>,
    hidden: usize,
    num_heads: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl NonlinAttention {
    pub fn load(dim: usize, num_heads: usize, vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let hidden = 3 * dim / 4;
        let in_proj_weight = vb.pp("in_proj").get((3 * hidden, dim), "weight")?;
        let in_proj_bias = Some(vb.pp("in_proj").get(3 * hidden, "bias")?);
        let out_proj_weight = vb.pp("out_proj").get((dim, hidden), "weight")?;
        let out_proj_bias = Some(vb.pp("out_proj").get(dim, "bias")?);

        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj_weight,
            out_proj_bias,
            hidden,
            num_heads,
            backend,
        })
    }

    /// Forward with precomputed attention weights.
    /// `attn_weights`: [batch, num_heads, seq, seq]
    pub fn forward(&self, x: &Tensor, attn_weights: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let head_dim = self.hidden / self.num_heads;

        // Project to 3 * hidden
        let projected = self.backend.linear_forward(x, &self.in_proj_weight, self.in_proj_bias.as_ref())?; // [batch, seq, 3*hidden]

        // Split into s, x_val, y
        let s = projected.narrow(candle_core::D::Minus1, 0, self.hidden)?;
        let x_val = projected.narrow(candle_core::D::Minus1, self.hidden, self.hidden)?;
        let y = projected.narrow(candle_core::D::Minus1, 2 * self.hidden, self.hidden)?;

        // s through tanh
        let s = s.tanh()?;

        // x = x_val * s (tanh gating)
        let x_gated = (x_val * s)?;

        // Reshape to multi-head: [batch, seq, nh, hd] -> [batch, nh, seq, hd]
        let x_mh = x_gated
            .reshape((batch, seq_len, self.num_heads, head_dim))?
            .transpose(1, 2)?;

        // Broadcast attn_weights if needed
        let attn_weights = attn_weights.broadcast_as((batch, self.num_heads, seq_len, seq_len))?;

        // matmul(attn_weights, x): [batch, nh, seq, seq] @ [batch, nh, seq, hd] -> [batch, nh, seq, hd]
        let x_attn = attn_weights.matmul(&x_mh)?;

        // Reshape back: [batch, nh, seq, hd] -> [batch, seq, hidden]
        let x_attn = x_attn
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.hidden))?;

        // Output gating: x = x * y
        let x_out = (x_attn * y)?;

        let x_out = self.backend.linear_forward(&x_out, &self.out_proj_weight, self.out_proj_bias.as_ref())?;
        Ok(x_out)
    }
}
