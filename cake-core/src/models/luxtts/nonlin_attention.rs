//! NonlinAttention -- tanh-gated attention.
//!
//! Python forward:
//!   s, x, y = in_proj(x).chunk(3)
//!   s = tanh(s)
//!   x = x * s  (tanh gating)
//!   x = reshape to multi-head, matmul(attn_weights, x), reshape back
//!   x = x * y  (output gating)
//!   x = out_proj(x)

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct NonlinAttention {
    in_proj: Linear,
    out_proj: Linear,
    hidden: usize,
    num_heads: usize,
}

impl NonlinAttention {
    pub fn load(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = 3 * dim / 4;
        let in_proj = candle_nn::linear(dim, 3 * hidden, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear(hidden, dim, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            out_proj,
            hidden,
            num_heads,
        })
    }

    /// Forward with precomputed attention weights.
    /// `attn_weights`: [batch, num_heads, seq, seq]
    pub fn forward(&self, x: &Tensor, attn_weights: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let head_dim = self.hidden / self.num_heads;

        // Project to 3 * hidden
        let projected = self.in_proj.forward(x)?; // [batch, seq, 3*hidden]

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

        let x_out = self.out_proj.forward(&x_out)?;
        Ok(x_out)
    }
}
