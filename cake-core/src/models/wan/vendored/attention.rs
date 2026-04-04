use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use super::rope::apply_wan_rope;

/// RMSNorm applied across all heads (not per-head).
/// Weight shape: [hidden_size] (not [head_dim]).
#[derive(Debug, Clone)]
pub struct WanRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl WanRmsNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        Ok(x_normed.broadcast_mul(&w)?.to_dtype(in_dtype)?)
    }
}

/// Multi-head attention for Wan transformer.
/// Supports both self-attention (with RoPE) and cross-attention (no RoPE).
#[derive(Debug, Clone)]
pub struct WanAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    norm_q: WanRmsNorm,
    norm_k: WanRmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl WanAttention {
    pub fn load(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        eps: f64,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;

        let q_proj = candle_nn::linear(dim, dim, vb.pp("q"))?;
        let k_proj = candle_nn::linear(dim, dim, vb.pp("k"))?;
        let v_proj = candle_nn::linear(dim, dim, vb.pp("v"))?;
        let o_proj = candle_nn::linear(dim, dim, vb.pp("o"))?;
        let norm_q = WanRmsNorm::load(dim, eps, vb.pp("norm_q"))?;
        let norm_k = WanRmsNorm::load(dim, eps, vb.pp("norm_k"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            norm_q,
            norm_k,
            num_heads,
            head_dim,
        })
    }

    /// Self-attention with 3D RoPE.
    pub fn forward_self(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Apply RMSNorm before reshape (across all heads)
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Reshape to [B, S, H, D]
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, s, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, s, self.num_heads, self.head_dim))?;

        // Apply RoPE
        let q = apply_wan_rope(&q, rope_cos, rope_sin)?;
        let k = apply_wan_rope(&k, rope_cos, rope_sin)?;

        // Transpose to [B, H, S, D] for attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let y = self.sdpa(&q, &k, &v)?;

        // Reshape back to [B, S, dim]
        let y = y.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        let y = self.o_proj.forward(&y)?;

        Ok(y)
    }

    /// Cross-attention (Q from video, K/V from text context). No RoPE.
    pub fn forward_cross(
        &self,
        x: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let ctx_len = context.dim(1)?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        // Apply RMSNorm
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Reshape
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, ctx_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, ctx_len, self.num_heads, self.head_dim))?;

        // Transpose to [B, H, S, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Attention
        let y = self.sdpa(&q, &k, &v)?;

        // Reshape
        let y = y.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        let y = self.o_proj.forward(&y)?;

        Ok(y)
    }

    fn sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let in_dtype = q.dtype();

        #[cfg(feature = "cuda")]
        if matches!(q.device(), candle_core::Device::Cuda(_))
            && matches!(q.dtype(), candle_core::DType::F16 | candle_core::DType::BF16)
        {
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            return Ok(crate::utils::flash_attn::flash_attention(q, k, v, scale, false)?);
        }

        // Fallback: F32 attention
        let scale = (self.head_dim as f64).sqrt();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v)?;
        Ok(y.to_dtype(in_dtype)?)
    }
}
