//! Multi-head attention for LTX-2 transformer.
//!
//! Matches HF diffusers `LTX2Attention` + `LTX2AudioVideoAttnProcessor`.
//! QK-norm is applied across all heads (before head reshape), then RoPE, then reshape.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use super::rope::apply_rotary_emb;

/// RMSNorm (with learned weight).
#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        xs.broadcast_mul(&self.weight)
    }
}

/// Standalone RMS normalization (no learned weight).
pub fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    x.to_dtype(dtype)
}

/// LayerNorm without learnable affine parameters (elementwise_affine=False).
/// Subtracts mean and divides by std, matching `nn.LayerNorm(..., elementwise_affine=False)`.
pub fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let variance = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
    let x = x_centered.broadcast_div(&(variance + eps)?.sqrt()?)?;
    x.to_dtype(dtype)
}

/// Multi-head attention with QK-norm across heads, split RoPE.
///
/// Matches HF `LTX2Attention`:
/// - norm_q/norm_k operate on `[B, T, heads*d_head]` (across all heads)
/// - Order: project → norm → RoPE → reshape to heads → SDPA → reshape back → project out
#[derive(Debug)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm, // normalizes heads*d_head dim
    norm_k: RmsNorm, // normalizes heads*d_head dim
    /// LTX-2.3: per-head gating (sigmoid gate on attention output).
    /// Linear(inner_dim, heads) -> sigmoid -> gate per head.
    to_gate_logits: Option<Linear>,
    heads: usize,
    d_head: usize,
}

impl Attention {
    pub fn new(
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        d_head: usize,
        norm_eps: f64,
        gated: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = heads * d_head;
        let kv_dim = context_dim.unwrap_or(query_dim);

        let to_q = candle_nn::linear(query_dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(kv_dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(kv_dim, inner_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(inner_dim, query_dim, vb.pp("to_out.0"))?;

        // QK norm across full inner dim (heads * d_head)
        let norm_q = RmsNorm::new(inner_dim, norm_eps, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(inner_dim, norm_eps, vb.pp("norm_k"))?;

        // LTX-2.3: per-head gated attention
        let to_gate_logits = if gated {
            Some(candle_nn::linear(inner_dim, heads, vb.pp("to_gate_logits"))?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            norm_q,
            norm_k,
            to_gate_logits,
            heads,
            d_head,
        })
    }

    /// Forward pass.
    ///
    /// `x`: query, `[B, T_q, D]`
    /// `context`: key/value (None = self-attention), `[B, T_kv, D_kv]`
    /// `pe`: RoPE `(cos, sin)` — applied BEFORE head reshape
    /// `k_pe`: separate K RoPE (for cross-modal attention)
    /// `mask`: attention mask `[B, T_q, T_kv]` (0=masked, 1=attend)
    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        pe: Option<&(Tensor, Tensor)>,
        k_pe: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t_q, _) = x.dims3()?;
        let kv_input = context.unwrap_or(x);

        // 1. Project Q, K, V — [B, T, inner_dim]
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(kv_input)?;
        let v = self.to_v.forward(kv_input)?;

        // 2. QK-norm across full inner dim (before head reshape)
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // 3. Apply split RoPE (q/k still flat [B, T, inner_dim])
        // cos/sin: [B, H, T, r] — apply_rotary_emb reshapes x per-head internally
        let (q, k) = if let Some((cos, sin)) = pe {
            let q = apply_rotary_emb(&q, cos, sin)?;
            let k = if let Some((k_cos, k_sin)) = k_pe {
                apply_rotary_emb(&k, k_cos, k_sin)?
            } else {
                apply_rotary_emb(&k, cos, sin)?
            };
            (q, k)
        } else {
            (q, k)
        };

        // 4. Reshape to heads: [B, T, H, D_head]
        let q = q.reshape((b, t_q, self.heads, self.d_head))?;
        let k = k.reshape((b, (), self.heads, self.d_head))?;
        let v = v.reshape((b, (), self.heads, self.d_head))?;

        // 5. Transpose to [B, H, T, D_head] for attention
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // 6. Scaled dot-product attention (compute scores in F32 for numerical stability,
        //    matching PyTorch's F.scaled_dot_product_attention which uses F32 internally)
        let input_dtype = q.dtype();
        let scale = (self.d_head as f64).sqrt();
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k.to_dtype(DType::F32)?;
        let attn = q_f32.matmul(&k_f32.transpose(2, 3)?.contiguous()?)?.affine(1.0 / scale, 0.0)?;

        // Apply mask (additive: masked positions get -inf)
        let attn = if let Some(mask) = mask {
            // mask: [B, T_q, T_kv] (1=attend, 0=masked) -> [B, 1, T_q, T_kv]
            let mask = mask.unsqueeze(1)?.to_dtype(DType::F32)?;
            // (1 - mask) * -1e9 gives 0 for attend positions, -1e9 for masked
            let additive_mask = mask.affine(-1.0, 1.0)?.affine(-1e9, 0.0)?;
            attn.broadcast_add(&additive_mask)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let v_f32 = v.to_dtype(DType::F32)?;
        let out = attn.matmul(&v_f32)?.to_dtype(input_dtype)?; // [B, H, T_q, D_head]

        // 7. Apply per-head gating (LTX-2.3)
        let out = if let Some(ref gate_proj) = self.to_gate_logits {
            // Compute gate from query input: [B, T_q, inner_dim] -> [B, T_q, H]
            let gate = gate_proj.forward(x)?;
            let gate = (candle_nn::ops::sigmoid(&gate)? * 2.0)?;
            // gate: [B, T_q, H] -> [B, H, T_q, 1] to broadcast with [B, H, T_q, D_head]
            let gate = gate.transpose(1, 2)?.unsqueeze(3)?;
            out.broadcast_mul(&gate)?
        } else {
            out
        };

        // 8. Transpose back and flatten: [B, T_q, H*D_head]
        let out = out.transpose(1, 2)?.contiguous()?;
        let out = out.flatten_from(2)?;

        // 9. Project out
        self.to_out.forward(&out)
    }

    /// STG forward: skip Q/K attention, pass V straight through.
    ///
    /// Computes `to_out(to_v(kv_input))` with gating but no attention.
    pub fn forward_skip_attn(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
    ) -> Result<Tensor> {
        let kv_input = context.unwrap_or(x);

        // Only V projection — skip Q, K, RoPE, softmax
        let v = self.to_v.forward(kv_input)?;

        // Apply per-head gating (LTX-2.3) — gate is computed from query input
        let out = if let Some(ref gate_proj) = self.to_gate_logits {
            let (b, _t_q, _) = x.dims3()?;
            let gate = gate_proj.forward(x)?;
            let gate = (candle_nn::ops::sigmoid(&gate)? * 2.0)?;
            // Reshape v to [B, H, T, D_head] then apply gate
            let v = v.reshape((b, (), self.heads, self.d_head))?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let gate = gate.transpose(1, 2)?.unsqueeze(3)?;
            let out = v.broadcast_mul(&gate)?;
            let out = out.transpose(1, 2)?.contiguous()?;
            out.flatten_from(2)?
        } else {
            v
        };

        self.to_out.forward(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_attention_self_attn_shape() {
        let device = Device::Cpu;
        let dim = 32;
        let heads = 2;
        let d_head = 16;

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let attn = Attention::new(dim, None, heads, d_head, 1e-6, false, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, 8, dim), &device).unwrap();
        let out = attn.forward(&x, None, None, None, None).unwrap();
        assert_eq!(out.dims(), &[1, 8, dim]);
    }

    #[test]
    fn test_attention_cross_attn_shape() {
        let device = Device::Cpu;
        let q_dim = 32;
        let kv_dim = 64;
        let heads = 2;
        let d_head = 16;

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let attn = Attention::new(q_dim, Some(kv_dim), heads, d_head, 1e-6, false, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, 8, q_dim), &device).unwrap();
        let ctx = Tensor::randn(0f32, 1f32, (1, 12, kv_dim), &device).unwrap();
        let out = attn.forward(&x, Some(&ctx), None, None, None).unwrap();
        assert_eq!(out.dims(), &[1, 8, q_dim]);
    }

    #[test]
    fn test_rms_norm_unit_variance() {
        let device = Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        let normed = rms_norm(&x, 1e-6).unwrap();
        // RMS norm: x / sqrt(mean(x^2))
        // mean(x^2) = (1+4+9+16)/4 = 7.5, sqrt = 2.7386
        let vals: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        let rms = (7.5f32).sqrt();
        assert!((vals[0] - 1.0 / rms).abs() < 1e-5);
        assert!((vals[3] - 4.0 / rms).abs() < 1e-5);
    }
}
