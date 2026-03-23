//! CUDA compute backend with fused PTX kernels.
//!
//! All fused operations dispatch to custom CUDA kernels compiled from
//! `fused_ops.cu` via PTX. GPU synchronization is a no-op on CUDA since
//! each kernel launch implicitly orders on the default stream.

mod ops;

use candle_core::{Device, Result, Tensor};

use super::ComputeBackend;

/// CUDA backend — uses custom PTX kernels for fused operations.
#[derive(Debug)]
pub struct CudaBackend {
    device: Device,
}

impl CudaBackend {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        if matches!(q.dtype(), candle_core::DType::F16 | candle_core::DType::BF16) {
            return crate::utils::flash_attn::flash_attention(q, k, v, scale, causal);
        }

        let q = q.to_dtype(candle_core::DType::F32)?;
        let k = k.to_dtype(candle_core::DType::F32)?;
        let v = v.to_dtype(candle_core::DType::F32)?;
        let attn = q.matmul(&k.t()?)?;
        let attn = (attn * scale as f64)?;
        let attn = if causal {
            let seq_len = q.dim(2)?;
            let kv_len = k.dim(2)?;
            // Build causal mask on GPU: row indices >= col indices (with offset for kv_len > seq_len)
            let offset = kv_len.saturating_sub(seq_len);
            let rows = Tensor::arange(0u32, seq_len as u32, q.device())?
                .reshape((seq_len, 1))?;
            let cols = Tensor::arange(0u32, kv_len as u32, q.device())?
                .reshape((1, kv_len))?;
            let offset_t = Tensor::full(offset as u32, (1, 1), q.device())?;
            // mask[i,j] = 1 where j <= i + offset, i.e. (i + offset) >= j
            let shifted = rows.broadcast_add(&offset_t)?;
            let mask = shifted.broadcast_ge(&cols)?;
            let mask = mask.reshape((1, 1, seq_len, kv_len))?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, attn.shape(), q.device())?;
            mask.broadcast_as(attn.shape())?
                .where_cond(&attn, &neg_inf)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        gate.apply_op2_no_bwd(up, &ops::SiluMul)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        x.apply_op1_no_bwd(&ops::StableSoftplus)
    }

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let x = x.contiguous()?;
        let z = z.contiguous()?.to_dtype(x.dtype())?;
        let w = weight.contiguous()?;
        x.apply_op3_no_bwd(&z, &w, &ops::RmsNormGated { eps })
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        let n_cols = *a.dims().last().unwrap();
        let a = a.contiguous()?;
        let b = b.contiguous()?;
        let w = weight.contiguous()?;
        let combined = a.apply_op3_no_bwd(&b, &w, &ops::AddRmsNorm { eps, n_cols })?;
        // Layout: (2*n_rows, n_cols) — narrow on dim 0 yields contiguous views (no copy)
        let n_rows = combined.dim(0)? / 2;
        let residual = combined.narrow(0, 0, n_rows)?;
        let normed = combined.narrow(0, n_rows, n_rows)?;
        Ok((residual, normed))
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let x = x.contiguous()?;
        let w = weight.contiguous()?;
        x.apply_op2_no_bwd(&w, &ops::RmsNormChannel { eps })
    }

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        window.apply_op2_no_bwd(weight, &ops::DepthwiseConv1dSilu { kernel_size, channels })
    }

    fn depthwise_conv1d_bias(&self, padded_input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        let input = padded_input.contiguous()?;
        let w = weight.contiguous()?;
        let b = bias.contiguous()?;
        input.apply_op3_no_bwd(&w, &b, &ops::DepthwiseConv1dBias { kernel_size, channels })
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        let ctx = ctx.contiguous()?;
        let inp = input.contiguous()?;
        let w = weight.contiguous()?;
        let b = bias.contiguous()?;
        ctx.apply_op3_no_bwd(&inp, &w, &ops::DepthwiseConv1dBiasCtx { kernel_size, channels, bias: b })
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        a.apply_op3_no_bwd(b, c, &ops::Add3)
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(y, &ops::ExpMul)
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        a.apply_op3_no_bwd(b, c, &ops::SubMul)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        let a = a.contiguous()?;
        let b = b.contiguous()?;
        let c = c.contiguous()?;
        a.apply_op3_no_bwd(&b, &c, &ops::AddScaled)
    }

    fn adaln_modulate(&self, x: &Tensor, norm_weight: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
        let x = x.contiguous()?;
        let w = norm_weight.contiguous()?;
        let sc = scale.contiguous()?;
        let sh = shift.contiguous()?;
        x.apply_op3_no_bwd(&w, &sc, &ops::AdaLnModulate { eps, shift: sh })
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() != candle_core::DType::F8E4M3 { return x.to_dtype(candle_core::DType::F32); }
        if let Ok(t) = x.to_dtype(candle_core::DType::F32) { return Ok(t); }
        x.apply_op1_no_bwd(&ops::F8E4M3ToF32)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() != candle_core::DType::F8E4M3 { return x.to_dtype(candle_core::DType::F16); }
        x.apply_op1_no_bwd(&ops::F8E4M3ToF16)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() != candle_core::DType::F8E4M3 { return x.to_dtype(candle_core::DType::BF16); }
        x.apply_op1_no_bwd(&ops::F8E4M3ToBF16)
    }

    // synchronize() — default no-op is correct for CUDA
}
