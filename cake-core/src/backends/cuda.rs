//! CUDA compute backend.
//!
//! Delegates fused operations to candle's CUDA CustomOp dispatch (PTX kernels)
//! via the existing `fused_ops` infrastructure. GPU synchronization is a no-op
//! on CUDA since each kernel launch implicitly orders on the default stream.

use candle_core::{Device, Result, Tensor};

use super::ComputeBackend;
use super::ops;

/// CUDA backend — uses candle CUDA CustomOp kernels for fused operations.
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
        // Flash Attention on CUDA for F16/BF16 — fused kernel, native GQA
        #[cfg(feature = "flash-attn")]
        if matches!(q.dtype(), candle_core::DType::F16 | candle_core::DType::BF16) {
            return crate::utils::flash_attn::flash_attention(q, k, v, scale, causal);
        }

        // Fallback: manual SDPA in F32
        let q = q.to_dtype(candle_core::DType::F32)?;
        let k = k.to_dtype(candle_core::DType::F32)?;
        let v = v.to_dtype(candle_core::DType::F32)?;
        let attn = q.matmul(&k.t()?)?;
        let attn = (attn * scale as f64)?;
        let attn = if causal {
            let seq_len = q.dim(2)?;
            let kv_len = k.dim(2)?;
            let mut mask_data = vec![0u8; seq_len * kv_len];
            for i in 0..seq_len {
                let max_j = kv_len.saturating_sub(seq_len) + i;
                for j in 0..=max_j.min(kv_len - 1) {
                    mask_data[i * kv_len + j] = 1;
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, kv_len), q.device())?;
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
        ops::silu_mul(gate, up)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        ops::stable_softplus(x)
    }

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops::rms_norm_gated(x, z, weight, eps)
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        ops::add_rms_norm(a, b, weight, eps)
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops::rms_norm_channel(x, weight, eps)
    }

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        ops::depthwise_conv1d_silu(window, weight, kernel_size, channels)
    }

    fn depthwise_conv1d_bias(&self, padded_input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        ops::depthwise_conv1d_bias(padded_input, weight, bias, kernel_size, channels)
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        ops::depthwise_conv1d_bias_ctx(ctx, input, weight, bias, kernel_size, channels)
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ops::add3(a, b, c)
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        ops::exp_mul(x, y)
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ops::sub_mul(a, b, c)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ops::add_scaled(a, b, c)
    }

    fn adaln_modulate(&self, x: &Tensor, norm_weight: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
        ops::adaln_modulate(x, norm_weight, scale, shift, eps)
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        ops::f8e4m3_to_f32(x)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        ops::f8e4m3_to_f16(x)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        ops::f8e4m3_to_bf16(x)
    }

    // synchronize() — default no-op is correct for CUDA
}
