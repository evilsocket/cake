//! Metal compute backend.
//!
//! Delegates fused operations to candle's Metal dispatch (MSL kernels) via the
//! existing `fused_ops` infrastructure. Overrides `attention()` to use candle's
//! fused SDPA kernel and `synchronize()` to flush Metal command buffers.
//!
//! Metal command buffer accumulation >25 commands causes catastrophic 50x slowdown.
//! The `synchronize()` method flushes the command buffer and is called at strategic
//! points during forward passes (see GatedDeltaNet, Qwen3_5FullAttention).

use candle_core::{DType, Device, Result, Tensor};

use super::ComputeBackend;
use super::ops;

/// Metal backend — MSL kernels for fused ops, SDPA for attention, command buffer sync.
#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
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
        // Fused SDPA on Metal — single kernel, native GQA (no repeat_kv needed)
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        candle_nn::ops::sdpa(&q, &k, &v, None, causal, scale, 1.0)
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

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}
