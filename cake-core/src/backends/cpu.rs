//! CPU compute backend — delegates to existing fused_ops functions.
//!
//! This is the default backend. All operations use candle's CPU tensor ops
//! with rayon parallelization where beneficial. GPU backends override specific
//! methods for acceleration.

use candle_core::{DType, Device, Result, Tensor};

use super::ComputeBackend;
use crate::utils::fused_ops;

/// CPU backend — uses candle CPU ops + rayon-parallelized fused kernels.
#[derive(Debug)]
pub struct CpuBackend {
    device: Device,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    pub fn with_device(device: Device) -> Self {
        Self { device }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
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
        // Manual SDPA: Q·K^T * scale → softmax → ·V
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let attn = q.matmul(&k.t()?)?;
        let attn = (attn * scale as f64)?;
        let attn = if causal {
            let seq_len = q.dim(2)?;
            // Build lower-triangular causal mask (u8 for where_cond)
            let mut mask_data = vec![0u8; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    mask_data[i * seq_len + j] = 1;
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), q.device())?;
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
        fused_ops::silu_mul(gate, up)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        fused_ops::stable_softplus(x)
    }

    fn rms_norm_gated(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        fused_ops::rms_norm_gated(x, z, weight, eps)
    }

    fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        fused_ops::add_rms_norm(a, b, weight, eps)
    }

    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor> {
        fused_ops::depthwise_conv1d_silu(window, weight, kernel_size, channels)
    }

    fn depthwise_conv1d_bias(
        &self,
        padded_input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor> {
        fused_ops::depthwise_conv1d_bias(padded_input, weight, bias, kernel_size, channels)
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        fused_ops::add3(a, b, c)
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        fused_ops::exp_mul(x, y)
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        fused_ops::sub_mul(a, b, c)
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        fused_ops::rms_norm_channel(x, weight, eps)
    }

    fn depthwise_conv1d_bias_ctx(
        &self,
        ctx: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor> {
        fused_ops::depthwise_conv1d_bias_ctx(ctx, input, weight, bias, kernel_size, channels)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        fused_ops::add_scaled(a, b, c)
    }

    fn adaln_modulate(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        fused_ops::adaln_modulate(x, norm_weight, scale, shift, eps)
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        fused_ops::f8e4m3_to_f32(x)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        fused_ops::f8e4m3_to_f16(x)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        fused_ops::f8e4m3_to_bf16(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_cpu_backend_device_is_cpu() {
        let backend = CpuBackend::new();
        assert!(backend.device().is_cpu());
    }

    #[test]
    fn test_cpu_backend_silu_mul() {
        let backend = CpuBackend::new();
        let gate = Tensor::new(&[1.0f32, 2.0, -1.0], &Device::Cpu).unwrap();
        let up = Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu).unwrap();
        let result = backend.silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dims(), &[3]);
    }

    #[test]
    fn test_cpu_backend_stable_softplus() {
        let backend = CpuBackend::new();
        let x = Tensor::new(&[0.0f32, 1.0, 100.0], &Device::Cpu).unwrap();
        let result = backend.stable_softplus(&x).unwrap();
        assert_eq!(result.dims(), &[3]);
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - std::f32::consts::LN_2).abs() < 0.01);
        assert!(vals[2] > 99.0); // large x ≈ x
    }

    #[test]
    fn test_cpu_backend_add3() {
        let backend = CpuBackend::new();
        let a = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[3.0f32, 4.0], &Device::Cpu).unwrap();
        let c = Tensor::new(&[5.0f32, 6.0], &Device::Cpu).unwrap();
        let result = backend.add3(&a, &b, &c).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert_eq!(vals, vec![9.0, 12.0]);
    }

    #[test]
    fn test_cpu_backend_synchronize_is_noop() {
        let backend = CpuBackend::new();
        assert!(backend.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_backend_attention_shape() {
        let backend = CpuBackend::new();
        // (batch=1, heads=2, seq_len=4, head_dim=8)
        let q = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let result = backend.attention(&q, &k, &v, 0.125, false).unwrap();
        assert_eq!(result.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_cpu_backend_attention_causal() {
        let backend = CpuBackend::new();
        let q = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
        let result = backend.attention(&q, &k, &v, 0.125, true).unwrap();
        assert_eq!(result.dims(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_cpu_backend_exp_mul() {
        let backend = CpuBackend::new();
        let x = Tensor::new(&[2.0f32, 3.0], &Device::Cpu).unwrap();
        let y = Tensor::new(&[0.0f32, 1.0], &Device::Cpu).unwrap();
        let result = backend.exp_mul(&x, &y).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        // x * exp(y): 2*exp(0) = 2.0, 3*exp(1) ≈ 8.15
        assert!((vals[0] - 2.0).abs() < 0.01);
        assert!((vals[1] - 3.0 * std::f32::consts::E).abs() < 0.1);
    }

    #[test]
    fn test_cpu_backend_sub_mul() {
        let backend = CpuBackend::new();
        let a = Tensor::new(&[5.0f32, 10.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[3.0f32, 4.0], &Device::Cpu).unwrap();
        let c = Tensor::new(&[2.0f32, 3.0], &Device::Cpu).unwrap();
        let result = backend.sub_mul(&a, &b, &c).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        // (5-3)*2=4, (10-4)*3=18
        assert!((vals[0] - 4.0).abs() < 0.01);
        assert!((vals[1] - 18.0).abs() < 0.01);
    }

    #[test]
    fn test_cpu_backend_add_scaled() {
        let backend = CpuBackend::new();
        // add_scaled with broadcast: a + b * c where c is (channels,)
        // Use (batch=1, channels=2, time=3) layout
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap(); // (1, 2, 3)
        let b = Tensor::new(&[[0.5f32, 0.5, 0.5], [1.0, 1.0, 1.0]], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap(); // (1, 2, 3)
        let c = Tensor::new(&[2.0f32, 3.0], &Device::Cpu).unwrap(); // (2,)
        let result = backend.add_scaled(&a, &b, &c).unwrap();
        assert_eq!(result.dims(), &[1, 2, 3]);
    }

    #[test]
    fn test_cpu_backend_rms_norm_channel() {
        let backend = CpuBackend::new();
        // (batch=1, channels=4, time=3)
        let x = Tensor::randn(0f32, 1.0, (1, 4, 3), &Device::Cpu).unwrap();
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let result = backend.rms_norm_channel(&x, &weight, 1e-6).unwrap();
        assert_eq!(result.dims(), &[1, 4, 3]);
    }

    #[test]
    fn test_cpu_backend_depthwise_conv1d_bias() {
        let backend = CpuBackend::new();
        // padded_input: (batch=1, channels=4, time=5), weight: (4, 3), bias: (4,)
        let input = Tensor::randn(0f32, 1.0, (1, 4, 5), &Device::Cpu).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (4, 3), &Device::Cpu).unwrap();
        let bias = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();
        let result = backend
            .depthwise_conv1d_bias(&input, &weight, &bias, 3, 4)
            .unwrap();
        assert_eq!(result.dims(), &[1, 4, 3]); // time - kernel + 1 = 5 - 3 + 1 = 3
    }

    #[test]
    fn test_cpu_backend_depthwise_conv1d_bias_ctx() {
        let backend = CpuBackend::new();
        // ctx: (1, 4, 2), input: (1, 4, 3), weight: (4, 3), bias: (4,)
        let ctx = Tensor::zeros((1, 4, 2), DType::F32, &Device::Cpu).unwrap();
        let input = Tensor::randn(0f32, 1.0, (1, 4, 3), &Device::Cpu).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (4, 3), &Device::Cpu).unwrap();
        let bias = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();
        let result = backend
            .depthwise_conv1d_bias_ctx(&ctx, &input, &weight, &bias, 3, 4)
            .unwrap();
        assert_eq!(result.dims(), &[1, 4, 3]); // output length = input time
    }

    #[test]
    fn test_cpu_backend_adaln_modulate() {
        let backend = CpuBackend::new();
        // (batch=1, seq=2, hidden=4)
        let x = Tensor::randn(0f32, 1.0, (1, 2, 4), &Device::Cpu).unwrap();
        let norm_weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let scale = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let shift = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let result = backend
            .adaln_modulate(&x, &norm_weight, &scale, &shift, 1e-6)
            .unwrap();
        assert_eq!(result.dims(), &[1, 2, 4]);
    }

    #[test]
    fn test_cpu_backend_rms_norm_gated() {
        let backend = CpuBackend::new();
        let x = Tensor::randn(0f32, 1.0, (1, 2, 4), &Device::Cpu).unwrap();
        let z = Tensor::randn(0f32, 1.0, (1, 2, 4), &Device::Cpu).unwrap();
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let result = backend.rms_norm_gated(&x, &z, &weight, 1e-6).unwrap();
        assert_eq!(result.dims(), &[1, 2, 4]);
    }

    #[test]
    fn test_cpu_backend_add_rms_norm() {
        let backend = CpuBackend::new();
        let a = Tensor::randn(0f32, 1.0, (1, 2, 4), &Device::Cpu).unwrap();
        let b = Tensor::randn(0f32, 1.0, (1, 2, 4), &Device::Cpu).unwrap();
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let (normed, residual) = backend.add_rms_norm(&a, &b, &weight, 1e-6).unwrap();
        assert_eq!(normed.dims(), &[1, 2, 4]);
        assert_eq!(residual.dims(), &[1, 2, 4]);
    }

    #[test]
    fn test_cpu_backend_depthwise_conv1d_silu() {
        let backend = CpuBackend::new();
        // window: (batch=1, channels=4, kernel_size=3)
        let window = Tensor::randn(0f32, 1.0, (1, 4, 3), &Device::Cpu).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (4, 3), &Device::Cpu).unwrap();
        let result = backend
            .depthwise_conv1d_silu(&window, &weight, 3, 4)
            .unwrap();
        assert_eq!(result.dims(), &[1, 4]);
    }

    #[test]
    fn test_cpu_backend_with_device() {
        let backend = CpuBackend::with_device(Device::Cpu);
        assert_eq!(backend.name(), "cpu");
        assert!(backend.device().is_cpu());
    }
}
