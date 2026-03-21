//! CPU compute backend — delegates to existing fused_ops functions.
//!
//! This is the default backend. All operations use candle's CPU tensor ops
//! with rayon parallelization where beneficial. GPU backends override specific
//! methods for acceleration.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::ComputeBackend;
use crate::utils::fused_ops;

/// CPU backend — uses candle CPU ops + rayon-parallelized fused kernels.
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
            // Build lower-triangular causal mask
            let mut mask_data = vec![0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    mask_data[i * seq_len + j] = 1.0;
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
        Ok(attn.matmul(&v)?)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::silu_mul(gate, up)?)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::stable_softplus(x)?)
    }

    fn rms_norm_gated(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        Ok(fused_ops::rms_norm_gated(x, z, weight, eps)?)
    }

    fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        Ok(fused_ops::add_rms_norm(a, b, weight, eps)?)
    }

    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor> {
        Ok(fused_ops::depthwise_conv1d_silu(window, weight, kernel_size, channels)?)
    }

    fn depthwise_conv1d_bias(
        &self,
        padded_input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor> {
        Ok(fused_ops::depthwise_conv1d_bias(padded_input, weight, bias, kernel_size, channels)?)
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::add3(a, b, c)?)
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::exp_mul(x, y)?)
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::sub_mul(a, b, c)?)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::add_scaled(a, b, c)?)
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::f8e4m3_to_f32(x)?)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::f8e4m3_to_f16(x)?)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        Ok(fused_ops::f8e4m3_to_bf16(x)?)
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
}
