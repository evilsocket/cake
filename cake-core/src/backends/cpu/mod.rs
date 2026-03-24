//! CPU compute backend — pure candle tensor operations, no CustomOp, no ops/ module.
//!
//! This is the default backend. All operations use candle's CPU tensor ops
//! with rayon parallelization where beneficial. GPU backends override specific
//! methods for acceleration.

use candle_core::{DType, Device, Result, Tensor, D};

use super::ComputeBackend;

/// CPU backend — uses candle CPU ops directly.
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
            // Build upper-triangular future mask (1 = future/masked position)
            let mut mask_data = vec![0u8; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = 1;
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), q.device())?;
            // Use scalar broadcast instead of allocating full neg_inf tensor
            let neg_inf = Tensor::new(f32::NEG_INFINITY, q.device())?
                .broadcast_as(attn.shape())?;
            mask.broadcast_as(attn.shape())?
                .where_cond(&neg_inf, &attn)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(&gate.contiguous()?)? * up.contiguous()?
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        // ln(1 + exp(clamp(x, -inf, 88))) with max(x, result)
        if x.dtype() == DType::F32 {
            let data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let shape = x.dims();
            let mut out = data;
            for v in out.iter_mut() {
                let clamped = v.min(88.0);
                let sp = (1.0 + clamped.exp()).ln();
                *v = v.max(sp);
            }
            return Tensor::from_vec(out, shape, x.device());
        }
        let t88 = Tensor::full(88.0f32, x.shape(), x.device())?.to_dtype(x.dtype())?;
        let clamped = x.minimum(&t88)?;
        let sp = (clamped.exp()? + 1.0)?.log()?;
        x.maximum(&sp)
    }

    fn rms_norm_gated(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)?;
        n * candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?
    }

    fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        let res = (a + b)?;
        let normed = candle_nn::ops::rms_norm(&res.contiguous()?, weight, eps)?;
        Ok((res, normed))
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        // x is (batch, channels, time) — norm over channels at each time step
        if x.dtype() == DType::F32 {
            let x = x.contiguous()?;
            let batch = x.dim(0)?;
            let channels = x.dim(1)?;
            let time = x.dim(2)?;
            let data = x.flatten_all()?.to_vec1::<f32>()?;
            let w = weight.to_vec1::<f32>()?;
            let eps64 = eps as f64;
            let mut out = vec![0f32; data.len()];
            for b in 0..batch {
                let batch_off = b * channels * time;
                for t in 0..time {
                    // Gather channel values at this time step
                    let mut sum_sq = 0f64;
                    for c in 0..channels {
                        let v = data[batch_off + c * time + t] as f64;
                        sum_sq += v * v;
                    }
                    let rms = (sum_sq / channels as f64 + eps64).sqrt();
                    let inv_rms = 1.0 / rms;
                    for (c, &wc) in w.iter().enumerate() {
                        let idx = batch_off + c * time + t;
                        out[idx] = (data[idx] as f64 * inv_rms * wc as f64) as f32;
                    }
                }
            }
            return Tensor::from_vec(out, (batch, channels, time), x.device());
        }
        // Fallback: transpose approach
        x.transpose(1, 2)?
            .contiguous()
            .and_then(|t| candle_nn::ops::rms_norm(&t, weight, eps))?
            .transpose(1, 2)
    }

    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        _kernel_size: usize,
        _channels: usize,
    ) -> Result<Tensor> {
        // window: (batch, channels, kernel_size), weight: (channels, kernel_size)
        let w = weight.unsqueeze(0)?;
        let conv = window.broadcast_mul(&w)?.sum(D::Minus1)?;
        candle_nn::ops::silu(&conv)
    }

    fn depthwise_conv1d_bias(
        &self,
        padded_input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        _channels: usize,
    ) -> Result<Tensor> {
        // padded_input: (batch, channels, padded_time)
        // weight: (channels, kernel_size)
        // bias: (channels,)
        // output: (batch, channels, output_time) where output_time = padded_time - kernel_size + 1
        let padded_input = padded_input.contiguous()?;
        let batch = padded_input.dim(0)?;
        let channels = padded_input.dim(1)?;
        let padded_time = padded_input.dim(2)?;
        let output_time = padded_time - kernel_size + 1;

        // Raw f32 path: single pass, no intermediate tensor allocations
        if padded_input.dtype() == DType::F32 {
            let input_data = padded_input.flatten_all()?.to_vec1::<f32>()?;
            let weight_data = weight.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let bias_data = bias.to_vec1::<f32>()?;
            let mut out = vec![0f32; batch * channels * output_time];

            for b in 0..batch {
                for (c, &bias_val) in bias_data.iter().enumerate() {
                    let in_off = (b * channels + c) * padded_time;
                    let out_off = (b * channels + c) * output_time;
                    let w_off = c * kernel_size;
                    for t in 0..output_time {
                        let mut sum = bias_val;
                        for k in 0..kernel_size {
                            sum += input_data[in_off + t + k] * weight_data[w_off + k];
                        }
                        out[out_off + t] = sum;
                    }
                }
            }
            return Tensor::from_vec(out, (batch, channels, output_time), padded_input.device());
        }

        // Fallback: tensor ops — sum shifted slices weighted by each kernel tap
        let w = weight.unsqueeze(0)?.unsqueeze(D::Minus1)?; // (1, channels, kernel_size, 1)
        let mut windows = Vec::with_capacity(kernel_size);
        for k in 0..kernel_size {
            windows.push(padded_input.narrow(2, k, output_time)?);
        }
        // Stack windows: (batch, channels, kernel_size, output_time)
        let stacked = Tensor::stack(&windows, 2)?;
        let conv = stacked.broadcast_mul(&w)?.sum(2)?;
        let bias_view = bias.unsqueeze(0)?.unsqueeze(2)?; // (1, channels, 1)
        conv.broadcast_add(&bias_view)
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
        // Concatenate [ctx, input] along time dimension, then run conv
        let combined = Tensor::cat(&[ctx, input], 2)?;
        self.depthwise_conv1d_bias(&combined, weight, bias, kernel_size, channels)
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b)? + c
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        x * y.exp()?
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a - b)? * c
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        // a + b * c where c is (channels,) or already broadcastable
        let c_broadcast = if c.dims().len() == 1 {
            c.unsqueeze(0)?.unsqueeze(2)?
        } else {
            c.clone()
        };
        a + b.broadcast_mul(&c_broadcast)?
    }

    fn adaln_modulate(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        // rms_norm(x) * (1 + scale) + shift
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, norm_weight, eps)?;
        n.broadcast_mul(&(scale + 1.0)?)? + shift
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        x.to_dtype(DType::F32)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        x.to_dtype(DType::F16)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        x.to_dtype(DType::BF16)
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
