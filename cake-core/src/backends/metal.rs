//! Metal compute backend with MSL shaders + candle tensor op fallbacks.
//!
//! Self-contained: owns the MSL shader source and CustomOp structs for the 2
//! MSL-accelerated ops (SiluMul, StableSoftplus). All other trait methods use
//! plain candle tensor operations with no dependency on the ops/ module.
//!
//! Metal command buffer accumulation >25 commands causes catastrophic 50x slowdown.
//! The `synchronize()` method flushes the command buffer and is called at strategic
//! points during forward passes (see GatedDeltaNet, Qwen3_5FullAttention).

use candle_core::{backend::BackendStorage as _, CpuStorage, DType, Device, Layout, Result, Shape, Tensor, D};

use super::ComputeBackend;

// ─── MSL shader source ──────────────────────────────────────────────
const FUSED_OPS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── stable_softplus: ln(1 + exp(clamp(x, -inf, 88))) with max(x, result) ───
kernel void stable_softplus_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float x = input[idx];
    float clamped = min(x, 88.0f);
    float sp = log(exp(clamped) + 1.0f);
    output[idx] = max(x, sp);
}

kernel void stable_softplus_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float x = float(input[idx]);
    float clamped = min(x, 88.0f);
    float sp = log(exp(clamped) + 1.0f);
    output[idx] = half(max(x, sp));
}

// ─── silu_mul: silu(gate) * up ───────────────────────────────────────────────
kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float g = float(gate[idx]);
    output[idx] = half((g / (1.0f + exp(-g))) * float(up[idx]));
}
"#;

// ─── CustomOp structs for MSL-accelerated ops (2 total) ─────────────

struct MetalSiluMul;

impl candle_core::CustomOp2 for MetalSiluMul {
    fn name(&self) -> &'static str {
        "metal_silu_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            gate: &[T],
            l1: &Layout,
            up: &[T],
            l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let gate = match l1.contiguous_offsets() {
                Some((o1, o2)) => &gate[o1..o2],
                None => candle_core::bail!("silu_mul: gate must be contiguous"),
            };
            let up = match l2.contiguous_offsets() {
                Some((o1, o2)) => &up[o1..o2],
                None => candle_core::bail!("silu_mul: up must be contiguous"),
            };
            let n = gate.len();
            let dst: Vec<T> = (0..n)
                .map(|i| {
                    let x = gate[i];
                    let y = up[i];
                    x / (T::one() + (-x).exp()) * y
                })
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l1.shape().clone()))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(a), C::BF16(b)) => inner(a, l1, b, l2),
            (C::F16(a), C::F16(b)) => inner(a, l1, b, l2),
            (C::F32(a), C::F32(b)) => inner(a, l1, b, l2),
            (C::F64(a), C::F64(b)) => inner(a, l1, b, l2),
            _ => candle_core::bail!("silu_mul: unsupported dtype {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s1.device();
        let el = l1.shape().elem_count();
        let kernel_name = match s1.dtype() {
            DType::F32 => "silu_mul_f32",
            DType::F16 => "silu_mul_f16",
            dt => candle_core::bail!("silu_mul metal: unsupported dtype {dt:?}"),
        };
        let lib = device.new_library_with_source(FUSED_OPS_MSL, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal shader compile: {e}")))?;
        let func = lib.get_function(kernel_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal get_function: {e}")))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("metal pipeline: {e}")))?;
        let output = device.new_buffer(el, s1.dtype(), "silu_mul")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s1.buffer(), off1));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s2.buffer(), off2));
        candle_metal_kernels::utils::set_param(&encoder, 2, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 3, el as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s1.dtype()), l1.shape().clone()))
    }
}

struct MetalStableSoftplus;

impl candle_core::CustomOp1 for MetalStableSoftplus {
    fn name(&self) -> &'static str {
        "metal_stable_softplus"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                Some((o1, o2)) => &src[o1..o2],
                None => candle_core::bail!("stable_softplus: input must be contiguous"),
            };
            let t88 = T::from(88.0).unwrap();
            let one = T::one();
            let dst: Vec<T> = src
                .iter()
                .map(|&x| {
                    let clamped = if x < t88 { x } else { t88 };
                    let sp = (clamped.exp() + one).ln();
                    if x > sp { x } else { sp }
                })
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, layout.shape().clone()))
        }

        use CpuStorage as C;
        match s {
            C::BF16(s) => inner(s, l),
            C::F16(s) => inner(s, l),
            C::F32(s) => inner(s, l),
            C::F64(s) => inner(s, l),
            _ => candle_core::bail!("stable_softplus: unsupported dtype {:?}", s.dtype()),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &candle_core::MetalStorage,
        l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s.device();
        let el = l.shape().elem_count();
        let kernel_name = match s.dtype() {
            DType::F32 => "stable_softplus_f32",
            DType::F16 => "stable_softplus_f16",
            dt => candle_core::bail!("stable_softplus metal: unsupported dtype {dt:?}"),
        };
        let lib = device.new_library_with_source(FUSED_OPS_MSL, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal shader compile: {e}")))?;
        let func = lib.get_function(kernel_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal get_function: {e}")))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("metal pipeline: {e}")))?;
        let output = device.new_buffer(el, s.dtype(), "stable_softplus")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = l.start_offset() * s.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s.buffer(), offset));
        candle_metal_kernels::utils::set_param(&encoder, 1, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 2, el as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s.dtype()), l.shape().clone()))
    }
}

// ─── MetalBackend ────────────────────────────────────────────────────

/// Metal backend -- MSL kernels for fused activations, candle tensor ops for everything else.
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

    // ── Attention ────────────────────────────────────────────────────

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        // Fused SDPA on Metal -- single kernel, native GQA (no repeat_kv needed)
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        candle_nn::ops::sdpa(&q, &k, &v, None, causal, scale, 1.0)
    }

    // ── Fused activations (MSL shaders) ──────────────────────────────

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        gate.apply_op2_no_bwd(up, &MetalSiluMul)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        x.apply_op1_no_bwd(&MetalStableSoftplus)
    }

    // ── Normalization (candle tensor ops) ────────────────────────────

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let normed = candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)?;
        let gate = candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?;
        normed.mul(&gate)
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        let res = (a + b)?;
        let normed = candle_nn::ops::rms_norm(&res.contiguous()?, weight, eps)?;
        Ok((res, normed))
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        // (batch, channels, time) -> transpose to (batch, time, channels) for rms_norm
        let xt = x.transpose(1, 2)?.contiguous()?;
        let normed = candle_nn::ops::rms_norm(&xt, weight, eps)?;
        normed.transpose(1, 2)?.contiguous()
    }

    // ── Convolutions (candle tensor ops) ─────────────────────────────

    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        _kernel_size: usize,
        _channels: usize,
    ) -> Result<Tensor> {
        let w = weight.unsqueeze(0)?;
        let dot = window.broadcast_mul(&w)?.sum(D::Minus1)?;
        candle_nn::ops::silu(&dot)
    }

    fn depthwise_conv1d_bias(
        &self,
        padded_input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        _channels: usize,
    ) -> Result<Tensor> {
        let (_b, _channels, t_padded) = padded_input.dims3()?;
        let out_len = t_padded - kernel_size + 1;
        let mut acc: Option<Tensor> = None;
        for k in 0..kernel_size {
            let slice = padded_input.narrow(2, k, out_len)?;
            let w_k = weight.narrow(1, k, 1)?.unsqueeze(0)?; // (1, channels, 1)
            let term = slice.broadcast_mul(&w_k)?;
            acc = Some(match acc {
                None => term,
                Some(a) => (a + term)?,
            });
        }
        let out = acc.unwrap();
        let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
        out.broadcast_add(&bias)
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
        let merged = Tensor::cat(&[ctx, input], 2)?;
        self.depthwise_conv1d_bias(&merged, weight, bias, kernel_size, channels)
    }

    // ── Elementwise (candle tensor ops) ──────────────────────────────

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ((a + b)? + c)?
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        (x * y.exp()?)?
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ((a - b)? * c)?
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?
    }

    // ── AdaLN (candle tensor ops) ────────────────────────────────────

    fn adaln_modulate(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, norm_weight, eps)?;
        (n.broadcast_mul(&(scale + 1.0)?)? + shift)?
    }

    // ── F8 dequantization ────────────────────────────────────────────

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        crate::backends::f8_dequant::f8e4m3_to_f32(x)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        crate::backends::f8_dequant::f8e4m3_to_f16(x)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        crate::backends::f8_dequant::f8e4m3_to_bf16(x)
    }

    // ── Device control ───────────────────────────────────────────────

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}
