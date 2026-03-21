//! Fused CUDA/Metal/CPU kernels for inference-critical operations.
//!
//! Each fused op replaces 2-5 separate kernel launches with a single launch,
//! saving ~4-7µs per launch on modern GPUs.
//!
//! Fused operations:
//! - `silu_mul(gate, up)` → silu(gate) * up (MLP activation, saves 1 launch)
//! - `stable_softplus(x)` → ln(1+exp(clamp(x))) clamped (GDN gates, saves 4 launches)
//! - `rms_norm_gated(x, z, weight, eps)` → rms_norm(x,w) * silu(z) (GDN norm, saves 2 launches)
//! - `depthwise_conv1d_silu(window, weight)` → dot(window, weight) + silu (GDN conv, saves 2 launches)

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

// ─── Metal shader source (compiled at runtime via new_library_with_source) ──
#[cfg(feature = "metal")]
mod metal_shaders {
    /// Metal Shading Language (MSL) kernels for fused ops.
    /// Compiled once at first use and cached by the Metal pipeline.
    pub const FUSED_OPS_MSL: &str = r#"
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

// ─── depthwise_conv1d_silu: dot(window, weight) per channel + silu ───────────
// window: (batch, channels, kernel_size), weight: (channels, kernel_size)
// output: (batch, channels) with silu activation
kernel void depthwise_conv1d_silu_f32(
    device const float* window [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    uint2 idx [[thread_position_in_grid]]  // (channel, batch)
) {
    uint b = idx.y;
    uint c = idx.x;
    if (c >= channels) return;
    float sum = 0.0f;
    uint win_offset = b * channels * kernel_size + c * kernel_size;
    uint w_offset = c * kernel_size;
    for (uint k = 0; k < kernel_size; k++) {
        sum += window[win_offset + k] * weight[w_offset + k];
    }
    // silu activation
    output[b * channels + c] = sum / (1.0f + exp(-sum));
}
"#;
}

// ─── PTX module (compiled from src/cuda/fused_ops.cu) ───────────────
#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/fused_ops_ptx.rs"));
}

#[cfg(feature = "cuda")]
const FUSED_OPS_PTX: &str = ptx::FUSED_OPS;

// ─── F8E4M3 → F32 dequantization (software, works on all SM) ───────

/// Dequantize F8E4M3 tensor to F32.
/// Uses our custom CUDA kernel on GPU (works on SM80/A100 where candle lacks native F8 support).
/// Falls back to CPU dequantization on non-CUDA devices.
pub fn f8e4m3_to_f32(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F8E4M3 {
        return x.to_dtype(candle_core::DType::F32);
    }
    // Try candle's native cast first (works on SM89+)
    if let Ok(t) = x.to_dtype(candle_core::DType::F32) {
        return Ok(t);
    }
    // Fall back to our software kernel
    x.apply_op1_no_bwd(&F8E4M3ToF32)
}

/// Dequantize F8E4M3 tensor to BF16.
/// Direct F8→BF16 avoids F16 intermediate when using BF16 compute.
pub fn f8e4m3_to_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F8E4M3 {
        return x.to_dtype(candle_core::DType::BF16);
    }
    x.apply_op1_no_bwd(&F8E4M3ToBF16)
}

struct F8E4M3ToBF16;

impl candle_core::CustomOp1 for F8E4M3ToBF16 {
    fn name(&self) -> &'static str {
        "f8e4m3_to_bf16"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (f32_storage, shape) = F8E4M3ToF32.cpu_fwd(s, l)?;
        match f32_storage {
            CpuStorage::F32(data) => {
                let bf16_data: Vec<half::bf16> =
                    data.iter().map(|&v| half::bf16::from_f32(v)).collect();
                Ok((CpuStorage::BF16(bf16_data), shape))
            }
            _ => candle_core::bail!("expected F32 from F8E4M3ToF32"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device();
        let src = match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F8E4M3(s) => s,
            _ => candle_core::bail!("f8e4m3_to_bf16: expected F8E4M3 storage"),
        };
        let src = match l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => candle_core::bail!("f8e4m3_to_bf16: input must be contiguous"),
        };
        let el = l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let func =
            dev.get_or_load_custom_func("f8e4m3_to_bf16", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<half::bf16>(el)? };
        let mut builder = func.builder();
        builder.arg(&el);
        builder.arg(&src);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }
}

/// Dequantize F8E4M3 tensor to F16.
/// Uses our custom CUDA kernel. F16 matmul is 2x faster than F32 on A100.
pub fn f8e4m3_to_f16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F8E4M3 {
        return x.to_dtype(candle_core::DType::F16);
    }
    x.apply_op1_no_bwd(&F8E4M3ToF16)
}

struct F8E4M3ToF16;

impl candle_core::CustomOp1 for F8E4M3ToF16 {
    fn name(&self) -> &'static str {
        "f8e4m3_to_f16"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        // CPU: decode F8 bytes to f32, then quantize to f16
        let (f32_storage, shape) = F8E4M3ToF32.cpu_fwd(s, l)?;
        match f32_storage {
            CpuStorage::F32(data) => {
                let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
                Ok((CpuStorage::F16(f16_data), shape))
            }
            _ => candle_core::bail!("expected F32 from F8E4M3ToF32"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device();
        let src = match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F8E4M3(s) => s,
            _ => candle_core::bail!("f8e4m3_to_f16: expected F8E4M3 storage"),
        };
        let src = match l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => candle_core::bail!("f8e4m3_to_f16: input must be contiguous"),
        };
        let el = l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let func =
            dev.get_or_load_custom_func("f8e4m3_to_f16", "cake_fused_ops", FUSED_OPS_PTX)?;
        // half::f16 is used by CudaStorageSlice::F16
        let out = unsafe { dev.alloc::<half::f16>(el)? };
        let mut builder = func.builder();
        builder.arg(&el);
        builder.arg(&src);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::F16(out),
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }
}

struct F8E4M3ToF32;

impl candle_core::CustomOp1 for F8E4M3ToF32 {
    fn name(&self) -> &'static str {
        "f8e4m3_to_f32"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let data = match s {
            CpuStorage::F8E4M3(data) => data,
            _ => candle_core::bail!("f8e4m3_to_f32: expected F8E4M3, got {:?}", s.dtype()),
        };
        let data = match l.contiguous_offsets() {
            Some((o1, o2)) => &data[o1..o2],
            None => candle_core::bail!("f8e4m3_to_f32: input must be contiguous"),
        };
        let dst: Vec<f32> = data
            .iter()
            .map(|&b| {
                let bits = b.to_bits();
                let sign = (bits >> 7) & 1;
                let exp = (bits >> 3) & 0xF;
                let mant = bits & 0x7;
                let result = if exp == 0 && mant == 0 {
                    0.0f32
                } else if exp == 0 {
                    f32::from(mant) / 8.0 * 2.0f32.powi(-6)
                } else if exp == 0xF && mant == 0x7 {
                    f32::NAN
                } else {
                    (1.0 + f32::from(mant) / 8.0) * 2.0f32.powi(i32::from(exp) - 7)
                };
                if sign == 1 { -result } else { result }
            })
            .collect();
        let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device();
        let src = match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F8E4M3(s) => s,
            _ => candle_core::bail!("f8e4m3_to_f32: expected F8E4M3 storage"),
        };
        let src = match l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => candle_core::bail!("f8e4m3_to_f32: input must be contiguous"),
        };
        let el = l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let func =
            dev.get_or_load_custom_func("f8e4m3_to_f32", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<f32>(el)? };
        let mut builder = func.builder();
        builder.arg(&el);
        builder.arg(&src);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }
}

// ─── silu_mul: silu(gate) * up ──────────────────────────────────────

struct SiluMul;

impl candle_core::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "silu_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use rayon::prelude::*;

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
            let mut dst = vec![T::zero(); n];
            const CHUNK: usize = 8192;
            dst.par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, dst_chunk)| {
                    let start = chunk_idx * CHUNK;
                    for (i, d) in dst_chunk.iter_mut().enumerate() {
                        let x = gate[start + i];
                        let y = up[start + i];
                        // silu(x) * y = x * sigmoid(x) * y
                        *d = x / (T::one() + (-x).exp()) * y;
                    }
                });
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

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map2, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        struct S;
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                gate: &CudaSlice<T>,
                l1: &Layout,
                up: &CudaSlice<T>,
                l2: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let gate = match l1.contiguous_offsets() {
                    Some((o1, o2)) => gate.slice(o1..o2),
                    None => candle_core::bail!("silu_mul: gate must be contiguous"),
                };
                let up = match l2.contiguous_offsets() {
                    Some((o1, o2)) => up.slice(o1..o2),
                    None => candle_core::bail!("silu_mul: up must be contiguous"),
                };
                let el = l1.shape().elem_count();
                let cfg = LaunchConfig::for_num_elems(el as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("silu_mul"),
                    "cake_fused_ops",
                    FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s1.device();
        let slice = S.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            l1.shape().clone(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::DType;
        let device = s1.device();
        let el = l1.shape().elem_count();
        let kernel_name = match s1.dtype() {
            DType::F32 => "silu_mul_f32",
            DType::F16 => "silu_mul_f16",
            dt => candle_core::bail!("silu_mul metal: unsupported dtype {dt:?}"),
        };
        let lib = device.new_library_with_source(metal_shaders::FUSED_OPS_MSL, None)
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

/// Fused silu(gate) * up — single kernel on CUDA and Metal.
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &SiluMul)
}

// ─── stable_softplus: ln(1+exp(clamp(x,−∞,88))) with max(x,·) ──────

struct StableSoftplus;

impl candle_core::CustomOp1 for StableSoftplus {
    fn name(&self) -> &'static str {
        "stable_softplus"
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

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map1, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    Some((o1, o2)) => src.slice(o1..o2),
                    None => candle_core::bail!("stable_softplus: input must be contiguous"),
                };
                let el = layout.shape().elem_count();
                let cfg = LaunchConfig::for_num_elems(el as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("stable_softplus"),
                    "cake_fused_ops",
                    FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&src);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s.device();
        let slice = S.map(&s.slice, dev, l)?;
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &candle_core::MetalStorage,
        l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::DType;
        let device = s.device();
        let el = l.shape().elem_count();
        let kernel_name = match s.dtype() {
            DType::F32 => "stable_softplus_f32",
            DType::F16 => "stable_softplus_f16",
            dt => candle_core::bail!("stable_softplus metal: unsupported dtype {dt:?}"),
        };
        let lib = device.new_library_with_source(metal_shaders::FUSED_OPS_MSL, None)
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

/// Fused stable softplus: ln(1 + exp(clamp(x, -inf, 88))) with max(x, result).
///
/// On CUDA: single fused kernel (1 dispatch).
/// On Metal: single fused kernel via MSL shader (1 dispatch).
/// On CPU: scalar implementation via CustomOp1.
pub fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&StableSoftplus)
}

// ─── rms_norm_gated: rms_norm(x, weight) * silu(z) ─────────────────

struct RmsNormGated {
    eps: f32,
}

impl candle_core::CustomOp3 for RmsNormGated {
    fn name(&self) -> &'static str {
        "rms_norm_gated"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_z: &CpuStorage,
        l_z: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use rayon::prelude::*;

        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            x: &[T],
            l_x: &Layout,
            z: &[T],
            l_z: &Layout,
            w: &[T],
            l_w: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("rms_norm_gated: x must be contiguous"),
            };
            let z = match l_z.contiguous_offsets() {
                Some((o1, o2)) => &z[o1..o2],
                None => candle_core::bail!("rms_norm_gated: z must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("rms_norm_gated: weight must be contiguous"),
            };
            let dims = l_x.shape().dims();
            let n_cols = dims[dims.len() - 1];
            let el = l_x.shape().elem_count();
            let mut dst = vec![T::zero(); el];

            dst.par_chunks_mut(n_cols)
                .enumerate()
                .for_each(|(row, dst_row)| {
                    let x_row = &x[row * n_cols..(row + 1) * n_cols];
                    let z_row = &z[row * n_cols..(row + 1) * n_cols];
                    let sum2: f32 = x_row.iter().map(|v| { let f: f32 = v.as_(); f * f }).sum();
                    let inv_rms = 1.0 / (sum2 / n_cols as f32 + eps).sqrt();
                    for i in 0..n_cols {
                        let xv: f32 = x_row[i].as_();
                        let wv: f32 = w[i].as_();
                        let zv: f32 = z_row[i].as_();
                        let silu_z = zv / (1.0 + (-zv).exp());
                        dst_row[i] = T::from_f32(xv * inv_rms * wv * silu_z)
                            .unwrap_or_else(T::nan);
                    }
                });

            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s_x, s_z, s_w) {
            (C::BF16(x), C::BF16(z), C::BF16(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            (C::F16(x), C::F16(z), C::F16(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            (C::F32(x), C::F32(z), C::F32(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            (C::F64(x), C::F64(z), C::F64(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            _ => candle_core::bail!("rms_norm_gated: unsupported dtype {:?}", s_x.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_x: &candle_core::CudaStorage,
        l_x: &Layout,
        s_z: &candle_core::CudaStorage,
        l_z: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            x: &CudaSlice<T>,
            l_x: &Layout,
            z: &CudaSlice<T>,
            l_z: &Layout,
            w: &CudaSlice<T>,
            l_w: &Layout,
            dev: &CudaDevice,
            eps: f32,
        ) -> Result<CudaSlice<T>> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => x.slice(o1..o2),
                None => candle_core::bail!("rms_norm_gated: x must be contiguous"),
            };
            let z = match l_z.contiguous_offsets() {
                Some((o1, o2)) => z.slice(o1..o2),
                None => candle_core::bail!("rms_norm_gated: z must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => w.slice(o1..o2),
                None => candle_core::bail!("rms_norm_gated: weight must be contiguous"),
            };

            let dims = l_x.shape().dims();
            let n_cols = dims[dims.len() - 1];
            let el = l_x.shape().elem_count();
            let n_rows = el / n_cols;

            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("rms_norm_gated"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&z);
            builder.arg(&w);
            builder.arg(&out);
            candle_core::builder_arg!(builder, n_cols as i32, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_x.device();

        let slice = match (&s_x.slice, &s_z.slice, &s_w.slice) {
            (S::BF16(x), S::BF16(z), S::BF16(w)) => {
                S::BF16(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            (S::F16(x), S::F16(z), S::F16(w)) => {
                S::F16(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            (S::F32(x), S::F32(z), S::F32(w)) => {
                S::F32(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            (S::F64(x), S::F64(z), S::F64(w)) => {
                S::F64(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            _ => candle_core::bail!("rms_norm_gated: unsupported dtype"),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            l_x.shape().clone(),
        ))
    }
}

/// Fused rms_norm(x, weight) * silu(z) — replaces 3 kernel launches with 1 on CUDA.
/// On Metal, uses candle's built-in ops (rms_norm + silu + mul).
pub fn rms_norm_gated(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        let normed = candle_nn::ops::rms_norm(x, weight, eps)?;
        let gate = candle_nn::ops::silu(z)?;
        return normed.mul(&gate);
    }
    x.apply_op3_no_bwd(z, weight, &RmsNormGated { eps })
}

// ─── add_rms_norm: rms_norm(a + b, weight, eps) with residual ────────

struct AddRmsNorm {
    eps: f32,
    n_cols: usize,
}

impl candle_core::CustomOp3 for AddRmsNorm {
    fn name(&self) -> &'static str {
        "add_rms_norm"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use rayon::prelude::*;

        #[allow(clippy::too_many_arguments)]
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            w: &[T],
            l_w: &Layout,
            eps: f32,
            n_cols: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("add_rms_norm: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("add_rms_norm: b must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("add_rms_norm: weight must be contiguous"),
            };
            let dims = l_a.shape().dims();
            let el = l_a.shape().elem_count();
            let n_rows = el / n_cols;
            // Output: [sum, normed] concatenated on last dim
            let out_cols = n_cols * 2;
            let mut dst = vec![T::zero(); n_rows * out_cols];

            dst.par_chunks_mut(out_cols)
                .enumerate()
                .for_each(|(row, dst_row)| {
                    let a_row = &a[row * n_cols..(row + 1) * n_cols];
                    let b_row = &b[row * n_cols..(row + 1) * n_cols];
                    // Compute sum and sum of squares
                    let mut sum2: f32 = 0.0;
                    for i in 0..n_cols {
                        let s: f32 = a_row[i].as_() + b_row[i].as_();
                        dst_row[i] = T::from_f32(s).unwrap_or_else(T::nan);
                        sum2 += s * s;
                    }
                    let inv_rms = 1.0 / (sum2 / n_cols as f32 + eps).sqrt();
                    for i in 0..n_cols {
                        let s: f32 = dst_row[i].as_();
                        let wv: f32 = w[i].as_();
                        dst_row[n_cols + i] =
                            T::from_f32(s * inv_rms * wv).unwrap_or_else(T::nan);
                    }
                });

            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = out_cols;
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(&out_dims)))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_w) {
            (C::BF16(a), C::BF16(b), C::BF16(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            (C::F16(a), C::F16(b), C::F16(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            (C::F32(a), C::F32(b), C::F32(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            (C::F64(a), C::F64(b), C::F64(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            _ => candle_core::bail!("add_rms_norm: unsupported dtype {:?}", s_a.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        let n_cols = self.n_cols;
        let eps = self.eps;
        let el = l_a.shape().elem_count();
        let n_rows = el / n_cols;
        let out_cols = n_cols * 2;

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>,
            l_a: &Layout,
            b: &CudaSlice<T>,
            l_b: &Layout,
            w: &CudaSlice<T>,
            l_w: &Layout,
            dev: &CudaDevice,
            n_cols: usize,
            n_rows: usize,
            eps: f32,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("add_rms_norm: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("add_rms_norm: b must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => w.slice(o1..o2),
                None => candle_core::bail!("add_rms_norm: weight must be contiguous"),
            };
            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("add_rms_norm"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out_el = n_rows * n_cols * 2;
            let out = unsafe { dev.alloc::<T>(out_el)? };
            let mut builder = func.builder();
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&w);
            builder.arg(&out);
            candle_core::builder_arg!(builder, n_cols as i32, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as SS;
        let dev = s_a.device();

        let slice = match (&s_a.slice, &s_b.slice, &s_w.slice) {
            (SS::BF16(a), SS::BF16(b), SS::BF16(w)) => {
                SS::BF16(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            (SS::F16(a), SS::F16(b), SS::F16(w)) => {
                SS::F16(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            (SS::F32(a), SS::F32(b), SS::F32(w)) => {
                SS::F32(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            (SS::F64(a), SS::F64(b), SS::F64(w)) => {
                SS::F64(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            _ => candle_core::bail!("add_rms_norm: unsupported dtype"),
        };

        let mut out_dims = l_a.shape().dims().to_vec();
        *out_dims.last_mut().unwrap() = out_cols;

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            Shape::from_dims(&out_dims),
        ))
    }
}

/// Fused residual add + RmsNorm — replaces 2 kernel launches with 1.
/// Returns (residual_sum, rms_normed) as contiguous tensors.
/// `a` and `b` are added, then the sum is RMS-normalized with `weight` and `eps`.
pub fn add_rms_norm(
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let n_cols = *a.dims().last().unwrap();
    let combined = a.apply_op3_no_bwd(b, weight, &AddRmsNorm { eps, n_cols })?;
    let residual = combined.narrow(candle_core::D::Minus1, 0, n_cols)?.contiguous()?;
    let normed = combined.narrow(candle_core::D::Minus1, n_cols, n_cols)?.contiguous()?;
    Ok((residual, normed))
}

/// Fused residual add + RmsNorm — returns ONLY the normed result.
/// Use when caller will reconstruct residual via add3 at the end of the block.
pub fn add_rms_norm_normed(
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    // Fuse add+norm into 1 kernel, return only the normed half
    let n_cols = *a.dims().last().unwrap();
    let combined = a.apply_op3_no_bwd(b, weight, &AddRmsNorm { eps, n_cols })?;
    combined.narrow(candle_core::D::Minus1, n_cols, n_cols)?.contiguous()
}

// ─── add3: a + b + c ────────────────────────────────────────────────

struct Add3;

impl candle_core::CustomOp3 for Add3 {
    fn name(&self) -> &'static str {
        "add3"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_c: &CpuStorage,
        l_c: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            c: &[T],
            l_c: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("add3: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("add3: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => &c[o1..o2],
                None => candle_core::bail!("add3: c must be contiguous"),
            };
            let dst: Vec<T> = a
                .iter()
                .zip(b)
                .zip(c)
                .map(|((&av, &bv), &cv)| av + bv + cv)
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_a.shape().clone()))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_c) {
            (C::BF16(a), C::BF16(b), C::BF16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F16(a), C::F16(b), C::F16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F32(a), C::F32(b), C::F32(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F64(a), C::F64(b), C::F64(c)) => inner(a, l_a, b, l_b, c, l_c),
            _ => candle_core::bail!("add3: unsupported dtype {:?}", s_a.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_c: &candle_core::CudaStorage,
        l_c: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>,
            l_a: &Layout,
            b: &CudaSlice<T>,
            l_b: &Layout,
            c: &CudaSlice<T>,
            l_c: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("add3: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("add3: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => c.slice(o1..o2),
                None => candle_core::bail!("add3: c must be contiguous"),
            };
            let el = l_a.shape().elem_count();
            let cfg = LaunchConfig::for_num_elems(el as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("add3"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&el);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as SS;
        let dev = s_a.device();

        let slice = match (&s_a.slice, &s_b.slice, &s_c.slice) {
            (SS::BF16(a), SS::BF16(b), SS::BF16(c)) => {
                SS::BF16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F16(a), SS::F16(b), SS::F16(c)) => {
                SS::F16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F32(a), SS::F32(b), SS::F32(c)) => {
                SS::F32(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F64(a), SS::F64(b), SS::F64(c)) => {
                SS::F64(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            _ => candle_core::bail!("add3: unsupported dtype"),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            l_a.shape().clone(),
        ))
    }
}

/// Fused a + b + c — replaces 2 kernel launches (add + add) with 1.
pub fn add3(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.apply_op3_no_bwd(b, c, &Add3)
}

// ─── exp_mul: x * exp(y) ────────────────────────────────────────────

struct ExpMul;

impl candle_core::CustomOp2 for ExpMul {
    fn name(&self) -> &'static str {
        "exp_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            x: &[T],
            l1: &Layout,
            y: &[T],
            l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l1.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("exp_mul: x must be contiguous"),
            };
            let y = match l2.contiguous_offsets() {
                Some((o1, o2)) => &y[o1..o2],
                None => candle_core::bail!("exp_mul: y must be contiguous"),
            };
            let dst: Vec<T> = x.iter().zip(y).map(|(&a, &b)| a * b.exp()).collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l1.shape().clone()))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(a), C::BF16(b)) => inner(a, l1, b, l2),
            (C::F16(a), C::F16(b)) => inner(a, l1, b, l2),
            (C::F32(a), C::F32(b)) => inner(a, l1, b, l2),
            (C::F64(a), C::F64(b)) => inner(a, l1, b, l2),
            _ => candle_core::bail!("exp_mul: unsupported dtype {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map2, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        struct S;
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                x: &CudaSlice<T>,
                l1: &Layout,
                y: &CudaSlice<T>,
                l2: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let x = match l1.contiguous_offsets() {
                    Some((o1, o2)) => x.slice(o1..o2),
                    None => candle_core::bail!("exp_mul: x must be contiguous"),
                };
                let y = match l2.contiguous_offsets() {
                    Some((o1, o2)) => y.slice(o1..o2),
                    None => candle_core::bail!("exp_mul: y must be contiguous"),
                };
                let el = l1.shape().elem_count();
                let cfg = LaunchConfig::for_num_elems(el as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("exp_mul"),
                    "cake_fused_ops",
                    FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&x);
                builder.arg(&y);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s1.device();
        let slice = S.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            l1.shape().clone(),
        ))
    }
}

/// Fused x * exp(y) — replaces 2 kernel launches (exp + mul) with 1.
pub fn exp_mul(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.apply_op2_no_bwd(y, &ExpMul)
}

// ─── sub_mul: (a - b) * c ───────────────────────────────────────────

struct SubMul;

impl candle_core::CustomOp3 for SubMul {
    fn name(&self) -> &'static str {
        "sub_mul"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_c: &CpuStorage,
        l_c: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            c: &[T],
            l_c: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("sub_mul: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("sub_mul: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => &c[o1..o2],
                None => candle_core::bail!("sub_mul: c must be contiguous"),
            };
            let dst: Vec<T> = a
                .iter()
                .zip(b)
                .zip(c)
                .map(|((&av, &bv), &cv)| (av - bv) * cv)
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_a.shape().clone()))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_c) {
            (C::BF16(a), C::BF16(b), C::BF16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F16(a), C::F16(b), C::F16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F32(a), C::F32(b), C::F32(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F64(a), C::F64(b), C::F64(c)) => inner(a, l_a, b, l_b, c, l_c),
            _ => candle_core::bail!("sub_mul: unsupported dtype {:?}", s_a.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_c: &candle_core::CudaStorage,
        l_c: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>,
            l_a: &Layout,
            b: &CudaSlice<T>,
            l_b: &Layout,
            c: &CudaSlice<T>,
            l_c: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("sub_mul: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("sub_mul: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => c.slice(o1..o2),
                None => candle_core::bail!("sub_mul: c must be contiguous"),
            };
            let el = l_a.shape().elem_count();
            let cfg = LaunchConfig::for_num_elems(el as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("sub_mul"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&el);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as SS;
        let dev = s_a.device();

        let slice = match (&s_a.slice, &s_b.slice, &s_c.slice) {
            (SS::BF16(a), SS::BF16(b), SS::BF16(c)) => {
                SS::BF16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F16(a), SS::F16(b), SS::F16(c)) => {
                SS::F16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F32(a), SS::F32(b), SS::F32(c)) => {
                SS::F32(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F64(a), SS::F64(b), SS::F64(c)) => {
                SS::F64(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            _ => candle_core::bail!("sub_mul: unsupported dtype"),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            l_a.shape().clone(),
        ))
    }
}

/// Fused (a - b) * c — replaces 2 kernel launches (sub + mul) with 1.
pub fn sub_mul(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.apply_op3_no_bwd(b, c, &SubMul)
}

// ─── depthwise_conv1d_silu: dot(window, weight) per channel + silu ──

struct DepthwiseConv1dSilu {
    kernel_size: usize,
    channels: usize,
}

impl candle_core::CustomOp2 for DepthwiseConv1dSilu {
    fn name(&self) -> &'static str {
        "depthwise_conv1d_silu"
    }

    fn cpu_fwd(
        &self,
        s_w: &CpuStorage,
        l_w: &Layout,
        s_wt: &CpuStorage,
        l_wt: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            window: &[T],
            l_w: &Layout,
            weight: &[T],
            l_wt: &Layout,
            kernel_size: usize,
            channels: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let window = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &window[o1..o2],
                None => candle_core::bail!("conv1d_silu: window must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => &weight[o1..o2],
                None => candle_core::bail!("conv1d_silu: weight must be contiguous"),
            };
            let dims = l_w.shape().dims();
            let batch = dims[0];
            let numel = batch * channels;
            let mut dst = vec![T::zero(); numel];
            for b in 0..batch {
                for c in 0..channels {
                    let mut acc = T::zero();
                    let w_off = b * channels * kernel_size + c * kernel_size;
                    let wt_off = c * kernel_size;
                    for k in 0..kernel_size {
                        acc += window[w_off + k] * weight[wt_off + k];
                    }
                    // silu(acc) = acc * sigmoid(acc)
                    let sig = T::one() / (T::one() + (-acc).exp());
                    dst[b * channels + c] = acc * sig;
                }
            }
            let out_shape = Shape::from_dims(&[batch, channels]);
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, out_shape))
        }

        use CpuStorage as C;
        match (s_w, s_wt) {
            (C::BF16(w), C::BF16(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            (C::F16(w), C::F16(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            (C::F32(w), C::F32(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            (C::F64(w), C::F64(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            _ => candle_core::bail!("conv1d_silu: unsupported dtype {:?}", s_w.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
        s_wt: &candle_core::CudaStorage,
        l_wt: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map2, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let kernel_size = self.kernel_size;
        let channels = self.channels;
        let batch = l_w.shape().dims()[0];

        struct S {
            kernel_size: i32,
            channels: i32,
            numel: usize,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                window: &CudaSlice<T>,
                l_w: &Layout,
                weight: &CudaSlice<T>,
                l_wt: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let window = match l_w.contiguous_offsets() {
                    Some((o1, o2)) => window.slice(o1..o2),
                    None => candle_core::bail!("conv1d_silu: window must be contiguous"),
                };
                let weight = match l_wt.contiguous_offsets() {
                    Some((o1, o2)) => weight.slice(o1..o2),
                    None => candle_core::bail!("conv1d_silu: weight must be contiguous"),
                };
                let cfg = LaunchConfig::for_num_elems(self.numel as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("depthwise_conv1d_silu"),
                    "cake_fused_ops",
                    FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(self.numel)? };
                let mut builder = func.builder();
                builder.arg(&self.numel);
                builder.arg(&window);
                builder.arg(&weight);
                builder.arg(&out);
                candle_core::builder_arg!(builder, self.kernel_size, self.channels);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s_w.device();
        let numel = batch * channels;
        let slice = S {
            kernel_size: kernel_size as i32,
            channels: channels as i32,
            numel,
        }
        .map(&s_w.slice, l_w, &s_wt.slice, l_wt, dev)?;
        let out_shape = Shape::from_dims(&[batch, channels]);
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            out_shape,
        ))
    }
}

/// Fused depthwise conv1d + silu — replaces 3 kernel launches with 1.
/// window: (batch, channels, kernel_size), weight: (channels, kernel_size)
/// Returns: (batch, channels)
/// Fused depthwise conv1d + SiLU on a single token window.
/// On Metal, falls back to broadcast_mul + sum + silu using candle built-in ops.
/// Fused depthwise conv1d + SiLU on a single token window.
/// On CUDA: single fused kernel. On Metal: candle built-in ops (3 dispatches).
pub fn depthwise_conv1d_silu(
    window: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
    channels: usize,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if window.device().is_metal() {
        let w = weight.unsqueeze(0)?;
        let dot = window.broadcast_mul(&w)?.sum(candle_core::D::Minus1)?;
        return candle_nn::ops::silu(&dot);
    }
    window.apply_op2_no_bwd(
        weight,
        &DepthwiseConv1dSilu {
            kernel_size,
            channels,
        },
    )
}

// ─── depthwise_conv1d_bias: full depthwise conv1d + bias (no activation) ──

struct DepthwiseConv1dBias {
    kernel_size: usize,
    channels: usize,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for DepthwiseConv1dBias {
    fn name(&self) -> &'static str {
        "depthwise_conv1d_bias"
    }

    fn cpu_fwd(
        &self,
        s_in: &CpuStorage,
        l_in: &Layout,
        s_wt: &CpuStorage,
        l_wt: &Layout,
        s_bi: &CpuStorage,
        l_bi: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            input: &[T],
            l_in: &Layout,
            weight: &[T],
            l_wt: &Layout,
            bias: &[T],
            l_bi: &Layout,
            kernel_size: usize,
            channels: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => &input[o1..o2],
                None => candle_core::bail!("conv1d_bias: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => &weight[o1..o2],
                None => candle_core::bail!("conv1d_bias: weight must be contiguous"),
            };
            let bias = match l_bi.contiguous_offsets() {
                Some((o1, o2)) => &bias[o1..o2],
                None => candle_core::bail!("conv1d_bias: bias must be contiguous"),
            };
            let dims = l_in.shape().dims();
            let batch = dims[0];
            let input_len = dims[2];
            let out_len = input_len - kernel_size + 1;
            let numel = batch * channels * out_len;
            let mut dst = vec![T::zero(); numel];
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..out_len {
                        let mut acc = T::zero();
                        let in_off = (b * channels + c) * input_len + t;
                        let wt_off = c * kernel_size;
                        for k in 0..kernel_size {
                            acc += input[in_off + k] * weight[wt_off + k];
                        }
                        acc += bias[c];
                        dst[(b * channels + c) * out_len + t] = acc;
                    }
                }
            }
            let out_shape = Shape::from_dims(&[batch, channels, out_len]);
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, out_shape))
        }

        use CpuStorage as C;
        match (s_in, s_wt, s_bi) {
            (C::BF16(i), C::BF16(w), C::BF16(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            (C::F16(i), C::F16(w), C::F16(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            (C::F32(i), C::F32(w), C::F32(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            (C::F64(i), C::F64(w), C::F64(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            _ => candle_core::bail!("conv1d_bias: unsupported dtype {:?}", s_in.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_in: &candle_core::CudaStorage,
        l_in: &Layout,
        s_wt: &candle_core::CudaStorage,
        l_wt: &Layout,
        s_bi: &candle_core::CudaStorage,
        l_bi: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let kernel_size = self.kernel_size;
        let channels = self.channels;
        let dims = l_in.shape().dims();
        let batch = dims[0];
        let input_len = dims[2];
        let out_len = input_len - kernel_size + 1;
        let numel = batch * channels * out_len;

        fn launch<T: DeviceRepr + WithDType>(
            input: &CudaSlice<T>,
            l_in: &Layout,
            weight: &CudaSlice<T>,
            l_wt: &Layout,
            bias: &CudaSlice<T>,
            l_bi: &Layout,
            dev: &CudaDevice,
            kernel_size: i32,
            channels: i32,
            input_len: i32,
            numel: usize,
        ) -> Result<CudaSlice<T>> {
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => input.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => weight.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias: weight must be contiguous"),
            };
            let bias = match l_bi.contiguous_offsets() {
                Some((o1, o2)) => bias.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias: bias must be contiguous"),
            };
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("depthwise_conv1d_bias"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(numel)? };
            let mut builder = func.builder();
            builder.arg(&numel);
            builder.arg(&input);
            builder.arg(&weight);
            builder.arg(&bias);
            builder.arg(&out);
            candle_core::builder_arg!(builder, kernel_size, channels, input_len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_in.device();
        let ks = kernel_size as i32;
        let ch = channels as i32;
        let il = input_len as i32;

        let slice = match (&s_in.slice, &s_wt.slice, &s_bi.slice) {
            (S::BF16(i), S::BF16(w), S::BF16(b)) => {
                S::BF16(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            (S::F16(i), S::F16(w), S::F16(b)) => {
                S::F16(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            (S::F32(i), S::F32(w), S::F32(b)) => {
                S::F32(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            (S::F64(i), S::F64(w), S::F64(b)) => {
                S::F64(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            _ => candle_core::bail!("conv1d_bias: unsupported dtype"),
        };

        let out_shape = Shape::from_dims(&[batch, channels, out_len]);
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            out_shape,
        ))
    }
}

/// Fused depthwise conv1d + bias — replaces 14 kernel launches with 1.
/// padded_input: (batch, channels, input_len) — already causal-padded
/// weight: (channels, kernel_size), bias: (channels,)
/// Returns: (batch, channels, out_len) where out_len = input_len - kernel_size + 1
pub fn depthwise_conv1d_bias(
    padded_input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    kernel_size: usize,
    channels: usize,
) -> Result<Tensor> {
    // Ensure all inputs are contiguous (weight may come from squeeze)
    let inp = padded_input.contiguous()?;
    let wt = weight.contiguous()?;
    let bi = bias.contiguous()?;
    inp.apply_op3_no_bwd(
        &wt,
        &bi,
        &DepthwiseConv1dBias {
            kernel_size,
            channels,
        },
    )
}

// ─── depthwise_conv1d_bias_ctx: conv with separate context + input ──

struct DepthwiseConv1dBiasCtx {
    kernel_size: usize,
    channels: usize,
    bias: Tensor,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for DepthwiseConv1dBiasCtx {
    fn name(&self) -> &'static str {
        "depthwise_conv1d_bias_ctx"
    }

    fn cpu_fwd(
        &self,
        s_ctx: &CpuStorage,
        l_ctx: &Layout,
        s_in: &CpuStorage,
        l_in: &Layout,
        s_wt: &CpuStorage,
        l_wt: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive>(
            ctx: &[T], l_ctx: &Layout,
            input: &[T], l_in: &Layout,
            weight: &[T], l_wt: &Layout,
            bias_t: &Tensor,
            kernel_size: usize, channels: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let ctx = match l_ctx.contiguous_offsets() {
                Some((o1, o2)) => &ctx[o1..o2],
                None => candle_core::bail!("conv1d_bias_ctx: ctx must be contiguous"),
            };
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => &input[o1..o2],
                None => candle_core::bail!("conv1d_bias_ctx: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => &weight[o1..o2],
                None => candle_core::bail!("conv1d_bias_ctx: weight must be contiguous"),
            };
            let bias_v: Vec<f32> = bias_t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
            let dims = l_in.shape().dims();
            let (batch, time_len) = (dims[0], dims[2]);
            let ctx_len = kernel_size - 1;
            let numel = batch * channels * time_len;
            let mut dst = vec![T::zero(); numel];
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..time_len {
                        let mut acc = 0f32;
                        for k in 0..kernel_size {
                            let pos = t + k;
                            let v: f32 = if pos < ctx_len {
                                num_traits::AsPrimitive::as_(ctx[(b * channels + c) * ctx_len + pos])
                            } else {
                                num_traits::AsPrimitive::as_(input[(b * channels + c) * time_len + (pos - ctx_len)])
                            };
                            let w: f32 = num_traits::AsPrimitive::as_(weight[c * kernel_size + k]);
                            acc += v * w;
                        }
                        acc += bias_v[c];
                        dst[(b * channels + c) * time_len + t] =
                            num_traits::FromPrimitive::from_f32(acc).unwrap_or(T::zero());
                    }
                }
            }
            let out_shape = Shape::from_dims(&[batch, channels, time_len]);
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, out_shape))
        }

        use CpuStorage as C;
        match (s_ctx, s_in, s_wt) {
            (C::BF16(c), C::BF16(i), C::BF16(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            (C::F16(c), C::F16(i), C::F16(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            (C::F32(c), C::F32(i), C::F32(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            (C::F64(c), C::F64(i), C::F64(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            _ => candle_core::bail!("conv1d_bias_ctx: unsupported dtype"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_ctx: &candle_core::CudaStorage,
        l_ctx: &Layout,
        s_in: &candle_core::CudaStorage,
        l_in: &Layout,
        s_wt: &candle_core::CudaStorage,
        l_wt: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let kernel_size = self.kernel_size;
        let channels = self.channels;
        let dims = l_in.shape().dims();
        let (batch, time_len) = (dims[0], dims[2]);
        let ctx_len = kernel_size - 1;
        let numel = batch * channels * time_len;

        // Get bias CUDA storage
        let bias_cuda = self.bias.storage_and_layout();
        let (bias_storage, bias_layout) = (&*bias_cuda.0, &bias_cuda.1);

        fn launch<T: DeviceRepr + WithDType>(
            ctx: &CudaSlice<T>, l_ctx: &Layout,
            input: &CudaSlice<T>, l_in: &Layout,
            weight: &CudaSlice<T>, l_wt: &Layout,
            bias: &CudaSlice<T>, l_bi: &Layout,
            dev: &CudaDevice,
            kernel_size: i32, channels: i32, ctx_len: i32, time_len: i32,
            numel: usize,
        ) -> Result<CudaSlice<T>> {
            let ctx = match l_ctx.contiguous_offsets() {
                Some((o1, o2)) => ctx.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: ctx must be contiguous"),
            };
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => input.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => weight.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: weight must be contiguous"),
            };
            let bias = match l_bi.contiguous_offsets() {
                Some((o1, o2)) => bias.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: bias must be contiguous"),
            };
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("depthwise_conv1d_bias_ctx"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(numel)? };
            let mut builder = func.builder();
            builder.arg(&numel);
            builder.arg(&ctx);
            builder.arg(&input);
            builder.arg(&weight);
            builder.arg(&bias);
            builder.arg(&out);
            candle_core::builder_arg!(builder, kernel_size, channels, ctx_len, time_len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_in.device();
        let ks = kernel_size as i32;
        let ch = channels as i32;
        let cl = ctx_len as i32;
        let tl = time_len as i32;

        let bias_s = match bias_storage {
            candle_core::Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("conv1d_bias_ctx: bias must be on CUDA"),
        };

        let slice = match (&s_ctx.slice, &s_in.slice, &s_wt.slice, &bias_s.slice) {
            (S::BF16(c), S::BF16(i), S::BF16(w), S::BF16(b)) => S::BF16(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            (S::F16(c), S::F16(i), S::F16(w), S::F16(b)) => S::F16(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            (S::F32(c), S::F32(i), S::F32(w), S::F32(b)) => S::F32(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            (S::F64(c), S::F64(i), S::F64(w), S::F64(b)) => S::F64(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            _ => candle_core::bail!("conv1d_bias_ctx: unsupported dtype"),
        };

        let out_shape = Shape::from_dims(&[batch, channels, time_len]);
        Ok((
            candle_core::CudaStorage { slice, device: dev.clone() },
            out_shape,
        ))
    }
}

// ─── adaln_modulate: rms_norm(x,w,eps) * (1+scale) + shift ──────────

struct AdaLnModulate {
    eps: f32,
    shift: Tensor,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for AdaLnModulate {
    fn name(&self) -> &'static str {
        "adaln_modulate"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
        s_sc: &CpuStorage,
        l_sc: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            x: &[T], l_x: &Layout,
            w: &[T], l_w: &Layout,
            scale: &[T], l_sc: &Layout,
            shift_t: &Tensor, eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("adaln: x contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("adaln: w contiguous"),
            };
            let scale = match l_sc.contiguous_offsets() {
                Some((o1, o2)) => &scale[o1..o2],
                None => candle_core::bail!("adaln: scale contiguous"),
            };
            let shift_v: Vec<f32> = shift_t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
            let dims = l_x.shape().dims();
            let n_cols = dims[dims.len() - 1];
            let el = l_x.shape().elem_count();
            let n_rows = el / n_cols;
            let mut dst = vec![T::zero(); el];
            for r in 0..n_rows {
                let off = r * n_cols;
                let mut sum2 = 0f32;
                for c in 0..n_cols {
                    let v: f32 = x[off + c].as_();
                    sum2 += v * v;
                }
                let inv_rms = 1.0f32 / (sum2 / n_cols as f32 + eps).sqrt();
                for c in 0..n_cols {
                    let xv: f32 = x[off + c].as_() * inv_rms;
                    let wv: f32 = w[c].as_();
                    let sv: f32 = scale[off + c].as_();
                    let shv = shift_v[off + c];
                    dst[off + c] = T::from_f32(xv * wv * (1.0 + sv) + shv).unwrap_or(T::zero());
                }
            }
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_x.shape().clone()))
        }

        use CpuStorage as C;
        match (s_x, s_w, s_sc) {
            (C::BF16(x), C::BF16(w), C::BF16(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            (C::F16(x), C::F16(w), C::F16(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            (C::F32(x), C::F32(w), C::F32(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            (C::F64(x), C::F64(w), C::F64(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            _ => candle_core::bail!("adaln: unsupported dtype"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_x: &candle_core::CudaStorage,
        l_x: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
        s_sc: &candle_core::CudaStorage,
        l_sc: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let dims = l_x.shape().dims();
        let n_cols = dims[dims.len() - 1];
        let el = l_x.shape().elem_count();
        let n_rows = el / n_cols;
        let eps = self.eps;

        let shift_sl = self.shift.storage_and_layout();
        let (shift_storage, shift_layout) = (&*shift_sl.0, &shift_sl.1);

        fn launch<T: DeviceRepr + WithDType>(
            x: &CudaSlice<T>, l_x: &Layout,
            w: &CudaSlice<T>, l_w: &Layout,
            sc: &CudaSlice<T>, l_sc: &Layout,
            sh: &CudaSlice<T>, l_sh: &Layout,
            dev: &CudaDevice,
            n_cols: i32, n_rows: usize, el: usize, eps: f32,
        ) -> Result<CudaSlice<T>> {
            let x = match l_x.contiguous_offsets() { Some((a,b)) => x.slice(a..b), None => candle_core::bail!("adaln: x") };
            let w = match l_w.contiguous_offsets() { Some((a,b)) => w.slice(a..b), None => candle_core::bail!("adaln: w") };
            let sc = match l_sc.contiguous_offsets() { Some((a,b)) => sc.slice(a..b), None => candle_core::bail!("adaln: sc") };
            let sh = match l_sh.contiguous_offsets() { Some((a,b)) => sh.slice(a..b), None => candle_core::bail!("adaln: sh") };
            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig { grid_dim: (n_rows as u32, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
            let func = dev.get_or_load_custom_func(&kernel_name::<T>("adaln_modulate"), "cake_fused_ops", FUSED_OPS_PTX)?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&x); builder.arg(&w); builder.arg(&sc); builder.arg(&sh); builder.arg(&out);
            candle_core::builder_arg!(builder, n_cols, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_x.device();
        let nc = n_cols as i32;

        let sh_s = match shift_storage {
            candle_core::Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("adaln: shift must be on CUDA"),
        };

        let slice = match (&s_x.slice, &s_w.slice, &s_sc.slice, &sh_s.slice) {
            (S::BF16(x), S::BF16(w), S::BF16(sc), S::BF16(sh)) => S::BF16(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            (S::F16(x), S::F16(w), S::F16(sc), S::F16(sh)) => S::F16(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            (S::F32(x), S::F32(w), S::F32(sc), S::F32(sh)) => S::F32(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            (S::F64(x), S::F64(w), S::F64(sc), S::F64(sh)) => S::F64(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            _ => candle_core::bail!("adaln: unsupported dtype"),
        };

        Ok((candle_core::CudaStorage { slice, device: dev.clone() }, l_x.shape().clone()))
    }
}

/// Fused AdaLN modulation: rms_norm(x, weight, eps) * (1 + scale) + shift.
/// Replaces 4 kernel launches (rms_norm + add_1 + mul + add_shift) with 1.
pub fn adaln_modulate(
    x: &Tensor,
    norm_weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let w = norm_weight.contiguous()?;
    let sc = scale.contiguous()?;
    let sh = shift.contiguous()?;
    x.apply_op3_no_bwd(
        &w,
        &sc,
        &AdaLnModulate { eps, shift: sh },
    )
}

/// Fused depthwise conv1d with separate context + input (no cat needed).
/// Replaces Tensor::zeros + Tensor::cat + depthwise_conv1d_bias (3 kernels → 1).
/// ctx: (batch, channels, kernel_size-1), input: (batch, channels, time_len)
/// weight: (channels, kernel_size), bias: (channels,)
/// Returns: (batch, channels, time_len)
pub fn depthwise_conv1d_bias_ctx(
    ctx: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    kernel_size: usize,
    channels: usize,
) -> Result<Tensor> {
    let ctx = ctx.contiguous()?;
    let inp = input.contiguous()?;
    let wt = weight.contiguous()?;
    ctx.apply_op3_no_bwd(
        &inp,
        &wt,
        &DepthwiseConv1dBiasCtx {
            kernel_size,
            channels,
            bias: bias.contiguous()?,
        },
    )
}

// ─── rms_norm_channel: RMS-normalize over channel dim of (b,c,t) ────

struct RmsNormChannel {
    eps: f32,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp2 for RmsNormChannel {
    fn name(&self) -> &'static str {
        "rms_norm_channel"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            x: &[T],
            l_x: &Layout,
            w: &[T],
            l_w: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("rms_norm_channel: x must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("rms_norm_channel: weight must be contiguous"),
            };
            let dims = l_x.shape().dims();
            let (batch, channels, time_len) = (dims[0], dims[1], dims[2]);
            let mut dst = vec![T::zero(); batch * channels * time_len];
            for b in 0..batch {
                for t in 0..time_len {
                    let mut sum2 = 0f32;
                    for c in 0..channels {
                        let v: f32 = x[b * channels * time_len + c * time_len + t].as_();
                        sum2 += v * v;
                    }
                    let inv_rms = 1.0f32 / (sum2 / channels as f32 + eps).sqrt();
                    for (c, wv_t) in w.iter().enumerate().take(channels) {
                        let off = b * channels * time_len + c * time_len + t;
                        let xv: f32 = x[off].as_();
                        let wv: f32 = (*wv_t).as_();
                        dst[off] = T::from_f32(xv * inv_rms * wv).unwrap_or(T::zero());
                    }
                }
            }
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_x.shape().clone()))
        }

        use CpuStorage as C;
        match (s_x, s_w) {
            (C::BF16(x), C::BF16(w)) => inner(x, l_x, w, l_w, self.eps),
            (C::F16(x), C::F16(w)) => inner(x, l_x, w, l_w, self.eps),
            (C::F32(x), C::F32(w)) => inner(x, l_x, w, l_w, self.eps),
            (C::F64(x), C::F64(w)) => inner(x, l_x, w, l_w, self.eps),
            _ => candle_core::bail!("rms_norm_channel: unsupported dtype"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_x: &candle_core::CudaStorage,
        l_x: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let dims = l_x.shape().dims();
        let (batch, channels, time_len) = (dims[0], dims[1], dims[2]);
        let n_rows = batch * time_len; // one block per (batch, time) position
        let el = l_x.shape().elem_count();
        let eps = self.eps;

        fn launch<T: DeviceRepr + WithDType>(
            x: &CudaSlice<T>,
            l_x: &Layout,
            w: &CudaSlice<T>,
            l_w: &Layout,
            dev: &CudaDevice,
            channels: i32,
            time_len: i32,
            n_rows: usize,
            el: usize,
            eps: f32,
        ) -> Result<CudaSlice<T>> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => x.slice(o1..o2),
                None => candle_core::bail!("rms_norm_channel: x must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => w.slice(o1..o2),
                None => candle_core::bail!("rms_norm_channel: weight must be contiguous"),
            };
            let block_size: u32 = if channels < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("rms_norm_channel"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&w);
            builder.arg(&out);
            candle_core::builder_arg!(builder, channels, time_len, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_x.device();
        let ch = channels as i32;
        let tl = time_len as i32;

        let slice = match (&s_x.slice, &s_w.slice) {
            (S::BF16(x), S::BF16(w)) => S::BF16(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            (S::F16(x), S::F16(w)) => S::F16(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            (S::F32(x), S::F32(w)) => S::F32(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            (S::F64(x), S::F64(w)) => S::F64(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            _ => candle_core::bail!("rms_norm_channel: unsupported dtype"),
        };

        Ok((
            candle_core::CudaStorage { slice, device: dev.clone() },
            l_x.shape().clone(),
        ))
    }
}

/// RMS-normalize over the channel dimension of a (batch, channels, time) tensor.
/// Replaces transpose + rms_norm + transpose (3 ops including copy) with 1 kernel.
pub fn rms_norm_channel(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let w = weight.contiguous()?;
    x.apply_op2_no_bwd(&w, &RmsNormChannel { eps })
}

// ─── add_scaled: a + b * c with broadcast on c ─────────────────────

struct AddScaled;

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for AddScaled {
    fn name(&self) -> &'static str {
        "add_scaled"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_c: &CpuStorage,
        l_c: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            c: &[T],
            l_c: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("add_scaled: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("add_scaled: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => &c[o1..o2],
                None => candle_core::bail!("add_scaled: c must be contiguous"),
            };
            let dims = l_a.shape().dims();
            let (channels, time_len) = (dims[1], dims[2]);
            let numel = l_a.shape().elem_count();
            let mut dst = vec![T::zero(); numel];
            for (i, d) in dst.iter_mut().enumerate() {
                let chan = (i / time_len) % channels;
                *d = a[i] + b[i] * c[chan];
            }
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_a.shape().clone()))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_c) {
            (C::BF16(a), C::BF16(b), C::BF16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F16(a), C::F16(b), C::F16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F32(a), C::F32(b), C::F32(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F64(a), C::F64(b), C::F64(c)) => inner(a, l_a, b, l_b, c, l_c),
            _ => candle_core::bail!("add_scaled: unsupported dtype"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_c: &candle_core::CudaStorage,
        l_c: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let dims = l_a.shape().dims();
        let (channels, time_len) = (dims[1], dims[2]);
        let numel = l_a.shape().elem_count();

        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>, l_a: &Layout,
            b: &CudaSlice<T>, l_b: &Layout,
            c: &CudaSlice<T>, l_c: &Layout,
            dev: &CudaDevice,
            channels: i32, time_len: i32, numel: usize,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("add_scaled: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("add_scaled: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => c.slice(o1..o2),
                None => candle_core::bail!("add_scaled: c must be contiguous"),
            };
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("add_scaled"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(numel)? };
            let mut builder = func.builder();
            builder.arg(&numel);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
            builder.arg(&out);
            candle_core::builder_arg!(builder, channels, time_len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_a.device();
        let ch = channels as i32;
        let tl = time_len as i32;

        let slice = match (&s_a.slice, &s_b.slice, &s_c.slice) {
            (S::BF16(a), S::BF16(b), S::BF16(c)) => S::BF16(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            (S::F16(a), S::F16(b), S::F16(c)) => S::F16(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            (S::F32(a), S::F32(b), S::F32(c)) => S::F32(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            (S::F64(a), S::F64(b), S::F64(c)) => S::F64(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            _ => candle_core::bail!("add_scaled: unsupported dtype"),
        };

        Ok((
            candle_core::CudaStorage { slice, device: dev.clone() },
            l_a.shape().clone(),
        ))
    }
}

/// Fused a + b * c where c is (channels,) broadcast over (batch, channels, time).
/// Replaces broadcast_mul + add (2 kernels) with 1.
pub fn add_scaled(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    a.apply_op3_no_bwd(&b, &c, &AddScaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_silu_mul_correctness() {
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.0, -2.0], &Device::Cpu).unwrap();
        let up = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &Device::Cpu).unwrap();

        let fused: Vec<f32> = silu_mul(&gate, &up).unwrap().to_vec1().unwrap();

        // Reference: silu(gate) * up
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate).unwrap() * &up)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-6),
            "silu_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[test]
    fn test_silu_mul_2d() {
        let gate = Tensor::new(&[[1.0f32, 2.0], [-1.0, 0.5]], &Device::Cpu).unwrap();
        let up = Tensor::new(&[[0.5f32, -1.0], [2.0, 3.0]], &Device::Cpu).unwrap();

        let fused = silu_mul(&gate, &up).unwrap();
        assert_eq!(fused.dims(), &[2, 2]);

        let fused: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate).unwrap() * &up)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(approx_eq(&fused, &reference, 1e-6));
    }

    #[test]
    fn test_stable_softplus_correctness() {
        let x = Tensor::new(&[-10.0f32, -1.0, 0.0, 1.0, 10.0, 100.0], &Device::Cpu).unwrap();
        let result: Vec<f32> = stable_softplus(&x).unwrap().to_vec1().unwrap();

        // Reference values
        let expected: Vec<f32> = vec![
            ((-10.0f32).exp() + 1.0).ln(),          // ~4.5e-5
            ((-1.0f32).exp() + 1.0).ln(),            // ~0.3133
            (0.0f32.exp() + 1.0).ln(),               // ~0.6931
            (1.0f32.exp() + 1.0).ln(),               // ~1.3133
            (10.0f32.exp() + 1.0).ln(),              // ~10.0
            100.0,                                     // clamped: max(100, softplus(88))
        ];
        assert!(
            approx_eq(&result, &expected, 1e-4),
            "softplus mismatch: result={result:?} expected={expected:?}"
        );
    }

    #[test]
    fn test_stable_softplus_matches_original() {
        let x = Tensor::new(&[-5.0f32, -1.0, 0.0, 1.0, 5.0, 50.0], &Device::Cpu).unwrap();

        let fused: Vec<f32> = stable_softplus(&x).unwrap().to_vec1().unwrap();

        // Original implementation from linear_attention.rs
        let sp = (x.minimum(88f64).unwrap().exp().unwrap() + 1.0)
            .unwrap()
            .log()
            .unwrap();
        let reference: Vec<f32> = x.maximum(&sp).unwrap().to_vec1().unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "softplus vs original: fused={fused:?} ref={reference:?}"
        );
    }

    #[test]
    fn test_rms_norm_gated_correctness() {
        // Small test: 2 rows, 4 cols
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [0.5, -1.0, 0.5, -1.0]], &Device::Cpu)
            .unwrap();
        let z = Tensor::new(
            &[[0.1f32, 0.2, 0.3, 0.4], [-0.5, 1.0, -0.5, 1.0]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap();
        let eps = 1e-6f32;

        let fused: Vec<f32> = rms_norm_gated(&x, &z, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference: rms_norm(x, weight, eps) * silu(z)
        let x_normed: Vec<f32> = candle_nn::ops::rms_norm(&x, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let silu_z: Vec<f32> = candle_nn::ops::silu(&z)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let reference: Vec<f32> = x_normed.iter().zip(&silu_z).map(|(a, b)| a * b).collect();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "rms_norm_gated mismatch:\nfused={fused:?}\nref  ={reference:?}"
        );
    }

    #[test]
    fn test_silu_mul_f16() {
        let gate = Tensor::new(&[1.0f32, -1.0, 2.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let up = Tensor::new(&[2.0f32, 3.0, 0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let result = silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dtype(), DType::F16);
        assert_eq!(result.dims(), &[3]);
    }

    // ── add_rms_norm shape and residual tests ──────────────────────

    #[test]
    fn test_add_rms_norm_shape_preservation_2d() {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &Device::Cpu).unwrap();
        let w = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
        let (residual, normed) = add_rms_norm(&a, &b, &w, 1e-6).unwrap();
        assert_eq!(residual.dims(), &[2, 2]);
        assert_eq!(normed.dims(), &[2, 2]);
    }

    #[test]
    fn test_add_rms_norm_residual_is_sum() {
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[10.0f32, 20.0, 30.0]], &Device::Cpu).unwrap();
        let w = Tensor::ones(3, DType::F32, &Device::Cpu).unwrap();
        let (residual, _) = add_rms_norm(&a, &b, &w, 1e-6).unwrap();
        let vals: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
        assert!(approx_eq(&vals, &[11.0, 22.0, 33.0], 1e-5));
    }

    // ── add3 tests ──────────────────────────────────────────────────

    #[test]
    fn test_add3_correctness() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &Device::Cpu).unwrap();
        let c = Tensor::new(&[100.0f32, 200.0, 300.0, 400.0], &Device::Cpu).unwrap();
        let result: Vec<f32> = add3(&a, &b, &c).unwrap().to_vec1().unwrap();
        assert!(approx_eq(&result, &[111.0, 222.0, 333.0, 444.0], 1e-5));
    }

    #[test]
    fn test_add3_2d_shape() {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &Device::Cpu).unwrap();
        let c = Tensor::new(&[[0.01f32, 0.02], [0.03, 0.04]], &Device::Cpu).unwrap();
        let result = add3(&a, &b, &c).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(approx_eq(&vals, &[1.11, 2.22, 3.33, 4.44], 1e-4));
    }

    // ── exp_mul tests ───────────────────────────────────────────────

    #[test]
    fn test_exp_mul_zero_exponent() {
        // exp(0) = 1, so x * exp(0) = x
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let y = Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap();
        let result: Vec<f32> = exp_mul(&x, &y).unwrap().to_vec1().unwrap();
        assert!(approx_eq(&result, &[1.0, 2.0, 3.0], 1e-6));
    }

    // ── sub_mul tests ───────────────────────────────────────────────

    #[test]
    fn test_sub_mul_zero_diff() {
        let a = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let c = Tensor::new(&[999.0f32, 999.0], &Device::Cpu).unwrap();
        let result: Vec<f32> = sub_mul(&a, &b, &c).unwrap().to_vec1().unwrap();
        assert!(approx_eq(&result, &[0.0, 0.0], 1e-6));
    }

    // ── add_scaled tests ────────────────────────────────────────────

    #[test]
    fn test_add_scaled_cpu_correctness() {
        // a: (1, 2, 3), b: (1, 2, 3), c: (2,)
        let a = Tensor::new(&[1f32, 2., 3., 4., 5., 6.], &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let b = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let c = Tensor::new(&[10.0f32, 20.0], &Device::Cpu).unwrap();

        let out = add_scaled(&a, &b, &c).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // chan0 (scale=10): [1+0.1*10, 2+0.2*10, 3+0.3*10] = [2, 4, 6]
        // chan1 (scale=20): [4+0.4*20, 5+0.5*20, 6+0.6*20] = [12, 15, 18]
        assert!(approx_eq(&vals, &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0], 1e-5));
    }

    // ── depthwise_conv1d_bias tests ─────────────────────────────────

    #[test]
    fn test_depthwise_conv1d_bias_cpu() {
        // batch=1, channels=2, input_len=4, kernel_size=3 → out_len=2
        let input = Tensor::new(
            &[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[1.0f32, 0.0, 0.0], [0.0, 0.0, 1.0]],
            &Device::Cpu,
        )
        .unwrap();
        let bias = Tensor::new(&[10.0f32, 20.0], &Device::Cpu).unwrap();

        let out = depthwise_conv1d_bias(&input, &weight, &bias, 3, 2).unwrap();
        assert_eq!(out.dims(), &[1, 2, 2]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // chan0, w=[1,0,0]: pos0=1*1+0+0+10=11, pos1=1*2+0+0+10=12
        // chan1, w=[0,0,1]: pos0=0+0+7*1+20=27, pos1=0+0+8*1+20=28
        assert!(approx_eq(&vals, &[11.0, 12.0, 27.0, 28.0], 1e-5));
    }

    #[test]
    fn test_depthwise_conv1d_bias_shape() {
        // batch=2, channels=3, input_len=6, kernel_size=2 → out_len=5
        let input = Tensor::zeros((2, 3, 6), DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::ones((3, 2), DType::F32, &Device::Cpu).unwrap();
        let bias = Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap();
        let out = depthwise_conv1d_bias(&input, &weight, &bias, 2, 3).unwrap();
        assert_eq!(out.dims(), &[2, 3, 5]);
    }

    // ── rms_norm_channel additional tests ───────────────────────────

    #[test]
    fn test_rms_norm_channel_with_nonunit_weight() {
        // (1, 2, 3) with weight=[2, 0.5]
        let x = Tensor::new(&[1f32, 1., 1., 2., 2., 2.], &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let w = Tensor::new(&[2.0f32, 0.5], &Device::Cpu).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "all values should be finite");
        assert!(vals.iter().any(|v| *v != 0.0), "output should be non-zero");
    }

    // ── depthwise_conv1d_silu additional tests ──────────────────────

    #[test]
    fn test_depthwise_conv1d_silu_batch2() {
        // batch=2, channels=2, kernel_size=2
        let window = Tensor::new(
            &[
                [[1.0f32, 2.0], [3.0, 4.0]],
                [[0.5, -0.5], [-1.0, 1.0]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(&[[1.0f32, 1.0], [0.5, 0.5]], &Device::Cpu).unwrap();
        let result = depthwise_conv1d_silu(&window, &weight, 2, 2).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    // ── CUDA tests ───────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_silu_mul_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return, // skip if no GPU
        };
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.0, -2.0], &dev).unwrap();
        let up = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &dev).unwrap();

        let fused: Vec<f32> = silu_mul(&gate, &up).unwrap().to_vec1().unwrap();

        // Reference on CPU
        let gate_cpu = gate.to_device(&Device::Cpu).unwrap();
        let up_cpu = up.to_device(&Device::Cpu).unwrap();
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate_cpu).unwrap() * &up_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA silu_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_silu_mul_cuda_f16() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let gate = Tensor::new(&[1.0f32, -1.0, 2.0, -0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let up = Tensor::new(&[2.0f32, 3.0, 0.5, -1.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&dev)
            .unwrap();

        let result = silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dtype(), DType::F16);

        let fused: Vec<f32> = result
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference
        let gate_cpu = gate.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
        let up_cpu = up.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate_cpu).unwrap() * &up_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-2), // F16 has less precision
            "CUDA F16 silu_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_stable_softplus_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(&[-10.0f32, -1.0, 0.0, 1.0, 10.0, 100.0], &dev).unwrap();
        let result: Vec<f32> = stable_softplus(&x).unwrap().to_vec1().unwrap();

        let expected: Vec<f32> = vec![
            ((-10.0f32).exp() + 1.0).ln(),
            ((-1.0f32).exp() + 1.0).ln(),
            (0.0f32.exp() + 1.0).ln(),
            (1.0f32.exp() + 1.0).ln(),
            (10.0f32.exp() + 1.0).ln(),
            100.0,
        ];
        assert!(
            approx_eq(&result, &expected, 1e-4),
            "CUDA softplus mismatch: result={result:?} expected={expected:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_gated_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [0.5, -1.0, 0.5, -1.0]], &dev).unwrap();
        let z = Tensor::new(
            &[[0.1f32, 0.2, 0.3, 0.4], [-0.5, 1.0, -0.5, 1.0]],
            &dev,
        )
        .unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &dev).unwrap();
        let eps = 1e-6f32;

        let fused: Vec<f32> = rms_norm_gated(&x, &z, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference on CPU
        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let z_cpu = z.to_device(&Device::Cpu).unwrap();
        let w_cpu = weight.to_device(&Device::Cpu).unwrap();
        let x_normed: Vec<f32> = candle_nn::ops::rms_norm(&x_cpu, &w_cpu, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let silu_z: Vec<f32> = candle_nn::ops::silu(&z_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let reference: Vec<f32> = x_normed.iter().zip(&silu_z).map(|(a, b)| a * b).collect();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA rms_norm_gated mismatch:\nfused={fused:?}\nref  ={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_silu_mul_cuda_large() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Test with realistic hidden size
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 1024;
        let gate_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let up_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();

        let gate_cpu = Tensor::new(gate_data.as_slice(), &Device::Cpu).unwrap();
        let up_cpu = Tensor::new(up_data.as_slice(), &Device::Cpu).unwrap();
        let gate_gpu = gate_cpu.to_device(&dev).unwrap();
        let up_gpu = up_cpu.to_device(&dev).unwrap();

        let gpu_result: Vec<f32> = silu_mul(&gate_gpu, &up_gpu).unwrap().to_vec1().unwrap();
        let cpu_result: Vec<f32> = silu_mul(&gate_cpu, &up_cpu).unwrap().to_vec1().unwrap();

        assert!(
            approx_eq(&gpu_result, &cpu_result, 1e-5),
            "CUDA vs CPU silu_mul mismatch on 1024 elements"
        );
    }

    // ── add_rms_norm tests ─────────────────────────────────────────

    #[test]
    fn test_add_rms_norm_correctness() {
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[0.5f32, -1.0, 0.5, -1.0]], &Device::Cpu).unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap();
        let eps = 1e-6f32;

        let (residual, normed) = add_rms_norm(&a, &b, &weight, eps).unwrap();

        // Check residual = a + b
        let expected_sum: Vec<f32> = (&a + &b).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let actual_sum: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            approx_eq(&actual_sum, &expected_sum, 1e-5),
            "residual mismatch: got={actual_sum:?} expected={expected_sum:?}"
        );

        // Check normed = rms_norm(a + b, weight, eps)
        let sum_tensor = (&a + &b).unwrap();
        let expected_norm: Vec<f32> = candle_nn::ops::rms_norm(&sum_tensor, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let actual_norm: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            approx_eq(&actual_norm, &expected_norm, 1e-5),
            "normed mismatch: got={actual_norm:?} expected={expected_norm:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_add_rms_norm_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [0.5, -1.0, 0.5, -1.0]], &dev).unwrap();
        let b = Tensor::new(&[[0.1f32, 0.2, 0.3, 0.4], [-0.5, 1.0, -0.5, 1.0]], &dev).unwrap();
        let weight = Tensor::new(&[1.0f32, 0.5, 2.0, 1.5], &dev).unwrap();
        let eps = 1e-6f32;

        let (residual, normed) = add_rms_norm(&a, &b, &weight, eps).unwrap();
        let res_vals: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
        let norm_vals: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();

        // CPU reference
        let a_cpu = a.to_device(&Device::Cpu).unwrap();
        let b_cpu = b.to_device(&Device::Cpu).unwrap();
        let w_cpu = weight.to_device(&Device::Cpu).unwrap();
        let sum_cpu = (&a_cpu + &b_cpu).unwrap();
        let expected_sum: Vec<f32> = sum_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let expected_norm: Vec<f32> = candle_nn::ops::rms_norm(&sum_cpu, &w_cpu, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(approx_eq(&res_vals, &expected_sum, 1e-5), "CUDA residual mismatch");
        assert!(approx_eq(&norm_vals, &expected_norm, 1e-4), "CUDA normed mismatch");
    }

    // ── depthwise_conv1d_silu tests ────────────────────────────────

    #[test]
    fn test_depthwise_conv1d_silu_correctness() {
        // batch=1, channels=4, kernel_size=3
        let window = Tensor::new(
            &[[[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.1, 0.0, 0.1], [1.0, -1.0, 0.5]]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[1.0f32, 0.5, 0.25], [0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            &Device::Cpu,
        )
        .unwrap();

        let fused = depthwise_conv1d_silu(&window, &weight, 3, 4).unwrap();
        assert_eq!(fused.dims(), &[1, 4]);

        // Reference: broadcast_mul + sum + silu
        let ref_y = window
            .broadcast_mul(&weight.unsqueeze(0).unwrap())
            .unwrap()
            .sum(candle_core::D::Minus1)
            .unwrap();
        let ref_y: Vec<f32> = candle_nn::ops::silu(&ref_y)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let fused_vals: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();

        assert!(
            approx_eq(&fused_vals, &ref_y, 1e-5),
            "conv1d_silu mismatch: fused={fused_vals:?} ref={ref_y:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_depthwise_conv1d_silu_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let window = Tensor::new(
            &[[[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [-0.1, 0.0, 0.1, 0.2]]],
            &dev,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[1.0f32, 0.5, 0.25, 0.1], [0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0]],
            &dev,
        )
        .unwrap();

        let fused_vals: Vec<f32> = depthwise_conv1d_silu(&window, &weight, 4, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // CPU reference
        let window_cpu = window.to_device(&Device::Cpu).unwrap();
        let weight_cpu = weight.to_device(&Device::Cpu).unwrap();
        let ref_vals: Vec<f32> = depthwise_conv1d_silu(&window_cpu, &weight_cpu, 4, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused_vals, &ref_vals, 1e-5),
            "CUDA conv1d_silu mismatch: fused={fused_vals:?} ref={ref_vals:?}"
        );
    }

    // ── exp_mul tests ────────────────────────────────────────────────

    #[test]
    fn test_exp_mul_correctness() {
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5], &Device::Cpu).unwrap();
        let y = Tensor::new(&[0.0f32, 1.0, -1.0, 0.5], &Device::Cpu).unwrap();

        let fused: Vec<f32> = exp_mul(&x, &y).unwrap().to_vec1().unwrap();
        let reference: Vec<f32> = (&x * y.exp().unwrap())
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-6),
            "exp_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_exp_mul_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &dev).unwrap();
        let y = Tensor::new(&[0.0f32, 1.0, -1.0, 0.5, -2.0], &dev).unwrap();

        let fused: Vec<f32> = exp_mul(&x, &y).unwrap().to_vec1().unwrap();

        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let y_cpu = y.to_device(&Device::Cpu).unwrap();
        let reference: Vec<f32> = (&x_cpu * y_cpu.exp().unwrap())
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA exp_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    // ── sub_mul tests ────────────────────────────────────────────────

    #[test]
    fn test_sub_mul_correctness() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5], &Device::Cpu).unwrap();
        let b = Tensor::new(&[0.5f32, 1.0, 4.0, -0.5], &Device::Cpu).unwrap();
        let c = Tensor::new(&[2.0f32, 0.5, -1.0, 3.0], &Device::Cpu).unwrap();

        let fused: Vec<f32> = sub_mul(&a, &b, &c).unwrap().to_vec1().unwrap();
        let reference: Vec<f32> = ((&a - &b).unwrap() * &c)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-6),
            "sub_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_sub_mul_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &dev).unwrap();
        let b = Tensor::new(&[0.5f32, 1.0, 4.0, -0.5, 2.0], &dev).unwrap();
        let c = Tensor::new(&[2.0f32, 0.5, -1.0, 3.0, 0.1], &dev).unwrap();

        let fused: Vec<f32> = sub_mul(&a, &b, &c).unwrap().to_vec1().unwrap();

        let a_cpu = a.to_device(&Device::Cpu).unwrap();
        let b_cpu = b.to_device(&Device::Cpu).unwrap();
        let c_cpu = c.to_device(&Device::Cpu).unwrap();
        let reference: Vec<f32> = ((&a_cpu - &b_cpu).unwrap() * &c_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA sub_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[test]
    fn test_rms_norm_channel_cpu() {
        // x: (1, 4, 3) — batch=1, channels=4, time=3
        let x = Tensor::new(
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &Device::Cpu,
        )
        .unwrap()
        .reshape((1, 4, 3))
        .unwrap();
        let w = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(!out_v.iter().any(|v| v.is_nan()), "NaN in rms_norm_channel: {out_v:?}");
        assert_eq!(out.dims(), &[1, 4, 3]);
        // For time=0: values [1,4,7,10], rms = sqrt((1+16+49+100)/4) = sqrt(41.5) ≈ 6.44
        // normalized: [1/6.44, 4/6.44, 7/6.44, 10/6.44] ≈ [0.155, 0.621, 1.087, 1.553]
        assert!((out_v[0] - 0.1553).abs() < 0.01, "wrong norm val: {}", out_v[0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_bf16_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Larger test: (1, 32, 100) in BF16
        let data: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.01) - 16.0).collect();
        let x = Tensor::new(data.as_slice(), &dev)
            .unwrap()
            .reshape((1, 32, 100))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::ones(32, DType::BF16, &dev).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        let out_v: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let nan_count = out_v.iter().filter(|v| v.is_nan()).count();
        assert!(
            nan_count == 0,
            "NaN in bf16 rms_norm_channel: {nan_count}/{} values",
            out_v.len()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_add_scaled_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // (1, 4, 3) tensor
        let a = Tensor::new(&[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], &dev)
            .unwrap()
            .reshape((1, 4, 3))
            .unwrap();
        let b = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], &dev)
            .unwrap()
            .reshape((1, 4, 3))
            .unwrap();
        let c = Tensor::new(&[2.0f32, 3.0, 4.0, 5.0], &dev).unwrap();
        let out: Vec<f32> = add_scaled(&a, &b, &c)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Expected: a[i] + b[i] * c[chan]
        // chan 0 (scale=2): [1+0.1*2, 2+0.2*2, 3+0.3*2] = [1.2, 2.4, 3.6]
        assert!((out[0] - 1.2).abs() < 1e-5, "wrong: {}", out[0]);
        assert!((out[1] - 2.4).abs() < 1e-5, "wrong: {}", out[1]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_large_channels_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Large channels like encoder stage 6: (1, 2048, 1) in BF16
        let n = 2048;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01) - 10.0).collect();
        let x = Tensor::new(data.as_slice(), &dev)
            .unwrap()
            .reshape((1, 2048, 1))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::ones(2048, DType::BF16, &dev).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        let out_v: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let nan_count = out_v.iter().filter(|v| v.is_nan()).count();
        assert!(
            nan_count == 0,
            "NaN in 2048-channel rms_norm_channel: {nan_count}/{n} values"
        );
        // Also check correctness vs CPU
        let x_cpu = x.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
        let w_cpu = w.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
        let ref_v: Vec<f32> = rms_norm_channel(&x_cpu, &w_cpu, 1e-5)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // BF16 precision: ~0.4% relative error, use wider tolerance
        assert!(
            approx_eq(&out_v, &ref_v, 0.5),
            "2048-channel CUDA/CPU mismatch: cuda[0..5]={:?} cpu[0..5]={:?}",
            &out_v[..5],
            &ref_v[..5]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_vs_candle_cuda() {
        use candle_core::Module;
        // Compare our rms_norm_channel against candle's transpose+rms_norm+transpose
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Test multiple channel counts that the VAE encoder uses
        for (channels, time_len) in [(32, 1000), (64, 500), (128, 200), (2048, 10)] {
            let n = channels * time_len;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 7.13) % 5.0) - 2.5)
                .collect();
            let x = Tensor::new(data.as_slice(), &dev)
                .unwrap()
                .reshape((1, channels, time_len))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let w_data: Vec<f32> = (0..channels)
                .map(|i| 0.5 + (i as f32) * 0.01)
                .collect();
            let w = Tensor::new(w_data.as_slice(), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

            // Our kernel
            let ours = rms_norm_channel(&x, &w, 1e-5).unwrap();
            let ours_v: Vec<f32> = ours
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            // candle's approach: transpose → rms_norm → transpose
            let norm = candle_nn::RmsNorm::new(w.clone(), 1e-5_f64);
            let ref_out = norm
                .forward(&x.transpose(1, 2).unwrap())
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let ref_v: Vec<f32> = ref_out
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            let nan_ours = ours_v.iter().filter(|v| v.is_nan()).count();
            let nan_ref = ref_v.iter().filter(|v| v.is_nan()).count();
            assert!(
                nan_ours == 0,
                "ch={channels} t={time_len}: {nan_ours} NaN in rms_norm_channel"
            );
            assert!(
                nan_ref == 0,
                "ch={channels} t={time_len}: {nan_ref} NaN in candle rms_norm"
            );

            // Check max absolute difference
            let max_diff = ours_v
                .iter()
                .zip(ref_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 0.5,
                "ch={channels} t={time_len}: max_diff={max_diff} (first 5: ours={:?} ref={:?})",
                &ours_v[..5],
                &ref_v[..5]
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &dev,
        )
        .unwrap()
        .reshape((1, 4, 3))
        .unwrap();
        let w = Tensor::ones(4, DType::F32, &dev).unwrap();
        let out_cuda: Vec<f32> = rms_norm_channel(&x, &w, 1e-5)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Compare with CPU
        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let w_cpu = w.to_device(&Device::Cpu).unwrap();
        let out_cpu: Vec<f32> = rms_norm_channel(&x_cpu, &w_cpu, 1e-5)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&out_cuda, &out_cpu, 1e-3),
            "CUDA rms_norm_channel mismatch:\n  cuda={out_cuda:?}\n  cpu ={out_cpu:?}"
        );
    }
}
