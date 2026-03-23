//! Metal compute backend with MSL shaders + candle tensor op fallbacks.
//!
//! Self-contained: owns the MSL shader source and CustomOp structs for the
//! MSL-accelerated ops. All other trait methods use plain candle tensor
//! operations with no dependency on the ops/ module.
//!
//! Metal command buffer accumulation >25 commands causes catastrophic 50x slowdown.
//! The `synchronize()` method flushes the command buffer and is called at strategic
//! points during forward passes (see GatedDeltaNet, Qwen3_5FullAttention).

use candle_core::{backend::BackendStorage as _, CpuStorage, DType, Device, Layout, Result, Shape, Tensor};

use super::ComputeBackend;

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

// ─── MSL shader source (loaded from ops.msl) ───────────────────────
const FUSED_OPS_MSL: &str = include_str!("ops.msl");

// ─── Pipeline cache ─────────────────────────────────────────────────

/// All kernel names in the MSL source — compiled eagerly on first access.
const ALL_KERNELS: &[&str] = &[
    "stable_softplus_f32", "stable_softplus_f16",
    "silu_mul_f32", "silu_mul_f16",
    "add3_f32", "add3_f16",
    "exp_mul_f32", "exp_mul_f16",
    "sub_mul_f32", "sub_mul_f16",
    "add_scaled_f32", "add_scaled_f16",
    "depthwise_conv1d_silu_f32", "depthwise_conv1d_silu_f16",
    "depthwise_conv1d_bias_f32", "depthwise_conv1d_bias_f16",
    "rms_norm_gated_f32", "rms_norm_gated_f16",
    "add_rms_norm_f32", "add_rms_norm_f16",
    "rms_norm_channel_f32", "rms_norm_channel_f16",
    "f8e4m3_to_f32", "f8e4m3_to_f16",
    "fused_vector_attention_f16",
];

struct PipelineCache {
    pipelines: RwLock<HashMap<&'static str, candle_metal_kernels::metal::ComputePipeline>>,
    compile_lock: Mutex<()>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
            compile_lock: Mutex::new(()),
        }
    }

    fn get_or_create(
        &self,
        device: &candle_core::MetalDevice,
        kernel_name: &'static str,
    ) -> Result<candle_metal_kernels::metal::ComputePipeline> {
        if let Ok(cache) = self.pipelines.read() {
            if let Some(pipeline) = cache.get(kernel_name) {
                return Ok(pipeline.clone());
            }
        }
        let _guard = self.compile_lock.lock().map_err(|e| candle_core::Error::Msg(format!("compile lock: {e}")))?;
        if let Ok(cache) = self.pipelines.read() {
            if let Some(pipeline) = cache.get(kernel_name) {
                return Ok(pipeline.clone());
            }
        }
        let lib = device.new_library_with_source(FUSED_OPS_MSL, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal shader compile: {e}")))?;
        let mut cache = self.pipelines.write().map_err(|e| candle_core::Error::Msg(format!("pipeline write lock: {e}")))?;
        for &name in ALL_KERNELS {
            if cache.contains_key(name) { continue; }
            if let Ok(func) = lib.get_function(name, None) {
                if let Ok(pipeline) = device.new_compute_pipeline_state_with_function(&func) {
                    cache.insert(name, pipeline);
                }
            }
        }
        cache.get(kernel_name).cloned().ok_or_else(|| {
            candle_core::Error::Msg(format!("metal kernel not found: {kernel_name}"))
        })
    }
}

static PIPELINE_CACHE: std::sync::LazyLock<PipelineCache> = std::sync::LazyLock::new(PipelineCache::new);

// ─── Helper: dispatch an elementwise 2-input kernel ─────────────────

#[inline]
fn dispatch_binary(
    s1: &candle_core::MetalStorage, l1: &Layout,
    s2: &candle_core::MetalStorage, l2: &Layout,
    f32_kernel: &'static str, f16_kernel: &'static str, label: &'static str,
) -> Result<(candle_core::MetalStorage, Shape)> {
    let device = s1.device();
    let el = l1.shape().elem_count();
    let kernel_name: &'static str = match s1.dtype() {
        DType::F32 => f32_kernel,
        DType::F16 => f16_kernel,
        dt => candle_core::bail!("{label} metal: unsupported dtype {dt:?}"),
    };
    let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
    let output = device.new_buffer(el, s1.dtype(), label)?;
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

// ─── Helper: dispatch an elementwise 3-input kernel ─────────────────

struct TernaryKernel { f32_kernel: &'static str, f16_kernel: &'static str, label: &'static str }

#[allow(clippy::too_many_arguments)]
#[inline]
fn dispatch_ternary(
    s1: &candle_core::MetalStorage, l1: &Layout,
    s2: &candle_core::MetalStorage, l2: &Layout,
    s3: &candle_core::MetalStorage, l3: &Layout,
    k: &TernaryKernel,
) -> Result<(candle_core::MetalStorage, Shape)> {
    let device = s1.device();
    let el = l1.shape().elem_count();
    let kernel_name: &'static str = match s1.dtype() {
        DType::F32 => k.f32_kernel,
        DType::F16 => k.f16_kernel,
        dt => candle_core::bail!("{} metal: unsupported dtype {dt:?}", k.label),
    };
    let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
    let output = device.new_buffer(el, s1.dtype(), k.label)?;
    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline);
    let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
    let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
    let off3 = l3.start_offset() * s3.dtype().size_in_bytes();
    candle_metal_kernels::utils::set_param(&encoder, 0, (s1.buffer(), off1));
    candle_metal_kernels::utils::set_param(&encoder, 1, (s2.buffer(), off2));
    candle_metal_kernels::utils::set_param(&encoder, 2, (s3.buffer(), off3));
    candle_metal_kernels::utils::set_param(&encoder, 3, (&*output, 0usize));
    candle_metal_kernels::utils::set_param(&encoder, 4, el as u32);
    let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
    let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
    encoder.dispatch_threads(grid, group);
    Ok((candle_core::MetalStorage::new(output, device.clone(), el, s1.dtype()), l1.shape().clone()))
}

const ADD3_KERNEL: TernaryKernel = TernaryKernel { f32_kernel: "add3_f32", f16_kernel: "add3_f16", label: "add3" };
const SUB_MUL_KERNEL: TernaryKernel = TernaryKernel { f32_kernel: "sub_mul_f32", f16_kernel: "sub_mul_f16", label: "sub_mul" };

// ─── CustomOp structs ───────────────────────────────────────────────

struct MetalSiluMul;
impl candle_core::CustomOp2 for MetalSiluMul {
    fn name(&self) -> &'static str { "metal_silu_mul" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalSiluMul: expected Metal device") }
    fn metal_fwd(&self, s1: &candle_core::MetalStorage, l1: &Layout, s2: &candle_core::MetalStorage, l2: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        dispatch_binary(s1, l1, s2, l2, "silu_mul_f32", "silu_mul_f16", "silu_mul")
    }
}

struct MetalStableSoftplus;
impl candle_core::CustomOp1 for MetalStableSoftplus {
    fn name(&self) -> &'static str { "metal_stable_softplus" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalStableSoftplus: expected Metal device") }
    fn metal_fwd(&self, s: &candle_core::MetalStorage, l: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s.device();
        let el = l.shape().elem_count();
        let kernel_name: &'static str = match s.dtype() { DType::F32 => "stable_softplus_f32", DType::F16 => "stable_softplus_f16", dt => candle_core::bail!("stable_softplus metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
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

struct MetalExpMul;
impl candle_core::CustomOp2 for MetalExpMul {
    fn name(&self) -> &'static str { "metal_exp_mul" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalExpMul: expected Metal device") }
    fn metal_fwd(&self, s1: &candle_core::MetalStorage, l1: &Layout, s2: &candle_core::MetalStorage, l2: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        dispatch_binary(s1, l1, s2, l2, "exp_mul_f32", "exp_mul_f16", "exp_mul")
    }
}

struct MetalAdd3;
impl candle_core::CustomOp3 for MetalAdd3 {
    fn name(&self) -> &'static str { "metal_add3" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalAdd3: expected Metal device") }
    fn metal_fwd(&self, s1: &candle_core::MetalStorage, l1: &Layout, s2: &candle_core::MetalStorage, l2: &Layout, s3: &candle_core::MetalStorage, l3: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        dispatch_ternary(s1, l1, s2, l2, s3, l3, &ADD3_KERNEL)
    }
}

struct MetalSubMul;
impl candle_core::CustomOp3 for MetalSubMul {
    fn name(&self) -> &'static str { "metal_sub_mul" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalSubMul: expected Metal device") }
    fn metal_fwd(&self, s1: &candle_core::MetalStorage, l1: &Layout, s2: &candle_core::MetalStorage, l2: &Layout, s3: &candle_core::MetalStorage, l3: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        dispatch_ternary(s1, l1, s2, l2, s3, l3, &SUB_MUL_KERNEL)
    }
}

struct MetalAddScaled;
impl candle_core::CustomOp3 for MetalAddScaled {
    fn name(&self) -> &'static str { "metal_add_scaled" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalAddScaled: expected Metal device") }
    #[allow(clippy::too_many_arguments)]
    fn metal_fwd(&self, s1: &candle_core::MetalStorage, l1: &Layout, s2: &candle_core::MetalStorage, l2: &Layout, s3: &candle_core::MetalStorage, l3: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s1.device();
        let el = l1.shape().elem_count();
        let dims = l1.shape().dims();
        let (channels, time_len) = if dims.len() >= 3 { (dims[dims.len() - 2], dims[dims.len() - 1]) } else { (1usize, el) };
        let kernel_name: &'static str = match s1.dtype() { DType::F32 => "add_scaled_f32", DType::F16 => "add_scaled_f16", dt => candle_core::bail!("add_scaled metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
        let output = device.new_buffer(el, s1.dtype(), "add_scaled")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        let off3 = l3.start_offset() * s3.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s1.buffer(), off1));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s2.buffer(), off2));
        candle_metal_kernels::utils::set_param(&encoder, 2, (s3.buffer(), off3));
        candle_metal_kernels::utils::set_param(&encoder, 3, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 4, el as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, channels as u32);
        candle_metal_kernels::utils::set_param(&encoder, 6, time_len as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s1.dtype()), l1.shape().clone()))
    }
}

struct MetalDepthwiseConv1dSilu;
impl candle_core::CustomOp2 for MetalDepthwiseConv1dSilu {
    fn name(&self) -> &'static str { "metal_depthwise_conv1d_silu" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalDepthwiseConv1dSilu: expected Metal device") }
    fn metal_fwd(&self, s_win: &candle_core::MetalStorage, l_win: &Layout, s_wt: &candle_core::MetalStorage, l_wt: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s_win.device();
        let win_dims = l_win.shape().dims();
        let (batch, channels, kernel_size) = (win_dims[0], win_dims[1], win_dims[2]);
        let out_count = batch * channels;
        let kernel_name: &'static str = match s_win.dtype() { DType::F32 => "depthwise_conv1d_silu_f32", DType::F16 => "depthwise_conv1d_silu_f16", dt => candle_core::bail!("depthwise_conv1d_silu metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
        let output = device.new_buffer(out_count, s_win.dtype(), "dw_conv1d_silu")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off_win = l_win.start_offset() * s_win.dtype().size_in_bytes();
        let off_wt = l_wt.start_offset() * s_wt.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s_win.buffer(), off_win));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s_wt.buffer(), off_wt));
        candle_metal_kernels::utils::set_param(&encoder, 2, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 3, out_count as u32);
        candle_metal_kernels::utils::set_param(&encoder, 4, channels as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, kernel_size as u32);
        let grid = objc2_metal::MTLSize { width: out_count, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(out_count, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), out_count, s_win.dtype()), Shape::from(vec![batch, channels])))
    }
}

struct MetalDepthwiseConv1dBias;
impl candle_core::CustomOp3 for MetalDepthwiseConv1dBias {
    fn name(&self) -> &'static str { "metal_depthwise_conv1d_bias" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalDepthwiseConv1dBias: expected Metal device") }
    #[allow(clippy::too_many_arguments)]
    fn metal_fwd(&self, s_in: &candle_core::MetalStorage, l_in: &Layout, s_wt: &candle_core::MetalStorage, l_wt: &Layout, s_bias: &candle_core::MetalStorage, l_bias: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s_in.device();
        let in_dims = l_in.shape().dims();
        let (batch, channels, t_padded) = (in_dims[0], in_dims[1], in_dims[2]);
        let wt_dims = l_wt.shape().dims();
        let kernel_size = wt_dims[wt_dims.len() - 1];
        let out_len = t_padded - kernel_size + 1;
        let out_count = batch * channels * out_len;
        let kernel_name: &'static str = match s_in.dtype() { DType::F32 => "depthwise_conv1d_bias_f32", DType::F16 => "depthwise_conv1d_bias_f16", dt => candle_core::bail!("depthwise_conv1d_bias metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
        let output = device.new_buffer(out_count, s_in.dtype(), "dw_conv1d_bias")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off_in = l_in.start_offset() * s_in.dtype().size_in_bytes();
        let off_wt = l_wt.start_offset() * s_wt.dtype().size_in_bytes();
        let off_bias = l_bias.start_offset() * s_bias.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s_in.buffer(), off_in));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s_wt.buffer(), off_wt));
        candle_metal_kernels::utils::set_param(&encoder, 2, (s_bias.buffer(), off_bias));
        candle_metal_kernels::utils::set_param(&encoder, 3, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 4, out_count as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, channels as u32);
        candle_metal_kernels::utils::set_param(&encoder, 6, out_len as u32);
        candle_metal_kernels::utils::set_param(&encoder, 7, t_padded as u32);
        candle_metal_kernels::utils::set_param(&encoder, 8, kernel_size as u32);
        let grid = objc2_metal::MTLSize { width: out_count, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(out_count, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), out_count, s_in.dtype()), Shape::from(vec![batch, channels, out_len])))
    }
}

struct MetalRmsNormGated { eps: f32 }
impl candle_core::CustomOp3 for MetalRmsNormGated {
    fn name(&self) -> &'static str { "metal_rms_norm_gated" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalRmsNormGated: expected Metal device") }
    #[allow(clippy::too_many_arguments)]
    fn metal_fwd(&self, s_x: &candle_core::MetalStorage, l_x: &Layout, s_z: &candle_core::MetalStorage, l_z: &Layout, s_w: &candle_core::MetalStorage, l_w: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s_x.device();
        let dims = l_x.shape().dims();
        let el = l_x.shape().elem_count();
        let hidden = *dims.last().ok_or_else(|| candle_core::Error::Msg("empty shape".into()))?;
        let num_rows = el / hidden;
        let kernel_name: &'static str = match s_x.dtype() { DType::F32 => "rms_norm_gated_f32", DType::F16 => "rms_norm_gated_f16", dt => candle_core::bail!("rms_norm_gated metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
        let output = device.new_buffer(el, s_x.dtype(), "rms_norm_gated")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off_x = l_x.start_offset() * s_x.dtype().size_in_bytes();
        let off_z = l_z.start_offset() * s_z.dtype().size_in_bytes();
        let off_w = l_w.start_offset() * s_w.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s_x.buffer(), off_x));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s_z.buffer(), off_z));
        candle_metal_kernels::utils::set_param(&encoder, 2, (s_w.buffer(), off_w));
        candle_metal_kernels::utils::set_param(&encoder, 3, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 4, hidden as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, self.eps);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg_width = hidden.min(max_threads);
        let grid = objc2_metal::MTLSize { width: hidden, height: num_rows, depth: 1 };
        let group = objc2_metal::MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s_x.dtype()), l_x.shape().clone()))
    }
}

struct MetalAddRmsNorm { eps: f32 }
impl candle_core::CustomOp3 for MetalAddRmsNorm {
    fn name(&self) -> &'static str { "metal_add_rms_norm" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalAddRmsNorm: expected Metal device") }
    #[allow(clippy::too_many_arguments)]
    fn metal_fwd(&self, s_a: &candle_core::MetalStorage, l_a: &Layout, s_b: &candle_core::MetalStorage, l_b: &Layout, s_w: &candle_core::MetalStorage, l_w: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s_a.device();
        let dims = l_a.shape().dims();
        let el = l_a.shape().elem_count();
        let hidden = *dims.last().ok_or_else(|| candle_core::Error::Msg("empty shape".into()))?;
        let num_rows = el / hidden;
        let kernel_name: &'static str = match s_a.dtype() { DType::F32 => "add_rms_norm_f32", DType::F16 => "add_rms_norm_f16", dt => candle_core::bail!("add_rms_norm metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
        let output = device.new_buffer(2 * el, s_a.dtype(), "add_rms_norm")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off_a = l_a.start_offset() * s_a.dtype().size_in_bytes();
        let off_b = l_b.start_offset() * s_b.dtype().size_in_bytes();
        let off_w = l_w.start_offset() * s_w.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s_a.buffer(), off_a));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s_b.buffer(), off_b));
        candle_metal_kernels::utils::set_param(&encoder, 2, (s_w.buffer(), off_w));
        candle_metal_kernels::utils::set_param(&encoder, 3, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 4, el as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, hidden as u32);
        candle_metal_kernels::utils::set_param(&encoder, 6, self.eps);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg_width = hidden.min(max_threads);
        let grid = objc2_metal::MTLSize { width: hidden, height: num_rows, depth: 1 };
        let group = objc2_metal::MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), 2 * el, s_a.dtype()), Shape::from(vec![2 * el])))
    }
}

struct MetalRmsNormChannel { eps: f32 }
impl candle_core::CustomOp2 for MetalRmsNormChannel {
    fn name(&self) -> &'static str { "metal_rms_norm_channel" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalRmsNormChannel: expected Metal device") }
    fn metal_fwd(&self, s_x: &candle_core::MetalStorage, l_x: &Layout, s_w: &candle_core::MetalStorage, l_w: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s_x.device();
        let dims = l_x.shape().dims();
        let el = l_x.shape().elem_count();
        let (batch, channels, time_len) = (dims[0], dims[1], dims[2]);
        let num_bt = batch * time_len;
        let kernel_name: &'static str = match s_x.dtype() { DType::F32 => "rms_norm_channel_f32", DType::F16 => "rms_norm_channel_f16", dt => candle_core::bail!("rms_norm_channel metal: unsupported dtype {dt:?}") };
        let pipeline = PIPELINE_CACHE.get_or_create(device, kernel_name)?;
        let output = device.new_buffer(el, s_x.dtype(), "rms_norm_channel")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off_x = l_x.start_offset() * s_x.dtype().size_in_bytes();
        let off_w = l_w.start_offset() * s_w.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s_x.buffer(), off_x));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s_w.buffer(), off_w));
        candle_metal_kernels::utils::set_param(&encoder, 2, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 3, channels as u32);
        candle_metal_kernels::utils::set_param(&encoder, 4, time_len as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, self.eps);
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg_width = channels.min(max_threads);
        let grid = objc2_metal::MTLSize { width: channels, height: num_bt, depth: 1 };
        let group = objc2_metal::MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s_x.dtype()), l_x.shape().clone()))
    }
}

struct MetalF8ToF32;
impl candle_core::CustomOp1 for MetalF8ToF32 {
    fn name(&self) -> &'static str { "metal_f8e4m3_to_f32" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalF8ToF32: expected Metal device") }
    fn metal_fwd(&self, s: &candle_core::MetalStorage, l: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s.device();
        let el = l.shape().elem_count();
        let pipeline = PIPELINE_CACHE.get_or_create(device, "f8e4m3_to_f32")?;
        let output = device.new_buffer(el, DType::F32, "f8_to_f32")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = l.start_offset() * s.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s.buffer(), offset));
        candle_metal_kernels::utils::set_param(&encoder, 1, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 2, el as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, DType::F32), l.shape().clone()))
    }
}

struct MetalF8ToF16;
impl candle_core::CustomOp1 for MetalF8ToF16 {
    fn name(&self) -> &'static str { "metal_f8e4m3_to_f16" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalF8ToF16: expected Metal device") }
    fn metal_fwd(&self, s: &candle_core::MetalStorage, l: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        let device = s.device();
        let el = l.shape().elem_count();
        let pipeline = PIPELINE_CACHE.get_or_create(device, "f8e4m3_to_f16")?;
        let output = device.new_buffer(el, DType::F16, "f8_to_f16")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = l.start_offset() * s.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s.buffer(), offset));
        candle_metal_kernels::utils::set_param(&encoder, 1, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 2, el as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, DType::F16), l.shape().clone()))
    }
}

struct MetalFusedVectorAttention { scale: f32, gqa_ratio: u32 }
impl candle_core::CustomOp3 for MetalFusedVectorAttention {
    fn name(&self) -> &'static str { "metal_fused_vector_attention" }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> { candle_core::bail!("MetalFusedVectorAttention: expected Metal device") }
    #[allow(clippy::too_many_arguments)]
    fn metal_fwd(&self, s_q: &candle_core::MetalStorage, l_q: &Layout, s_k: &candle_core::MetalStorage, l_k: &Layout, s_v: &candle_core::MetalStorage, l_v: &Layout) -> Result<(candle_core::MetalStorage, Shape)> {
        // Q: (batch*heads, head_dim), K/V: (batch*kv_heads, kv_len, head_dim)
        let device = s_q.device();
        let q_dims = l_q.shape().dims();
        let k_dims = l_k.shape().dims();
        let (bh, head_dim) = (q_dims[0], q_dims[1]);
        let kv_len = k_dims[1];
        let pipeline = PIPELINE_CACHE.get_or_create(device, "fused_vector_attention_f16")?;
        let output = device.new_buffer(bh * head_dim, DType::F16, "fused_vec_attn")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off_q = l_q.start_offset() * s_q.dtype().size_in_bytes();
        let off_k = l_k.start_offset() * s_k.dtype().size_in_bytes();
        let off_v = l_v.start_offset() * s_v.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s_q.buffer(), off_q));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s_k.buffer(), off_k));
        candle_metal_kernels::utils::set_param(&encoder, 2, (s_v.buffer(), off_v));
        candle_metal_kernels::utils::set_param(&encoder, 3, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 4, head_dim as u32);
        candle_metal_kernels::utils::set_param(&encoder, 5, kv_len as u32);
        candle_metal_kernels::utils::set_param(&encoder, 6, self.scale);
        candle_metal_kernels::utils::set_param(&encoder, 7, self.gqa_ratio);
        // Grid: (head_dim, batch*heads) — one column per head_dim element, one row per head
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let tg_width = head_dim.min(max_threads);
        let grid = objc2_metal::MTLSize { width: head_dim, height: bh, depth: 1 };
        let group = objc2_metal::MTLSize { width: tg_width, height: 1, depth: 1 };
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), bh * head_dim, DType::F16), Shape::from(vec![bh, head_dim])))
    }
}

// ─── MetalBackend ────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    pub fn new(device: Device) -> Self {
        if let Device::Metal(ref metal_dev) = device {
            let _ = PIPELINE_CACHE.get_or_create(metal_dev, ALL_KERNELS[0]);
        }
        Self { device }
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str { "metal" }
    fn device(&self) -> &Device { &self.device }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let q_dims = q.dims();
        // Generation case: seq_len=1 + F16 → use fused MSL kernel (F32 internal precision)
        if q_dims.len() == 4 && q_dims[2] == 1 && q.dtype() == DType::F16 && !causal {
            let (batch, heads, _, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
            let k_dims = k.dims();
            let kv_heads = k_dims[1];
            let gqa_ratio = (heads / kv_heads) as u32;
            // Flatten Q: (batch, heads, 1, head_dim) → (batch*heads, head_dim)
            let q_flat = q.contiguous()?.reshape((batch * heads, head_dim))?;
            // Flatten K/V: (batch, kv_heads, kv_len, head_dim) → (batch*kv_heads, kv_len, head_dim)
            let k_flat = k.contiguous()?.reshape((batch * kv_heads, k_dims[2], head_dim))?;
            let v_flat = v.contiguous()?.reshape((batch * kv_heads, k_dims[2], head_dim))?;
            let out = q_flat.apply_op3_no_bwd(&k_flat, &v_flat, &MetalFusedVectorAttention { scale, gqa_ratio })?;
            return out.reshape((batch, heads, 1, head_dim));
        }
        // Promote to F32 if needed (F16 SDPA produces imprecise results on Metal)
        let q = q.to_dtype(DType::F32)?; // no-op if already F32
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        // Try fused SDPA first, fall back to manual attention if threadgroup memory exceeded
        match candle_nn::ops::sdpa(&q, &k, &v, None, causal, scale, 1.0) {
            Ok(result) => Ok(result),
            Err(_) => {
                let att = (q.matmul(&k.t()?)? * (scale as f64))?;
                let att = if causal {
                    let seq_len = att.dim(candle_core::D::Minus1)?;
                    let tril = Tensor::tril2(seq_len, DType::F32, att.device())?
                        .broadcast_as(att.shape())?;
                    let mask = ((tril - 1.0)? * 1e9)?;
                    (att + mask)?
                } else {
                    att
                };
                let att = candle_nn::ops::softmax_last_dim(&att)?;
                att.matmul(&v.contiguous()?)
            }
        }
    }

    // ── MSL-accelerated ops (validated by Metal vs CPU tests) ────────

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        gate.apply_op2_no_bwd(up, &MetalSiluMul)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        x.apply_op1_no_bwd(&MetalStableSoftplus)
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        a.apply_op3_no_bwd(b, c, &MetalAdd3)
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        x.apply_op2_no_bwd(y, &MetalExpMul)
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        a.apply_op3_no_bwd(b, c, &MetalSubMul)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        if c.dims().len() == 1 {
            a.apply_op3_no_bwd(b, c, &MetalAddScaled)
        } else {
            (a + b.broadcast_mul(c)?)?.contiguous()
        }
    }

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, _kernel_size: usize, _channels: usize) -> Result<Tensor> {
        window.apply_op2_no_bwd(weight, &MetalDepthwiseConv1dSilu)
    }

    fn depthwise_conv1d_bias(&self, padded_input: &Tensor, weight: &Tensor, bias: &Tensor, _kernel_size: usize, _channels: usize) -> Result<Tensor> {
        let weight_flat = if weight.dims().len() == 3 { weight.contiguous()?.flatten(1, 2)? } else { weight.contiguous()? };
        padded_input.apply_op3_no_bwd(&weight_flat, bias, &MetalDepthwiseConv1dBias)
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        let merged = Tensor::cat(&[ctx, input], 2)?;
        self.depthwise_conv1d_bias(&merged, weight, bias, kernel_size, channels)
    }

    // ── Candle tensor ops (norm + remaining ops) ─────────────────────

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let x = x.contiguous()?;
        let z = z.contiguous()?.to_dtype(x.dtype())?;
        x.apply_op3_no_bwd(&z, weight, &MetalRmsNormGated { eps })
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        let a = a.contiguous()?;
        let b = b.contiguous()?;
        let shape = a.shape().clone();
        let el = shape.elem_count();
        let packed = a.apply_op3_no_bwd(&b, weight, &MetalAddRmsNorm { eps })?;
        let residual = packed.narrow(0, 0, el)?.reshape(&shape)?;
        let normed = packed.narrow(0, el, el)?.reshape(&shape)?;
        Ok((residual, normed))
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let x = x.contiguous()?;
        x.apply_op2_no_bwd(weight, &MetalRmsNormChannel { eps })
    }

    fn adaln_modulate(&self, x: &Tensor, norm_weight: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, norm_weight, eps)?;
        (n.broadcast_mul(&(scale + 1.0)?)? + shift)?.contiguous()
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() != DType::F8E4M3 { return x.to_dtype(DType::F32); }
        x.apply_op1_no_bwd(&MetalF8ToF32)
    }

    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() != DType::F8E4M3 { return x.to_dtype(DType::F16); }
        x.apply_op1_no_bwd(&MetalF8ToF16)
    }

    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        if x.dtype() != DType::F8E4M3 { return x.to_dtype(DType::BF16); }
        let dev = x.device().clone();
        x.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_dtype(DType::BF16)?.to_device(&dev)
    }

    fn synchronize(&self) -> Result<()> { self.device.synchronize() }
}
