//! Compute backend abstraction.
//!
//! The [`ComputeBackend`] trait defines all backend-specific operations (fused kernels,
//! attention, device control). Implementations exist for CPU (default), CUDA, Metal,
//! and Vulkan (via wgpu).
//!
//! Models call `ctx.backend().method()` instead of backend-specific code paths,
//! making it trivial to add new GPU backends.

use anyhow::Result;
use candle_core::{Device, Tensor};

mod cpu;
pub use cpu::CpuBackend;

/// Abstraction over compute backends (CPU, CUDA, Metal, Vulkan).
///
/// Each method has a CPU-based default via [`CpuBackend`]. GPU backends override
/// specific methods for acceleration while falling back to CPU for unimplemented ops.
pub trait ComputeBackend: Send + Sync {
    /// Human-readable backend name for logging.
    fn name(&self) -> &str;

    /// The candle device this backend operates on.
    fn device(&self) -> &Device;

    // ── Attention ────────────────────────────────────────────────────

    /// Scaled dot-product attention.
    ///
    /// Backends may use flash-attn (CUDA), fused SDPA (Metal), wgpu matmul (Vulkan),
    /// or manual matmul + softmax (CPU).
    ///
    /// Input layout: `(batch, heads, seq_len, head_dim)`.
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor>;

    // ── Fused activations ────────────────────────────────────────────

    /// `silu(gate) * up` — MLP activation gate.
    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// `ln(1 + exp(clamp(x, -inf, 88)))` with `max(x, result)` — GDN gate.
    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor>;

    // ── Fused normalization ──────────────────────────────────────────

    /// `rms_norm(x, weight, eps) * silu(z)` — GDN output gating.
    fn rms_norm_gated(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor>;

    /// `rms_norm(a + b, weight, eps)` — residual + norm fusion.
    /// Returns `(normed, residual)` where `residual = a + b`.
    fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)>;

    // ── Fused convolutions (GDN) ─────────────────────────────────────

    /// Depthwise conv1d + SiLU on a single token window.
    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor>;

    /// Depthwise conv1d + bias (no activation).
    fn depthwise_conv1d_bias(
        &self,
        padded_input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor>;

    // ── Elementwise fusions ──────────────────────────────────────────

    /// `a + b + c` — three-way element-wise add.
    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor>;

    /// `exp(x) * y`.
    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor>;

    /// `(a - b) * c`.
    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor>;

    /// `a + b * c` — scaled addition.
    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor>;

    // ── F8 dequantization ────────────────────────────────────────────

    /// Dequantize F8E4M3 tensor to F32.
    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor>;

    /// Dequantize F8E4M3 tensor to F16.
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor>;

    /// Dequantize F8E4M3 tensor to BF16.
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor>;

    // ── Device control ───────────────────────────────────────────────

    /// Flush GPU command buffer. No-op on CPU/CUDA, required on Metal
    /// to prevent command buffer accumulation (>25 commands = 50x slowdown).
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Create the appropriate backend for the given device.
pub fn create_backend(device: &Device) -> Box<dyn ComputeBackend> {
    match device {
        Device::Cpu => Box::new(CpuBackend::new()),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Box::new(CpuBackend::with_device(device.clone())),
        #[cfg(feature = "metal")]
        Device::Metal(_) => Box::new(CpuBackend::with_device(device.clone())),
        #[allow(unreachable_patterns)]
        _ => Box::new(CpuBackend::new()),
    }
}
