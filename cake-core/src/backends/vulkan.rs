//! Vulkan compute backend via wgpu.
//!
//! Provides GPU compute on Vulkan-capable devices (Steam Deck, AMD GPUs, etc.)
//! without requiring vendor-specific toolchains (CUDA, Metal).
//!
//! Current implementation delegates all tensor operations to candle CPU ops
//! (identical to [`CpuBackend`]). The wgpu device and queue are initialized
//! at construction time, ready for future WGSL compute shader dispatch.
//!
//! **Target**: Steam Deck (AMD Van Gogh APU, RDNA 2, Vulkan 1.3, 16GB unified RAM).
//! On unified memory architectures, CPU↔GPU copies are near-free.

use candle_core::{Device, Result, Tensor};

use super::ComputeBackend;
use super::fused_ops;

/// Vulkan backend — wgpu-based GPU compute with CPU fallback.
///
/// Tensor data lives in candle CPU tensors. Compute-intensive operations
/// will be offloaded to Vulkan via wgpu compute shaders in future iterations.
pub struct VulkanBackend {
    device: Device,
    #[allow(dead_code)]
    gpu_device: wgpu::Device,
    #[allow(dead_code)]
    gpu_queue: wgpu::Queue,
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("device", &"vulkan")
            .finish()
    }
}

impl VulkanBackend {
    /// Create a Vulkan backend, initializing the wgpu device on the default adapter.
    /// Falls back to any available backend if Vulkan is not available.
    pub fn new() -> std::result::Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no Vulkan/GL adapter found".to_string())?;

        let adapter_info = adapter.get_info();
        log::info!(
            "Vulkan backend: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let (gpu_device, gpu_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cake-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("failed to create wgpu device: {e}"))?;

        Ok(Self {
            device: Device::Cpu, // tensors live on CPU; wgpu handles compute dispatch
            gpu_device,
            gpu_queue,
        })
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    // All operations delegate to CPU fused_ops. Future iterations will
    // dispatch compute-intensive ops (matmul, attention, elementwise) to
    // wgpu compute shaders via the gpu_device/gpu_queue.

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
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
        fused_ops::silu_mul(gate, up)
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        fused_ops::stable_softplus(x)
    }

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        fused_ops::rms_norm_gated(x, z, weight, eps)
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        fused_ops::add_rms_norm(a, b, weight, eps)
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        fused_ops::rms_norm_channel(x, weight, eps)
    }

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        fused_ops::depthwise_conv1d_silu(window, weight, kernel_size, channels)
    }

    fn depthwise_conv1d_bias(&self, padded_input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        fused_ops::depthwise_conv1d_bias(padded_input, weight, bias, kernel_size, channels)
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        fused_ops::depthwise_conv1d_bias_ctx(ctx, input, weight, bias, kernel_size, channels)
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

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        fused_ops::add_scaled(a, b, c)
    }

    fn adaln_modulate(&self, x: &Tensor, norm_weight: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
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

    // synchronize() — default no-op is correct (CPU tensors, no GPU command buffer)
}
