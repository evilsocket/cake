//! Vulkan compute backend via wgpu.
//!
//! GPU-accelerated elementwise ops + tiled matmul via WGSL compute shaders.
//! Complex ops (normalization, convolution) fall back to candle CPU.
//! Pipelines are compiled once and cached for the lifetime of the backend.
//!
//! **Target**: Steam Deck (AMD Van Gogh, RDNA 2, Vulkan 1.3, 16GB unified RAM).

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Result, Tensor, D};

use super::ComputeBackend;

const WGSL_SOURCE: &str = include_str!("ops.wgsl");
const WG_ELEM: u32 = 256; // workgroup size for elementwise ops

/// Vulkan backend with cached pipelines and GPU matmul.
pub struct VulkanBackend {
    device: Device,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
    /// Cached compute pipelines keyed by entry point name.
    pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
    /// Pre-compiled shader module (shared across all pipelines).
    module: wgpu::ShaderModule,
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend").field("device", &"vulkan").finish()
    }
}

impl VulkanBackend {
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

        let info = adapter.get_info();
        log::info!("Vulkan backend: {} ({:?})", info.name, info.backend);

        let (gpu, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cake-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("wgpu device: {e}"))?;

        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cake_ops"),
            source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });

        Ok(Self {
            device: Device::Cpu,
            gpu,
            queue,
            pipelines: Mutex::new(HashMap::new()),
            module,
        })
    }

    // ── Pipeline cache ──────────────────────────────────────────────

    fn get_pipeline(&self, entry: &str) -> wgpu::ComputePipeline {
        let mut cache = self.pipelines.lock().unwrap();
        if let Some(p) = cache.get(entry) {
            return p.clone();
        }
        let pipeline = self.gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &self.module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        cache.insert(entry.to_string(), pipeline.clone());
        pipeline
    }

    // ── Buffer helpers ──────────────────────────────────────────────

    fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn alloc_output(&self, count: usize) -> wgpu::Buffer {
        self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn download(&self, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.gpu.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        result
    }

    // ── Tensor ↔ GPU helpers ────────────────────────────────────────

    fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()
    }

    fn from_f32_vec(data: Vec<f32>, shape: &[usize], dtype: DType) -> Result<Tensor> {
        Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(dtype)
    }

    // ── Dispatch: elementwise binary ────────────────────────────────

    fn dispatch_binary(&self, a: &Tensor, b: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = a.dtype();
        let shape = a.shape().clone();
        let a_data = Self::to_f32_vec(a)?;
        let b_data = Self::to_f32_vec(b)?;
        let n = a_data.len();

        let buf_a = self.upload(&a_data);
        let buf_b = self.upload(&b_data);
        let buf_out = self.alloc_output(n);
        let buf_p = self.uniform(&[n as u32]);

        let pipeline = self.get_pipeline(entry);
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_p.as_entire_binding() },
            ],
        });

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default());
          p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]);
          p.dispatch_workgroups((n as u32).div_ceil(WG_ELEM), 1, 1); }
        self.queue.submit(Some(enc.finish()));

        Self::from_f32_vec(self.download(&buf_out, n), shape.dims(), dtype)
    }

    // ── Dispatch: elementwise ternary ───────────────────────────────

    fn dispatch_ternary(&self, a: &Tensor, b: &Tensor, c: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = a.dtype();
        let shape = a.shape().clone();
        let a_data = Self::to_f32_vec(a)?;
        let b_data = Self::to_f32_vec(b)?;
        let c_data = Self::to_f32_vec(c)?;
        let n = a_data.len();

        let buf_a = self.upload(&a_data);
        let buf_b = self.upload(&b_data);
        let buf_out = self.alloc_output(n);
        let buf_p = self.uniform(&[n as u32]);
        let buf_c = self.upload(&c_data);

        let pipeline = self.get_pipeline(entry);
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_p.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_c.as_entire_binding() },
            ],
        });

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default());
          p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]);
          p.dispatch_workgroups((n as u32).div_ceil(WG_ELEM), 1, 1); }
        self.queue.submit(Some(enc.finish()));

        Self::from_f32_vec(self.download(&buf_out, n), shape.dims(), dtype)
    }

    // ── Dispatch: elementwise unary ─────────────────────────────────

    fn dispatch_unary(&self, x: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = x.dtype();
        let shape = x.shape().clone();
        let x_data = Self::to_f32_vec(x)?;
        let n = x_data.len();

        let buf_a = self.upload(&x_data);
        let buf_b = self.upload(&[0.0f32]); // dummy for binding 1
        let buf_out = self.alloc_output(n);
        let buf_p = self.uniform(&[n as u32]);

        let pipeline = self.get_pipeline(entry);
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_p.as_entire_binding() },
            ],
        });

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default());
          p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]);
          p.dispatch_workgroups((n as u32).div_ceil(WG_ELEM), 1, 1); }
        self.queue.submit(Some(enc.finish()));

        Self::from_f32_vec(self.download(&buf_out, n), shape.dims(), dtype)
    }

    // ── GPU matmul: C[M,N] = A[M,K] × B[K,N] ──────────────────────

    fn gpu_matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let buf_a = self.upload(a);
        let buf_b = self.upload(b);
        let buf_c = self.alloc_output(m * n);
        let buf_p = self.uniform(&[m as u32, n as u32, k as u32, 0]);

        let pipeline = self.get_pipeline("matmul");
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_p.as_entire_binding() },
            ],
        });

        let wg_m = (m as u32).div_ceil(16);
        let wg_n = (n as u32).div_ceil(16);
        let mut enc = self.gpu.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default());
          p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]);
          p.dispatch_workgroups(wg_m, wg_n, 1); }
        self.queue.submit(Some(enc.finish()));

        self.download(&buf_c, m * n)
    }

    /// Tensor matmul via GPU. Handles batched matmul by iterating batches.
    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = a.to_dtype(DType::F32)?.contiguous()?;
        let b = b.to_dtype(DType::F32)?.contiguous()?;

        let a_dims = a.dims();
        let b_dims = b.dims();
        let a_rank = a_dims.len();
        let b_rank = b_dims.len();

        // Extract M, K, N
        let m = a_dims[a_rank - 2];
        let k = a_dims[a_rank - 1];
        let n = b_dims[b_rank - 1];

        // Batch dimensions
        let a_batch: usize = a_dims[..a_rank - 2].iter().product();
        let b_batch: usize = b_dims[..b_rank - 2].iter().product();
        let batch = a_batch.max(b_batch);

        let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;

        let mk = m * k;
        let kn = k * n;
        let mn = m * n;

        let mut out = Vec::with_capacity(batch * mn);
        for i in 0..batch {
            let a_off = if a_batch == 1 { 0 } else { i * mk };
            let b_off = if b_batch == 1 { 0 } else { i * kn };
            let a_slice = &a_data[a_off..a_off + mk];
            let b_slice = &b_data[b_off..b_off + kn];
            out.extend_from_slice(&self.gpu_matmul(a_slice, b_slice, m, k, n));
        }

        // Build output shape
        let mut out_shape: Vec<usize> = a_dims[..a_rank - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        Tensor::from_vec(out, out_shape.as_slice(), &Device::Cpu)
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str { "vulkan" }
    fn device(&self) -> &Device { &self.device }

    // ── GPU-accelerated attention with matmul ────────────────────────

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let orig_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Q @ K^T via GPU matmul
        let attn = self.tensor_matmul(&q, &k.t()?)?;
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
            mask.broadcast_as(attn.shape())?.where_cond(&attn, &neg_inf)?
        } else {
            attn
        };

        // softmax on CPU (reduction op, not great for naive GPU dispatch)
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // attn @ V via GPU matmul
        let out = self.tensor_matmul(&attn, &v)?;
        out.to_dtype(orig_dtype)
    }

    // ── GPU-accelerated elementwise ops ──────────────────────────────

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        self.dispatch_binary(gate, up, "silu_mul")
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        self.dispatch_unary(x, "stable_softplus")
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        self.dispatch_ternary(a, b, c, "add3")
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        self.dispatch_binary(x, y, "exp_mul")
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        self.dispatch_ternary(a, b, c, "sub_mul")
    }

    // ── CPU fallback for complex ops ────────────────────────────────

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)?;
        (n * candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?)?.contiguous()
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        let res = (a + b)?;
        let normed = candle_nn::ops::rms_norm(&res.contiguous()?, weight, eps)?;
        Ok((res, normed))
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        x.transpose(1, 2)?.contiguous()
            .and_then(|t| candle_nn::ops::rms_norm(&t, weight, eps))?
            .transpose(1, 2)?.contiguous()
    }

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, _ks: usize, _ch: usize) -> Result<Tensor> {
        candle_nn::ops::silu(&window.broadcast_mul(&weight.unsqueeze(0)?)?.sum(D::Minus1)?)
    }

    fn depthwise_conv1d_bias(&self, input: &Tensor, weight: &Tensor, bias: &Tensor, ks: usize, _ch: usize) -> Result<Tensor> {
        let out_t = input.dim(2)? - ks + 1;
        let mut slices = Vec::with_capacity(out_t);
        for t in 0..out_t {
            let w = input.narrow(2, t, ks)?.broadcast_mul(&weight.unsqueeze(0)?)?.sum(D::Minus1)?.broadcast_add(bias)?;
            slices.push(w.unsqueeze(2)?);
        }
        Tensor::cat(&slices, 2)
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, ks: usize, ch: usize) -> Result<Tensor> {
        self.depthwise_conv1d_bias(&Tensor::cat(&[ctx, input], 2)?, weight, bias, ks, ch)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?.contiguous()
    }

    fn adaln_modulate(&self, x: &Tensor, nw: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
        (candle_nn::ops::rms_norm(&x.contiguous()?, nw, eps)?.broadcast_mul(&(scale + 1.0)?)? + shift)?.contiguous()
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32) }
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F16) }
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::BF16) }
}
