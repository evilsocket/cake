//! Vulkan compute backend via wgpu.
//!
//! GPU-accelerated elementwise ops + tiled matmul via WGSL compute shaders.
//! Complex ops (normalization, convolution) fall back to candle CPU.
//! Pipelines are compiled once and cached for the lifetime of the backend.
//!
//! **Buffer management**: Weight tensors are uploaded once and cached by candle
//! `TensorId` (stable across forward passes). Output and staging buffers are
//! pooled by power-of-2 byte size to eliminate per-dispatch allocation.
//!
//! **Target**: Steam Deck (AMD Van Gogh, RDNA 2, Vulkan 1.3, 16GB unified RAM).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor, TensorId, D};

use super::ComputeBackend;

const WGSL_SOURCE: &str = include_str!("ops.wgsl");
const WG_ELEM: u32 = 256; // workgroup size for elementwise ops

// ─── GPU buffer cache ────────────────────────────────────────────────

/// Caches GPU buffers keyed by candle `TensorId`.
///
/// Weight tensors have stable IDs across forward passes (they are fields in
/// model structs, not recreated). Upload happens once; subsequent calls return
/// the cached `Arc<wgpu::Buffer>`.
///
/// Activation tensors get new `TensorId`s each forward pass and will not
/// accumulate because they are short-lived and dropped before the next pass.
/// We do not evict — for a 1.6GB F16 model the f32 cache is ~3.2GB, well
/// within the Steam Deck's 16GB unified RAM.
struct GpuBufferCache {
    buffers: HashMap<TensorId, Arc<wgpu::Buffer>>,
}

impl GpuBufferCache {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Return a cached buffer or upload `data` into a new one.
    fn get_or_upload(
        &mut self,
        id: TensorId,
        data: &[f32],
        gpu: &wgpu::Device,
    ) -> Arc<wgpu::Buffer> {
        if let Some(buf) = self.buffers.get(&id) {
            return buf.clone();
        }
        let size = (data.len() * 4) as u64;
        let buf = gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        buf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
        buf.unmap();
        let buf = Arc::new(buf);
        self.buffers.insert(id, buf.clone());
        buf
    }
}

// ─── Buffer pool ─────────────────────────────────────────────────────

/// Power-of-2 buffer pool to avoid per-dispatch allocation.
///
/// Two pools: **storage** (compute output) and **staging** (CPU readback).
/// After a dispatch completes the caller releases buffers back into the pool.
struct BufferPool {
    /// `STORAGE | COPY_SRC` buffers keyed by bucket size.
    storage: HashMap<u64, Vec<wgpu::Buffer>>,
    /// `MAP_READ | COPY_DST` buffers keyed by bucket size.
    staging: HashMap<u64, Vec<wgpu::Buffer>>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            storage: HashMap::new(),
            staging: HashMap::new(),
        }
    }

    /// Round `bytes` up to the next power of 2 (minimum 256).
    fn bucket(bytes: u64) -> u64 {
        bytes.next_power_of_two().max(256)
    }

    /// Acquire a storage buffer of at least `bytes` bytes.
    fn acquire_storage(&mut self, gpu: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
        let key = Self::bucket(bytes);
        self.storage
            .entry(key)
            .or_default()
            .pop()
            .unwrap_or_else(|| {
                gpu.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: key,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
    }

    /// Release a storage buffer back to the pool.
    fn release_storage(&mut self, buf: wgpu::Buffer) {
        let key = Self::bucket(buf.size());
        self.storage.entry(key).or_default().push(buf);
    }

    /// Acquire a staging (readback) buffer of at least `bytes` bytes.
    fn acquire_staging(&mut self, gpu: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
        let key = Self::bucket(bytes);
        self.staging
            .entry(key)
            .or_default()
            .pop()
            .unwrap_or_else(|| {
                gpu.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: key,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
    }

    /// Release a staging buffer back to the pool.
    fn release_staging(&mut self, buf: wgpu::Buffer) {
        let key = Self::bucket(buf.size());
        self.staging.entry(key).or_default().push(buf);
    }
}

// ─── Vulkan backend ──────────────────────────────────────────────────

/// Vulkan backend with cached pipelines, persistent GPU weight buffers, and
/// pooled output/staging buffers.
pub struct VulkanBackend {
    device: Device,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
    /// Cached compute pipelines keyed by entry point name.
    pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
    /// Pre-compiled shader module (shared across all pipelines).
    module: wgpu::ShaderModule,
    /// Persistent GPU buffer cache for weight tensors.
    buffer_cache: Mutex<GpuBufferCache>,
    /// Pool of reusable output and staging buffers.
    buffer_pool: Mutex<BufferPool>,
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("device", &"vulkan")
            .finish()
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
            buffer_cache: Mutex::new(GpuBufferCache::new()),
            buffer_pool: Mutex::new(BufferPool::new()),
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

    /// Get or upload a tensor's data to GPU. Weight tensors (stable TensorId)
    /// are uploaded once; activation tensors are uploaded each time.
    fn get_or_upload(&self, tensor: &Tensor) -> Result<Arc<wgpu::Buffer>> {
        let id = tensor.id();
        let data = Self::to_f32_vec(tensor)?;
        let mut cache = self.buffer_cache.lock().unwrap();
        Ok(cache.get_or_upload(id, &data, &self.gpu))
    }

    /// Upload raw f32 data without caching (for data not tied to a tensor).
    fn upload_uncached(&self, data: &[f32]) -> wgpu::Buffer {
        let size = (data.len() * 4) as u64;
        let buf = self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        buf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
        buf.unmap();
        buf
    }

    /// Acquire a storage output buffer from the pool.
    fn alloc_output(&self, count: usize) -> wgpu::Buffer {
        let bytes = (count * 4) as u64;
        self.buffer_pool.lock().unwrap().acquire_storage(&self.gpu, bytes)
    }

    /// Release a storage buffer back to the pool.
    fn release_output(&self, buf: wgpu::Buffer) {
        self.buffer_pool.lock().unwrap().release_storage(buf);
    }

    /// Create a uniform buffer (small, not pooled).
    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Download `count` f32 values from a GPU buffer using a pooled staging buffer.
    fn download(&self, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let size = (count * 4) as u64;
        let staging = self.buffer_pool.lock().unwrap().acquire_staging(&self.gpu, size);

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.gpu.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view)[..count].to_vec();
        drop(view);
        staging.unmap();

        self.buffer_pool.lock().unwrap().release_staging(staging);
        result
    }

    // ── Tensor ↔ GPU helpers ────────────────────────────────────────

    fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()
    }

    fn from_f32_vec(data: Vec<f32>, shape: &[usize], dtype: DType) -> Result<Tensor> {
        Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(dtype)
    }

    // ── Dispatch: elementwise binary ────────────────────────────────

    fn dispatch_binary(&self, a: &Tensor, b: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = a.dtype();
        let shape = a.shape().clone();
        let n = a.elem_count();

        let buf_a = self.get_or_upload(a)?;
        let buf_b = self.get_or_upload(b)?;
        let buf_out = self.alloc_output(n);
        let buf_p = self.uniform(&[n as u32]);

        let pipeline = self.get_pipeline(entry);
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups((n as u32).div_ceil(WG_ELEM), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        let result = Self::from_f32_vec(self.download(&buf_out, n), shape.dims(), dtype);
        self.release_output(buf_out);
        result
    }

    // ── Dispatch: elementwise ternary ───────────────────────────────

    fn dispatch_ternary(
        &self,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        entry: &str,
    ) -> Result<Tensor> {
        let dtype = a.dtype();
        let shape = a.shape().clone();
        let n = a.elem_count();

        let buf_a = self.get_or_upload(a)?;
        let buf_b = self.get_or_upload(b)?;
        let buf_c = self.get_or_upload(c)?;
        let buf_out = self.alloc_output(n);
        let buf_p = self.uniform(&[n as u32]);

        let pipeline = self.get_pipeline(entry);
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_c.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups((n as u32).div_ceil(WG_ELEM), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        let result = Self::from_f32_vec(self.download(&buf_out, n), shape.dims(), dtype);
        self.release_output(buf_out);
        result
    }

    // ── Dispatch: elementwise unary ─────────────────────────────────

    fn dispatch_unary(&self, x: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = x.dtype();
        let shape = x.shape().clone();
        let n = x.elem_count();

        let buf_a = self.get_or_upload(x)?;
        let buf_b = self.upload_uncached(&[0.0f32]); // dummy for binding 1
        let buf_out = self.alloc_output(n);
        let buf_p = self.uniform(&[n as u32]);

        let pipeline = self.get_pipeline(entry);
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let mut enc = self.gpu.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups((n as u32).div_ceil(WG_ELEM), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));

        let result = Self::from_f32_vec(self.download(&buf_out, n), shape.dims(), dtype);
        self.release_output(buf_out);
        result
    }

    // ── GPU matmul: C[M,N] = A[M,K] × B[K,N] ──────────────────────

    fn gpu_matmul(
        &self,
        buf_a: &wgpu::Buffer,
        buf_b: &wgpu::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let buf_c = self.alloc_output(m * n);
        let buf_p = self.uniform(&[m as u32, n as u32, k as u32, 0]);

        let pipeline = self.get_pipeline("matmul");
        let bg = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        let wg_m = (m as u32).div_ceil(16);
        let wg_n = (n as u32).div_ceil(16);
        let mut enc = self.gpu.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(wg_m, wg_n, 1);
        }
        self.queue.submit(Some(enc.finish()));

        let result = self.download(&buf_c, m * n);
        self.release_output(buf_c);
        result
    }

    /// Tensor matmul via GPU. Handles batched matmul by iterating batches.
    /// Weight tensors (B) are cached on GPU across forward passes.
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

        let mk = m * k;
        let kn = k * n;
        let mn = m * n;

        if batch == 1 {
            // Single batch: upload full tensors via cache, dispatch once.
            let buf_a = self.get_or_upload(&a)?;
            let buf_b = self.get_or_upload(&b)?;
            let out = self.gpu_matmul(&buf_a, &buf_b, m, k, n);

            let mut out_shape: Vec<usize> = a_dims[..a_rank - 2].to_vec();
            out_shape.push(m);
            out_shape.push(n);
            return Tensor::from_vec(out, out_shape.as_slice(), &Device::Cpu);
        }

        // Multi-batch: extract slices and dispatch per batch.
        let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;

        let mut out = Vec::with_capacity(batch * mn);
        for i in 0..batch {
            let a_off = if a_batch == 1 { 0 } else { i * mk };
            let b_off = if b_batch == 1 { 0 } else { i * kn };
            let a_slice = &a_data[a_off..a_off + mk];
            let b_slice = &b_data[b_off..b_off + kn];
            let buf_a = self.upload_uncached(a_slice);
            let buf_b = self.upload_uncached(b_slice);
            out.extend_from_slice(&self.gpu_matmul(&buf_a, &buf_b, m, k, n));
        }

        // Build output shape
        let mut out_shape: Vec<usize> = a_dims[..a_rank - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        Tensor::from_vec(out, out_shape.as_slice(), &Device::Cpu)
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }
    fn device(&self) -> &Device {
        &self.device
    }

    // ── GPU-accelerated attention with matmul ────────────────────────

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
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
            mask.broadcast_as(attn.shape())?
                .where_cond(&attn, &neg_inf)?
        } else {
            attn
        };

        // softmax on CPU (reduction op, not great for naive GPU dispatch)
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // attn @ V via GPU matmul
        let out = self.tensor_matmul(&attn, &v)?;
        out.to_dtype(orig_dtype)
    }

    // ── Elementwise ops (GPU for large tensors, CPU for small) ────────
    // GPU dispatch overhead (~50µs) only pays off for tensors > 8K elements.

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if gate.elem_count() > 8192 {
            self.dispatch_binary(gate, up, "silu_mul")
        } else {
            (candle_nn::ops::silu(&gate.contiguous()?)? * up.contiguous()?)?.contiguous()
        }
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        if x.elem_count() > 8192 {
            self.dispatch_unary(x, "stable_softplus")
        } else {
            let t88 = Tensor::full(88.0f32, x.shape(), x.device())?.to_dtype(x.dtype())?;
            let clamped = x.minimum(&t88)?;
            let sp = (clamped.exp()? + 1.0)?.log()?;
            x.maximum(&sp)
        }
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        if a.elem_count() > 8192 {
            self.dispatch_ternary(a, b, c, "add3")
        } else {
            ((a + b)? + c)?.contiguous()
        }
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        if x.elem_count() > 8192 {
            self.dispatch_binary(x, y, "exp_mul")
        } else {
            (x * y.exp()?)?.contiguous()
        }
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        if a.elem_count() > 8192 {
            self.dispatch_ternary(a, b, c, "sub_mul")
        } else {
            ((a - b)? * c)?.contiguous()
        }
    }

    // ── GPU matmul ───────────────────────────────────────────────────

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // GPU matmul only wins for large matrices. For the small activations
        // in token-by-token generation (M=1), CPU GEMM with AVX2 is faster
        // than the GPU dispatch overhead.
        let m = a.dims()[a.dims().len() - 2];
        if m <= 4 {
            return a.matmul(b); // CPU fast path for generation
        }
        let orig_dtype = a.dtype();
        let result = self.tensor_matmul(a, b)?;
        result.to_dtype(orig_dtype)
    }

    // ── CPU fallback for complex ops ────────────────────────────────

    fn rms_norm_gated(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)?;
        (n * candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?)?.contiguous()
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
        x.transpose(1, 2)?
            .contiguous()
            .and_then(|t| candle_nn::ops::rms_norm(&t, weight, eps))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        _ks: usize,
        _ch: usize,
    ) -> Result<Tensor> {
        candle_nn::ops::silu(&window.broadcast_mul(&weight.unsqueeze(0)?)?.sum(D::Minus1)?)
    }

    fn depthwise_conv1d_bias(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        ks: usize,
        _ch: usize,
    ) -> Result<Tensor> {
        let out_t = input.dim(2)? - ks + 1;
        let mut slices = Vec::with_capacity(out_t);
        for t in 0..out_t {
            let w = input
                .narrow(2, t, ks)?
                .broadcast_mul(&weight.unsqueeze(0)?)?
                .sum(D::Minus1)?
                .broadcast_add(bias)?;
            slices.push(w.unsqueeze(2)?);
        }
        Tensor::cat(&slices, 2)
    }

    fn depthwise_conv1d_bias_ctx(
        &self,
        ctx: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        ks: usize,
        ch: usize,
    ) -> Result<Tensor> {
        self.depthwise_conv1d_bias(&Tensor::cat(&[ctx, input], 2)?, weight, bias, ks, ch)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?.contiguous()
    }

    fn adaln_modulate(
        &self,
        x: &Tensor,
        nw: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        (candle_nn::ops::rms_norm(&x.contiguous()?, nw, eps)?
            .broadcast_mul(&(scale + 1.0)?)?
            + shift)?
            .contiguous()
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
