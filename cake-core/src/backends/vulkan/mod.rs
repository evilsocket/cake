//! Vulkan compute backend via wgpu.
//!
//! Dispatches elementwise operations to Vulkan GPU via WGSL compute shaders.
//! Complex ops (attention, normalization, convolution) fall back to candle CPU.
//!
//! On unified memory architectures (Steam Deck), CPU↔GPU copies are near-free.
//! **Target**: Steam Deck (AMD Van Gogh APU, RDNA 2, Vulkan 1.3, 16GB unified RAM).

use candle_core::{DType, Device, Result, Tensor, D};

use super::ComputeBackend;

const WGSL_SOURCE: &str = include_str!("ops.wgsl");
const WORKGROUP_SIZE: u32 = 256;

/// Vulkan backend — wgpu compute shaders for elementwise ops, CPU fallback for the rest.
pub struct VulkanBackend {
    device: Device,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
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

        Ok(Self {
            device: Device::Cpu,
            gpu,
            queue,
        })
    }

    // ── GPU dispatch helpers ─────────────────────────────────────────

    /// Run a 2-input WGSL shader: output[i] = f(a[i], b[i])
    fn dispatch_binary(&self, a: &Tensor, b: &Tensor, entry: &str) -> Result<Tensor> {
        let a = a.to_dtype(DType::F32)?.contiguous()?;
        let b = b.to_dtype(DType::F32)?.contiguous()?;
        let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;
        let count = a_data.len() as u32;

        let buf_a = self.create_storage_buffer(&a_data);
        let buf_b = self.create_storage_buffer(&b_data);
        let buf_out = self.create_output_buffer(count as u64);
        let buf_params = self.create_uniform_buffer(count);

        let module = self.gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });
        let pipeline = self.gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        });

        let workgroups = count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = self.gpu.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let result = self.read_buffer(&buf_out, count as usize);
        Tensor::from_vec(result, a.shape(), &Device::Cpu)
    }

    /// Run a 3-input WGSL shader: output[i] = f(a[i], b[i], c[i])
    fn dispatch_ternary(&self, a: &Tensor, b: &Tensor, c: &Tensor, entry: &str) -> Result<Tensor> {
        let a = a.to_dtype(DType::F32)?.contiguous()?;
        let b = b.to_dtype(DType::F32)?.contiguous()?;
        let c = c.to_dtype(DType::F32)?.contiguous()?;
        let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;
        let c_data: Vec<f32> = c.flatten_all()?.to_vec1()?;
        let count = a_data.len() as u32;

        let buf_a = self.create_storage_buffer(&a_data);
        let buf_b = self.create_storage_buffer(&b_data);
        let buf_out = self.create_output_buffer(count as u64);
        let buf_params = self.create_uniform_buffer(count);
        let buf_c = self.create_storage_buffer(&c_data);

        let module = self.gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });
        let pipeline = self.gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: buf_c.as_entire_binding() },
            ],
        });

        let workgroups = count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = self.gpu.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let result = self.read_buffer(&buf_out, count as usize);
        Tensor::from_vec(result, a.shape(), &Device::Cpu)
    }

    /// Run a 1-input WGSL shader: output[i] = f(a[i])
    /// Uses the binary layout with a dummy second buffer.
    fn dispatch_unary(&self, x: &Tensor, entry: &str) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let x_data: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let count = x_data.len() as u32;

        let buf_a = self.create_storage_buffer(&x_data);
        // Dummy buffer for binding 1 (shader reads input_a only for unary ops)
        let buf_b = self.create_storage_buffer(&[0.0f32]);
        let buf_out = self.create_output_buffer(count as u64);
        let buf_params = self.create_uniform_buffer(count);

        let module = self.gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });
        let pipeline = self.gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = self.gpu.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        });

        let workgroups = count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = self.gpu.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let result = self.read_buffer(&buf_out, count as usize);
        Tensor::from_vec(result, x.shape(), &Device::Cpu)
    }

    fn create_storage_buffer(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn create_output_buffer(&self, count: u64) -> wgpu::Buffer {
        self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: count * 4, // f32 = 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn create_uniform_buffer(&self, count: u32) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.gpu.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[count]),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn read_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let staging = self.gpu.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.gpu.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).unwrap(); });
        self.gpu.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str { "vulkan" }
    fn device(&self) -> &Device { &self.device }

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

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
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
            mask.broadcast_as(attn.shape())?.where_cond(&attn, &neg_inf)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)
    }

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

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, _kernel_size: usize, _channels: usize) -> Result<Tensor> {
        let w = weight.unsqueeze(0)?;
        candle_nn::ops::silu(&window.broadcast_mul(&w)?.sum(D::Minus1)?)
    }

    fn depthwise_conv1d_bias(&self, padded_input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, _channels: usize) -> Result<Tensor> {
        let output_time = padded_input.dim(2)? - kernel_size + 1;
        let mut slices = Vec::with_capacity(output_time);
        for t in 0..output_time {
            let window = padded_input.narrow(2, t, kernel_size)?;
            let w = weight.unsqueeze(0)?;
            let conv_t = window.broadcast_mul(&w)?.sum(D::Minus1)?.broadcast_add(bias)?;
            slices.push(conv_t.unsqueeze(2)?);
        }
        Tensor::cat(&slices, 2)
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize, channels: usize) -> Result<Tensor> {
        let combined = Tensor::cat(&[ctx, input], 2)?;
        self.depthwise_conv1d_bias(&combined, weight, bias, kernel_size, channels)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?.contiguous()
    }

    fn adaln_modulate(&self, x: &Tensor, norm_weight: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, norm_weight, eps)?;
        (n.broadcast_mul(&(scale + 1.0)?)? + shift)?.contiguous()
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32) }
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F16) }
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::BF16) }
}
