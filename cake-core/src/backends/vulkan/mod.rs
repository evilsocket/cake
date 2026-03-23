//! Vulkan compute backend via ash (raw Vulkan bindings).
//!
//! GPU-accelerated elementwise ops + tiled matmul via SPIR-V compute shaders
//! (compiled from WGSL at build time via naga). Complex ops (normalization,
//! convolution) fall back to candle CPU.
//!
//! **Key design**: On UMA hardware (Steam Deck), all GPU buffers use
//! DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory, enabling zero-copy
//! reads and writes. No staging buffers, no map/unmap — just memcpy and
//! fence waits.
//!
//! **Target**: Steam Deck (AMD Van Gogh, RDNA 2, Vulkan 1.3, 16GB unified RAM).

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;
use candle_core::{DType, Device, Result, Tensor, TensorId, D};

use super::ComputeBackend;

// SPIR-V modules compiled from ops.wgsl at build time
include!(concat!(env!("OUT_DIR"), "/spirv_ops.rs"));

const WG_ELEM: u32 = 256; // workgroup size for elementwise ops

// ─── Mapped GPU buffer ──────────────────────────────────────────────

/// A GPU buffer with a persistently mapped host pointer (UMA).
struct MappedBuffer {
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    mapped_ptr: *mut u8,
    size: u64,
}

// Safety: MappedBuffer is only accessed while holding a Mutex.
unsafe impl Send for MappedBuffer {}
unsafe impl Sync for MappedBuffer {}

impl MappedBuffer {
    /// Write f32 data into the buffer via mapped pointer.
    fn write_f32(&self, data: &[f32]) {
        let bytes = bytemuck::cast_slice::<f32, u8>(data);
        assert!(bytes.len() as u64 <= self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped_ptr, bytes.len());
        }
    }

    /// Read f32 data from the buffer via mapped pointer.
    fn read_f32(&self, count: usize) -> Vec<f32> {
        let byte_count = count * 4;
        assert!(byte_count as u64 <= self.size);
        unsafe {
            let slice = std::slice::from_raw_parts(self.mapped_ptr as *const f32, count);
            slice.to_vec()
        }
    }

    /// Write u32 data into the buffer.
    fn write_u32(&self, data: &[u32]) {
        let bytes = bytemuck::cast_slice::<u32, u8>(data);
        assert!(bytes.len() as u64 <= self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped_ptr, bytes.len());
        }
    }
}

// ─── Buffer pool ─────────────────────────────────────────────────────

/// Power-of-2 buffer pool for output and activation buffers.
struct BufferPool {
    free: HashMap<u64, Vec<MappedBuffer>>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
        }
    }

    fn bucket(bytes: u64) -> u64 {
        bytes.next_power_of_two().max(256)
    }
}

// ─── Weight cache ────────────────────────────────────────────────────

type ViewKey = (usize, usize, usize, u64);

struct WeightCache {
    buffers: HashMap<TensorId, Arc<MappedBuffer>>,
    /// F32 view cache for weight.t() views.
    views: HashMap<ViewKey, Arc<MappedBuffer>>,
    /// F16 (raw bytes) view cache for GEMV — halves bandwidth.
    f16_views: HashMap<ViewKey, Arc<MappedBuffer>>,
}

impl WeightCache {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            views: HashMap::new(),
            f16_views: HashMap::new(),
        }
    }
}

// ─── Vulkan backend ──────────────────────────────────────────────────

pub struct VulkanBackend {
    device: Device,
    // Vulkan handles
    _entry: ash::Entry,
    _instance: ash::Instance,
    vk_device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    // Memory
    allocator: Mutex<Option<gpu_allocator::vulkan::Allocator>>,
    uma_memory_type: Option<u32>,
    // Pipelines: entry_point -> (pipeline, pipeline_layout, ds_layout, descriptor_set, num_bindings)
    // Descriptor sets are pre-allocated at init (one per pipeline, reused across dispatches).
    // ds_layout is kept for allocating additional sets during batched dispatch.
    pipelines: HashMap<String, (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout, vk::DescriptorSet, u32)>,
    descriptor_pool: vk::DescriptorPool,
    // Params uniform buffer (16 bytes, persistently mapped)
    params_buf: MappedBuffer,
    // Dummy buffer for unary dispatches (binding slot filler)
    dummy_buf: MappedBuffer,
    // Buffer management
    weight_cache: Mutex<WeightCache>,
    buffer_pool: Mutex<BufferPool>,
    // Activation cache: recent dispatch outputs cached by TensorId.
    // When the next dispatch consumes an output (e.g., silu_mul → matmul),
    // it finds the GPU buffer here and skips re-uploading.
    // Bounded FIFO (max 16 entries), evicts oldest on overflow.
    activation_cache: Mutex<Vec<(TensorId, Arc<MappedBuffer>)>>,
    // Dispatch lock (serializes GPU submissions)
    dispatch_lock: Mutex<()>,
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("device", &"vulkan/ash")
            .finish()
    }
}

impl VulkanBackend {
    pub fn new() -> std::result::Result<Self, String> {
        unsafe { Self::init_vulkan() }
    }

    unsafe fn init_vulkan() -> std::result::Result<Self, String> {
        // 1. Entry + Instance
        let entry = ash::Entry::linked();
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"cake-vulkan")
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry
            .create_instance(&instance_ci, None)
            .map_err(|e| format!("vkCreateInstance: {e}"))?;

        // 2. Physical device
        let phys_devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices: {e}"))?;
        if phys_devices.is_empty() {
            return Err("no Vulkan physical devices found".into());
        }
        // Prefer discrete, fall back to first
        let physical_device = phys_devices
            .iter()
            .find(|&&pd| {
                let props = instance.get_physical_device_properties(pd);
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .unwrap_or(phys_devices[0]);

        let props = instance.get_physical_device_properties(physical_device);
        let dev_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .to_string();
        log::info!("Vulkan backend (ash): {dev_name}");

        // 3. Queue family
        let queue_families =
            instance.get_physical_device_queue_family_properties(physical_device);
        let queue_family_index = queue_families
            .iter()
            .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("no compute queue family")? as u32;

        // 4. Logical device + queue
        let queue_priorities = [1.0f32];
        let queue_ci =
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);
        let device_ci = vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_ci));
        let vk_device = instance
            .create_device(physical_device, &device_ci, None)
            .map_err(|e| format!("vkCreateDevice: {e}"))?;
        let queue = vk_device.get_device_queue(queue_family_index, 0);

        // 5. Command pool + buffer
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = vk_device
            .create_command_pool(&pool_ci, None)
            .map_err(|e| format!("create_command_pool: {e}"))?;
        let alloc_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = vk_device
            .allocate_command_buffers(&alloc_ci)
            .map_err(|e| format!("allocate_command_buffers: {e}"))?[0];

        // 6. Fence
        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = vk_device
            .create_fence(&fence_ci, None)
            .map_err(|e| format!("create_fence: {e}"))?;

        // 7. Memory allocator
        let mut allocator = gpu_allocator::vulkan::Allocator::new(
            &gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: instance.clone(),
                device: vk_device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            },
        )
        .map_err(|e| format!("gpu-allocator: {e}"))?;

        // 8. Detect UMA memory type
        let mem_props = instance.get_physical_device_memory_properties(physical_device);
        let uma_memory_type = (0..mem_props.memory_type_count).find(|&i| {
            let flags = mem_props.memory_types[i as usize].property_flags;
            flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        });
        if uma_memory_type.is_some() {
            log::info!("UMA detected — using zero-copy DEVICE_LOCAL|HOST_VISIBLE buffers");
        } else {
            log::warn!("No UMA memory — falling back to HOST_VISIBLE buffers (slower)");
        }

        // 9. Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 256,
        }, vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 64,
        }];
        let dp_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(256)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = vk_device
            .create_descriptor_pool(&dp_ci, None)
            .map_err(|e| format!("create_descriptor_pool: {e}"))?;

        // 10. Create pipelines from SPIR-V modules, pre-allocate one descriptor set each
        let mut pipelines = HashMap::new();
        for &(name, spv_bytes) in SPIRV_MODULES {
            let spv_words: Vec<u32> = spv_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let shader_ci =
                vk::ShaderModuleCreateInfo::default().code(&spv_words);
            let shader_module = vk_device
                .create_shader_module(&shader_ci, None)
                .map_err(|e| format!("create_shader_module({name}): {e}"))?;

            // Determine bindings: 3-input ops (sub_mul, add3) have 5 bindings, rest have 4
            let num_bindings: u32 = if name == "sub_mul" || name == "add3" { 5 } else { 4 };

            // Create descriptor set layout
            let mut bindings_vec = Vec::new();
            for i in 0..num_bindings {
                let (ty, binding_idx) = if i == 3 {
                    (vk::DescriptorType::UNIFORM_BUFFER, 3)
                } else {
                    let idx = if i == 4 { 4 } else { i }; // binding 4 = input_c
                    (vk::DescriptorType::STORAGE_BUFFER, idx)
                };
                bindings_vec.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding_idx)
                        .descriptor_type(ty)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            let dsl_ci =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings_vec);
            let ds_layout = vk_device
                .create_descriptor_set_layout(&dsl_ci, None)
                .map_err(|e| format!("create_descriptor_set_layout({name}): {e}"))?;

            // Pre-allocate descriptor set (reused across all dispatches for this pipeline)
            let ds_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&ds_layout));
            let descriptor_set = vk_device
                .allocate_descriptor_sets(&ds_alloc_info)
                .map_err(|e| format!("allocate_descriptor_sets({name}): {e}"))?[0];

            // Pipeline layout
            let pl_ci = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&ds_layout));
            let pipe_layout = vk_device
                .create_pipeline_layout(&pl_ci, None)
                .map_err(|e| format!("create_pipeline_layout({name}): {e}"))?;

            // Compute pipeline — naga preserves the original entry point name
            let entry_name = std::ffi::CString::new(name).unwrap();
            let stage_ci = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_name);
            let pipeline_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage_ci)
                .layout(pipe_layout);
            let pipeline = vk_device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .map_err(|e| format!("create_compute_pipeline({name}): {e:?}"))?[0];

            vk_device.destroy_shader_module(shader_module, None);
            pipelines.insert(name.to_string(), (pipeline, pipe_layout, ds_layout, descriptor_set, num_bindings));
        }

        // 11. Params uniform buffer (16 bytes)
        let params_buf = Self::alloc_mapped_buffer(
            &vk_device,
            &mut allocator,
            16,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            uma_memory_type,
        )?;

        // 12. Dummy buffer for unary dispatches (avoids per-dispatch allocation)
        let dummy_buf = Self::alloc_mapped_buffer(
            &vk_device,
            &mut allocator,
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            uma_memory_type,
        )?;

        let backend = Self {
            device: Device::Cpu,
            _entry: entry,
            _instance: instance,
            vk_device,
            queue,
            command_pool,
            command_buffer,
            fence,
            allocator: Mutex::new(Some(allocator)),
            uma_memory_type,
            dummy_buf,
            pipelines,
            descriptor_pool,
            params_buf,
            weight_cache: Mutex::new(WeightCache::new()),
            buffer_pool: Mutex::new(BufferPool::new()),
            activation_cache: Mutex::new(Vec::with_capacity(16)),
            dispatch_lock: Mutex::new(()),
        };

        // Warm-up: dispatch every pipeline once to eliminate cold-start penalties.
        // Each first dispatch per pipeline pays a one-time cost (driver JIT, etc).
        {
            let out = backend.alloc_output(4);
            let dummy = backend.dummy_buf.buffer;
            // Elementwise pipelines (2-input)
            for entry in &["silu_mul", "exp_mul"] {
                let _ = backend.dispatch_compute(
                    entry,
                    &[dummy, dummy, out.buffer],
                    &out, 4, &[4, 0, 0, 0], (1, 1, 1),
                );
            }
            // Elementwise pipelines (3-input)
            for entry in &["add3", "sub_mul"] {
                let _ = backend.dispatch_compute(
                    entry,
                    &[dummy, dummy, out.buffer, dummy],
                    &out, 4, &[4, 0, 0, 0], (1, 1, 1),
                );
            }
            // Unary pipeline
            let _ = backend.dispatch_compute(
                "stable_softplus",
                &[dummy, dummy, out.buffer],
                &out, 4, &[4, 0, 0, 0], (1, 1, 1),
            );
            // GEMV: 1×4 * 4×4 = 1×4
            let _ = backend.dispatch_compute(
                "gemv",
                &[dummy, dummy, out.buffer],
                &out, 4, &[4, 4, 0, 0], (1, 1, 1),
            );
            // GEMM: 2×2 * 2×2 = 2×2
            let _ = backend.dispatch_compute(
                "matmul",
                &[dummy, dummy, out.buffer],
                &out, 4, &[2, 2, 2, 0], (1, 1, 1),
            );
            // Small GEMM: 2×2 * 2×2 = 2×2
            let _ = backend.dispatch_compute(
                "matmul_small",
                &[dummy, dummy, out.buffer],
                &out, 4, &[2, 2, 2, 0], (1, 1, 1),
            );
            // Scaled softmax: 1 row × 4 cols
            let _ = backend.dispatch_compute(
                "scaled_softmax",
                &[dummy, dummy, out.buffer],
                &out, 4,
                &[1, 4, 1.0f32.to_bits(), 0],
                (1, 1, 1),
            );
            // GEMV F16: 1×4 * 4×4 = 1×4
            let _ = backend.dispatch_compute(
                "gemv_f16",
                &[dummy, dummy, out.buffer],
                &out, 4, &[4, 4, 0, 0], (1, 1, 1),
            );
            backend.release_output(out);

            // Pre-warm buffer pool with common output sizes (avoids allocation on first real dispatch).
            // Covers: GEMM outputs (M*N), elementwise ops, GEMV outputs.
            for &count in &[1024, 4096, 8192, 16384, 32768, 65536, 262144] {
                let buf = backend.alloc_output(count);
                backend.release_output(buf);
            }
        }

        Ok(backend)
    }

    // ── Buffer allocation ────────────────────────────────────────────

    fn alloc_mapped_buffer(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        _uma_memory_type: Option<u32>,
    ) -> std::result::Result<MappedBuffer, String> {
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size.max(256))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            device
                .create_buffer(&buf_ci, None)
                .map_err(|e| format!("create_buffer: {e}"))?
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "cake-buf",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| format!("allocate: {e}"))?;

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| format!("bind_buffer_memory: {e}"))?;
        }

        let mapped_ptr = allocation
            .mapped_ptr()
            .ok_or("buffer not mapped — UMA required")?
            .as_ptr() as *mut u8;

        Ok(MappedBuffer {
            buffer,
            allocation,
            mapped_ptr,
            size: size.max(256),
        })
    }

    fn alloc_output(&self, count: usize) -> MappedBuffer {
        let bytes = (count * 4) as u64;
        let key = BufferPool::bucket(bytes);
        let mut pool = self.buffer_pool.lock().unwrap();
        if let Some(bufs) = pool.free.get_mut(&key) {
            if let Some(buf) = bufs.pop() {
                return buf;
            }
        }
        drop(pool);
        let mut alloc_guard = self.allocator.lock().unwrap();
        let alloc = alloc_guard.as_mut().unwrap();
        Self::alloc_mapped_buffer(
            &self.vk_device,
            alloc,
            key,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            self.uma_memory_type,
        )
        .expect("failed to allocate output buffer")
    }

    fn release_output(&self, buf: MappedBuffer) {
        let key = BufferPool::bucket(buf.size);
        let mut pool = self.buffer_pool.lock().unwrap();
        pool.free.entry(key).or_default().push(buf);
    }

    // ── Weight cache ─────────────────────────────────────────────────

    /// Cache a dispatch output buffer by the returned Tensor's TensorId.
    /// If the next dispatch consumes this tensor, `get_or_upload` will find
    /// the buffer here and skip the re-upload.
    fn cache_activation(&self, tensor_id: TensorId, buf: MappedBuffer) {
        let buf = Arc::new(buf);
        let mut act_cache = self.activation_cache.lock().unwrap();
        if act_cache.len() >= 16 {
            // Evict oldest entry, release its buffer back to pool
            let (_, old_buf) = act_cache.remove(0);
            if let Ok(old) = Arc::try_unwrap(old_buf) {
                self.release_output(old);
            }
        }
        act_cache.push((tensor_id, buf));
    }

    /// Compute a stable cache key from tensor's storage pointer + layout.
    /// Survives `.t()` calls which create new TensorIds but share storage.
    /// Works for any dtype by extracting raw Vec pointer from CpuStorage.
    #[allow(dead_code)]
    fn view_key(tensor: &Tensor) -> Option<(usize, usize, usize, u64)> {
        let (storage, layout) = tensor.storage_and_layout();
        let ptr = match &*storage {
            candle_core::Storage::Cpu(cpu) => {
                // Extract raw pointer regardless of dtype
                match cpu {
                    candle_core::CpuStorage::F16(v) => v.as_ptr() as usize,
                    candle_core::CpuStorage::F32(v) => v.as_ptr() as usize,
                    candle_core::CpuStorage::BF16(v) => v.as_ptr() as usize,
                    candle_core::CpuStorage::F64(v) => v.as_ptr() as usize,
                    _ => return None,
                }
            }
            _ => return None,
        };
        let offset = layout.start_offset();
        let count = layout.shape().elem_count();
        let strides = layout.stride();
        let stride_hash = strides
            .iter()
            .fold(0u64, |h, &s| h.wrapping_mul(0x517cc1b727220a95).wrapping_add(s as u64));
        Some((ptr, offset, count, stride_hash))
    }

    fn get_or_upload(&self, tensor: &Tensor) -> Result<Arc<MappedBuffer>> {
        let id = tensor.id();
        // Check activation cache first (ephemeral outputs from recent dispatches)
        {
            let mut act_cache = self.activation_cache.lock().unwrap();
            if let Some(pos) = act_cache.iter().position(|(tid, _)| *tid == id) {
                let (_, buf) = act_cache.remove(pos);
                return Ok(buf);
            }
        }
        // Check weight cache by TensorId (fast path for stable tensors)
        {
            let cache = self.weight_cache.lock().unwrap();
            if let Some(buf) = cache.buffers.get(&id) {
                return Ok(buf.clone());
            }
        }
        // View cache: only for NON-CONTIGUOUS tensors (weight.t() views).
        // Contiguous tensors (activations) must NOT use the view cache because
        // freed tensors can be reallocated at the same address, causing stale hits.
        let is_view = !tensor.is_contiguous();
        let vk = if is_view { Self::view_key(tensor) } else { None };
        if let Some(ref key) = vk {
            let cache = self.weight_cache.lock().unwrap();
            if let Some(buf) = cache.views.get(key) {
                return Ok(buf.clone());
            }
        }
        // Upload: convert to f32 contiguous, copy to GPU
        let tensor = if tensor.dtype() == DType::F32 { tensor.clone() } else { tensor.to_dtype(DType::F32)? };
        let tensor = if tensor.is_contiguous() { tensor } else { tensor.contiguous()? };
        let n = tensor.elem_count();
        let bytes = (n * 4) as u64;
        let mut alloc_guard = self.allocator.lock().unwrap();
        let alloc = alloc_guard.as_mut().unwrap();
        let buf = Self::alloc_mapped_buffer(
            &self.vk_device,
            alloc,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            self.uma_memory_type,
        )
        .map_err(candle_core::Error::Msg)?;
        // Write directly from tensor storage — avoids intermediate Vec allocation
        let (storage, layout) = tensor.storage_and_layout();
        if let candle_core::Storage::Cpu(cpu) = &*storage {
            let slice: &[f32] = cpu.as_slice()?;
            let offset = layout.start_offset();
            buf.write_f32(&slice[offset..offset + n]);
        } else {
            drop(storage);
            let data = Self::to_f32_vec(&tensor)?;
            buf.write_f32(&data);
        }
        let buf = Arc::new(buf);
        let mut cache = self.weight_cache.lock().unwrap();
        if let Some(key) = vk {
            // Non-contiguous view (e.g., weight.t()): store in view cache only.
            // Don't store by TensorId since .t() creates new IDs each call (would leak).
            cache.views.insert(key, buf.clone());
        } else {
            // Contiguous tensor with stable TensorId: store in buffers cache.
            cache.buffers.insert(id, buf.clone());
        }
        Ok(buf)
    }

    fn upload_uncached(&self, data: &[f32]) -> MappedBuffer {
        let bytes = (data.len() * 4) as u64;
        let key = BufferPool::bucket(bytes);
        let mut alloc_guard = self.allocator.lock().unwrap();
        let alloc = alloc_guard.as_mut().unwrap();
        let buf = Self::alloc_mapped_buffer(
            &self.vk_device,
            alloc,
            key,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            self.uma_memory_type,
        )
        .expect("failed to allocate upload buffer");
        buf.write_f32(data);
        buf
    }

    // ── Tensor helpers ───────────────────────────────────────────────

    fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        let t = if t.dtype() == DType::F32 { t.clone() } else { t.to_dtype(DType::F32)? };
        let t = if t.is_contiguous() { t } else { t.contiguous()? };
        t.flatten_all()?.to_vec1()
    }

    fn from_f32_vec(data: Vec<f32>, shape: &[usize], dtype: DType) -> Result<Tensor> {
        Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(dtype)
    }

    // ── GPU dispatch ─────────────────────────────────────────────────

    /// Core dispatch: bind pipeline, bind buffers, dispatch workgroups, fence wait, read output.
    fn dispatch_compute(
        &self,
        entry: &str,
        storage_buffers: &[vk::Buffer], // bindings 0, 1, 2, [4]
        output_buf: &MappedBuffer,
        output_count: usize,
        params: &[u32],
        workgroups: (u32, u32, u32),
    ) -> Vec<f32> {
        let _lock = self.dispatch_lock.lock().unwrap();
        let (pipeline, pipe_layout, _ds_layout, descriptor_set, _num_bindings) =
            self.pipelines.get(entry).unwrap_or_else(|| panic!("unknown pipeline: {entry}"));
        let descriptor_set = *descriptor_set;

        // Write params
        self.params_buf.write_u32(params);

        unsafe {
            // Build descriptor writes with stack-allocated arrays (max 5 storage + 1 uniform = 6)
            let num_storage = storage_buffers.len();
            let mut buf_infos: [vk::DescriptorBufferInfo; 6] = Default::default();
            for (i, &buf) in storage_buffers.iter().enumerate() {
                buf_infos[i] = vk::DescriptorBufferInfo::default()
                    .buffer(buf)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
            }
            buf_infos[num_storage] = vk::DescriptorBufferInfo::default()
                .buffer(self.params_buf.buffer)
                .offset(0)
                .range(16);

            let mut writes: [vk::WriteDescriptorSet; 6] = Default::default();
            for i in 0..num_storage {
                let binding = if i == 3 { 4 } else { i as u32 };
                writes[i] = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_infos[i]));
            }
            writes[num_storage] = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_infos[num_storage]));

            self.vk_device.update_descriptor_sets(&writes[..num_storage + 1], &[]);

            // Reset fence + command buffer.
            // Fence is always signaled here: either from SIGNALED init flag (first call)
            // or from the wait_for_fences at the bottom of the previous dispatch.
            self.vk_device.reset_fences(&[self.fence]).expect("reset_fences");
            self.vk_device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("reset_command_buffer");

            // Record
            let begin_ci = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.vk_device
                .begin_command_buffer(self.command_buffer, &begin_ci)
                .expect("begin_command_buffer");
            self.vk_device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                *pipeline,
            );
            self.vk_device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                *pipe_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.vk_device.cmd_dispatch(
                self.command_buffer,
                workgroups.0,
                workgroups.1,
                workgroups.2,
            );

            // No explicit compute→host barrier needed: fence wait provides
            // execution dependency, and HOST_COHERENT memory (used by all our
            // buffers via CpuToGpu allocation) makes shader writes visible
            // to the host automatically after execution completes.

            self.vk_device
                .end_command_buffer(self.command_buffer)
                .expect("end_command_buffer");

            // Submit
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.vk_device
                .queue_submit(self.queue, &[submit_info], self.fence)
                .expect("queue_submit");

            // Wait
            self.vk_device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .expect("wait_for_fences");
        }

        // Read output directly from mapped pointer
        output_buf.read_f32(output_count)
    }

    // ── Elementwise dispatch helpers ─────────────────────────────────

    fn dispatch_binary_vec4(&self, a: &Tensor, b: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = a.dtype();
        let shape = a.shape().clone();
        let n = a.elem_count();

        let buf_a = self.get_or_upload(a)?;
        let buf_b = self.get_or_upload(b)?;
        let buf_out = self.alloc_output(n);

        let threads_needed = (n as u32).div_ceil(4);
        let result = self.dispatch_compute(
            entry,
            &[buf_a.buffer, buf_b.buffer, buf_out.buffer],
            &buf_out,
            n,
            &[n as u32, 0, 0, 0],
            (threads_needed.div_ceil(WG_ELEM), 1, 1),
        );

        let tensor = Self::from_f32_vec(result, shape.dims(), dtype)?;
        self.cache_activation(tensor.id(), buf_out);
        Ok(tensor)
    }

    fn dispatch_ternary_vec4(
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

        let threads_needed = (n as u32).div_ceil(4);
        let result = self.dispatch_compute(
            entry,
            &[buf_a.buffer, buf_b.buffer, buf_out.buffer, buf_c.buffer],
            &buf_out,
            n,
            &[n as u32, 0, 0, 0],
            (threads_needed.div_ceil(WG_ELEM), 1, 1),
        );

        let tensor = Self::from_f32_vec(result, shape.dims(), dtype)?;
        self.cache_activation(tensor.id(), buf_out);
        Ok(tensor)
    }

    fn dispatch_unary_vec4(&self, x: &Tensor, entry: &str) -> Result<Tensor> {
        let dtype = x.dtype();
        let shape = x.shape().clone();
        let n = x.elem_count();

        let buf_a = self.get_or_upload(x)?;
        let buf_out = self.alloc_output(n);

        let threads_needed = (n as u32).div_ceil(4);
        let result = self.dispatch_compute(
            entry,
            &[buf_a.buffer, self.dummy_buf.buffer, buf_out.buffer],
            &buf_out,
            n,
            &[n as u32, 0, 0, 0],
            (threads_needed.div_ceil(WG_ELEM), 1, 1),
        );

        let tensor = Self::from_f32_vec(result, shape.dims(), dtype)?;
        self.cache_activation(tensor.id(), buf_out);
        Ok(tensor)
    }

    // ── GPU softmax ──────────────────────────────────────────────────

    /// Dispatch scaled softmax on GPU: output[row,j] = softmax(input[row,j] * scale).
    /// If seq_len > 0, applies causal mask.
    fn dispatch_softmax(
        &self,
        input: &Tensor,
        rows: usize,
        cols: usize,
        scale: f32,
        seq_len: usize,
    ) -> Result<Tensor> {
        let dtype = input.dtype();
        let shape = input.shape().clone();
        let n = input.elem_count();

        let buf_in = self.get_or_upload(input)?;
        let buf_out = self.alloc_output(n);

        let result = self.dispatch_compute(
            "scaled_softmax",
            &[buf_in.buffer, self.dummy_buf.buffer, buf_out.buffer],
            &buf_out,
            n,
            &[rows as u32, cols as u32, scale.to_bits(), seq_len as u32],
            (rows as u32, 1, 1), // one workgroup per row
        );

        let tensor = Self::from_f32_vec(result, shape.dims(), dtype)?;
        self.cache_activation(tensor.id(), buf_out);
        Ok(tensor)
    }

    // ── GPU matmul ───────────────────────────────────────────────────

    /// Returns (result_data, output_buffer) so caller can cache the buffer.
    #[allow(dead_code)]
    fn gpu_gemv(
        &self,
        buf_x: &MappedBuffer,
        buf_w: &MappedBuffer,
        buf_w_f16: Option<&Arc<MappedBuffer>>,
        k: usize,
        n: usize,
    ) -> (Vec<f32>, MappedBuffer) {
        let buf_y = self.alloc_output(n);
        // Use F16 kernel if available — halves memory bandwidth
        let (entry, w_buf) = if let Some(f16_buf) = buf_w_f16 {
            ("gemv_f16", f16_buf.buffer)
        } else {
            ("gemv", buf_w.buffer)
        };
        let result = self.dispatch_compute(
            entry,
            &[buf_x.buffer, w_buf, buf_y.buffer],
            &buf_y,
            n,
            &[n as u32, k as u32, 0, 0],
            ((n as u32).div_ceil(256), 1, 1),
        );
        (result, buf_y)
    }

    fn gpu_gemm(
        &self,
        buf_a: &MappedBuffer,
        buf_b: &MappedBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f32>, MappedBuffer) {
        let buf_c = self.alloc_output(m * n);
        // Use small-M kernel (16×64 tile) for M<=16, large kernel (32×64) otherwise
        let (entry, wg_m, wg_n) = if m <= 8 {
            ("matmul_small", (m as u32).div_ceil(8), (n as u32).div_ceil(64))
        } else {
            ("matmul", (m as u32).div_ceil(32), (n as u32).div_ceil(64))
        };
        let result = self.dispatch_compute(
            entry,
            &[buf_a.buffer, buf_b.buffer, buf_c.buffer],
            &buf_c,
            m * n,
            &[m as u32, n as u32, k as u32, 0],
            (wg_m, wg_n, 1),
        );
        (result, buf_c)
    }

    fn gpu_matmul(
        &self,
        buf_a: &MappedBuffer,
        buf_b: &MappedBuffer,
        buf_b_f16: Option<&Arc<MappedBuffer>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f32>, MappedBuffer) {
        if m == 1 {
            self.gpu_gemv(buf_a, buf_b, buf_b_f16, k, n)
        } else {
            self.gpu_gemm(buf_a, buf_b, m, k, n)
        }
    }

    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.dims();
        let b_dims = b.dims();
        let a_rank = a_dims.len();
        let b_rank = b_dims.len();

        let m = a_dims[a_rank - 2];
        let k = a_dims[a_rank - 1];
        let n = b_dims[b_rank - 1];

        let a_batch: usize = a_dims[..a_rank - 2].iter().product();
        let b_batch: usize = b_dims[..b_rank - 2].iter().product();
        let batch = a_batch.max(b_batch);

        let mk = m * k;
        let kn = k * n;
        let mn = m * n;

        if batch == 1 {
            // get_or_upload handles dtype conversion + contiguity
            let buf_a = self.get_or_upload(a)?;
            let buf_b = self.get_or_upload(b)?;
            // Check for F16 weight buffer (halves GEMV bandwidth)
            let f16_buf = if m == 1 {
                let vk = Self::view_key(b);
                vk.and_then(|key| {
                    let cache = self.weight_cache.lock().unwrap();
                    cache.f16_views.get(&key).cloned()
                })
            } else {
                None
            };
            let (out, buf_out) = self.gpu_matmul(&buf_a, &buf_b, f16_buf.as_ref(), m, k, n);
            let mut out_shape: Vec<usize> = a_dims[..a_rank - 2].to_vec();
            out_shape.push(m);
            out_shape.push(n);
            let tensor = Tensor::from_vec(out, out_shape.as_slice(), &Device::Cpu)?;
            self.cache_activation(tensor.id(), buf_out);
            return Ok(tensor);
        }

        // Batch path: record ALL batch dispatches in one command buffer (1 fence wait).
        let a = a.to_dtype(DType::F32)?.contiguous()?;
        let b = b.to_dtype(DType::F32)?.contiguous()?;
        let a_data: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;

        // Upload all batch slices first
        let mut a_bufs = Vec::with_capacity(batch);
        let mut b_bufs = Vec::with_capacity(batch);
        let mut out_bufs = Vec::with_capacity(batch);
        for i in 0..batch {
            let a_off = if a_batch == 1 { 0 } else { i * mk };
            let b_off = if b_batch == 1 { 0 } else { i * kn };
            a_bufs.push(self.upload_uncached(&a_data[a_off..a_off + mk]));
            b_bufs.push(self.upload_uncached(&b_data[b_off..b_off + kn]));
            out_bufs.push(self.alloc_output(mn));
        }

        // Determine pipeline and workgroup dims
        let is_gemv = m == 1;
        let entry = if is_gemv { "gemv" } else if m <= 8 { "matmul_small" } else { "matmul" };
        let params: [u32; 4] = if is_gemv {
            [n as u32, k as u32, 0, 0]
        } else {
            [m as u32, n as u32, k as u32, 0]
        };
        let workgroups = if is_gemv {
            ((n as u32).div_ceil(256), 1, 1)
        } else if m <= 8 {
            ((m as u32).div_ceil(8), (n as u32).div_ceil(64), 1)
        } else {
            ((m as u32).div_ceil(32), (n as u32).div_ceil(64), 1)
        };

        // Record all dispatches in one command buffer
        let _lock = self.dispatch_lock.lock().unwrap();
        let (pipeline, pipe_layout, ds_layout, _default_ds, _) =
            self.pipelines.get(entry).unwrap_or_else(|| panic!("unknown pipeline: {entry}"));

        unsafe {
            // Allocate batch descriptor sets
            let ds_layouts: Vec<vk::DescriptorSetLayout> = vec![*ds_layout; batch];
            let ds_alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&ds_layouts);
            let descriptor_sets = self.vk_device
                .allocate_descriptor_sets(&ds_alloc)
                .expect("allocate batch descriptor sets");

            // Write params (shared across all dispatches)
            self.params_buf.write_u32(&params);

            // Update descriptor sets for all batch elements
            for i in 0..batch {
                let buf_infos = [
                    vk::DescriptorBufferInfo::default().buffer(a_bufs[i].buffer).offset(0).range(vk::WHOLE_SIZE),
                    vk::DescriptorBufferInfo::default().buffer(b_bufs[i].buffer).offset(0).range(vk::WHOLE_SIZE),
                    vk::DescriptorBufferInfo::default().buffer(out_bufs[i].buffer).offset(0).range(vk::WHOLE_SIZE),
                    vk::DescriptorBufferInfo::default().buffer(self.params_buf.buffer).offset(0).range(16),
                ];
                let writes = [
                    vk::WriteDescriptorSet::default().dst_set(descriptor_sets[i]).dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&buf_infos[0])),
                    vk::WriteDescriptorSet::default().dst_set(descriptor_sets[i]).dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&buf_infos[1])),
                    vk::WriteDescriptorSet::default().dst_set(descriptor_sets[i]).dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&buf_infos[2])),
                    vk::WriteDescriptorSet::default().dst_set(descriptor_sets[i]).dst_binding(3)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(std::slice::from_ref(&buf_infos[3])),
                ];
                self.vk_device.update_descriptor_sets(&writes, &[]);
            }

            // Reset fence + command buffer
            self.vk_device.wait_for_fences(&[self.fence], true, u64::MAX).expect("wait_for_fences");
            self.vk_device.reset_fences(&[self.fence]).expect("reset_fences");
            self.vk_device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("reset_command_buffer");

            // Record all dispatches
            let begin_ci = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.vk_device.begin_command_buffer(self.command_buffer, &begin_ci)
                .expect("begin_command_buffer");

            let compute_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            for (i, &ds) in descriptor_sets.iter().enumerate() {
                if i > 0 {
                    // Compute-to-compute barrier between dispatches
                    self.vk_device.cmd_pipeline_barrier(
                        self.command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[compute_barrier], &[], &[],
                    );
                }
                self.vk_device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, *pipeline);
                self.vk_device.cmd_bind_descriptor_sets(
                    self.command_buffer, vk::PipelineBindPoint::COMPUTE, *pipe_layout,
                    0, &[ds], &[],
                );
                self.vk_device.cmd_dispatch(self.command_buffer, workgroups.0, workgroups.1, workgroups.2);
            }

            // Final compute-to-host barrier
            let host_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.vk_device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host_barrier], &[], &[],
            );

            self.vk_device.end_command_buffer(self.command_buffer).expect("end_command_buffer");

            // Single submit + single fence wait for ALL batch dispatches
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));
            self.vk_device.queue_submit(self.queue, &[submit_info], self.fence)
                .expect("queue_submit");
            self.vk_device.wait_for_fences(&[self.fence], true, u64::MAX)
                .expect("wait_for_fences");

            // Free batch descriptor sets
            self.vk_device.free_descriptor_sets(self.descriptor_pool, &descriptor_sets)
                .expect("free batch descriptor sets");
        }

        // Read all outputs
        let mut out = Vec::with_capacity(batch * mn);
        for buf in &out_bufs {
            out.extend_from_slice(&buf.read_f32(mn));
        }
        for buf in out_bufs {
            self.release_output(buf);
        }

        let mut out_shape: Vec<usize> = a_dims[..a_rank - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Tensor::from_vec(out, out_shape.as_slice(), &Device::Cpu)
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.vk_device.device_wait_idle();
            // Take the allocator out and leak it — its Drop accesses the
            // Vulkan device which causes segfaults during process exit cleanup.
            let allocator = self.allocator.lock().unwrap().take();
            if let Some(a) = allocator {
                std::mem::forget(a);
            }
        }
        return;
        #[allow(unreachable_code)]
        unsafe {
            let _ = self.vk_device.device_wait_idle();
            // Pipelines (descriptor sets freed implicitly with pool)
            for (pipeline, layout, _dsl, _ds, _) in self.pipelines.values() {
                self.vk_device.destroy_pipeline(*pipeline, None);
                self.vk_device.destroy_pipeline_layout(*layout, None);
            }
            self.vk_device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            // Free all allocations via gpu-allocator before destroying device.
            // We must free allocations before destroying buffers.
            let mut alloc_opt = self.allocator.lock().unwrap();
            let mut alloc = alloc_opt.take().expect("allocator already dropped");

            // Params buffer
            let params_alloc = std::mem::take(&mut self.params_buf.allocation);
            if !params_alloc.is_null() {
                self.vk_device.destroy_buffer(self.params_buf.buffer, None);
                let _ = alloc.free(params_alloc);
            }

            // Free cached weight buffers
            let mut cache = self.weight_cache.lock().unwrap();
            for (_, buf) in cache.buffers.drain() {
                if let Ok(mut buf) = Arc::try_unwrap(buf) {
                    let a = std::mem::take(&mut buf.allocation);
                    self.vk_device.destroy_buffer(buf.buffer, None);
                    if !a.is_null() {
                        let _ = alloc.free(a);
                    }
                }
            }
            drop(cache);

            // Free pooled buffers
            let mut pool = self.buffer_pool.lock().unwrap();
            for (_, bufs) in pool.free.drain() {
                for mut buf in bufs {
                    let a = std::mem::take(&mut buf.allocation);
                    self.vk_device.destroy_buffer(buf.buffer, None);
                    if !a.is_null() {
                        let _ = alloc.free(a);
                    }
                }
            }
            drop(pool);
            drop(alloc);

            // Command pool + fence
            self.vk_device.destroy_fence(self.fence, None);
            self.vk_device
                .destroy_command_pool(self.command_pool, None);
            // Note: vk_device and _instance are NOT destroyed here.
            // The gpu-allocator holds clones of these handles internally.
            // ash types don't implement Drop, so no double-free risk.
        }
    }
}

impl ComputeBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
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
        let orig_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Q @ K^T
        let attn = self.tensor_matmul(&q, &k.t()?)?;

        // GPU scaled softmax (fuses scale + optional causal mask + softmax)
        let seq_len = q.dim(q.dims().len() - 2)?;
        let kv_len = k.dim(k.dims().len() - 2)?;
        let total_rows = attn.elem_count() / kv_len;
        let causal_seq_len = if causal { seq_len } else { 0 };
        let attn = self.dispatch_softmax(&attn, total_rows, kv_len, scale, causal_seq_len)?;

        // Attn @ V
        let out = self.tensor_matmul(&attn, &v)?;
        out.to_dtype(orig_dtype)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let n = gate.elem_count();
        if n > 32768 {
            log::debug!("silu_mul GPU: {n} elements");
            self.dispatch_binary_vec4(gate, up, "silu_mul")
        } else {
            log::debug!("silu_mul CPU: {n} elements (<=32768)");
            (candle_nn::ops::silu(&gate.contiguous()?)? * up.contiguous()?)?.contiguous()
        }
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.elem_count();
        if n > 32768 {
            log::debug!("stable_softplus GPU: {n} elements");
            self.dispatch_unary_vec4(x, "stable_softplus")
        } else {
            log::debug!("stable_softplus CPU: {n} elements (<=32768)");
            let t88 = Tensor::full(88.0f32, x.shape(), x.device())?.to_dtype(x.dtype())?;
            let clamped = x.minimum(&t88)?;
            let sp = (clamped.exp()? + 1.0)?.log()?;
            x.maximum(&sp)
        }
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        let n = a.elem_count();
        if n > 32768 {
            log::debug!("add3 GPU: {n} elements");
            self.dispatch_ternary_vec4(a, b, c, "add3")
        } else {
            log::debug!("add3 CPU: {n} elements (<=32768)");
            ((a + b)? + c)?.contiguous()
        }
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let n = x.elem_count();
        if n > 32768 {
            log::debug!("exp_mul GPU: {n} elements");
            self.dispatch_binary_vec4(x, y, "exp_mul")
        } else {
            log::debug!("exp_mul CPU: {n} elements (<=32768)");
            (x * y.exp()?)?.contiguous()
        }
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        let n = a.elem_count();
        if n > 32768 {
            log::debug!("sub_mul GPU: {n} elements");
            self.dispatch_ternary_vec4(a, b, c, "sub_mul")
        } else {
            log::debug!("sub_mul CPU: {n} elements (<=32768)");
            ((a - b)? * c)?.contiguous()
        }
    }

    fn preprocess_linear_weight(&self, weight: &Tensor) -> Result<Tensor> {
        // Pre-convert to F32 contiguous and pre-upload to GPU.
        let w = weight.to_dtype(DType::F32)?.contiguous()?;
        // Pre-upload F32 to GPU weight cache
        let _ = self.get_or_upload(&w)?;

        // Also pre-upload the TRANSPOSED F16 data for GEMV (halves bandwidth).
        // weight.t() is what linear_forward passes to matmul.
        if weight.dtype() == DType::F16 {
            let wt_f16 = weight.t()?.contiguous()?; // [in, out] F16 contiguous
            let f16_data: Vec<f32> = wt_f16.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            // Store as raw F16 bytes (u32 packed pairs)
            let n = wt_f16.elem_count();
            let f16_vals: Vec<half::f16> = f16_data.iter().map(|&v| half::f16::from_f32(v)).collect();
            let f16_bytes: &[u8] = bytemuck::cast_slice(&f16_vals);
            // Upload as u32 buffer
            let buf_bytes = (f16_bytes.len()) as u64;
            let mut alloc_guard = self.allocator.lock().unwrap();
            let alloc = alloc_guard.as_mut().unwrap();
            let buf = Self::alloc_mapped_buffer(
                &self.vk_device, alloc, buf_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER, self.uma_memory_type,
            ).map_err(candle_core::Error::Msg)?;
            unsafe {
                std::ptr::copy_nonoverlapping(f16_bytes.as_ptr(), buf.mapped_ptr, f16_bytes.len());
            }
            // Cache by the view key of the F32 weight's .t() (what get_or_upload will see)
            let wt_f32 = w.t()?; // F32 non-contiguous [in, out]
            if let Some(key) = Self::view_key(&wt_f32) {
                let mut cache = self.weight_cache.lock().unwrap();
                cache.f16_views.insert(key, Arc::new(buf));
                log::debug!("pre-uploaded F16 GEMV weight: {} elements", n);
            }
            drop(alloc_guard);
        }

        Ok(w)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.dims();
        let b_dims = b.dims();
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let n = b_dims[b_dims.len() - 1];
        // GPU for all sizes. M=1 uses coalesced GEMV. View cache handles weight.t().
        // Return F32 always — avoids F32→F16→F32 round-trips between ops.
        log::debug!("matmul GPU: M={m} K={k} N={n}");
        self.tensor_matmul(a, b)
    }

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
