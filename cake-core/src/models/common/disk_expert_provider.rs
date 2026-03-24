//! Disk-backed expert provider — streams expert weights from safetensors via pread.
//!
//! Instead of loading all expert weights into RAM at startup, this provider
//! reads individual expert weight matrices on demand using [`TensorStorageProvider`].
//! The OS page cache handles caching — no application-level LRU needed (Flash-MoE
//! showed this is 38% faster than custom caching).
//!
//! Memory usage: O(num_experts_per_tok × expert_size) for the buffer pool,
//! plus whatever the OS decides to keep in the page cache.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

use crate::utils::tensor_storage::TensorStorageProvider;

use super::expert_provider::{ExpertProvider, ExpertWeights};

/// Pre-computed tensor names for a single expert (avoids format! on hot path).
struct ExpertNames {
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

/// Streams expert weights from disk via `TensorStorageProvider`.
///
/// Expert tensors are read from safetensors files by name, e.g.:
/// `"{layer_prefix}.experts.{idx}.gate_proj.weight"`.
///
/// Supports GPTQ-quantized experts: when `gptq_group_size` is set, reads
/// `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros` and dequantizes
/// on the fly using `dequantize_gptq_4bit`.
///
/// Also supports stacked switch_mlp format: `{prefix}.switch_mlp.{proj}.weight`
/// with shape `[num_experts, ...]`. Reads the full stacked tensor, dequantizes
/// once per projection, and slices per expert.
pub struct DiskExpertProvider {
    storage: Arc<dyn TensorStorageProvider>,
    layer_prefix: String,
    /// Pre-computed tensor name strings for each expert index.
    expert_names: Vec<ExpertNames>,
    num_experts: usize,
    device: Device,
    dtype: DType,
    /// Pre-computed: whether device transfer is needed.
    needs_device_transfer: bool,
    /// Pre-computed: whether F32 zero-copy path can be used (storage and target both F32).
    use_f32_zerocopy: bool,
    /// GPTQ group size — when Some, experts are GPTQ-quantized and need dequantization.
    gptq_group_size: Option<usize>,
    /// Stacked switch_mlp tensor names (for models that use stacked 3D expert format).
    stacked_names: Option<StackedNames>,
}

/// Pre-computed names for stacked switch_mlp format.
struct StackedNames {
    gate_proj: String,
    up_proj: String,
    down_proj: String,
}

impl std::fmt::Debug for DiskExpertProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskExpertProvider")
            .field("prefix", &self.layer_prefix)
            .field("num_experts", &self.num_experts)
            .finish()
    }
}

impl DiskExpertProvider {
    /// Create a disk-backed expert provider.
    ///
    /// - `storage`: safetensors storage with pread access
    /// - `layer_prefix`: e.g., "model.layers.5.mlp" — experts are at
    ///   `"{prefix}.experts.{idx}.{gate_proj,up_proj,down_proj}.weight"`
    /// - `num_experts`: total number of experts in this layer
    /// - `device`: target device (CPU for inference, or GPU for transfer)
    /// - `dtype`: target dtype for the loaded tensors
    /// - `gptq_group_size`: if Some, experts are GPTQ-quantized and will be dequantized on read
    pub fn new(
        storage: Arc<dyn TensorStorageProvider>,
        layer_prefix: String,
        num_experts: usize,
        device: Device,
        dtype: DType,
        gptq_group_size: Option<usize>,
    ) -> Self {
        let expert_names: Vec<ExpertNames> = (0..num_experts)
            .map(|idx| {
                let prefix = format!("{}.experts.{}", layer_prefix, idx);
                ExpertNames {
                    gate_proj: format!("{prefix}.gate_proj.weight"),
                    up_proj: format!("{prefix}.up_proj.weight"),
                    down_proj: format!("{prefix}.down_proj.weight"),
                }
            })
            .collect();
        let needs_device_transfer = !device.is_cpu();
        // Auto-detect GPTQ: if caller says GPTQ, verify first expert has qweight.
        let gptq_group_size = if let Some(gs) = gptq_group_size {
            let qw_name = format!("{}.experts.0.gate_proj.qweight", layer_prefix);
            if storage.has_tensor(&qw_name) {
                log::info!("expert offload: GPTQ 4-bit detected (group_size={gs})");
                Some(gs)
            } else {
                log::info!("expert offload: GPTQ requested but no qweight found, using plain weights");
                None
            }
        } else {
            None
        };
        // Detect storage dtype from first expert (skip for GPTQ — qweight is int32, not weights)
        let storage_dtype = if gptq_group_size.is_none() {
            expert_names
                .first()
                .and_then(|names| storage.read_tensor(&names.gate_proj).ok().map(|d| d.dtype))
        } else {
            None
        };
        // For GPTQ experts, disable F32 zerocopy (qweight is int32, not F32)
        let use_f32_zerocopy = gptq_group_size.is_none()
            && dtype == DType::F32
            && storage_dtype.is_some_and(|sd| sd == DType::F32);
        Self {
            storage,
            layer_prefix,
            expert_names,
            num_experts,
            device,
            dtype,
            needs_device_transfer,
            use_f32_zerocopy,
            gptq_group_size,
            stacked_names: None,
        }
    }

    /// Create a disk-backed expert provider for stacked switch_mlp format.
    ///
    /// In this format, all experts are stored in a single 3D tensor per projection:
    /// `{prefix}.switch_mlp.gate_proj.weight` shape `[num_experts, intermediate, packed_hidden]`
    ///
    /// Each expert is read by loading the full stacked tensor, dequantizing, and slicing.
    #[allow(clippy::too_many_arguments)]
    pub fn new_stacked(
        storage: Arc<dyn TensorStorageProvider>,
        layer_prefix: String,
        num_experts: usize,
        intermediate: usize,
        hidden: usize,
        device: Device,
        dtype: DType,
        gptq_group_size: Option<usize>,
    ) -> Self {
        let needs_device_transfer = !device.is_cpu();
        let _ = (intermediate, hidden); // used by callers for shape info
        let stacked_names = StackedNames {
            gate_proj: format!("{layer_prefix}.switch_mlp.gate_proj.weight"),
            up_proj: format!("{layer_prefix}.switch_mlp.up_proj.weight"),
            down_proj: format!("{layer_prefix}.switch_mlp.down_proj.weight"),
        };
        log::info!("expert offload: stacked switch_mlp format (GPTQ={:?})", gptq_group_size);
        Self {
            storage,
            layer_prefix,
            expert_names: Vec::new(), // not used for stacked format
            num_experts,
            device,
            dtype,
            needs_device_transfer,
            use_f32_zerocopy: false,
            gptq_group_size,
            stacked_names: Some(stacked_names),
        }
    }

    /// Read a single expert's 2D slice from a stacked 3D tensor using targeted pread.
    /// Only reads the bytes for one expert, not the entire [256, ...] tensor.
    fn read_stacked_expert_slice(
        &self,
        tensor_name: &str,
        expert_idx: usize,
    ) -> Result<Tensor> {
        if let Some(group_size) = self.gptq_group_size {
            // Affine quantized: read slices of weight + scales + biases
            let prefix = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
            let scales_name = format!("{prefix}.scales");
            if self.storage.has_tensor(&scales_name) {
                let biases_name = format!("{prefix}.biases");

                // Get full tensor metadata for shape info (no data read)
                let (w_dtype, w_shape) = self.storage.tensor_meta(tensor_name)
                    .map_err(|e| candle_core::Error::Msg(format!("meta: {e}")))?;
                let (s_dtype, s_shape) = self.storage.tensor_meta(&scales_name)
                    .map_err(|e| candle_core::Error::Msg(format!("meta: {e}")))?;
                let (b_dtype, b_shape) = self.storage.tensor_meta(&biases_name)
                    .map_err(|e| candle_core::Error::Msg(format!("meta: {e}")))?;

                // Compute per-expert byte ranges from 3D shapes
                let w_elem_size = w_dtype.size_in_bytes();
                let w_slice_elems: usize = w_shape[1..].iter().product();
                let w_slice_bytes = w_slice_elems * w_elem_size;
                let w_offset = expert_idx * w_slice_bytes;

                let s_elem_size = s_dtype.size_in_bytes();
                let s_slice_elems: usize = s_shape[1..].iter().product();
                let s_slice_bytes = s_slice_elems * s_elem_size;
                let s_offset = expert_idx * s_slice_bytes;

                let b_elem_size = b_dtype.size_in_bytes();
                let b_slice_elems: usize = b_shape[1..].iter().product();
                let b_slice_bytes = b_slice_elems * b_elem_size;
                let b_offset = expert_idx * b_slice_bytes;

                // Targeted pread: only read this expert's bytes
                let w_bytes = self.storage.read_tensor_slice(tensor_name, w_offset, w_slice_bytes)
                    .map_err(|e| candle_core::Error::Msg(format!("slice weight: {e}")))?;
                let s_bytes = self.storage.read_tensor_slice(&scales_name, s_offset, s_slice_bytes)
                    .map_err(|e| candle_core::Error::Msg(format!("slice scales: {e}")))?;
                let b_bytes = self.storage.read_tensor_slice(&biases_name, b_offset, b_slice_bytes)
                    .map_err(|e| candle_core::Error::Msg(format!("slice biases: {e}")))?;

                // Build 2D tensors from sliced bytes
                let w_2d_shape: Vec<usize> = w_shape[1..].to_vec();
                let s_2d_shape: Vec<usize> = s_shape[1..].to_vec();
                let b_2d_shape: Vec<usize> = b_shape[1..].to_vec();

                let packed = Tensor::from_raw_buffer(&w_bytes, w_dtype, &w_2d_shape, &Device::Cpu)?;
                let scales = Tensor::from_raw_buffer(&s_bytes, s_dtype, &s_2d_shape, &Device::Cpu)?;
                let biases = Tensor::from_raw_buffer(&b_bytes, b_dtype, &b_2d_shape, &Device::Cpu)?;

                let weight = crate::utils::gptq::dequantize_packed_4bit(&packed, &scales, &biases, group_size)?;
                let weight = weight.to_dtype(self.dtype)?;
                return if self.needs_device_transfer {
                    weight.to_device(&self.device)
                } else {
                    Ok(weight)
                };
            }
        }

        // Non-quantized: targeted pread of one expert's 2D slice
        let (dtype, shape) = self.storage.tensor_meta(tensor_name)
            .map_err(|e| candle_core::Error::Msg(format!("meta: {e}")))?;
        let elem_size = dtype.size_in_bytes();
        let slice_elems: usize = shape[1..].iter().product();
        let slice_bytes = slice_elems * elem_size;
        let offset = expert_idx * slice_bytes;

        let bytes = self.storage.read_tensor_slice(tensor_name, offset, slice_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("slice: {e}")))?;
        let slice_shape: Vec<usize> = shape[1..].to_vec();
        let tensor = Tensor::from_raw_buffer(&bytes, dtype, &slice_shape, &Device::Cpu)?;
        let tensor = if tensor.dtype() != self.dtype { tensor.to_dtype(self.dtype)? } else { tensor };
        if self.needs_device_transfer { tensor.to_device(&self.device) } else { Ok(tensor) }
    }

    /// Reinterpret a `Vec<u8>` as `Vec<f32>` without copying.
    ///
    /// # Safety
    /// The buffer must:
    /// - Have length divisible by 4
    /// - Be properly aligned for f32 (alignment >= 4, guaranteed by std allocator)
    /// - Contain valid f32 bytes (guaranteed by pread from safetensors F32 data)
    #[inline]
    fn bytes_to_f32_vec(bytes: Vec<u8>) -> Vec<f32> {
        let byte_len = bytes.len();
        let f32_len = byte_len / 4;
        let ptr = bytes.as_ptr();
        let cap = bytes.capacity();
        std::mem::forget(bytes);
        // SAFETY: Vec<u8> from global allocator is aligned >= 8 on 64-bit,
        // satisfying f32's alignment of 4. Length and capacity are adjusted
        // from u8 to f32 units. Data is valid F32 from safetensors.
        unsafe { Vec::from_raw_parts(ptr as *mut f32, f32_len, cap / 4) }
    }

    /// Read an expert weight, handling GPTQ dequantization if needed.
    fn read_expert_weight(&self, weight_name: &str) -> Result<Tensor> {
        if let Some(group_size) = self.gptq_group_size {
            let prefix = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
            let qw_name = format!("{prefix}.qweight");
            if self.storage.has_tensor(&qw_name) {
                let sc_name = format!("{prefix}.scales");
                let qz_name = format!("{prefix}.qzeros");
                // Read the GPTQ triplet
                let qw_data = self.storage.read_tensor(&qw_name)
                    .map_err(|e| candle_core::Error::Msg(format!("read qweight: {e}")))?;
                let sc_data = self.storage.read_tensor(&sc_name)
                    .map_err(|e| candle_core::Error::Msg(format!("read scales: {e}")))?;
                let qz_data = self.storage.read_tensor(&qz_name)
                    .map_err(|e| candle_core::Error::Msg(format!("read qzeros: {e}")))?;
                // Materialize to CPU tensors
                let qw = Tensor::from_raw_buffer(&qw_data.bytes, qw_data.dtype, &qw_data.shape, &Device::Cpu)?;
                let sc = Tensor::from_raw_buffer(&sc_data.bytes, sc_data.dtype, &sc_data.shape, &Device::Cpu)?;
                let qz = Tensor::from_raw_buffer(&qz_data.bytes, qz_data.dtype, &qz_data.shape, &Device::Cpu)?;
                // Dequantize
                let weight = crate::utils::gptq::dequantize_gptq_4bit(&qw, &sc, &qz, group_size)?;
                let weight = weight.to_dtype(self.dtype)?;
                return if self.needs_device_transfer {
                    weight.to_device(&self.device)
                } else {
                    Ok(weight)
                };
            }
        }
        // Non-GPTQ: read plain weight tensor
        let data = self.storage.read_tensor(weight_name)
            .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
        self.materialize(data)
    }

    /// Convert raw TensorData to a candle Tensor with target dtype/device.
    #[inline]
    fn materialize(&self, data: crate::utils::tensor_storage::TensorData) -> Result<Tensor> {
        let target_device = if self.needs_device_transfer {
            &self.device
        } else {
            &Device::Cpu
        };

        // Fast path: F32 storage → F32 target, zero-copy via Vec ownership transfer
        if data.dtype == DType::F32 && self.dtype == DType::F32 {
            let f32_data = Self::bytes_to_f32_vec(data.bytes);
            return Tensor::from_vec(f32_data, &*data.shape, target_device);
        }

        let tensor = if data.dtype == self.dtype {
            Tensor::from_raw_buffer(&data.bytes, data.dtype, &data.shape, target_device)?
        } else {
            let tensor = Tensor::from_raw_buffer(&data.bytes, data.dtype, &data.shape, &Device::Cpu)?;
            let tensor = tensor.to_dtype(self.dtype)?;
            if self.needs_device_transfer {
                tensor.to_device(&self.device)?
            } else {
                tensor
            }
        };

        Ok(tensor)
    }
}

impl ExpertProvider for DiskExpertProvider {
    fn get_expert(&self, idx: usize) -> Result<ExpertWeights> {
        if idx >= self.num_experts {
            return Err(candle_core::Error::Msg(format!(
                "expert index {idx} out of range (num_experts={})",
                self.num_experts
            )));
        }

        // Stacked switch_mlp path: read 3D tensor, slice per expert
        if let Some(stacked) = &self.stacked_names {
            return Ok(ExpertWeights {
                gate_proj: self.read_stacked_expert_slice(&stacked.gate_proj, idx)?,
                up_proj: self.read_stacked_expert_slice(&stacked.up_proj, idx)?,
                down_proj: self.read_stacked_expert_slice(&stacked.down_proj, idx)?,
            });
        }

        let names = &self.expert_names[idx];

        // GPTQ path: read and dequantize each weight individually
        if self.gptq_group_size.is_some() {
            return Ok(ExpertWeights {
                gate_proj: self.read_expert_weight(&names.gate_proj)?,
                up_proj: self.read_expert_weight(&names.up_proj)?,
                down_proj: self.read_expert_weight(&names.down_proj)?,
            });
        }

        // Non-GPTQ: optimized read paths
        if self.use_f32_zerocopy {
            // F32→F32: try zero-copy mmap path first (avoids allocation + memcpy)
            let target_device = if self.needs_device_transfer { &self.device } else { &Device::Cpu };
            if let (Some((gb, _, gs)), Some((ub, _, us)), Some((db, _, ds))) = (
                self.storage.tensor_bytes(&names.gate_proj),
                self.storage.tensor_bytes(&names.up_proj),
                self.storage.tensor_bytes(&names.down_proj),
            ) {
                // Reinterpret &[u8] as &[f32] directly from mmap — single memcpy into Tensor
                let gate_f32 = unsafe { std::slice::from_raw_parts(gb.as_ptr() as *const f32, gb.len() / 4) };
                let up_f32 = unsafe { std::slice::from_raw_parts(ub.as_ptr() as *const f32, ub.len() / 4) };
                let down_f32 = unsafe { std::slice::from_raw_parts(db.as_ptr() as *const f32, db.len() / 4) };
                return Ok(ExpertWeights {
                    gate_proj: Tensor::from_slice(gate_f32, &*gs, target_device)?,
                    up_proj: Tensor::from_slice(up_f32, &*us, target_device)?,
                    down_proj: Tensor::from_slice(down_f32, &*ds, target_device)?,
                });
            }
            // Fallback: read via TensorData (non-mmap storage)
            let g = self.storage.read_tensor(&names.gate_proj)
                .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
            let u = self.storage.read_tensor(&names.up_proj)
                .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
            let d = self.storage.read_tensor(&names.down_proj)
                .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
            return Ok(ExpertWeights {
                gate_proj: self.materialize(g)?,
                up_proj: self.materialize(u)?,
                down_proj: self.materialize(d)?,
            });
        }

        // Dtype conversion path: try zero-copy mmap first, fallback to read_tensors
        let target_device = if self.needs_device_transfer { &self.device } else { &Device::Cpu };
        if let (Some((gb, gdt, gs)), Some((ub, udt, us)), Some((db, ddt, ds))) = (
            self.storage.tensor_bytes(&names.gate_proj),
            self.storage.tensor_bytes(&names.up_proj),
            self.storage.tensor_bytes(&names.down_proj),
        ) {
            // Build tensors directly from mmap'd bytes — avoids allocation + copy
            let gate = Tensor::from_raw_buffer(gb, gdt, &gs, &Device::Cpu)?;
            let up = Tensor::from_raw_buffer(ub, udt, &us, &Device::Cpu)?;
            let down = Tensor::from_raw_buffer(db, ddt, &ds, &Device::Cpu)?;
            let (gate, up, down) = if gdt != self.dtype {
                (gate.to_dtype(self.dtype)?, up.to_dtype(self.dtype)?, down.to_dtype(self.dtype)?)
            } else {
                (gate, up, down)
            };
            return if self.needs_device_transfer {
                Ok(ExpertWeights {
                    gate_proj: gate.to_device(target_device)?,
                    up_proj: up.to_device(target_device)?,
                    down_proj: down.to_device(target_device)?,
                })
            } else {
                Ok(ExpertWeights { gate_proj: gate, up_proj: up, down_proj: down })
            };
        }

        // Fallback: batch read via TensorData
        let tensor_names = [
            names.gate_proj.as_str(),
            names.up_proj.as_str(),
            names.down_proj.as_str(),
        ];
        let mut data = self.storage.read_tensors(&tensor_names)
            .map_err(|e| candle_core::Error::Msg(format!("read_tensors: {e}")))?;
        let d = data.pop().unwrap();
        let u = data.pop().unwrap();
        let g = data.pop().unwrap();

        Ok(ExpertWeights {
            gate_proj: self.materialize(g)?,
            up_proj: self.materialize(u)?,
            down_proj: self.materialize(d)?,
        })
    }

    fn num_experts(&self) -> usize {
        self.num_experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tensor_storage::TensorData;

    /// Mock storage for testing.
    #[derive(Debug)]
    struct MockStorage {
        tensors: std::collections::HashMap<String, TensorData>,
    }

    impl TensorStorageProvider for MockStorage {
        fn read_tensor(&self, name: &str) -> anyhow::Result<TensorData> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("not found: {name}"))
        }

        fn has_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn tensor_names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }
    }

    impl Clone for TensorData {
        fn clone(&self) -> Self {
            Self {
                bytes: self.bytes.clone(),
                dtype: self.dtype,
                shape: self.shape.clone(),
            }
        }
    }

    fn make_mock_storage(prefix: &str, num_experts: usize, i: usize, h: usize) -> MockStorage {
        let mut tensors = std::collections::HashMap::new();
        for e in 0..num_experts {
            // gate_proj: (i, h) in F32
            let gate_bytes = vec![0u8; i * h * 4];
            tensors.insert(
                format!("{prefix}.experts.{e}.gate_proj.weight"),
                TensorData {
                    bytes: gate_bytes,
                    dtype: DType::F32,
                    shape: vec![i, h],
                },
            );
            // up_proj: (i, h)
            let up_bytes = vec![0u8; i * h * 4];
            tensors.insert(
                format!("{prefix}.experts.{e}.up_proj.weight"),
                TensorData {
                    bytes: up_bytes,
                    dtype: DType::F32,
                    shape: vec![i, h],
                },
            );
            // down_proj: (h, i)
            let down_bytes = vec![0u8; h * i * 4];
            tensors.insert(
                format!("{prefix}.experts.{e}.down_proj.weight"),
                TensorData {
                    bytes: down_bytes,
                    dtype: DType::F32,
                    shape: vec![h, i],
                },
            );
        }
        MockStorage { tensors }
    }

    #[test]
    fn test_disk_provider_get_expert_shapes() {
        let storage = Arc::new(make_mock_storage("model.layers.0.mlp", 4, 64, 32));
        let provider = DiskExpertProvider::new(
            storage,
            "model.layers.0.mlp".to_string(),
            4,
            Device::Cpu,
            DType::F32,
            None,
        );

        assert_eq!(provider.num_experts(), 4);

        let ew = provider.get_expert(0).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[64, 32]);
        assert_eq!(ew.up_proj.dims(), &[64, 32]);
        assert_eq!(ew.down_proj.dims(), &[32, 64]);
    }

    #[test]
    fn test_disk_provider_all_experts() {
        let storage = Arc::new(make_mock_storage("model.layers.0.mlp", 8, 64, 32));
        let provider = DiskExpertProvider::new(
            storage,
            "model.layers.0.mlp".to_string(),
            8,
            Device::Cpu,
            DType::F32,
            None,
        );

        for i in 0..8 {
            let ew = provider.get_expert(i).unwrap();
            assert_eq!(ew.gate_proj.dims(), &[64, 32]);
        }
    }

    #[test]
    fn test_disk_provider_out_of_range() {
        let storage = Arc::new(make_mock_storage("model.layers.0.mlp", 4, 64, 32));
        let provider = DiskExpertProvider::new(
            storage,
            "model.layers.0.mlp".to_string(),
            4,
            Device::Cpu,
            DType::F32,
            None,
        );

        assert!(provider.get_expert(4).is_err());
        assert!(provider.get_expert(100).is_err());
    }
}
