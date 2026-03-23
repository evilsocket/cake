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
    pub fn new(
        storage: Arc<dyn TensorStorageProvider>,
        layer_prefix: String,
        num_experts: usize,
        device: Device,
        dtype: DType,
    ) -> Self {
        let expert_names = (0..num_experts)
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
        Self {
            storage,
            layer_prefix,
            expert_names,
            num_experts,
            device,
            dtype,
            needs_device_transfer,
        }
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

        let names = &self.expert_names[idx];

        // For F32→F32 (zero-copy path), individual reads are faster than
        // batch+split since each read's buffer transfers directly into the
        // Tensor without copying. The batch path would add 3 memcpys to split
        // the contiguous buffer.
        let (gate_data, up_data, down_data) = if self.dtype == DType::F32 {
            let g = self.storage.read_tensor(&names.gate_proj)
                .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
            let u = self.storage.read_tensor(&names.up_proj)
                .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
            let d = self.storage.read_tensor(&names.down_proj)
                .map_err(|e| candle_core::Error::Msg(format!("read_tensor: {e}")))?;
            (g, u, d)
        } else {
            // For dtype conversion path, batch read saves pread syscalls
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
            (g, u, d)
        };

        Ok(ExpertWeights {
            gate_proj: self.materialize(gate_data)?,
            up_proj: self.materialize(up_data)?,
            down_proj: self.materialize(down_data)?,
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
        );

        assert!(provider.get_expert(4).is_err());
        assert!(provider.get_expert(100).is_err());
    }
}
