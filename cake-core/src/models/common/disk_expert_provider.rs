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

/// Streams expert weights from disk via `TensorStorageProvider`.
///
/// Expert tensors are read from safetensors files by name, e.g.:
/// `"{layer_prefix}.experts.{idx}.gate_proj.weight"`.
pub struct DiskExpertProvider {
    storage: Arc<dyn TensorStorageProvider>,
    layer_prefix: String,
    num_experts: usize,
    device: Device,
    dtype: DType,
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
        Self {
            storage,
            layer_prefix,
            num_experts,
            device,
            dtype,
        }
    }

    /// Read a single tensor from storage, wrap as candle Tensor.
    fn read_weight(&self, name: &str) -> Result<Tensor> {
        let data = self
            .storage
            .read_tensor(name)
            .map_err(|e| candle_core::Error::Msg(format!("read_tensor({name}): {e}")))?;

        // Create tensor from raw bytes
        let tensor = Tensor::from_raw_buffer(
            &data.bytes,
            data.dtype,
            &data.shape,
            &Device::Cpu,
        )?;

        // Convert to target dtype and device if needed
        let tensor = if tensor.dtype() != self.dtype {
            tensor.to_dtype(self.dtype)?
        } else {
            tensor
        };

        if !self.device.is_cpu() {
            tensor.to_device(&self.device)
        } else {
            Ok(tensor)
        }
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

        let prefix = format!("{}.experts.{}", self.layer_prefix, idx);

        let gate_proj = self.read_weight(&format!("{prefix}.gate_proj.weight"))?;
        let up_proj = self.read_weight(&format!("{prefix}.up_proj.weight"))?;
        let down_proj = self.read_weight(&format!("{prefix}.down_proj.weight"))?;

        Ok(ExpertWeights {
            gate_proj,
            up_proj,
            down_proj,
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
