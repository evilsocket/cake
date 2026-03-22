//! Abstraction for reading individual tensors from storage by name.
//!
//! [`TensorStorageProvider`] enables on-demand tensor reads without loading
//! the entire model into memory. The primary implementation [`SafetensorsStorage`]
//! reads directly from `.safetensors` files using `pread()` at byte offsets
//! parsed from each shard's header.
//!
//! This is the foundation for Flash-MoE style expert offloading: non-expert
//! weights (embeddings, attention, norms) are loaded normally via VarBuilder,
//! while expert weights are streamed from disk on demand during inference.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use candle_core::DType;

/// Raw tensor data read from storage.
#[derive(Debug)]
pub struct TensorData {
    /// Raw bytes in the tensor's native dtype layout.
    pub bytes: Vec<u8>,
    /// Data type of the tensor.
    pub dtype: DType,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
}

/// Abstraction for reading individual tensors from storage.
///
/// Implementations handle the details of locating and reading tensor data
/// from different storage formats. The key property: individual tensors can
/// be read independently without loading the entire file.
pub trait TensorStorageProvider: Send + Sync {
    /// Read a tensor's raw bytes and metadata by name.
    fn read_tensor(&self, name: &str) -> Result<TensorData>;
    /// Check if a tensor exists in this storage.
    fn has_tensor(&self, name: &str) -> bool;
    /// List all tensor names available in this storage.
    fn tensor_names(&self) -> Vec<String>;
}

/// Metadata for a single tensor within a safetensors shard file.
#[derive(Debug, Clone)]
struct TensorMeta {
    /// Path to the shard file containing this tensor.
    shard_path: PathBuf,
    /// Absolute byte offset in the shard file (8 + header_len + data_offset).
    abs_offset: u64,
    /// Size in bytes.
    byte_size: u64,
    /// Data type.
    dtype: DType,
    /// Shape.
    shape: Vec<usize>,
}

/// Reads individual tensors from safetensors files using `pread()`.
///
/// At construction, parses all shard file headers to build an in-memory index
/// mapping tensor names to `(file, offset, size, dtype, shape)`. During inference,
/// `read_tensor()` calls `pread()` at the computed offset — no mmap, no full file load.
///
/// On NVMe SSDs, sequential `pread()` achieves near-peak bandwidth. The OS page cache
/// provides transparent caching of frequently-accessed expert weights.
pub struct SafetensorsStorage {
    /// Tensor name → metadata (shard file, offset, size, dtype, shape).
    index: HashMap<String, TensorMeta>,
    /// Open file handles per shard (kept open for pread lifetime).
    files: HashMap<PathBuf, File>,
}

impl SafetensorsStorage {
    /// Build storage from a model directory containing `model.safetensors.index.json`
    /// or a single `model.safetensors` file.
    pub fn from_model_path(model_path: &Path) -> Result<Self> {
        let index_path = model_path.join("model.safetensors.index.json");
        let single_path = model_path.join("model.safetensors");

        if index_path.exists() {
            Self::from_index_json(&index_path)
        } else if single_path.exists() {
            Self::from_single_file(&single_path)
        } else {
            bail!(
                "no safetensors files found in {}",
                model_path.display()
            )
        }
    }

    /// Build from a multi-shard model with index JSON.
    fn from_index_json(index_path: &Path) -> Result<Self> {
        let parent = index_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("no parent dir"))?;

        // Parse the weight_map: tensor_name → shard_filename
        let json: serde_json::Value =
            serde_json::from_reader(File::open(index_path)?)?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("missing weight_map in index"))?;

        // Group tensors by shard file
        let mut shard_tensors: HashMap<String, Vec<String>> = HashMap::new();
        for (tensor_name, shard_val) in weight_map {
            let shard = shard_val
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("bad weight_map entry"))?;
            shard_tensors
                .entry(shard.to_string())
                .or_default()
                .push(tensor_name.clone());
        }

        let mut index = HashMap::new();
        let mut files = HashMap::new();

        for shard_name in shard_tensors.keys() {
            let shard_path = parent.join(shard_name);
            let (header_len, header) = Self::parse_shard_header(&shard_path)?;
            let f = File::open(&shard_path)?;
            files.insert(shard_path.clone(), f);

            let obj = header
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("shard header not object"))?;

            for (name, meta) in obj {
                if name == "__metadata__" {
                    continue;
                }
                if let Some(tm) = Self::parse_tensor_meta(name, meta, &shard_path, header_len)? {
                    index.insert(name.clone(), tm);
                }
            }
        }

        log::info!(
            "SafetensorsStorage: indexed {} tensors across {} shards",
            index.len(),
            files.len()
        );

        Ok(Self { index, files })
    }

    /// Build from a single safetensors file (non-sharded model).
    fn from_single_file(path: &Path) -> Result<Self> {
        let (header_len, header) = Self::parse_shard_header(path)?;
        let f = File::open(path)?;
        let path_buf = path.to_path_buf();

        let mut index = HashMap::new();
        let obj = header
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("header not object"))?;

        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            if let Some(tm) = Self::parse_tensor_meta(name, meta, &path_buf, header_len)? {
                index.insert(name.clone(), tm);
            }
        }

        let mut files = HashMap::new();
        files.insert(path_buf, f);

        log::info!("SafetensorsStorage: indexed {} tensors from single file", index.len());

        Ok(Self { index, files })
    }

    /// Parse a shard file's header. Returns (header_len, parsed JSON).
    fn parse_shard_header(path: &Path) -> Result<(usize, serde_json::Value)> {
        let mut f = File::open(path)?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        if header_len > 50 * 1024 * 1024 {
            bail!("safetensors header too large: {} bytes", header_len);
        }

        let mut header_buf = vec![0u8; header_len];
        f.read_exact(&mut header_buf)?;
        let header: serde_json::Value = serde_json::from_slice(&header_buf)?;

        Ok((header_len, header))
    }

    /// Parse tensor metadata from a shard header entry.
    fn parse_tensor_meta(
        name: &str,
        meta: &serde_json::Value,
        shard_path: &Path,
        header_len: usize,
    ) -> Result<Option<TensorMeta>> {
        let offsets = match meta.get("data_offsets").and_then(|v| v.as_array()) {
            Some(o) if o.len() == 2 => o,
            _ => return Ok(None),
        };

        let data_start = offsets[0].as_u64().unwrap_or(0);
        let data_end = offsets[1].as_u64().unwrap_or(0);
        let byte_size = data_end.saturating_sub(data_start);

        // Absolute offset in file: 8 (length prefix) + header_len + data_start
        let abs_offset = 8 + header_len as u64 + data_start;

        let dtype_str = meta
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("F32");
        let dtype = match dtype_str {
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            "F64" => DType::F64,
            "I64" => DType::I64,
            "U32" => DType::U32,
            "U8" => DType::U8,
            "F8_E4M3" | "F8E4M3" => DType::F8E4M3,
            other => {
                log::warn!("unknown dtype '{other}' for tensor '{name}', skipping");
                return Ok(None);
            }
        };

        let shape: Vec<usize> = meta
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(TensorMeta {
            shard_path: shard_path.to_path_buf(),
            abs_offset,
            byte_size,
            dtype,
            shape,
        }))
    }
}

impl TensorStorageProvider for SafetensorsStorage {
    fn read_tensor(&self, name: &str) -> Result<TensorData> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in storage", name))?;

        let file = self
            .files
            .get(&meta.shard_path)
            .ok_or_else(|| anyhow::anyhow!("shard file not open: {:?}", meta.shard_path))?;

        let mut buf = vec![0u8; meta.byte_size as usize];

        // pread: thread-safe, no seek, optimal for concurrent reads
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_at(&mut buf, meta.abs_offset)?;
        }

        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(meta.abs_offset))?;
            f.read_exact(&mut buf)?;
        }

        Ok(TensorData {
            bytes: buf,
            dtype: meta.dtype,
            shape: meta.shape.clone(),
        })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_offset_calculation() {
        // header_len=100, data_offsets=[0, 512]
        // abs_offset = 8 + 100 + 0 = 108
        let data_start: u64 = 0;
        let abs = 8u64 + 100 + data_start;
        assert_eq!(abs, 108);

        // second tensor at data_offsets=[512, 1024]
        let abs2 = 8u64 + 100 + 512u64;
        assert_eq!(abs2, 620);
    }

    #[test]
    fn test_parse_tensor_meta_valid() {
        let meta_json: serde_json::Value = serde_json::json!({
            "dtype": "F16",
            "shape": [1024, 512],
            "data_offsets": [0, 1048576]
        });
        let result =
            SafetensorsStorage::parse_tensor_meta("test", &meta_json, Path::new("/tmp/x.safetensors"), 200)
                .unwrap();
        let tm = result.unwrap();
        assert_eq!(tm.abs_offset, 8 + 200);
        assert_eq!(tm.byte_size, 1048576);
        assert_eq!(tm.dtype, DType::F16);
        assert_eq!(tm.shape, vec![1024, 512]);
    }

    #[test]
    fn test_parse_tensor_meta_skips_metadata() {
        let meta_json: serde_json::Value = serde_json::json!({
            "format": "pt"
        });
        let result =
            SafetensorsStorage::parse_tensor_meta("__metadata__", &meta_json, Path::new("/tmp/x"), 100)
                .unwrap();
        // No data_offsets → None
        assert!(result.is_none());
    }

    #[test]
    fn test_has_tensor() {
        let storage = SafetensorsStorage {
            index: {
                let mut m = HashMap::new();
                m.insert(
                    "weight".to_string(),
                    TensorMeta {
                        shard_path: PathBuf::from("/tmp/test"),
                        abs_offset: 108,
                        byte_size: 1024,
                        dtype: DType::F32,
                        shape: vec![16, 16],
                    },
                );
                m
            },
            files: HashMap::new(),
        };
        assert!(storage.has_tensor("weight"));
        assert!(!storage.has_tensor("missing"));
    }
}
