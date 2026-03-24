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
use std::path::Path;

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

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
    /// Read multiple tensors at once. Default calls read_tensor in a loop.
    /// Implementations may optimize by batching I/O for contiguous tensors.
    fn read_tensors(&self, names: &[&str]) -> Result<Vec<TensorData>> {
        names.iter().map(|n| self.read_tensor(n)).collect()
    }
    /// Check if a tensor exists in this storage.
    fn has_tensor(&self, name: &str) -> bool;
    /// List all tensor names available in this storage.
    fn tensor_names(&self) -> Vec<String>;
}

/// Max tensor dimensions supported (covers all practical shapes).
const MAX_DIMS: usize = 4;

/// Inline shape storage — avoids Vec heap allocation for shapes up to 4 dims.
#[derive(Debug, Clone, Copy)]
struct InlineShape {
    dims: [usize; MAX_DIMS],
    ndim: u8,
}

impl InlineShape {
    fn from_vec(v: &[usize]) -> Self {
        let mut dims = [0usize; MAX_DIMS];
        let ndim = v.len().min(MAX_DIMS);
        dims[..ndim].copy_from_slice(&v[..ndim]);
        Self {
            dims,
            ndim: ndim as u8,
        }
    }

    fn as_slice(&self) -> &[usize] {
        &self.dims[..self.ndim as usize]
    }

    fn to_vec(self) -> Vec<usize> {
        self.as_slice().to_vec()
    }
}

/// Metadata for a single tensor within a safetensors shard file.
#[derive(Debug, Clone)]
struct TensorMeta {
    /// Index into `SafetensorsStorage::files` Vec for the shard containing this tensor.
    shard_idx: u16,
    /// Absolute byte offset in the shard file (8 + header_len + data_offset).
    abs_offset: u64,
    /// Size in bytes.
    byte_size: u64,
    /// Data type.
    dtype: DType,
    /// Shape (inline, no heap allocation for shapes <= 4 dims).
    shape: InlineShape,
}

/// Memory-mapped shard file — provides zero-syscall access to tensor bytes.
struct MappedShard {
    /// Memory-mapped region of the shard file.
    #[cfg(unix)]
    mmap_ptr: *const u8,
    #[cfg(unix)]
    mmap_len: usize,
    /// Fallback file handle for non-Unix systems.
    #[cfg(not(unix))]
    file: File,
}

// SAFETY: mmap'd memory is read-only and the mapping lives for the lifetime of the storage.
unsafe impl Send for MappedShard {}
unsafe impl Sync for MappedShard {}

impl MappedShard {
    #[cfg(unix)]
    fn new(file: &File) -> anyhow::Result<Self> {
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self { mmap_ptr: std::ptr::null(), mmap_len: 0 });
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            anyhow::bail!("mmap failed: {}", std::io::Error::last_os_error());
        }
        Ok(Self { mmap_ptr: ptr as *const u8, mmap_len: len })
    }

    #[cfg(not(unix))]
    fn new(file: File) -> anyhow::Result<Self> {
        Ok(Self { file })
    }

    /// Read bytes from the mapped file at the given offset and size.
    #[cfg(unix)]
    #[inline]
    fn read_bytes(&self, offset: u64, size: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(size);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mmap_ptr.add(offset as usize),
                buf.as_mut_ptr(),
                size,
            );
            buf.set_len(size);
        }
        buf
    }
}

#[cfg(unix)]
impl Drop for MappedShard {
    fn drop(&mut self) {
        if !self.mmap_ptr.is_null() && self.mmap_len > 0 {
            unsafe { libc::munmap(self.mmap_ptr as *mut _, self.mmap_len); }
        }
    }
}

pub struct SafetensorsStorage {
    /// Tensor name → metadata (shard index, offset, size, dtype, shape).
    index: HashMap<String, TensorMeta>,
    /// Memory-mapped shard files (indexed by shard_idx in TensorMeta).
    shards: Vec<MappedShard>,
    /// File handles for non-mmap fallback.
    #[cfg(not(unix))]
    files: Vec<File>,
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

    /// Build from a single safetensors file path (not a directory).
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_single_file(path)
    }

    /// Load a tensor by name and return it as a candle Tensor on the given device.
    pub fn load_tensor(
        &self,
        name: &str,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<candle_core::Tensor> {
        let data = self.read_tensor(name)?;
        let tensor = candle_core::Tensor::from_raw_buffer(
            &data.bytes,
            data.dtype,
            &data.shape,
            &candle_core::Device::Cpu,
        ).map_err(|e| anyhow::anyhow!("from_raw_buffer({name}): {e}"))?;
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype).map_err(|e| anyhow::anyhow!("to_dtype({name}): {e}"))?
        } else {
            tensor
        };
        if !device.is_cpu() {
            tensor.to_device(device).map_err(|e| anyhow::anyhow!("to_device({name}): {e}"))
        } else {
            Ok(tensor)
        }
    }

    /// Load all tensors as a HashMap suitable for VarBuilder::from_tensors().
    pub fn load_all(
        &self,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<HashMap<String, candle_core::Tensor>> {
        let mut map = HashMap::new();
        for name in self.tensor_names() {
            let tensor = self.load_tensor(&name, dtype, device)?;
            map.insert(name, tensor);
        }
        Ok(map)
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

        let mut index = HashMap::with_capacity(weight_map.len());
        let mut shards = Vec::with_capacity(shard_tensors.len());

        for shard_name in shard_tensors.keys() {
            let shard_path = parent.join(shard_name);
            let (header_len, header) = Self::parse_shard_header(&shard_path)?;
            let shard_idx = shards.len() as u16;
            let f = File::open(&shard_path)?;
            #[cfg(unix)]
            {
                let shard = MappedShard::new(&f)?;
                shards.push(shard);
            }
            #[cfg(not(unix))]
            {
                shards.push(MappedShard::new(f)?);
            }

            let obj = header
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("shard header not object"))?;

            for (name, meta) in obj {
                if name == "__metadata__" {
                    continue;
                }
                if let Some(tm) = Self::parse_tensor_meta(name, meta, shard_idx, header_len)? {
                    index.insert(name.clone(), tm);
                }
            }
        }

        log::info!(
            "SafetensorsStorage: indexed {} tensors across {} shards",
            index.len(),
            shards.len()
        );

        Ok(Self { index, shards })
    }

    /// Build from a single safetensors file (non-sharded model).
    fn from_single_file(path: &Path) -> Result<Self> {
        let (header_len, header) = Self::parse_shard_header(path)?;
        let f = File::open(path)?;
        let shard_idx = 0u16;

        let obj = header
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("header not object"))?;
        let mut index = HashMap::with_capacity(obj.len());

        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            if let Some(tm) = Self::parse_tensor_meta(name, meta, shard_idx, header_len)? {
                index.insert(name.clone(), tm);
            }
        }

        #[cfg(unix)]
        let shards = vec![MappedShard::new(&f)?];
        #[cfg(not(unix))]
        let shards = vec![MappedShard::new(f)?];

        log::info!("SafetensorsStorage: indexed {} tensors from single file", index.len());

        Ok(Self { index, shards })
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
        shard_idx: u16,
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
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U32" => DType::U32,
            "U8" => DType::U8,
            "F8_E4M3" | "F8E4M3" => DType::F8E4M3,
            other => {
                log::warn!("unknown dtype '{other}' for tensor '{name}', skipping");
                return Ok(None);
            }
        };

        let shape_vec: Vec<usize> = meta
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(TensorMeta {
            shard_idx,
            abs_offset,
            byte_size,
            dtype,
            shape: InlineShape::from_vec(&shape_vec),
        }))
    }
}

impl TensorStorageProvider for SafetensorsStorage {
    fn read_tensor(&self, name: &str) -> Result<TensorData> {
        let meta = self
            .index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in storage", name))?;

        let shard = self
            .shards
            .get(meta.shard_idx as usize)
            .ok_or_else(|| anyhow::anyhow!("shard index {} out of range", meta.shard_idx))?;

        let size = meta.byte_size as usize;

        // mmap: zero-syscall read — just memcpy from mapped memory
        #[cfg(unix)]
        let buf = shard.read_bytes(meta.abs_offset, size);

        #[cfg(not(unix))]
        let buf = {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = shard.file.try_clone()?;
            let mut buf = vec![0u8; size];
            f.seek(SeekFrom::Start(meta.abs_offset))?;
            f.read_exact(&mut buf)?;
            buf
        };

        Ok(TensorData {
            bytes: buf,
            dtype: meta.dtype,
            shape: meta.shape.to_vec(),
        })
    }

    fn read_tensors(&self, names: &[&str]) -> Result<Vec<TensorData>> {
        if names.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve all metadata first
        let metas: Vec<&TensorMeta> = names
            .iter()
            .map(|n| {
                self.index
                    .get(*n)
                    .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in storage", n))
            })
            .collect::<Result<_>>()?;

        // Check if all tensors are in the same shard and contiguous
        let same_shard = metas.windows(2).all(|w| w[0].shard_idx == w[1].shard_idx);
        let contiguous = same_shard
            && metas
                .windows(2)
                .all(|w| w[0].abs_offset + w[0].byte_size == w[1].abs_offset);

        if contiguous && metas.len() > 1 {
            // Single pread for the entire contiguous region
            let first = metas[0];
            let last = metas[metas.len() - 1];
            let total_size = (last.abs_offset + last.byte_size - first.abs_offset) as usize;

            let shard = self
                .shards
                .get(first.shard_idx as usize)
                .ok_or_else(|| anyhow::anyhow!("shard index {} out of range", first.shard_idx))?;

            // mmap: zero-syscall read — just memcpy from mapped memory
            #[cfg(unix)]
            let buf = shard.read_bytes(first.abs_offset, total_size);

            #[cfg(not(unix))]
            let buf = {
                use std::io::{Read as _, Seek, SeekFrom};
                let mut f = shard.file.try_clone()?;
                let mut buf = vec![0u8; total_size];
                f.seek(SeekFrom::Start(first.abs_offset))?;
                f.read_exact(&mut buf)?;
                buf
            };

            // Split the buffer into individual tensor data
            let mut results = Vec::with_capacity(metas.len());
            let mut offset = 0usize;
            for meta in &metas {
                let size = meta.byte_size as usize;
                let bytes = buf[offset..offset + size].to_vec();
                results.push(TensorData {
                    bytes,
                    dtype: meta.dtype,
                    shape: meta.shape.to_vec(),
                });
                offset += size;
            }
            Ok(results)
        } else {
            // Fall back to individual reads
            names.iter().map(|n| self.read_tensor(n)).collect()
        }
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
            SafetensorsStorage::parse_tensor_meta("test", &meta_json, 0, 200)
                .unwrap();
        let tm = result.unwrap();
        assert_eq!(tm.abs_offset, 8 + 200);
        assert_eq!(tm.byte_size, 1048576);
        assert_eq!(tm.dtype, DType::F16);
        assert_eq!(tm.shape.to_vec(), vec![1024, 512]);
    }

    #[test]
    fn test_parse_tensor_meta_skips_metadata() {
        let meta_json: serde_json::Value = serde_json::json!({
            "format": "pt"
        });
        let result =
            SafetensorsStorage::parse_tensor_meta("__metadata__", &meta_json, 0, 100)
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
                        shard_idx: 0,
                        abs_offset: 108,
                        byte_size: 1024,
                        dtype: DType::F32,
                        shape: InlineShape::from_vec(&[16, 16]),
                    },
                );
                m
            },
            shards: Vec::new(),
        };
        assert!(storage.has_tensor("weight"));
        assert!(!storage.has_tensor("missing"));
    }
}
