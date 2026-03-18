//! Utility functions and abstractions.

pub mod fp8;
#[cfg(feature = "cuda")]
pub mod flash_attn;
pub mod fused_ops;
pub mod gguf;
pub mod gptq;
pub mod hf;
pub mod models;
pub mod native_dtype_backend;
pub mod split;

use std::path::{Path, PathBuf};

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};

use anyhow::{anyhow, bail, Result};

use candle_nn::VarBuilder;

// ─── Quantization trait ─────────────────────────────────────────────────────

/// A quantization strategy that can create a VarBuilder with appropriate
/// dequantization and estimate in-memory VRAM from on-disk size.
///
/// Implementations: `NoQuantization`, `Fp8Quantization`, `GptqQuantization`.
pub trait Quantization: Send + Sync {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Create a VarBuilder from safetensors files with this quantization.
    ///
    /// # Safety
    /// Inherits the mmap safety requirements from candle's safetensors loading.
    unsafe fn load_var_builder<'a>(
        &self,
        filenames: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'a>>;

    /// Estimate in-memory layer size given on-disk size and target dtype bytes.
    /// Default: no expansion (on-disk size = in-memory size).
    fn estimate_layer_vram(&self, on_disk_bytes: u64, _dtype_bytes: u64) -> u64 {
        on_disk_bytes
    }
}

/// No quantization — plain safetensors, loaded as-is.
pub struct NoQuantization;

impl Quantization for NoQuantization {
    fn name(&self) -> &str {
        "none"
    }

    unsafe fn load_var_builder<'a>(
        &self,
        filenames: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'a>> {
        VarBuilder::from_mmaped_safetensors(filenames, dtype, device)
            .map_err(|e| anyhow!("can't create varbuilder: {e:?}"))
    }
}

/// FP8 block-wise quantization — wraps `fp8::load_fp8_var_builder`.
pub struct Fp8Quantization;

impl Quantization for Fp8Quantization {
    fn name(&self) -> &str {
        "fp8"
    }

    unsafe fn load_var_builder<'a>(
        &self,
        filenames: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'a>> {
        fp8::load_fp8_var_builder(filenames, dtype, device)
            .map_err(|e| anyhow!("can't create fp8 varbuilder: {e:?}"))
    }

    fn estimate_layer_vram(&self, on_disk_bytes: u64, dtype_bytes: u64) -> u64 {
        // FP8 is 1 byte per element on disk, expands to dtype_bytes in memory
        on_disk_bytes * dtype_bytes
    }
}

/// GPTQ 4-bit quantization — wraps `gptq::load_gptq_var_builder`.
pub struct GptqQuantization {
    pub group_size: usize,
}

impl Quantization for GptqQuantization {
    fn name(&self) -> &str {
        "gptq"
    }

    unsafe fn load_var_builder<'a>(
        &self,
        filenames: &[PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'a>> {
        gptq::load_gptq_var_builder(filenames, dtype, device, self.group_size)
            .map_err(|e| anyhow!("can't create gptq varbuilder: {e:?}"))
    }
}

/// Detect quantization strategy from a model's config.json.
pub fn detect_quantization(config_path: &Path) -> Box<dyn Quantization> {
    if fp8::is_fp8_quantized(config_path) {
        log::info!("model uses FP8 quantization — weights will be dequantized at load time");
        Box::new(Fp8Quantization)
    } else if gptq::is_gptq_quantized(config_path) {
        let gs = gptq::gptq_group_size(config_path);
        log::info!("model uses GPTQ quantization (group_size={gs}) — weights will be dequantized at load time");
        Box::new(GptqQuantization { group_size: gs })
    } else {
        Box::new(NoQuantization)
    }
}

// ─── End quantization trait ──────────────────────────────────────────────────

/// Returns the best available device at `ordinal` index (in case of multiple GPUs), or CPU if `force_cpu` is true.
pub fn get_inference_device(force_cpu: bool, ordinal: usize) -> Result<Device> {
    if force_cpu {
        log::debug!("device is forced cpu");
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        log::debug!("device is cuda {ordinal}");
        Ok(Device::new_cuda(ordinal)?)
    } else if metal_is_available() {
        log::debug!("device is metal {ordinal}");
        Ok(Device::new_metal(ordinal)?)
    } else {
        log::debug!("device is cpu");
        // fallback to cpu if nothing else available
        Ok(Device::Cpu)
    }
}

pub fn load_safetensors_from_model(path: &Path) -> Result<Vec<std::path::PathBuf>> {
    log::info!("loading tensors from {} ...", "model.safetensors");
    let result = vec![path.join("model.safetensors")];
    Ok(result)
}

/// Load the safetensors files for a model from the hub based on a json index file.
pub fn load_safetensors_paths_from_index(
    tensors_index_json_filename: PathBuf,
) -> Result<Vec<std::path::PathBuf>> {
    log::info!(
        "loading tensors from {} ...",
        tensors_index_json_filename.display()
    );

    let parent_dir = tensors_index_json_filename.parent().unwrap();
    let json_file = std::fs::File::open(&tensors_index_json_filename).map_err(|e| {
        anyhow!(
            "can't open {}: {:?}",
            tensors_index_json_filename.display(),
            e
        )
    })?;
    let json: serde_json::Value = serde_json::from_reader(&json_file).map_err(|e| {
        anyhow!(
            "can't parse {}: {:?}",
            tensors_index_json_filename.display(),
            e
        )
    })?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| parent_dir.join(v))
        .collect::<Vec<std::path::PathBuf>>();

    Ok(safetensors_files)
}

/// Pre-read safetensor files into the OS page cache so that subsequent
/// mmap access doesn't trigger per-tensor page faults during layer loading.
/// Uses OnceLock to skip redundant calls (e.g. multi-GPU VarBuilder creation).
fn prefetch_safetensors(filenames: &[PathBuf]) -> Result<()> {
    use std::sync::OnceLock;
    static DONE: OnceLock<()> = OnceLock::new();

    if DONE.get().is_some() {
        log::info!("safetensor files already in page cache, skipping prefetch");
        return Ok(());
    }

    use std::io::Read;
    let start = std::time::Instant::now();
    let mut total_bytes: u64 = 0;
    let mut buf = Vec::new();
    for (i, filename) in filenames.iter().enumerate() {
        log::info!(
            "caching shard {}/{} ({}) ...",
            i + 1,
            filenames.len(),
            filename.file_name().unwrap_or_default().to_string_lossy()
        );
        buf.clear();
        std::fs::File::open(filename)
            .map_err(|e| anyhow!("prefetch: can't open {}: {e}", filename.display()))?
            .read_to_end(&mut buf)
            .map_err(|e| anyhow!("prefetch: can't read {}: {e}", filename.display()))?;
        total_bytes += buf.len() as u64;
    }
    log::info!(
        "pre-cached {} in {:.1}s",
        human_bytes::human_bytes(total_bytes as f64),
        start.elapsed().as_secs_f64()
    );

    DONE.set(()).ok();
    Ok(())
}

/// Create a VarBuilder with the tensors loaded from the index.
pub fn load_var_builder_from_index<'a>(
    tensor_index: PathBuf,
    dtype: DType,
    device: Device,
    quant: &dyn Quantization,
) -> Result<VarBuilder<'a>> {
    let filenames: Vec<std::path::PathBuf> = if tensor_index.exists() {
        load_safetensors_paths_from_index(tensor_index)
            .map_err(|e| anyhow!("can't load tensors index: {:?}", e))?
    } else {
        load_safetensors_from_model(tensor_index.parent().unwrap())
            .map_err(|e| anyhow!("can't load tensors index: {:?}", e))?
    };

    prefetch_safetensors(&filenames)?;
    unsafe { quant.load_var_builder(&filenames, dtype, &device) }
}

/// Create a VarBuilder that only loads safetensors shards needed for the given
/// local layers. Shards containing only remote-worker tensors are excluded,
/// reducing GPU memory usage on the master.
pub fn load_var_builder_for_local_layers<'a>(
    tensor_index: PathBuf,
    dtype: DType,
    device: Device,
    worker_layers: &std::collections::HashSet<String>,
    quant: &dyn Quantization,
) -> Result<VarBuilder<'a>> {
    if !tensor_index.exists() {
        return load_var_builder_from_index(tensor_index, dtype, device, quant);
    }

    if worker_layers.is_empty() {
        return load_var_builder_from_index(tensor_index, dtype, device, quant);
    }

    let parent_dir = tensor_index.parent().unwrap();
    let json_data = std::fs::read_to_string(&tensor_index)
        .map_err(|e| anyhow!("can't read {}: {:?}", tensor_index.display(), e))?;
    let json: serde_json::Value = serde_json::from_str(&json_data)
        .map_err(|e| anyhow!("can't parse {}: {:?}", tensor_index.display(), e))?;
    let weight_map = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("no weight_map in {}", tensor_index.display()))?;

    // Find shard files that contain at least one tensor NOT belonging to a worker layer.
    // A tensor belongs to a worker layer if its name starts with "<layer_name>."
    let mut needed_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (tensor_name, shard_file) in weight_map {
        let is_worker_tensor = worker_layers
            .iter()
            .any(|layer| tensor_name.starts_with(&format!("{}.", layer)));
        if !is_worker_tensor {
            if let Some(filename) = shard_file.as_str() {
                needed_shards.insert(filename.to_string());
            }
        }
    }

    let filenames: Vec<PathBuf> = needed_shards
        .iter()
        .map(|f| parent_dir.join(f))
        .collect();

    log::info!(
        "loading {} of {} shard file(s) for local layers",
        filenames.len(),
        weight_map
            .values()
            .filter_map(|v| v.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    prefetch_safetensors(&filenames)?;
    unsafe { quant.load_var_builder(&filenames, dtype, &device) }
}

/// Create a VarBuilder that only loads safetensors shards containing tensors
/// for the given layer prefixes. Workers use this to skip shards that only
/// contain layers assigned to other nodes.
pub fn load_var_builder_for_specific_layers<'a>(
    tensor_index: PathBuf,
    dtype: DType,
    device: Device,
    layer_prefixes: &[String],
    quant: &dyn Quantization,
) -> Result<VarBuilder<'a>> {
    if !tensor_index.exists() || layer_prefixes.is_empty() {
        return load_var_builder_from_index(tensor_index, dtype, device, quant);
    }

    let parent_dir = tensor_index.parent().unwrap();
    let json_data = std::fs::read_to_string(&tensor_index)
        .map_err(|e| anyhow!("can't read {}: {:?}", tensor_index.display(), e))?;
    let json: serde_json::Value = serde_json::from_str(&json_data)
        .map_err(|e| anyhow!("can't parse {}: {:?}", tensor_index.display(), e))?;
    let weight_map = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("no weight_map in {}", tensor_index.display()))?;

    let mut needed_shards: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (tensor_name, shard_file) in weight_map {
        let is_needed = layer_prefixes
            .iter()
            .any(|prefix| tensor_name.starts_with(&format!("{}.", prefix)));
        if is_needed {
            if let Some(filename) = shard_file.as_str() {
                needed_shards.insert(filename.to_string());
            }
        }
    }

    let total_shards = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len();

    let filenames: Vec<PathBuf> = needed_shards.iter().map(|f| parent_dir.join(f)).collect();

    log::info!(
        "loading {} of {} shard file(s) for {} layers",
        filenames.len(),
        total_shards,
        layer_prefixes.len()
    );

    prefetch_safetensors(&filenames)?;
    unsafe { quant.load_var_builder(&filenames, dtype, &device) }
}

/// Nasty hack to debug NaN in tensors.
#[allow(dead_code)]
pub(crate) fn panic_on_nan(t: &Tensor, name: &str) {
    if t.to_string().contains("NaN") {
        panic!("\ntensor '{name}' contains NaN: \n{t}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn load_safetensors_from_model_returns_single_path() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_safetensors_from_model(tmp.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], tmp.path().join("model.safetensors"));
    }

    #[test]
    fn load_safetensors_paths_from_index_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.weight": "shard-00001.safetensors",
                "model.layers.1.weight": "shard-00001.safetensors",
                "model.layers.2.weight": "shard-00002.safetensors"
            }
        });
        let index_path = tmp.path().join("model.safetensors.index.json");
        fs::write(&index_path, serde_json::to_string(&index).unwrap()).unwrap();

        let paths = load_safetensors_paths_from_index(index_path).unwrap();
        // Two unique shard files
        assert_eq!(paths.len(), 2);
        let names: std::collections::HashSet<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains("shard-00001.safetensors"));
        assert!(names.contains("shard-00002.safetensors"));
        // All paths should be under the temp dir
        for p in &paths {
            assert_eq!(p.parent().unwrap(), tmp.path());
        }
    }

    #[test]
    fn load_safetensors_paths_from_index_missing_file() {
        let result = load_safetensors_paths_from_index(PathBuf::from("/nonexistent/index.json"));
        assert!(result.is_err());
    }

    #[test]
    fn load_safetensors_paths_from_index_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().join("model.safetensors.index.json");
        fs::write(&index_path, "not valid json").unwrap();
        let result = load_safetensors_paths_from_index(index_path);
        assert!(result.is_err());
    }

    #[test]
    fn load_safetensors_paths_from_index_no_weight_map() {
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().join("model.safetensors.index.json");
        fs::write(&index_path, r#"{"metadata": {}}"#).unwrap();
        let result = load_safetensors_paths_from_index(index_path);
        assert!(result.is_err());
    }

    #[test]
    fn load_safetensors_paths_from_index_empty_weight_map() {
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().join("model.safetensors.index.json");
        fs::write(&index_path, r#"{"weight_map": {}}"#).unwrap();
        let paths = load_safetensors_paths_from_index(index_path).unwrap();
        assert!(paths.is_empty());
    }
}
