//! Utility functions and abstractions.

pub mod fp8;
pub mod gguf;
pub mod hf;
pub mod models;
pub mod split;

use std::path::{Path, PathBuf};

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};

use anyhow::{bail, Result};

use candle_nn::VarBuilder;

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
    for filename in filenames {
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
    fp8: bool,
) -> Result<VarBuilder<'a>> {
    let filenames: Vec<std::path::PathBuf> = if tensor_index.exists() {
        load_safetensors_paths_from_index(tensor_index)
            .map_err(|e| anyhow!("can't load tensors index: {:?}", e))?
    } else {
        load_safetensors_from_model(tensor_index.parent().unwrap())
            .map_err(|e| anyhow!("can't load tensors index: {:?}", e))?
    };

    prefetch_safetensors(&filenames)?;

    if fp8 {
        unsafe {
            fp8::load_fp8_var_builder(&filenames, dtype, &device)
                .map_err(|e| anyhow!("can't create fp8 varbuilder from tensors: {:?}", e))
        }
    } else {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)
                .map_err(|e| anyhow!("can't create varbuilder from tensors: {:?}", e))
        }
    }
}

/// Create a VarBuilder that only loads safetensors shards needed for the given
/// local layers. Shards containing only remote-worker tensors are excluded,
/// reducing GPU memory usage on the master.
pub fn load_var_builder_for_local_layers<'a>(
    tensor_index: PathBuf,
    dtype: DType,
    device: Device,
    worker_layers: &std::collections::HashSet<String>,
    fp8: bool,
) -> Result<VarBuilder<'a>> {
    if !tensor_index.exists() {
        // Single safetensors file — can't filter, load all
        return load_var_builder_from_index(tensor_index, dtype, device, fp8);
    }

    if worker_layers.is_empty() {
        // No workers — load everything
        return load_var_builder_from_index(tensor_index, dtype, device, fp8);
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

    if fp8 {
        unsafe {
            fp8::load_fp8_var_builder(&filenames, dtype, &device)
                .map_err(|e| anyhow!("can't create fp8 varbuilder from tensors: {:?}", e))
        }
    } else {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)
                .map_err(|e| anyhow!("can't create varbuilder from tensors: {:?}", e))
        }
    }
}

/// Create a VarBuilder that only loads safetensors shards containing tensors
/// for the given layer prefixes. Workers use this to skip shards that only
/// contain layers assigned to other nodes.
pub fn load_var_builder_for_specific_layers<'a>(
    tensor_index: PathBuf,
    dtype: DType,
    device: Device,
    layer_prefixes: &[String],
    fp8: bool,
) -> Result<VarBuilder<'a>> {
    if !tensor_index.exists() || layer_prefixes.is_empty() {
        return load_var_builder_from_index(tensor_index, dtype, device, fp8);
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

    if fp8 {
        unsafe {
            fp8::load_fp8_var_builder(&filenames, dtype, &device)
                .map_err(|e| anyhow!("can't create fp8 varbuilder from tensors: {:?}", e))
        }
    } else {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)
                .map_err(|e| anyhow!("can't create varbuilder from tensors: {:?}", e))
        }
    }
}

/// Nasty hack to debug NaN in tensors.
#[allow(dead_code)]
pub(crate) fn panic_on_nan(t: &Tensor, name: &str) {
    if t.to_string().contains("NaN") {
        panic!("\ntensor '{name}' contains NaN: \n{t}");
    }
}
