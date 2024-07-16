//! Utility functions and abstractions.

use std::path::PathBuf;

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Tensor,
};

use anyhow::{bail, Result};

mod token_output_stream;

use candle_nn::VarBuilder;
pub use token_output_stream::*;

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

/// Create a VarBuilder with the tensors loaded from the index.
pub fn load_var_builder_from_index<'a>(
    tensor_index: PathBuf,
    dtype: DType,
    device: Device,
) -> Result<VarBuilder<'a>> {
    log::info!("loading tensors in {}", tensor_index.display());

    let filenames: Vec<std::path::PathBuf> = load_safetensors_paths_from_index(tensor_index)
        .map_err(|e| anyhow!("can't load tensors index: {:?}", e))?;

    unsafe {
        VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)
            .map_err(|e| anyhow!("can't create varbuilder from tensors: {:?}", e))
    }
}

/// Nasty hack to debug NaN in tensors.
#[allow(dead_code)]
pub(crate) fn panic_on_nan(t: &Tensor, name: &str) {
    if t.to_string().contains("NaN") {
        panic!("\ntensor '{name}' contains NaN: \n{t}");
    }
}
