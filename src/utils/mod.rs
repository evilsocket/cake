use std::path::PathBuf;

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device,
};

use anyhow::{bail, Result};

mod token_output_stream;

pub use token_output_stream::*;

pub fn device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn load_safetensors_from_index(
    tensors_index_json_filename: PathBuf,
) -> Result<Vec<std::path::PathBuf>> {
    let parent_dir = tensors_index_json_filename.parent().unwrap();
    let json_file = std::fs::File::open(&tensors_index_json_filename)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
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
