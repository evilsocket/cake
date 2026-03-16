//! Model splitting utility — creates per-worker model bundles from a full model.

use std::{
    collections::HashMap,
    fs::File,
    path::Path,
};

use anyhow::Result;
use safetensors::{Dtype, SafeTensors, View};
use serde::{Deserialize, Serialize};

use crate::{
    cake::{Node, Topology},
    utils, ModelType,
};

#[derive(Debug, Serialize, Deserialize)]
struct Index {
    pub weight_map: HashMap<String, String>,
}

impl Index {
    pub fn new() -> Self {
        let weight_map = HashMap::new();
        Self { weight_map }
    }
}

#[derive(Debug)]
struct TensorStore {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for TensorStore {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::from(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn load_index(data_path: &Path) -> Result<Index> {
    let tensors_index_path = data_path.join("model.safetensors.index.json");

    if tensors_index_path.exists() {
        let tensors_index_data = std::fs::read_to_string(tensors_index_path)?;
        let tensors_index: Index = serde_json::from_str(&tensors_index_data)?;
        Ok(tensors_index)
    } else {
        let single_path = data_path.join("model.safetensors");
        if !single_path.exists() {
            anyhow::bail!(
                "neither model.safetensors.index.json nor model.safetensors found in {}",
                data_path.display()
            );
        }

        log::info!("no index file found, generating from model.safetensors ...");

        let file = File::open(&single_path)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&buffer)?;

        let mut index = Index::new();
        for (name, _) in tensors.tensors() {
            index
                .weight_map
                .insert(name.to_string(), "model.safetensors".to_string());
        }

        Ok(index)
    }
}

fn reduce_for_worker(
    index: &Index,
    worker: &Node,
) -> Result<(Index, HashMap<String, Vec<String>>)> {
    log::info!("worker: {}", &worker.host);

    let mut reduced: HashMap<String, Vec<String>> = HashMap::new();
    let mut new_index = Index::new();

    for (layer_full_name, filename) in &index.weight_map {
        if worker.is_text_model_layer_owner(layer_full_name) {
            if let Some(layers) = reduced.get_mut(filename) {
                layers.push(layer_full_name.to_string());
            } else {
                reduced.insert(filename.to_string(), vec![layer_full_name.to_string()]);
            }

            new_index.weight_map.insert(
                layer_full_name.to_string(),
                "reduced.safetensors".to_string(),
            );
        }
    }

    Ok((new_index, reduced))
}

fn create_new_metadata(
    data_path: &Path,
    reduced: &HashMap<String, Vec<String>>,
) -> Result<HashMap<String, TensorStore>> {
    let mut metadata: HashMap<String, TensorStore> = HashMap::new();

    for (filename, tensor_names) in reduced {
        let filepath = data_path.join(filename);

        log::info!("loading {} ...", filepath.display());

        let file = File::open(&filepath)?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&buffer)?;

        log::info!("  extracting {} tensors", tensor_names.len());

        for tensor_name in tensor_names {
            let tensor = tensors.tensor(tensor_name)?;
            metadata.insert(
                tensor_name.to_string(),
                TensorStore {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    data: tensor.data().to_vec(),
                },
            );
        }

        drop(tensors);
        drop(buffer);
    }

    Ok(metadata)
}

/// Split a model into per-worker bundles.
///
/// Each bundle contains a reduced safetensors file with only the worker's assigned tensors,
/// a matching index file, and the worker's topology.
pub fn split_model(
    model_path: &Path,
    topology_path: &str,
    worker: Option<&str>,
    output: &Path,
) -> Result<()> {
    let topology = Topology::from_path(topology_path, &ModelType::TextModel)?;
    let index = load_index(model_path)?;

    log::info!("index has {} tensors", index.weight_map.len());

    let selected_workers: Vec<String> = if let Some(name) = worker {
        vec![name.to_string()]
    } else {
        topology.keys().map(|s| s.to_string()).collect()
    };

    log::info!("processing {} workers", selected_workers.len());

    for worker_name in &selected_workers {
        log::info!("processing worker {worker_name} ...");

        let worker_node = topology
            .get(worker_name)
            .ok_or_else(|| anyhow!("can't find worker '{}' in topology", worker_name))?;

        let (new_index, reduced) = reduce_for_worker(&index, worker_node)?;

        log::info!("compacting {} tensors ...", new_index.weight_map.len());

        let metadata = create_new_metadata(model_path, &reduced)?;

        let bundle_name = format!("{worker_name}-node");
        let output_path = output.join(&bundle_name);
        let model_output_path = output_path.join("model");
        if !output_path.exists() {
            log::info!("creating {}", model_output_path.display());
            std::fs::create_dir_all(&model_output_path)?;
        } else {
            log::info!("saving model to {}", model_output_path.display());
        }

        let new_index_path = model_output_path.join("model.safetensors.index.json");

        log::info!("saving new index to {} ...", new_index_path.display());

        let new_index_data = serde_json::to_string_pretty(&new_index)?;
        std::fs::write(&new_index_path, new_index_data)?;

        let new_tensors_path = model_output_path.join("reduced.safetensors");

        log::info!(
            "saving reduced tensors to {} ...",
            new_tensors_path.display()
        );

        safetensors::serialize_to_file(metadata, None, &new_tensors_path)?;

        // Verify the output is readable.
        let loaded = utils::load_safetensors_paths_from_index(new_index_path)?;
        assert_eq!(loaded.len(), 1);
        let file = File::open(&loaded[0])?;
        let buffer = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let _ = SafeTensors::deserialize(&buffer)?;

        let new_topology_path = output_path.join("topology.yml");

        log::info!(
            "saving worker topology to {} ...",
            new_topology_path.display()
        );

        let mut new_topology: HashMap<String, &Node> = HashMap::new();
        new_topology.insert(worker_name.to_string(), worker_node);
        let new_topology_data = serde_yaml::to_string(&new_topology)?;
        std::fs::write(&new_topology_path, new_topology_data)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_node(layers: Vec<&str>) -> Node {
        Node {
            host: "test:10128".to_string(),
            description: None,
            layers: layers.into_iter().map(|s| s.to_string()).collect(),
            vram_bytes: 0,
            tflops: 0.0,
            backend: String::new(),
            hostname: String::new(),
            os: String::new(),
        }
    }

    #[test]
    fn load_index_from_json_file() {
        let tmp = tempfile::tempdir().unwrap();
        let index_json = serde_json::json!({
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "shard-00001.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "shard-00001.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index_json).unwrap(),
        )
        .unwrap();

        let index = load_index(tmp.path()).unwrap();
        assert_eq!(index.weight_map.len(), 3);
        assert_eq!(
            index.weight_map["model.layers.0.self_attn.q_proj.weight"],
            "shard-00001.safetensors"
        );
    }

    #[test]
    fn load_index_missing_both_files() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_index(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn reduce_for_worker_filters_correctly() {
        let mut weight_map = HashMap::new();
        weight_map.insert(
            "model.layers.0.attn.weight".to_string(),
            "shard-00001.safetensors".to_string(),
        );
        weight_map.insert(
            "model.layers.0.mlp.weight".to_string(),
            "shard-00001.safetensors".to_string(),
        );
        weight_map.insert(
            "model.layers.1.attn.weight".to_string(),
            "shard-00002.safetensors".to_string(),
        );
        weight_map.insert(
            "model.layers.2.attn.weight".to_string(),
            "shard-00002.safetensors".to_string(),
        );
        weight_map.insert(
            "model.embed_tokens.weight".to_string(),
            "shard-00001.safetensors".to_string(),
        );
        let index = Index { weight_map };

        // Worker owns layers 0 and 1
        let worker = make_node(vec!["model.layers.0", "model.layers.1"]);
        let (new_index, reduced) = reduce_for_worker(&index, &worker).unwrap();

        // Should have 3 tensors (layer 0 attn, layer 0 mlp, layer 1 attn)
        assert_eq!(new_index.weight_map.len(), 3);
        assert!(new_index
            .weight_map
            .contains_key("model.layers.0.attn.weight"));
        assert!(new_index
            .weight_map
            .contains_key("model.layers.0.mlp.weight"));
        assert!(new_index
            .weight_map
            .contains_key("model.layers.1.attn.weight"));
        // Should NOT contain layer 2 or embed_tokens
        assert!(!new_index
            .weight_map
            .contains_key("model.layers.2.attn.weight"));
        assert!(!new_index
            .weight_map
            .contains_key("model.embed_tokens.weight"));

        // All entries in new_index should point to "reduced.safetensors"
        for val in new_index.weight_map.values() {
            assert_eq!(val, "reduced.safetensors");
        }

        // reduced map should have source shard filenames as keys
        assert!(reduced.contains_key("shard-00001.safetensors"));
        assert!(reduced.contains_key("shard-00002.safetensors"));
    }

    #[test]
    fn reduce_for_worker_no_matching_layers() {
        let mut weight_map = HashMap::new();
        weight_map.insert(
            "model.layers.5.attn.weight".to_string(),
            "shard.safetensors".to_string(),
        );
        let index = Index { weight_map };

        let worker = make_node(vec!["model.layers.0"]);
        let (new_index, reduced) = reduce_for_worker(&index, &worker).unwrap();
        assert!(new_index.weight_map.is_empty());
        assert!(reduced.is_empty());
    }

    #[test]
    fn index_serialization_roundtrip() {
        let mut index = Index::new();
        index.weight_map.insert(
            "tensor.weight".to_string(),
            "file.safetensors".to_string(),
        );
        let json = serde_json::to_string(&index).unwrap();
        let deserialized: Index = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.weight_map.len(), 1);
        assert_eq!(deserialized.weight_map["tensor.weight"], "file.safetensors");
    }
}
