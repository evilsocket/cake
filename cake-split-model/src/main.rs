//! This is a utility to split a single, safetensors based model into parts
//! that are smaller and can be distributed to the workers instead of the entire model.
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::Result;
use cake_core::{
    cake::{Node, Topology},
    utils,
};
use clap::Parser;
use safetensors::{Dtype, SafeTensors, View};
use serde::{Deserialize, Serialize};

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

    fn data(&self) -> std::borrow::Cow<[u8]> {
        std::borrow::Cow::from(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Input model path.
    #[arg(long, default_value = "./cake-data/Meta-Llama-3-8B/")]
    pub model_path: String,
    /// Topology file.
    #[arg(long, default_value = "./cake-data/topology.yml")]
    pub topology: String,
    /// Worker name or empty for all.
    #[arg(long)]
    pub worker: Option<String>,
    /// Output folder.
    #[arg(long)]
    pub output: String,
}

fn load_index(data_path: &Path) -> Result<Index> {
    let tensors_index_path = data_path.join("model.safetensors.index.json");
    let tensors_index_data = std::fs::read_to_string(tensors_index_path)?;
    let tensors_index: Index = serde_json::from_str(&tensors_index_data)?;

    Ok(tensors_index)
}

fn reduce_for_worker(
    index: &Index,
    worker: &Node,
) -> Result<(Index, HashMap<String, Vec<String>>)> {
    println!("worker: {}", &worker.host);

    let mut reduced: HashMap<String, Vec<String>> = HashMap::new();
    let mut new_index = Index::new();

    for (layer_full_name, filename) in &index.weight_map {
        if worker.is_layer_owner(layer_full_name) {
            //println!("{} {}", layer_full_name, filename);
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

        println!("loading {} ...", filepath.display());

        let file = File::open(&filepath).unwrap();
        let buffer = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();

        println!("  extracting {} tensors", tensor_names.len());

        for tensor_name in tensor_names {
            let tensor = tensors.tensor(tensor_name).unwrap();
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

fn main() {
    let args = Args::parse();
    let data_path = PathBuf::from(&args.model_path);

    let topology = Topology::from_path(&args.topology).expect("can't load topology");
    let index = load_index(&data_path).expect("can't load index");

    println!("index has {} tensors", index.weight_map.len());

    let selected_workers = if let Some(name) = &args.worker {
        vec![name.to_string()]
    } else {
        topology.keys().map(|s| s.to_string()).collect()
    };

    println!("processing {} workers", selected_workers.len());

    for worker_name in &selected_workers {
        println!("processing worker {worker_name} ...");

        let worker_node = topology
            .get(worker_name)
            .expect("can't find worker topology");

        let (new_index, reduced) =
            reduce_for_worker(&index, worker_node).expect("can't reduce for worker");

        println!("compacting {} tensors ...", new_index.weight_map.len());

        let metadata =
            create_new_metadata(&data_path, &reduced).expect("can't create metadata for worker");

        let bundle_name = format!("{worker_name}-node");
        let output_path = PathBuf::from(&args.output).join(bundle_name);
        let model_output_path = output_path.join("model");
        if !output_path.exists() {
            println!("creating {}", model_output_path.display());
            std::fs::create_dir_all(&model_output_path).unwrap();
        } else {
            println!("saving model to {}", model_output_path.display());
        }

        let new_index_path = model_output_path.join("model.safetensors.index.json");

        println!("saving new index to {} ...", new_index_path.display());

        let new_index_data = serde_json::to_string_pretty(&new_index).unwrap();
        std::fs::write(&new_index_path, new_index_data).unwrap();

        let new_tensors_path = model_output_path.join("reduced.safetensors");

        println!(
            "saving reduced tensors to {} ...",
            new_tensors_path.display()
        );

        safetensors::serialize_to_file(metadata, &None, &new_tensors_path).unwrap();

        let loaded = utils::load_safetensors_paths_from_index(new_index_path).unwrap();

        assert_eq!(loaded.len(), 1);

        let file = File::open(&loaded[0]).unwrap();
        let buffer = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let _ = SafeTensors::deserialize(&buffer).unwrap();

        let new_topology_path = output_path.join("topology.yml");

        println!(
            "saving worker topology to {} ...",
            new_topology_path.display()
        );

        let mut new_topology: HashMap<String, &Node> = HashMap::new();

        new_topology.insert(worker_name.to_string(), worker_node);

        let new_topology_data = serde_yaml::to_string(&new_topology).unwrap();

        std::fs::write(&new_topology_path, new_topology_data).unwrap();
    }
}
