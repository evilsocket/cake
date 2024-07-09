use std::{collections::HashMap, fs::File, path::PathBuf};

use cake_core::{cake::Topology, utils};
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
    /// Input data path.
    #[arg(long, default_value = "./cake-data/Meta-Llama-3-8B/")]
    pub data_path: String,
    /// Topology file.
    #[arg(long, default_value = "./cake-data/topology.yml")]
    pub topology: String,
    /// Worker name.
    #[arg(long)]
    pub worker: String,
    /// Output folder.
    #[arg(long)]
    pub output: String,
}

fn main() {
    let args = Args::parse();
    let data_path = PathBuf::from(&args.data_path);

    let topology = Topology::from_path(&args.topology).expect("can't load topology");
    let worker_node = topology
        .get(&args.worker)
        .expect("can't find worker topology");

    println!("worker: {}", &args.worker);

    // println!("layers: {:?}", &worker_node.layers);

    let tensors_index_path = data_path.join("model.safetensors.index.json");
    let tensors_index_data =
        std::fs::read_to_string(tensors_index_path).expect("can't read tensors index");
    let tensors_index: Index =
        serde_json::from_str(&tensors_index_data).expect("can't parse index");

    let mut reduced: HashMap<String, Vec<String>> = HashMap::new();
    let mut new_index = Index::new();

    println!("index has {} tensors", tensors_index.weight_map.len());

    for (layer_full_name, filename) in &tensors_index.weight_map {
        if worker_node.is_layer_owner(layer_full_name) {
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

    println!(
        "reducing to {} tensors in {} files",
        new_index.weight_map.len(),
        reduced.len()
    );

    let mut metadata: HashMap<String, TensorStore> = HashMap::new();

    for (filename, tensor_names) in &reduced {
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

    let output_path = PathBuf::from(&args.output);
    if !output_path.exists() {
        println!("creating {}", output_path.display());
        std::fs::create_dir_all(&output_path).unwrap();
    }

    let new_index_path = output_path.join("model.safetensors.index.json");
    let new_index_data = serde_json::to_string_pretty(&new_index).unwrap();

    println!("saving new index to {} ...", new_index_path.display());

    std::fs::write(&new_index_path, new_index_data).unwrap();

    let new_tensors_path = output_path.join("reduced.safetensors");

    println!(
        "saving reduced tensors to {} ...",
        new_tensors_path.display()
    );

    safetensors::serialize_to_file(metadata, &None, &new_tensors_path).unwrap();

    let loaded = utils::load_safetensors_from_index(new_index_path).unwrap();

    assert_eq!(loaded.len(), 1);

    let file = File::open(&loaded[0]).unwrap();
    let buffer = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
    let _ = SafeTensors::deserialize(&buffer).unwrap();

    println!("done");
}
