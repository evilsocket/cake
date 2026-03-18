use std::collections::HashMap;

use crate::ModelType;
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::{WorkerCapacity, max_layers_for_gpus, estimate_tflops_for_gpus};
use super::discovery;

lazy_static! {
    static ref LAYER_RANGE_PARSER: Regex = Regex::new(r"(?m)^(.+[^\d])(\d+)-(\d+)$").unwrap();
}

/// A single node (worker).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /// Address and port of the worker.
    pub host: String,
    /// Optional description.
    pub description: Option<String>,
    pub layers: Vec<String>,
    /// Total VRAM (or system RAM) in bytes available on this node.
    #[serde(default)]
    pub vram_bytes: u64,
    /// Approximate FP16 TFLOPS for this node.
    #[serde(default)]
    pub tflops: f64,
    /// Backend description (e.g. "CUDA 12.4", "Apple M2 Max", "CPU").
    #[serde(default)]
    pub backend: String,
    /// Hostname of the node.
    #[serde(default)]
    pub hostname: String,
    /// Operating system (e.g. "linux", "macos", "windows").
    #[serde(default)]
    pub os: String,
}

impl Node {
    /// Return true if this node hosts the specified layer.
    pub fn is_text_model_layer_owner(&self, full_layer_name: &str) -> bool {
        for prefix in self.layers.iter() {
            if full_layer_name.starts_with(&format!("{}.", prefix)) {
                return true;
            }
        }

        false
    }

    /// Build a synthetic [`discovery::GpuInfo`] list from this node's VRAM/TFLOPS.
    ///
    /// Used when a topology file specifies nodes without layer assignments,
    /// so the sharding algorithm can estimate capacity.
    fn as_gpu_info(&self) -> Vec<discovery::GpuInfo> {
        if self.vram_bytes == 0 {
            return vec![];
        }
        vec![discovery::GpuInfo {
            name: self
                .description
                .clone()
                .or_else(|| {
                    if !self.backend.is_empty() {
                        Some(self.backend.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("GPU ({})", self.hostname)),
            vram_bytes: self.vram_bytes,
            tflops: self.tflops as f32,
        }]
    }
}

/// Wrapper that pairs a worker name with a [`Node`] reference, implementing
/// [`WorkerCapacity`] so topology nodes can be fed to the sharding algorithm.
pub struct NamedNode<'a> {
    /// Worker name (topology map key).
    pub name: &'a str,
    /// Reference to the node.
    pub node: &'a Node,
}

impl WorkerCapacity for NamedNode<'_> {
    fn name(&self) -> &str {
        self.name
    }
    fn total_vram(&self) -> u64 {
        self.node.vram_bytes
    }
    fn total_tflops(&self) -> f64 {
        let gpus = self.node.as_gpu_info();
        estimate_tflops_for_gpus(&gpus)
    }
    fn max_layers_for_size(&self, layer_size_bytes: u64) -> usize {
        let gpus = self.node.as_gpu_info();
        max_layers_for_gpus(&gpus, layer_size_bytes)
    }
}

/// The topology is a worker-name -> worker-info map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topology(HashMap<String, Node>);

impl Default for Topology {
    fn default() -> Self {
        Self::new()
    }
}

impl Topology {
    /// Create a new empty topology.
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Load the topology from a yaml file.
    pub fn from_path(path: &str, model_type: &ModelType) -> Result<Self> {
        log::info!("loading topology from {}", path);

        let mut topology: Self = serde_yaml::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| anyhow!("can't read {path}: {e}"))?;

        if *model_type == ModelType::TextModel || *model_type == ModelType::AudioModel {
            // check for range expressions
            for (_worker_name, node) in topology.iter_mut() {
                let mut layers = vec![];
                for layer_name in &node.layers {
                    if let Some(caps) = LAYER_RANGE_PARSER.captures_iter(layer_name).next() {
                        let base = caps.get(1).unwrap().as_str().to_string();
                        let start = caps.get(2).unwrap().as_str().to_string().parse::<usize>()?;
                        let stop = caps.get(3).unwrap().as_str().to_string().parse::<usize>()?;

                        if stop < start {
                            return Err(anyhow!(
                                "invalid range expression {layer_name}, end must be >= start"
                            ));
                        }

                        for n in start..=stop {
                            layers.push(format!("{}{}", base, n));
                        }
                    } else {
                        layers.push(layer_name.to_string());
                    }
                }

                node.layers = layers;
            }
        }

        Ok(topology)
    }

    /// Return a set of all layer names assigned to workers in this topology.
    pub fn all_worker_layers(&self) -> std::collections::HashSet<String> {
        let mut layers = std::collections::HashSet::new();
        for node in self.0.values() {
            for layer in &node.layers {
                layers.insert(layer.clone());
            }
        }
        layers
    }

    /// Return the node serving the specified layer, or None if not found.
    pub fn get_node_for_layer(&self, layer_name: &str) -> Option<(&str, &Node)> {
        for (node_name, node) in &self.0 {
            for node_layer_name in &node.layers {
                if layer_name == node_layer_name {
                    return Some((node_name, node));
                }
            }
        }
        None
    }

    /// Returns `true` if any node in this topology has an empty `layers` list,
    /// meaning layer assignments should be computed automatically.
    pub fn needs_auto_sharding(&self) -> bool {
        !self.0.is_empty() && self.0.values().all(|n| n.layers.is_empty())
    }

    /// Automatically assign layers to nodes using the provided [`Strategy`].
    ///
    /// This is used when a topology file specifies worker addresses but
    /// leaves `layers` empty — letting the master decide distribution
    /// without requiring mDNS discovery.
    pub fn auto_assign_layers(
        &mut self,
        num_layers: usize,
        master_tflops: f64,
        layer_size_bytes: u64,
        master_max_layers: usize,
        layer_prefix: &str,
    ) {
        self.auto_assign_layers_with_strategy(
            &super::DefaultStrategy,
            num_layers,
            master_tflops,
            layer_size_bytes,
            master_max_layers,
            layer_prefix,
        )
    }

    /// Automatically assign layers to nodes using the specified [`Strategy`].
    pub fn auto_assign_layers_with_strategy(
        &mut self,
        strategy: &dyn super::Strategy,
        num_layers: usize,
        master_tflops: f64,
        layer_size_bytes: u64,
        master_max_layers: usize,
        layer_prefix: &str,
    ) {
        // Build a stable ordering of worker names (sorted alphabetically)
        // so assignments are deterministic.
        let mut worker_names: Vec<String> = self.0.keys().cloned().collect();
        worker_names.sort();

        let named_nodes: Vec<NamedNode<'_>> = worker_names
            .iter()
            .map(|name| NamedNode {
                name: name.as_str(),
                node: &self.0[name],
            })
            .collect();

        let dyn_workers: Vec<&dyn WorkerCapacity> = named_nodes.iter().map(|n| n as &dyn WorkerCapacity).collect();
        let assignments = strategy.assign_layers(
            &dyn_workers,
            num_layers,
            master_tflops,
            layer_size_bytes,
            master_max_layers,
            layer_prefix,
        );

        for (worker_idx, layers) in assignments {
            let name = &worker_names[worker_idx];
            if let Some(node) = self.0.get_mut(name) {
                node.layers = layers;
            }
        }
    }
}

impl std::ops::Deref for Topology {
    type Target = HashMap<String, Node>;
    fn deref(&self) -> &HashMap<String, Node> {
        &self.0
    }
}

impl std::ops::DerefMut for Topology {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
