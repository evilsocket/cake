use std::collections::HashMap;

use crate::ModelType;
use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

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
}

/// The topology is a worker-name -> worker-info map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topology(HashMap<String, Node>);

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

        if *model_type == ModelType::TextModel {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(host: &str, layers: &[&str]) -> Node {
        Node {
            host: host.to_string(),
            description: None,
            layers: layers.iter().map(|s| s.to_string()).collect(),
            vram_bytes: 0,
            tflops: 0.0,
            backend: String::new(),
            hostname: String::new(),
            os: String::new(),
        }
    }

    #[test]
    fn test_empty_topology() {
        let topo = Topology::new();
        assert!(topo.is_empty());
        assert!(topo.all_worker_layers().is_empty());
        assert!(topo.get_node_for_layer("model.layers.0").is_none());
    }

    #[test]
    fn test_node_layer_ownership() {
        let node = make_node("worker1:10128", &["model.layers.0", "model.layers.1"]);

        // Full layer name with sub-path matches
        assert!(node.is_text_model_layer_owner("model.layers.0.self_attn"));
        assert!(node.is_text_model_layer_owner("model.layers.1.mlp"));

        // Layer not assigned
        assert!(!node.is_text_model_layer_owner("model.layers.2.self_attn"));

        // Exact name without trailing dot doesn't match
        assert!(!node.is_text_model_layer_owner("model.layers.0"));
    }

    #[test]
    fn test_get_node_for_layer() {
        let mut topo = Topology::new();
        topo.insert("gpu1".into(), make_node("10.0.0.1:10128", &["model.layers.0", "model.layers.1"]));
        topo.insert("gpu2".into(), make_node("10.0.0.2:10128", &["model.layers.2"]));

        let (name, node) = topo.get_node_for_layer("model.layers.0").unwrap();
        assert_eq!(name, "gpu1");
        assert_eq!(node.host, "10.0.0.1:10128");

        let (name, node) = topo.get_node_for_layer("model.layers.2").unwrap();
        assert_eq!(name, "gpu2");
        assert_eq!(node.host, "10.0.0.2:10128");

        assert!(topo.get_node_for_layer("model.layers.99").is_none());
    }

    #[test]
    fn test_all_worker_layers() {
        let mut topo = Topology::new();
        topo.insert("w1".into(), make_node("a:1", &["model.layers.0", "model.layers.1"]));
        topo.insert("w2".into(), make_node("b:1", &["model.layers.2"]));

        let layers = topo.all_worker_layers();
        assert_eq!(layers.len(), 3);
        assert!(layers.contains("model.layers.0"));
        assert!(layers.contains("model.layers.1"));
        assert!(layers.contains("model.layers.2"));
    }

    #[test]
    fn test_topology_yaml_parsing() {
        let yaml = r#"
gpu1:
  host: "10.0.0.1:10128"
  layers:
    - "model.layers.0-2"
gpu2:
  host: "10.0.0.2:10128"
  layers:
    - "model.layers.3"
    - "model.layers.4"
"#;
        let mut topo: Topology = serde_yaml::from_str(yaml).unwrap();

        // Before range expansion, gpu1 has the raw range string
        assert_eq!(topo["gpu1"].layers.len(), 1);
        assert_eq!(topo["gpu1"].layers[0], "model.layers.0-2");

        // Simulate range expansion (from_path does this for TextModel)
        let re = regex::Regex::new(r"(?m)^(.+[^\d])(\d+)-(\d+)$").unwrap();
        for (_name, node) in topo.iter_mut() {
            let mut expanded = vec![];
            for layer in &node.layers {
                if let Some(caps) = re.captures(layer) {
                    let base = caps.get(1).unwrap().as_str();
                    let start: usize = caps.get(2).unwrap().as_str().parse().unwrap();
                    let stop: usize = caps.get(3).unwrap().as_str().parse().unwrap();
                    for n in start..=stop {
                        expanded.push(format!("{}{}", base, n));
                    }
                } else {
                    expanded.push(layer.clone());
                }
            }
            node.layers = expanded;
        }

        // After expansion: gpu1 should have 3 layers
        assert_eq!(topo["gpu1"].layers, vec![
            "model.layers.0", "model.layers.1", "model.layers.2"
        ]);
        // gpu2 unchanged
        assert_eq!(topo["gpu2"].layers, vec!["model.layers.3", "model.layers.4"]);
    }

    #[test]
    fn test_topology_component_layers() {
        // Non-text-model topology (Flux/LTX/HunyuanVideo style)
        let yaml = r#"
worker1:
  host: "10.0.0.1:10128"
  layers:
    - "flux-t5"
    - "flux-clip"
worker2:
  host: "10.0.0.2:10128"
  layers:
    - "flux-transformer"
"#;
        let topo: Topology = serde_yaml::from_str(yaml).unwrap();

        assert!(topo.get_node_for_layer("flux-t5").is_some());
        assert!(topo.get_node_for_layer("flux-transformer").is_some());
        assert!(topo.get_node_for_layer("flux-vae").is_none());
    }

    #[test]
    fn test_node_optional_fields_default() {
        let yaml = r#"
worker:
  host: "10.0.0.1:10128"
  layers: ["layer0"]
"#;
        let topo: Topology = serde_yaml::from_str(yaml).unwrap();
        let node = &topo["worker"];
        assert_eq!(node.vram_bytes, 0);
        assert_eq!(node.tflops, 0.0);
        assert_eq!(node.backend, "");
        assert!(node.description.is_none());
    }
}
