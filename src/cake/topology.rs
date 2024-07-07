use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Node {
    pub host: String,
    pub description: Option<String>,
    pub layers: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Topology(HashMap<String, Node>);

impl Topology {
    pub fn from_path(path: &str) -> Result<Self> {
        serde_yaml::from_str(&std::fs::read_to_string(path)?).map_err(|e| anyhow!(e))
    }

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
