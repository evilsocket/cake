use std::collections::HashMap;

use anyhow::Result;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref LAYER_RANGE_PARSER: Regex = Regex::new(r"(?m)^(.+[^\d])(\d+)-(\d+)$").unwrap();
}

#[derive(Serialize, Deserialize)]
pub struct Node {
    pub host: String,
    pub description: Option<String>,
    pub layers: Vec<String>,
}

impl Node {
    pub fn is_layer_owner(&self, full_layer_name: &str) -> bool {
        for prefix in &self.layers {
            if full_layer_name.starts_with(&format!("{}.", prefix)) {
                return true;
            }
        }
        false
    }
}

#[derive(Serialize, Deserialize)]
pub struct Topology(HashMap<String, Node>);

impl Topology {
    pub fn from_path(path: &str) -> Result<Self> {
        let mut topology: Self =
            serde_yaml::from_str(&std::fs::read_to_string(path)?).map_err(|e| anyhow!(e))?;

        // check for range expressions
        for (_worker_name, node) in topology.iter_mut() {
            let mut layers = vec![];
            for layer_name in &node.layers {
                if let Some(caps) = LAYER_RANGE_PARSER.captures_iter(layer_name).next() {
                    let base = caps.get(1).unwrap().as_str().to_string();
                    let start = caps.get(2).unwrap().as_str().to_string().parse::<usize>()?;
                    let stop = caps.get(3).unwrap().as_str().to_string().parse::<usize>()?;

                    if stop <= start {
                        return Err(anyhow!(
                            "invalid range expression {layer_name}, end must be > start"
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

        Ok(topology)
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

impl std::ops::DerefMut for Topology {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
