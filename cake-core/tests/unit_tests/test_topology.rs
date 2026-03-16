use cake_core::cake::Topology;
use cake_core::ModelType;
use std::io::Write;

fn write_temp_topology(content: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f
}

#[test]
fn test_topology_layer_range_expansion() {
    let yaml = r#"
worker1:
  host: "192.168.1.1:10128"
  layers:
    - "model.layers.0-3"
"#;
    let f = write_temp_topology(yaml);
    let topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel).unwrap();
    let layers = topo.all_worker_layers();
    assert_eq!(layers.len(), 4);
    assert!(layers.contains("model.layers.0"));
    assert!(layers.contains("model.layers.1"));
    assert!(layers.contains("model.layers.2"));
    assert!(layers.contains("model.layers.3"));
}

#[test]
fn test_topology_multiple_workers() {
    let yaml = r#"
worker1:
  host: "192.168.1.1:10128"
  layers:
    - "model.layers.0-1"
worker2:
  host: "192.168.1.2:10128"
  layers:
    - "model.layers.2-3"
"#;
    let f = write_temp_topology(yaml);
    let topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel).unwrap();
    let layers = topo.all_worker_layers();
    assert_eq!(layers.len(), 4);
}

#[test]
fn test_topology_node_layer_ownership() {
    let yaml = r#"
worker1:
  host: "192.168.1.1:10128"
  layers:
    - "model.layers.0"
    - "model.layers.1"
"#;
    let f = write_temp_topology(yaml);
    let topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel).unwrap();

    // Check layer ownership via get_node_for_layer
    assert!(topo.get_node_for_layer("model.layers.0").is_some());
    assert!(topo.get_node_for_layer("model.layers.1").is_some());
    assert!(topo.get_node_for_layer("model.layers.2").is_none());
}

#[test]
fn test_topology_empty() {
    let topo = Topology::new();
    let layers = topo.all_worker_layers();
    assert!(layers.is_empty());
}

#[test]
fn test_topology_invalid_range() {
    // end < start should error
    let yaml = r#"
worker1:
  host: "192.168.1.1:10128"
  layers:
    - "model.layers.5-2"
"#;
    let f = write_temp_topology(yaml);
    let result = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel);
    assert!(result.is_err(), "reversed range should fail");
}

#[test]
fn test_topology_non_range_layers() {
    // Plain layer names without range expression
    let yaml = r#"
worker1:
  host: "192.168.1.1:10128"
  layers:
    - "model.embed_tokens"
    - "model.norm"
"#;
    let f = write_temp_topology(yaml);
    let topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel).unwrap();
    let layers = topo.all_worker_layers();
    assert_eq!(layers.len(), 2);
    assert!(layers.contains("model.embed_tokens"));
    assert!(layers.contains("model.norm"));
}

#[test]
fn test_topology_get_node_returns_name() {
    let yaml = r#"
bahamut:
  host: "192.168.1.1:10128"
  layers:
    - "model.layers.0"
"#;
    let f = write_temp_topology(yaml);
    let topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel).unwrap();
    let (name, node) = topo.get_node_for_layer("model.layers.0").unwrap();
    assert_eq!(name, "bahamut");
    assert_eq!(node.host, "192.168.1.1:10128");
}

#[test]
fn test_topology_image_model_no_range_expansion() {
    // ImageModel mode should NOT expand ranges
    let yaml = r#"
worker1:
  host: "192.168.1.1:10128"
  layers:
    - "model.layers.0-3"
"#;
    let f = write_temp_topology(yaml);
    let topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::ImageModel).unwrap();
    let layers = topo.all_worker_layers();
    assert_eq!(layers.len(), 1); // not expanded
    assert!(layers.contains("model.layers.0-3"));
}

#[test]
fn test_topology_node_is_layer_owner() {
    use cake_core::cake::Node;
    let node = Node {
        host: "localhost:10128".to_string(),
        description: None,
        layers: vec!["model.layers.0".to_string(), "model.layers.1".to_string()],
        vram_bytes: 0,
        tflops: 0.0,
        backend: String::new(),
        hostname: String::new(),
        os: String::new(),
    };
    assert!(node.is_text_model_layer_owner("model.layers.0.self_attn.q_proj"));
    assert!(node.is_text_model_layer_owner("model.layers.1.mlp.gate_proj"));
    assert!(!node.is_text_model_layer_owner("model.layers.2.mlp.gate_proj"));
    assert!(!node.is_text_model_layer_owner("model.layers.0")); // exact match without "." suffix should NOT match
}
