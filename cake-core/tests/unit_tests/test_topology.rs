use cake_core::cake::{Node, Topology};
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

// ── auto-sharding ──────────────────────────────────────────────

fn make_node(host: &str, vram_gb: f64, tflops: f64, description: &str) -> Node {
    Node {
        host: host.to_string(),
        description: Some(description.to_string()),
        layers: vec![],
        vram_bytes: (vram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
        tflops,
        backend: String::new(),
        hostname: String::new(),
        os: "linux".to_string(),
    }
}

#[test]
fn test_needs_auto_sharding_empty_topology() {
    let topo = Topology::new();
    assert!(!topo.needs_auto_sharding());
}

#[test]
fn test_needs_auto_sharding_all_empty_layers() {
    let mut topo = Topology::new();
    topo.insert("w1".into(), make_node("192.168.1.1:10128", 24.0, 30.0, "RTX 3090"));
    topo.insert("w2".into(), make_node("192.168.1.2:10128", 36.0, 14.0, "Apple M3 Pro"));
    assert!(topo.needs_auto_sharding());
}

#[test]
fn test_needs_auto_sharding_with_layers() {
    let mut topo = Topology::new();
    let mut node = make_node("192.168.1.1:10128", 24.0, 30.0, "GPU");
    node.layers = vec!["model.layers.0".into()];
    topo.insert("w1".into(), node);
    assert!(!topo.needs_auto_sharding());
}

#[test]
fn test_needs_auto_sharding_mixed_empty_and_filled() {
    // If some nodes have layers and some don't, it should not auto-shard
    let mut topo = Topology::new();
    let mut node_with = make_node("192.168.1.1:10128", 24.0, 30.0, "GPU");
    node_with.layers = vec!["model.layers.0".into()];
    topo.insert("w1".into(), node_with);
    topo.insert("w2".into(), make_node("192.168.1.2:10128", 24.0, 30.0, "GPU"));
    assert!(!topo.needs_auto_sharding());
}

#[test]
fn test_auto_assign_layers_single_worker() {
    let mut topo = Topology::new();
    topo.insert("w1".into(), make_node("192.168.1.1:10128", 24.0, 30.0, "NVIDIA RTX 3090"));
    topo.auto_assign_layers(24, 30.0, 0, usize::MAX, "model.layers");

    let w1 = topo.get("w1").unwrap();
    assert!(!w1.layers.is_empty(), "worker should have layers assigned");
    // With equal TFLOPS (30 each), worker gets ~12 layers
    assert_eq!(w1.layers.len(), 12);
    assert_eq!(w1.layers[0], "model.layers.0");
}

#[test]
fn test_auto_assign_layers_two_workers() {
    let mut topo = Topology::new();
    topo.insert("fast".into(), make_node("192.168.1.1:10128", 24.0, 40.0, "NVIDIA A100"));
    topo.insert("slow".into(), make_node("192.168.1.2:10128", 36.0, 14.0, "Apple M3 Pro"));

    topo.auto_assign_layers(24, 10.0, 0, usize::MAX, "model.layers");

    let fast = topo.get("fast").unwrap();
    let slow = topo.get("slow").unwrap();
    assert!(!fast.layers.is_empty());
    assert!(!slow.layers.is_empty());

    // Fast worker (40 TFLOPS) should get more layers than slow (14 TFLOPS)
    assert!(
        fast.layers.len() >= slow.layers.len(),
        "fast ({}) >= slow ({})",
        fast.layers.len(),
        slow.layers.len()
    );

    // Total assigned should be less than num_layers (master keeps some)
    let total = fast.layers.len() + slow.layers.len();
    assert!(total <= 24);
    assert!(total > 0);
}

#[test]
fn test_auto_assign_layers_deterministic() {
    // Same topology should produce same assignments every time
    let build = || {
        let mut topo = Topology::new();
        topo.insert("b".into(), make_node("192.168.1.2:10128", 24.0, 20.0, "GPU B"));
        topo.insert("a".into(), make_node("192.168.1.1:10128", 16.0, 30.0, "GPU A"));
        topo.auto_assign_layers(24, 10.0, 0, usize::MAX, "model.layers");
        topo
    };
    let t1 = build();
    let t2 = build();
    assert_eq!(
        t1.get("a").unwrap().layers,
        t2.get("a").unwrap().layers,
    );
    assert_eq!(
        t1.get("b").unwrap().layers,
        t2.get("b").unwrap().layers,
    );
}

#[test]
fn test_auto_assign_layers_all_accounted() {
    let mut topo = Topology::new();
    topo.insert("w1".into(), make_node("192.168.1.1:10128", 24.0, 20.0, "GPU"));
    topo.insert("w2".into(), make_node("192.168.1.2:10128", 24.0, 20.0, "GPU"));
    topo.auto_assign_layers(32, 10.0, 100_000_000, usize::MAX, "model.layers");

    let total: usize = topo.values().map(|n| n.layers.len()).sum();
    assert!(total <= 32);
    // All assigned layer names should be valid
    for node in topo.values() {
        for layer in &node.layers {
            let num: usize = layer.strip_prefix("model.layers.").unwrap().parse().unwrap();
            assert!(num < 32);
        }
    }
}

#[test]
fn test_auto_assign_from_topology_file() {
    // Simulate a topology file with empty layers
    let yaml = r#"
bahamut:
  host: "192.168.50.147:10128"
  description: "2x TITAN X Pascal"
  layers: []
  vram_bytes: 25769803776
  tflops: 40.0
stevie:
  host: "192.168.50.32:10128"
  description: "Apple M3 Pro"
  layers: []
  vram_bytes: 38654705664
  tflops: 14.0
"#;
    let f = write_temp_topology(yaml);
    let mut topo = Topology::from_path(f.path().to_str().unwrap(), &ModelType::TextModel).unwrap();
    assert!(topo.needs_auto_sharding());

    topo.auto_assign_layers(24, 30.0, 0, usize::MAX, "model.layers");

    assert!(!topo.needs_auto_sharding());
    let bahamut = topo.get("bahamut").unwrap();
    let stevie = topo.get("stevie").unwrap();
    assert!(!bahamut.layers.is_empty());
    assert!(!stevie.layers.is_empty());
    // bahamut (40 TFLOPS) should get more than stevie (14 TFLOPS)
    assert!(bahamut.layers.len() >= stevie.layers.len());
}
