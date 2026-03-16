use cake_core::cake::{Node, Topology};

fn make_topology(num_layers: usize) -> Topology {
    let mut topo = Topology::new();
    // Split layers across 3 workers
    let per_worker = num_layers / 3;
    for w in 0..3 {
        let start = w * per_worker;
        let end = if w == 2 { num_layers } else { (w + 1) * per_worker };
        let layers: Vec<String> = (start..end).map(|i| format!("model.layers.{i}")).collect();
        topo.insert(
            format!("worker-{w}"),
            Node {
                host: format!("192.168.50.{}:10128", 100 + w),
                description: None,
                layers,
                vram_bytes: 16 * 1024 * 1024 * 1024,
                tflops: 30.0,
                backend: "CUDA 12.4".into(),
                hostname: format!("worker-{w}"),
                os: "linux".into(),
            },
        );
    }
    topo
}

#[divan::bench(args = [24, 64, 128])]
fn get_node_for_layer(bencher: divan::Bencher, num_layers: usize) {
    let topo = make_topology(num_layers);
    // Look up a layer in the middle
    let target = format!("model.layers.{}", num_layers / 2);
    bencher.bench_local(|| topo.get_node_for_layer(&target));
}

#[divan::bench(args = [24, 64, 128])]
fn all_worker_layers(bencher: divan::Bencher, num_layers: usize) {
    let topo = make_topology(num_layers);
    bencher.bench_local(|| topo.all_worker_layers());
}

#[divan::bench(args = [24, 64, 128])]
fn is_text_model_layer_owner(bencher: divan::Bencher, num_layers: usize) {
    let topo = make_topology(num_layers);
    let target = format!("model.layers.{}.self_attn.q_proj", num_layers / 2);
    bencher.bench_local(|| {
        for (_name, node) in topo.iter() {
            if node.is_text_model_layer_owner(&target) {
                return true;
            }
        }
        false
    });
}
