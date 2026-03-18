use cake_core::cake::{NamedNode, Node, Topology};
use cake_core::cake::sharding::{Strategy, DefaultStrategy, WorkerCapacity};

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

fn make_empty_topology(num_workers: usize) -> Topology {
    let mut topo = Topology::new();
    for w in 0..num_workers {
        topo.insert(
            format!("worker-{w}"),
            Node {
                host: format!("192.168.50.{}:10128", 100 + w),
                description: Some(format!("NVIDIA GPU {w}")),
                layers: vec![],
                vram_bytes: 24 * 1024 * 1024 * 1024,
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

#[divan::bench(args = [24, 64, 128])]
fn auto_assign_layers(bencher: divan::Bencher, num_layers: usize) {
    bencher.bench_local(|| {
        let mut topo = make_empty_topology(3);
        topo.auto_assign_layers(num_layers, 30.0, 100_000_000, usize::MAX, "model.layers");
    });
}

#[divan::bench(args = [1, 3, 8])]
fn compute_assignments_named_nodes(bencher: divan::Bencher, num_workers: usize) {
    let topo = make_empty_topology(num_workers);
    let worker_names: Vec<String> = topo.keys().cloned().collect();
    let named: Vec<NamedNode<'_>> = worker_names
        .iter()
        .map(|n| NamedNode { name: n.as_str(), node: &topo[n] })
        .collect();
    let dyn_workers: Vec<&dyn WorkerCapacity> = named.iter().map(|n| n as &dyn WorkerCapacity).collect();
    bencher.bench_local(|| {
        DefaultStrategy.assign_layers(&dyn_workers, 64, 30.0, 100_000_000, usize::MAX, "model.layers")
    });
}

#[divan::bench(args = [1, 4, 8])]
fn estimate_tflops_for_gpus(bencher: divan::Bencher, num_gpus: usize) {
    let gpus: Vec<_> = (0..num_gpus).map(|i| cake_core::cake::sharding::discovery::GpuInfo {
        name: format!("NVIDIA RTX {}", 3080 + i),
        vram_bytes: 24 * 1024 * 1024 * 1024,
        tflops: 30.0,
    }).collect();
    bencher.bench_local(|| cake_core::cake::sharding::estimate_tflops_for_gpus(&gpus));
}

#[divan::bench(args = [1, 4, 8])]
fn max_layers_for_gpus(bencher: divan::Bencher, num_gpus: usize) {
    let gpus: Vec<_> = (0..num_gpus).map(|i| cake_core::cake::sharding::discovery::GpuInfo {
        name: format!("NVIDIA RTX {}", 3080 + i),
        vram_bytes: 24 * 1024 * 1024 * 1024,
        tflops: 30.0,
    }).collect();
    bencher.bench_local(|| cake_core::cake::sharding::max_layers_for_gpus(&gpus, 100_000_000));
}
