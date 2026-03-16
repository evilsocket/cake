use cake_core::cake::discovery;

#[divan::bench]
fn cluster_hash(bencher: divan::Bencher) {
    bencher.bench_local(|| discovery::cluster_hash("my-cluster-key-2026"));
}

#[divan::bench(args = [1_000_000_000u64, 12_000_000_000, 36_000_000_000])]
fn max_layers_for_size(bencher: divan::Bencher, vram: u64) {
    let worker = discovery::DiscoveredWorker {
        name: "bench-worker".into(),
        host: "127.0.0.1".into(),
        port: 10128,
        gpus: vec![discovery::GpuInfo {
            name: "NVIDIA RTX 3080".into(),
            vram_bytes: vram,
            tflops: 30.0,
        }],
        backend: "CUDA 12.4".into(),
        hostname: "bench".into(),
        os: "linux".into(),
    };
    let layer_size = 100_000_000u64; // 100 MB per layer
    bencher.bench_local(|| worker.max_layers_for_size(layer_size));
}

#[divan::bench]
fn encode_decode_packet(bencher: divan::Bencher) {
    let payload = b"{\"cluster_hash\":\"abcd1234\"}";
    bencher
        .counter(divan::counter::BytesCount::new(payload.len()))
        .bench_local(|| {
            let pkt = discovery::encode_packet(payload);
            let decoded = discovery::decode_packet(&pkt);
            assert!(decoded.is_some());
            pkt // return owned value to avoid borrow issue
        });
}

#[divan::bench]
fn total_tflops_reported(bencher: divan::Bencher) {
    let worker = discovery::DiscoveredWorker {
        name: "bench-worker".into(),
        host: "127.0.0.1".into(),
        port: 10128,
        gpus: vec![
            discovery::GpuInfo {
                name: "NVIDIA TITAN X".into(),
                vram_bytes: 12 * 1024 * 1024 * 1024,
                tflops: 22.0,
            },
            discovery::GpuInfo {
                name: "NVIDIA TITAN X".into(),
                vram_bytes: 12 * 1024 * 1024 * 1024,
                tflops: 22.0,
            },
        ],
        backend: "CUDA 12.4".into(),
        hostname: "bench".into(),
        os: "linux".into(),
    };
    bencher.bench_local(|| worker.total_tflops());
}

#[divan::bench]
fn total_tflops_estimated(bencher: divan::Bencher) {
    let worker = discovery::DiscoveredWorker {
        name: "bench-worker".into(),
        host: "127.0.0.1".into(),
        port: 10128,
        gpus: vec![discovery::GpuInfo {
            name: "Apple M3 Pro".into(),
            vram_bytes: 36 * 1024 * 1024 * 1024,
            tflops: 0.0, // force fallback estimation
        }],
        backend: "Metal".into(),
        hostname: "bench".into(),
        os: "macos".into(),
    };
    bencher.bench_local(|| worker.total_tflops());
}
