use cake_core::cake::{Message, WorkerInfo};
use candle_core::DType;

fn make_f16_tensor(size: usize) -> candle_core::Tensor {
    super::bench_helpers::make_tensor(&[1, size], 60)
        .to_dtype(DType::F16)
        .unwrap()
}

#[divan::bench]
fn single_op_encode(bencher: divan::Bencher) {
    let t = make_f16_tensor(1024);
    let msg = Message::single_op("model.layers.5", &t, 42, 7);
    bencher
        .counter(divan::counter::BytesCount::new(1024usize * 2))
        .bench_local(|| msg.to_bytes().unwrap());
}

#[divan::bench]
fn single_op_decode(bencher: divan::Bencher) {
    let t = make_f16_tensor(1024);
    let msg = Message::single_op("model.layers.5", &t, 42, 7);
    let bytes = msg.to_bytes().unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(bytes.len()))
        .bench_local(|| Message::from_bytes(&bytes).unwrap());
}

#[divan::bench]
fn batch_encode(bencher: divan::Bencher) {
    let t = make_f16_tensor(4096);
    let batch = vec![
        ("model.layers.0".into(), 0usize, 0usize),
        ("model.layers.1".into(), 1, 1),
        ("model.layers.2".into(), 2, 2),
    ];
    let msg = Message::from_batch(&t, batch);
    bencher
        .counter(divan::counter::BytesCount::new(4096usize * 2))
        .bench_local(|| msg.to_bytes().unwrap());
}

#[divan::bench]
fn batch_decode(bencher: divan::Bencher) {
    let t = make_f16_tensor(4096);
    let batch = vec![
        ("model.layers.0".into(), 0usize, 0usize),
        ("model.layers.1".into(), 1, 1),
        ("model.layers.2".into(), 2, 2),
    ];
    let msg = Message::from_batch(&t, batch);
    let bytes = msg.to_bytes().unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(bytes.len()))
        .bench_local(|| Message::from_bytes(&bytes).unwrap());
}

#[divan::bench]
fn hello_roundtrip(bencher: divan::Bencher) {
    bencher.bench_local(|| {
        let bytes = Message::Hello.to_bytes().unwrap();
        Message::from_bytes(&bytes).unwrap()
    });
}

#[divan::bench]
fn worker_info_roundtrip(bencher: divan::Bencher) {
    let info = WorkerInfo {
        version: "0.1.0".into(),
        dtype: "F16".into(),
        os: "linux".into(),
        arch: "x86_64".into(),
        device: "cuda".into(),
        device_idx: 0,
        latency: 42,
    };
    let msg = Message::WorkerInfo(info);
    bencher.bench_local(|| {
        let enc = msg.to_bytes().unwrap();
        Message::from_bytes(&enc).unwrap()
    });
}

#[divan::bench]
fn layer_assignment_roundtrip(bencher: divan::Bencher) {
    let layers: Vec<String> = (0..24).map(|i| format!("model.layers.{i}")).collect();
    let msg = Message::LayerAssignment {
        layers,
        model_hash: "abc12345".into(),
    };
    bencher.bench_local(|| {
        let enc = msg.to_bytes().unwrap();
        Message::from_bytes(&enc).unwrap()
    });
}

#[divan::bench]
fn full_pipe_roundtrip(bencher: divan::Bencher) {
    let t = make_f16_tensor(4096);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(4096usize * 2))
        .bench_local(|| {
            rt.block_on(async {
                let (mut w, mut r) = tokio::io::duplex(128 * 1024);
                Message::from_tensor(&t).to_writer(&mut w).await.unwrap();
                drop(w);
                Message::from_reader(&mut r).await.unwrap()
            })
        });
}
