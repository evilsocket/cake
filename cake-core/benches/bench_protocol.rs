use cake_core::cake::{Message, WorkerInfo};
use candle_core::DType;
use divan::counter::BytesCount;

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

// ── Larger tensor benchmarks (realistic inference sizes) ──────

#[divan::bench(args = [1024, 4096, 16384])]
fn large_tensor_encode(bencher: divan::Bencher, size: usize) {
    let t = make_f16_tensor(size);
    let msg = Message::from_tensor(&t);
    bencher
        .counter(divan::counter::BytesCount::new(size * 2))
        .bench_local(|| msg.to_bytes().unwrap());
}

#[divan::bench(args = [1024, 4096, 16384])]
fn large_tensor_decode(bencher: divan::Bencher, size: usize) {
    let t = make_f16_tensor(size);
    let msg = Message::from_tensor(&t);
    let bytes = msg.to_bytes().unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(bytes.len()))
        .bench_local(|| Message::from_bytes(&bytes).unwrap());
}

#[divan::bench(args = [3, 8, 14])]
fn batch_layers_encode(bencher: divan::Bencher, num_layers: usize) {
    let t = make_f16_tensor(1024);
    let batch: Vec<(String, usize, usize)> = (0..num_layers)
        .map(|i| (format!("model.layers.{i}"), i, i))
        .collect();
    let msg = Message::from_batch(&t, batch);
    bencher
        .counter(divan::counter::BytesCount::new(1024usize * 2))
        .bench_local(|| msg.to_bytes().unwrap());
}

#[divan::bench]
fn buf_reuse_pipe_roundtrip(bencher: divan::Bencher) {
    let t = make_f16_tensor(4096);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut write_buf = Vec::with_capacity(64 * 1024);
    let mut read_buf = Vec::with_capacity(64 * 1024);
    bencher
        .counter(divan::counter::BytesCount::new(4096usize * 2))
        .bench_local(|| {
            rt.block_on(async {
                let (mut w, mut r) = tokio::io::duplex(128 * 1024);
                Message::from_tensor(&t)
                    .to_writer_buf(&mut w, &mut write_buf)
                    .await
                    .unwrap();
                drop(w);
                Message::from_reader_buf(&mut r, &mut read_buf).await.unwrap()
            })
        });
}

// ── Model data transfer benchmarks ───────────────────────────────

#[divan::bench(args = [1024, 65536, 1048576])]
fn zstd_compress_chunk(bencher: divan::Bencher, size: usize) {
    // Simulate safetensors data (pseudo-random F16 weights)
    let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    bencher
        .counter(BytesCount::new(size))
        .bench_local(|| zstd::encode_all(data.as_slice(), 1).unwrap());
}

#[divan::bench(args = [1024, 65536, 1048576])]
fn zstd_decompress_chunk(bencher: divan::Bencher, size: usize) {
    let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    let compressed = zstd::encode_all(data.as_slice(), 1).unwrap();
    bencher
        .counter(BytesCount::new(compressed.len()))
        .bench_local(|| zstd::decode_all(compressed.as_slice()).unwrap());
}

#[divan::bench(args = [1024, 65536, 1048576])]
fn crc32_checksum(bencher: divan::Bencher, size: usize) {
    let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    bencher
        .counter(BytesCount::new(size))
        .bench_local(|| crc32fast::hash(&data));
}

#[divan::bench]
fn model_data_chunk_roundtrip_compressed() {
    let data: Vec<u8> = (0..16384).map(|i| (i % 251) as u8).collect();
    let compressed = zstd::encode_all(data.as_slice(), 1).unwrap();
    let checksum = crc32fast::hash(&compressed);
    let msg = Message::ModelDataChunk {
        filename: "model.safetensors".into(),
        offset: 0,
        total_size: 1_000_000,
        compressed: true,
        checksum,
        data: compressed,
    };
    let bytes = msg.to_bytes().unwrap();
    let _decoded = Message::from_bytes(&bytes).unwrap();
}

#[divan::bench]
fn model_data_chunk_roundtrip_uncompressed() {
    let data: Vec<u8> = (0..16384).map(|i| (i % 251) as u8).collect();
    let checksum = crc32fast::hash(&data);
    let msg = Message::ModelDataChunk {
        filename: "model.safetensors".into(),
        offset: 0,
        total_size: 1_000_000,
        compressed: false,
        checksum,
        data,
    };
    let bytes = msg.to_bytes().unwrap();
    let _decoded = Message::from_bytes(&bytes).unwrap();
}
