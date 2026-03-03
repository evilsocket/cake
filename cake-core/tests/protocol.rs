//! Protocol-level integration tests and benchmarks for Cake worker/client communication.
//!
//! Uses a mock worker that echoes tensors without loading any model.
//!
//! Run tests:     cargo test --test protocol
//! Run benchmarks: cargo test --test protocol -- --ignored --nocapture

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use safetensors::View;
use tokio::net::{TcpListener, TcpStream};

use cake_core::cake::{Message, WorkerInfo};

// ============================================================================
// Helpers
// ============================================================================

fn make_f16_tensor(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
}

fn mock_worker_info(latency: u128) -> WorkerInfo {
    WorkerInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        dtype: "F16".to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        device: "cpu".to_string(),
        device_idx: 0,
        latency,
    }
}

// ============================================================================
// Mock Worker
// ============================================================================

/// A mock worker that speaks the Cake protocol but echoes tensors back
/// instead of running model inference.
struct MockWorker {
    listener: TcpListener,
}

impl MockWorker {
    async fn bind() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        Self { listener }
    }

    fn addr(&self) -> SocketAddr {
        self.listener.local_addr().unwrap()
    }

    /// Handle exactly one client connection, then return.
    async fn handle_one(self, cluster_key: Option<&str>) -> anyhow::Result<()> {
        let (mut socket, _client) = self.listener.accept().await?;

        // Auth
        if let Some(key) = cluster_key {
            cake_core::cake::auth::authenticate_as_worker(&mut socket, key).await?;
        }

        // First message
        let (_, first) = Message::from_reader(&mut socket).await?;
        match first {
            Message::Hello => {
                Message::WorkerInfo(mock_worker_info(0))
                    .to_writer(&mut socket)
                    .await?;
            }
            other => return Err(anyhow::anyhow!("unexpected first message: {:?}", other)),
        }

        // Message loop
        loop {
            let result = Message::from_reader(&mut socket).await;
            let (_, msg) = match result {
                Ok(m) => m,
                Err(_) => break, // client disconnected
            };

            match msg {
                Message::SingleOp { x, .. } => {
                    log::debug!("single op");
                    Message::Tensor(x).to_writer(&mut socket).await?;
                }
                Message::Batch { x, .. } => {
                    log::debug!("batch");
                    Message::Tensor(x).to_writer(&mut socket).await?;
                }
                Message::Goodbye => {
                    log::debug!("goodbye");
                    Message::WorkerInfo(mock_worker_info(0))
                        .to_writer(&mut socket)
                        .await?;
                }
                other => {
                    return Err(anyhow::anyhow!("unhandled in mock loop: {:?}", other));
                }
            }
        }

        Ok(())
    }
}

/// Helper: connect to addr, do Hello handshake, return stream + WorkerInfo.
async fn connect_and_handshake(
    addr: SocketAddr,
    cluster_key: Option<&str>,
) -> (TcpStream, WorkerInfo) {
    let mut stream = TcpStream::connect(addr).await.unwrap();

    if let Some(key) = cluster_key {
        cake_core::cake::auth::authenticate_as_master(&mut stream, key)
            .await
            .unwrap();
    }

    Message::Hello.to_writer(&mut stream).await.unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    let info = match resp {
        Message::WorkerInfo(wi) => wi,
        other => panic!("expected WorkerInfo, got {:?}", other),
    };
    (stream, info)
}

// ============================================================================
// Connection & Handshake Tests
// ============================================================================

#[tokio::test]
async fn test_handshake_hello_worker_info() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(None).await });

    let (mut stream, info) = connect_and_handshake(addr, None).await;

    assert_eq!(info.device, "cpu");
    assert!(!info.version.is_empty());

    // Goodbye
    Message::Goodbye.to_writer(&mut stream).await.unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    assert!(matches!(resp, Message::WorkerInfo(_)));

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn test_handshake_with_auth() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let key = "test-cluster-key-123";
    let key_owned = key.to_string();
    let worker = tokio::spawn(async move { mock.handle_one(Some(&key_owned)).await });

    let (stream, info) = connect_and_handshake(addr, Some(key)).await;
    assert_eq!(info.device, "cpu");

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn test_handshake_auth_wrong_key() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(Some("correct-key")).await });

    let mut stream = TcpStream::connect(addr).await.unwrap();
    let result = cake_core::cake::auth::authenticate_as_master(&mut stream, "wrong-key").await;
    assert!(result.is_err());

    drop(stream);
    // Worker should also fail
    let worker_result = worker.await.unwrap();
    assert!(worker_result.is_err());
}

// ============================================================================
// Forward (Echo) Tests
// ============================================================================

#[tokio::test]
async fn test_single_op_echo() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(None).await });

    let (mut stream, _) = connect_and_handshake(addr, None).await;

    let tensor = make_f16_tensor(&[1, 128]);
    let orig_bytes: Vec<u8> = tensor.data().to_vec();

    Message::single_op("model.layers.0", &tensor, 0, 0)
        .to_writer(&mut stream)
        .await
        .unwrap();

    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    match resp {
        Message::Tensor(raw) => {
            let recovered = raw.to_tensor(&Device::Cpu).unwrap();
            assert_eq!(recovered.dtype(), DType::F16);
            assert_eq!(recovered.shape().dims(), &[1, 128]);
            assert_eq!(recovered.data().to_vec(), orig_bytes);
        }
        other => panic!("expected Tensor, got {:?}", other),
    }

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn test_batch_echo() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(None).await });

    let (mut stream, _) = connect_and_handshake(addr, None).await;

    let tensor = make_f16_tensor(&[1, 64]);
    let orig_bytes: Vec<u8> = tensor.data().to_vec();
    let batch = vec![
        ("model.layers.0".into(), 0usize, 0usize),
        ("model.layers.1".into(), 1, 1),
    ];

    Message::from_batch(&tensor, batch)
        .to_writer(&mut stream)
        .await
        .unwrap();

    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    match resp {
        Message::Tensor(raw) => {
            let recovered = raw.to_tensor(&Device::Cpu).unwrap();
            assert_eq!(recovered.shape().dims(), &[1, 64]);
            assert_eq!(recovered.data().to_vec(), orig_bytes);
        }
        other => panic!("expected Tensor, got {:?}", other),
    }

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn test_multiple_sequential_ops() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(None).await });

    let (mut stream, _) = connect_and_handshake(addr, None).await;

    for i in 0..10 {
        let tensor = make_f16_tensor(&[1, 32 + i]);
        let orig_bytes: Vec<u8> = tensor.data().to_vec();

        Message::single_op("model.layers.0", &tensor, i, 0)
            .to_writer(&mut stream)
            .await
            .unwrap();

        let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
        match resp {
            Message::Tensor(raw) => {
                assert_eq!(raw.data, orig_bytes);
            }
            other => panic!("iteration {}: expected Tensor, got {:?}", i, other),
        }
    }

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn test_goodbye_and_continue() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(None).await });

    let (mut stream, _) = connect_and_handshake(addr, None).await;

    // First op
    let tensor = make_f16_tensor(&[1, 64]);
    Message::single_op("model.layers.0", &tensor, 0, 0)
        .to_writer(&mut stream)
        .await
        .unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    assert!(matches!(resp, Message::Tensor(_)));

    // Goodbye
    Message::Goodbye.to_writer(&mut stream).await.unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    assert!(matches!(resp, Message::WorkerInfo(_)));

    // Another op after goodbye (worker should still be alive)
    Message::single_op("model.layers.0", &tensor, 1, 0)
        .to_writer(&mut stream)
        .await
        .unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    assert!(matches!(resp, Message::Tensor(_)));

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn test_large_tensor_transfer() {
    let mock = MockWorker::bind().await;
    let addr = mock.addr();
    let worker = tokio::spawn(async move { mock.handle_one(None).await });

    let (mut stream, _) = connect_and_handshake(addr, None).await;

    // Realistic hidden state size: (1, 5120) x F16 = 10,240 bytes
    let tensor = make_f16_tensor(&[1, 5120]);
    let orig_bytes: Vec<u8> = tensor.data().to_vec();
    assert_eq!(orig_bytes.len(), 10240);

    Message::single_op("model.layers.0", &tensor, 0, 0)
        .to_writer(&mut stream)
        .await
        .unwrap();

    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    match resp {
        Message::Tensor(raw) => {
            assert_eq!(raw.data, orig_bytes);
        }
        other => panic!("expected Tensor, got {:?}", other),
    }

    drop(stream);
    let _ = worker.await;
}

// ============================================================================
// Benchmarks (run with: cargo test --test protocol -- --ignored --nocapture)
// ============================================================================

fn human_bytes(n: usize) -> String {
    if n >= 1024 * 1024 {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{} B", n)
    }
}

fn print_bench(label: &str, size_bytes: usize, duration: Duration, iterations: usize) {
    let total_bytes = size_bytes * iterations;
    let throughput = total_bytes as f64 / duration.as_secs_f64();
    let per_op = duration / iterations as u32;
    println!(
        "  {:<45} {:>8} x {:>5} = {:>10} in {:>8.2}ms  ({}/s, {:.1}us/op)",
        label,
        human_bytes(size_bytes),
        iterations,
        human_bytes(total_bytes),
        duration.as_secs_f64() * 1000.0,
        human_bytes(throughput as usize),
        per_op.as_secs_f64() * 1_000_000.0,
    );
}

#[tokio::test]
#[ignore]
async fn bench_serialization() {
    println!("\n=== Serialization Benchmark (to_writer + from_reader over duplex) ===\n");

    let sizes: &[(usize, &str)] = &[
        (64, "[1,64]"),
        (512, "[1,512]"),
        (2048, "[1,2048]"),
        (5120, "[1,5120]"),
        (8192, "[1,8192]"),
    ];
    let iterations = 1000;

    for &(hidden_dim, label) in sizes {
        let tensor = make_f16_tensor(&[1, hidden_dim]);
        let tensor_bytes = hidden_dim * 2;

        let (mut writer, mut reader) = tokio::io::duplex(256 * 1024);

        let msg_for_write = Message::single_op("model.layers.0", &tensor, 0, 0);

        // Spawn writer in background
        let write_handle = tokio::spawn(async move {
            for _ in 0..iterations {
                // Re-create to include serialization cost
                msg_for_write.to_writer(&mut writer).await.unwrap();
            }
            drop(writer);
        });

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Message::from_reader(&mut reader).await.unwrap();
        }
        let elapsed = start.elapsed();

        write_handle.await.unwrap();

        print_bench(
            &format!("SingleOp F16 {}", label),
            tensor_bytes,
            elapsed,
            iterations,
        );
    }
}

#[tokio::test]
#[ignore]
async fn bench_tcp_roundtrip() {
    println!("\n=== TCP Round-Trip Benchmark (loopback, mock echo worker) ===\n");

    let sizes: &[(usize, &str)] = &[
        (64, "[1,64]"),
        (512, "[1,512]"),
        (2048, "[1,2048]"),
        (5120, "[1,5120]"),
        (8192, "[1,8192]"),
    ];
    let warmup = 50;
    let iterations = 500;

    for &(hidden_dim, label) in sizes {
        let mock = MockWorker::bind().await;
        let addr = mock.addr();
        let worker = tokio::spawn(async move { mock.handle_one(None).await });

        let (mut stream, _) = connect_and_handshake(addr, None).await;
        let tensor = make_f16_tensor(&[1, hidden_dim]);
        let tensor_bytes = hidden_dim * 2;

        // Warmup
        for _ in 0..warmup {
            Message::single_op("model.layers.0", &tensor, 0, 0)
                .to_writer(&mut stream)
                .await
                .unwrap();
            let _ = Message::from_reader(&mut stream).await.unwrap();
        }

        // Timed
        let start = Instant::now();
        for _ in 0..iterations {
            Message::single_op("model.layers.0", &tensor, 0, 0)
                .to_writer(&mut stream)
                .await
                .unwrap();
            let _ = Message::from_reader(&mut stream).await.unwrap();
        }
        let elapsed = start.elapsed();

        print_bench(
            &format!("TCP echo SingleOp F16 {}", label),
            tensor_bytes,
            elapsed,
            iterations,
        );

        drop(stream);
        let _ = worker.await;
    }
}

#[tokio::test]
#[ignore]
async fn bench_tcp_roundtrip_batch() {
    println!("\n=== TCP Round-Trip Benchmark (Batch, 10 ops) ===\n");

    let sizes: &[(usize, &str)] = &[
        (64, "[1,64]"),
        (512, "[1,512]"),
        (2048, "[1,2048]"),
        (5120, "[1,5120]"),
        (8192, "[1,8192]"),
    ];
    let warmup = 50;
    let iterations = 500;

    for &(hidden_dim, label) in sizes {
        let mock = MockWorker::bind().await;
        let addr = mock.addr();
        let worker = tokio::spawn(async move { mock.handle_one(None).await });

        let (mut stream, _) = connect_and_handshake(addr, None).await;
        let tensor = make_f16_tensor(&[1, hidden_dim]);
        let tensor_bytes = hidden_dim * 2;

        let batch: Vec<(String, usize, usize)> = (0..10)
            .map(|i| (format!("model.layers.{}", i), 0, i))
            .collect();

        // Warmup
        for _ in 0..warmup {
            Message::from_batch(&tensor, batch.clone())
                .to_writer(&mut stream)
                .await
                .unwrap();
            let _ = Message::from_reader(&mut stream).await.unwrap();
        }

        // Timed
        let start = Instant::now();
        for _ in 0..iterations {
            Message::from_batch(&tensor, batch.clone())
                .to_writer(&mut stream)
                .await
                .unwrap();
            let _ = Message::from_reader(&mut stream).await.unwrap();
        }
        let elapsed = start.elapsed();

        print_bench(
            &format!("TCP echo Batch(10) F16 {}", label),
            tensor_bytes,
            elapsed,
            iterations,
        );

        drop(stream);
        let _ = worker.await;
    }
}
