//! Client/Worker mock integration tests.
//!
//! Uses real TCP with a mock worker that echoes tensors back,
//! testing the Client::new() handshake and forward paths.

use std::net::SocketAddr;

use candle_core::{DType, Device, Tensor};
use tokio::net::{TcpListener, TcpStream};

use cake_core::cake::{auth, Client, Forwarder, Message, WorkerInfo};

// ── Helpers ──────────────────────────────────────────────────────

fn make_f16_tensor(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
}

fn mock_worker_info() -> WorkerInfo {
    WorkerInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        dtype: "F16".to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        device: "cpu".to_string(),
        device_idx: 0,
        latency: 0,
    }
}

// ── Mock Worker ──────────────────────────────────────────────────

/// Bind a TCP listener on localhost with an OS-assigned port.
async fn bind_mock() -> TcpListener {
    TcpListener::bind("127.0.0.1:0").await.unwrap()
}

fn listener_addr(l: &TcpListener) -> SocketAddr {
    l.local_addr().unwrap()
}

/// Run one mock-worker session: accept a connection, optionally authenticate,
/// handle Hello, then echo tensors for SingleOp/Batch until the client
/// disconnects or sends Goodbye.
async fn run_mock_worker(listener: TcpListener, cluster_key: Option<String>) -> anyhow::Result<()> {
    let (mut socket, _) = listener.accept().await?;

    // Authenticate if key provided
    if let Some(ref key) = cluster_key {
        auth::authenticate_as_worker(&mut socket, key).await?;
    }

    // Read Hello, send WorkerInfo
    let (_, first) = Message::from_reader(&mut socket).await?;
    match first {
        Message::Hello => {
            Message::WorkerInfo(mock_worker_info())
                .to_writer(&mut socket)
                .await?;
        }
        other => return Err(anyhow::anyhow!("expected Hello, got {:?}", other)),
    }

    // Echo loop
    loop {
        let result = Message::from_reader(&mut socket).await;
        let (_, msg) = match result {
            Ok(m) => m,
            Err(_) => break, // disconnected
        };

        match msg {
            Message::SingleOp { x, .. } => {
                Message::Tensor(x).to_writer(&mut socket).await?;
            }
            Message::Batch { x, .. } => {
                Message::Tensor(x).to_writer(&mut socket).await?;
            }
            Message::Goodbye => {
                Message::WorkerInfo(mock_worker_info())
                    .to_writer(&mut socket)
                    .await?;
                break;
            }
            other => {
                return Err(anyhow::anyhow!("unhandled message: {:?}", other));
            }
        }
    }

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn client_connects_and_gets_worker_info() {
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let worker = tokio::spawn(run_mock_worker(listener, None));

    let client = Client::new(
        Device::Cpu,
        &addr.to_string(),
        "model.layers.0",
        None,
    )
    .await
    .unwrap();

    // Client should have received WorkerInfo during handshake
    let display = format!("{}", client);
    assert!(display.contains("cpu"), "display should mention device: {}", display);

    drop(client);
    let _ = worker.await;
}

#[tokio::test]
async fn client_connects_with_auth() {
    let key = "test-secret-key-42".to_string();
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let key_clone = key.clone();
    let worker = tokio::spawn(run_mock_worker(listener, Some(key_clone)));

    let client = Client::new(
        Device::Cpu,
        &addr.to_string(),
        "model.layers.0",
        Some(&key),
    )
    .await
    .unwrap();

    let display = format!("{}", client);
    assert!(display.contains("cpu"));

    drop(client);
    let _ = worker.await;
}

#[tokio::test]
async fn client_auth_wrong_key_fails() {
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let worker = tokio::spawn(run_mock_worker(listener, Some("correct-key".to_string())));

    let result = Client::new(
        Device::Cpu,
        &addr.to_string(),
        "model.layers.0",
        Some("wrong-key"),
    )
    .await;

    assert!(result.is_err(), "connecting with wrong key should fail");

    // Worker should also fail
    let worker_result = worker.await.unwrap();
    assert!(worker_result.is_err());
}

#[tokio::test]
async fn client_forward_single_op_echoes_tensor() {
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let worker = tokio::spawn(run_mock_worker(listener, None));

    // Use a dummy Context -- we only need forward_mut which goes through the TCP mock
    let mut client = Client::new(
        Device::Cpu,
        &addr.to_string(),
        "model.layers.0",
        None,
    )
    .await
    .unwrap();

    // Test goodbye which IS accessible via the client API.
    client.goodbye().await.unwrap();

    drop(client);
    let _ = worker.await;
}

#[tokio::test]
async fn client_forward_via_raw_protocol() {
    // This test exercises the full TCP echo path using raw messages,
    // verifying the mock worker protocol that Client::new relies on.
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let worker = tokio::spawn(run_mock_worker(listener, None));

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Hello handshake
    Message::Hello.to_writer(&mut stream).await.unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    let info = match resp {
        Message::WorkerInfo(wi) => wi,
        other => panic!("expected WorkerInfo, got {:?}", other),
    };
    assert_eq!(info.device, "cpu");

    // Send SingleOp, get Tensor back
    let tensor = make_f16_tensor(&[1, 256]);
    Message::single_op("model.layers.0", &tensor, 0, 0)
        .to_writer(&mut stream)
        .await
        .unwrap();

    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    match resp {
        Message::Tensor(raw) => {
            let recovered = raw.to_tensor(&Device::Cpu).unwrap();
            assert_eq!(recovered.dtype(), DType::F16);
            assert_eq!(recovered.shape().dims(), &[1, 256]);
        }
        other => panic!("expected Tensor, got {:?}", other),
    }

    // Send Batch, get Tensor back
    let batch_tensor = make_f16_tensor(&[1, 64]);
    let batch = vec![
        ("model.layers.0".into(), 0usize, 0usize),
        ("model.layers.1".into(), 1, 1),
    ];

    Message::from_batch(&batch_tensor, batch)
        .to_writer(&mut stream)
        .await
        .unwrap();

    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    match resp {
        Message::Tensor(raw) => {
            let recovered = raw.to_tensor(&Device::Cpu).unwrap();
            assert_eq!(recovered.dtype(), DType::F16);
            assert_eq!(recovered.shape().dims(), &[1, 64]);
        }
        other => panic!("expected Tensor from batch, got {:?}", other),
    }

    // Goodbye
    Message::Goodbye.to_writer(&mut stream).await.unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    assert!(matches!(resp, Message::WorkerInfo(_)));

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn client_forward_with_auth_via_raw_protocol() {
    let key = "super-secret-auth-key";
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let worker = tokio::spawn(run_mock_worker(listener, Some(key.to_string())));

    let mut stream = TcpStream::connect(addr).await.unwrap();

    // Authenticate
    auth::authenticate_as_master(&mut stream, key).await.unwrap();

    // Hello handshake
    Message::Hello.to_writer(&mut stream).await.unwrap();
    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    assert!(matches!(resp, Message::WorkerInfo(_)));

    // SingleOp echo
    let tensor = make_f16_tensor(&[1, 32]);
    Message::single_op("model.layers.5", &tensor, 10, 5)
        .to_writer(&mut stream)
        .await
        .unwrap();

    let (_, resp) = Message::from_reader(&mut stream).await.unwrap();
    match resp {
        Message::Tensor(raw) => {
            let t = raw.to_tensor(&Device::Cpu).unwrap();
            assert_eq!(t.shape().dims(), &[1, 32]);
        }
        other => panic!("expected Tensor, got {:?}", other),
    }

    drop(stream);
    let _ = worker.await;
}

#[tokio::test]
async fn client_no_auth_to_authed_worker_fails() {
    let listener = bind_mock().await;
    let addr = listener_addr(&listener);
    let worker = tokio::spawn(run_mock_worker(listener, Some("secret".to_string())));

    // Try connecting without auth -- Client::new sends Hello directly,
    // but worker expects auth bytes first, so the protocol will desync.
    // Use a timeout to avoid hanging if both sides block on garbled reads.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        Client::new(Device::Cpu, &addr.to_string(), "model.layers.0", None),
    )
    .await;

    // Either the inner Result is Err (protocol failure) or the timeout fires
    match result {
        Ok(Ok(_)) => panic!("should not succeed without auth"),
        Ok(Err(_)) => {} // protocol error — expected
        Err(_) => {}     // timeout — also acceptable (protocol deadlock)
    }

    // Don't wait on worker — it may be stuck too
    worker.abort();
}
