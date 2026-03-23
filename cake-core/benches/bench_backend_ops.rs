//! Benchmarks for ComputeBackend inference primitive methods.
//! Covers: linear_forward, softmax, rms_norm, rope, silu, gelu, sigmoid,
//! embedding, causal_mask, topk.

use candle_core::{DType, Device, Tensor};
use cake_core::backends::create_backend;

fn backend() -> std::sync::Arc<dyn cake_core::backends::ComputeBackend> {
    create_backend(&Device::Cpu)
}

// ── linear_forward ─────────────────────────────────────────────────────

#[divan::bench(args = [64, 256, 1024])]
fn linear_forward_2d(bencher: divan::Bencher, hidden: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, hidden), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (hidden, hidden), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.linear_forward(&x, &w, None).unwrap();
    });
}

#[divan::bench(args = [64, 256])]
fn linear_forward_3d_batched(bencher: divan::Bencher, hidden: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 8, hidden), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (hidden, hidden), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (hidden,), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.linear_forward(&x, &w, Some(&bias)).unwrap();
    });
}

// ── softmax ────────────────────────────────────────────────────────────

#[divan::bench(args = [64, 256, 1024])]
fn softmax_last_dim(bencher: divan::Bencher, dim: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (4, dim), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.softmax(&x, 1).unwrap();
    });
}

// ── rms_norm ───────────────────────────────────────────────────────────

#[divan::bench(args = [64, 256, 1024])]
fn rms_norm(bencher: divan::Bencher, hidden: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 1, hidden), &Device::Cpu).unwrap();
    let w = Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.rms_norm(&x, &w, 1e-5).unwrap();
    });
}

// ── rope ───────────────────────────────────────────────────────────────

#[divan::bench(args = [32, 64, 128])]
fn rope(bencher: divan::Bencher, head_dim: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 8, 1, head_dim), &Device::Cpu).unwrap();
    let half = head_dim / 2;
    let cos = Tensor::ones((1, half), DType::F32, &Device::Cpu).unwrap();
    let sin = Tensor::zeros((1, half), DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.rope(&x, &cos, &sin).unwrap();
    });
}

// ── activations ────────────────────────────────────────────────────────

#[divan::bench(args = [256, 1024, 4096])]
fn silu(bencher: divan::Bencher, size: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, size), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.silu(&x).unwrap();
    });
}

#[divan::bench(args = [256, 1024, 4096])]
fn gelu(bencher: divan::Bencher, size: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, size), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.gelu(&x).unwrap();
    });
}

#[divan::bench(args = [256, 1024, 4096])]
fn sigmoid(bencher: divan::Bencher, size: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, size), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.sigmoid(&x).unwrap();
    });
}

// ── embedding ──────────────────────────────────────────────────────────

#[divan::bench(args = [1, 8, 64])]
fn embedding_lookup(bencher: divan::Bencher, seq_len: usize) {
    let b = backend();
    let vocab = 1000;
    let hidden = 256;
    let weight = Tensor::randn(0f32, 0.1, (vocab, hidden), &Device::Cpu).unwrap();
    let ids_data: Vec<u32> = (0..seq_len).map(|i| (i * 37 % vocab) as u32).collect();
    let ids = Tensor::from_vec(ids_data, (seq_len,), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.embedding(&ids, &weight).unwrap();
    });
}

// ── causal_mask ────────────────────────────────────────────────────────

#[divan::bench(args = [1, 32, 128])]
fn causal_mask(bencher: divan::Bencher, seq_len: usize) {
    let b = backend();
    bencher.bench_local(|| {
        let _ = b.causal_mask(seq_len, seq_len, &Device::Cpu).unwrap();
    });
}

// ── topk ───────────────────────────────────────────────────────────────

#[divan::bench(args = [32, 128, 256])]
fn topk(bencher: divan::Bencher, num_experts: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, num_experts), &Device::Cpu).unwrap();
    let k = 8.min(num_experts);
    bencher.bench_local(|| {
        let _ = b.topk(&x, k).unwrap();
    });
}

// ── conv1d ─────────────────────────────────────────────────────────────

#[divan::bench(args = [16, 64, 256])]
fn conv1d(bencher: divan::Bencher, channels: usize) {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (channels, channels, 3), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (channels,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, channels, 64), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.conv1d(&x, &w, Some(&bias), 1, 1, 1, 1).unwrap();
    });
}

// ── conv_transpose1d ───────────────────────────────────────────────────

#[divan::bench(args = [16, 64])]
fn conv_transpose1d(bencher: divan::Bencher, channels: usize) {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (channels, channels, 4), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (channels,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, channels, 16), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.conv_transpose1d(&x, &w, Some(&bias), 1, 0, 2, 1, 1).unwrap();
    });
}

// ── sdpa ───────────────────────────────────────────────────────────────

#[divan::bench(args = [1, 8])]
fn sdpa(bencher: divan::Bencher, seq_len: usize) {
    let b = backend();
    let q = Tensor::randn(0f32, 0.5, (1, 4, seq_len, 32), &Device::Cpu).unwrap();
    let k = Tensor::randn(0f32, 0.5, (1, 4, seq_len, 32), &Device::Cpu).unwrap();
    let v = Tensor::randn(0f32, 0.5, (1, 4, seq_len, 32), &Device::Cpu).unwrap();
    let scale = 1.0 / (32.0f32).sqrt();
    // SDPA may not have a CPU impl — skip gracefully
    if b.sdpa(&q, &k, &v, None, false, scale).is_err() {
        return;
    }
    bencher.bench_local(|| {
        let _ = b.sdpa(&q, &k, &v, None, seq_len > 1, scale);
    });
}

// ── layer_norm ─────────────────────────────────────────────────────────
#[divan::bench(args = [64, 256, 1024])]
fn layer_norm(bencher: divan::Bencher, hidden: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 8, hidden), &Device::Cpu).unwrap();
    let w = Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap();
    let bias = Tensor::zeros(hidden, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.layer_norm(&x, &w, Some(&bias), 1e-5).unwrap();
    });
}

// ── group_norm ─────────────────────────────────────────────────────────
#[divan::bench(args = [32, 128, 512])]
fn group_norm(bencher: divan::Bencher, channels: usize) {
    let b = backend();
    let groups = (channels / 4).max(1);
    let x = Tensor::randn(0f32, 1.0, (1, channels, 8, 8), &Device::Cpu).unwrap();
    let w = Tensor::ones(channels, DType::F32, &Device::Cpu).unwrap();
    let bias = Tensor::zeros(channels, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.group_norm(&x, &w, &bias, groups, 1e-5).unwrap();
    });
}

// ── conv2d ─────────────────────────────────────────────────────────────
#[divan::bench(args = [32, 64, 128])]
fn conv2d(bencher: divan::Bencher, channels: usize) {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (channels, channels, 3, 3), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (channels,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, channels, 16, 16), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.conv2d(&x, &w, Some(&bias), 1, 1, 1, 1).unwrap();
    });
}

// ── linear_forward 4D ──────────────────────────────────────────────────
#[divan::bench(args = [64, 256])]
fn linear_forward_4d(bencher: divan::Bencher, hidden: usize) {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 2, 4, hidden), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (hidden, hidden), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let _ = b.linear_forward(&x, &w, None).unwrap();
    });
}
