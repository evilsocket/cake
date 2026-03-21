//! Tests for the ComputeBackend trait and backend implementations.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use cake_core::backends::{self, ComputeBackend, CpuBackend};

// ── Factory tests ──────────────────────────────────────────────

#[test]
fn create_backend_cpu_returns_cpu_backend() {
    let backend = backends::create_backend(&Device::Cpu);
    assert_eq!(backend.name(), "cpu");
    assert!(backend.device().is_cpu());
}

#[test]
fn create_backend_returns_arc_dyn() {
    // Verify the returned backend is usable as Arc<dyn ComputeBackend>
    let backend: Arc<dyn ComputeBackend> = backends::create_backend(&Device::Cpu);
    let _cloned = backend.clone();
    assert_eq!(backend.name(), "cpu");
}

// ── Trait object dispatch ──────────────────────────────────────

#[test]
fn backend_trait_object_silu_mul() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
    let gate = Tensor::new(&[1.0f32, 2.0, -1.0], &Device::Cpu).unwrap();
    let up = Tensor::new(&[1.0f32, 1.0, 1.0], &Device::Cpu).unwrap();
    let result = backend.silu_mul(&gate, &up).unwrap();
    assert_eq!(result.dims(), &[3]);
    // silu(1) * 1 ≈ 0.731, silu(2) * 1 ≈ 1.762, silu(-1) * 1 ≈ -0.269
    let vals: Vec<f32> = result.to_vec1().unwrap();
    assert!((vals[0] - 0.731).abs() < 0.01);
}

#[test]
fn backend_trait_object_attention() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
    let q = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
    let k = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
    let v = Tensor::randn(0f32, 1.0, (1, 2, 4, 8), &Device::Cpu).unwrap();
    let result = backend.attention(&q, &k, &v, 0.125, true).unwrap();
    assert_eq!(result.dims(), &[1, 2, 4, 8]);
}

#[test]
fn backend_trait_object_synchronize() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuBackend::new());
    assert!(backend.synchronize().is_ok());
}

// ── New trait methods ──────────────────────────────────────────

#[test]
fn backend_rms_norm_channel_shape() {
    let backend = CpuBackend::new();
    let x = Tensor::randn(0f32, 1.0, (2, 4, 8), &Device::Cpu).unwrap();
    let w = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
    let result = backend.rms_norm_channel(&x, &w, 1e-6).unwrap();
    assert_eq!(result.dims(), &[2, 4, 8]);
}

#[test]
fn backend_depthwise_conv1d_bias_ctx_shape() {
    let backend = CpuBackend::new();
    let ctx = Tensor::zeros((1, 4, 6), DType::F32, &Device::Cpu).unwrap();
    let input = Tensor::randn(0f32, 1.0, (1, 4, 10), &Device::Cpu).unwrap();
    let weight = Tensor::randn(0f32, 1.0, (4, 7), &Device::Cpu).unwrap();
    let bias = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();
    let result = backend
        .depthwise_conv1d_bias_ctx(&ctx, &input, &weight, &bias, 7, 4)
        .unwrap();
    assert_eq!(result.dims(), &[1, 4, 10]);
}

#[test]
fn backend_adaln_modulate_shape() {
    let backend = CpuBackend::new();
    let x = Tensor::randn(0f32, 1.0, (2, 4, 16), &Device::Cpu).unwrap();
    let norm_w = Tensor::ones(16, DType::F32, &Device::Cpu).unwrap();
    let scale = Tensor::zeros((2, 4, 16), DType::F32, &Device::Cpu).unwrap();
    let shift = Tensor::zeros((2, 4, 16), DType::F32, &Device::Cpu).unwrap();
    let result = backend
        .adaln_modulate(&x, &norm_w, &scale, &shift, 1e-6)
        .unwrap();
    assert_eq!(result.dims(), &[2, 4, 16]);
}

// ── Numerical correctness ──────────────────────────────────────

#[test]
fn backend_attention_causal_mask_works() {
    // With causal masking, position 0 should only attend to itself
    let backend = CpuBackend::new();
    // Create identity-like Q and K so attention scores are clear
    let q = Tensor::new(
        &[[[[1.0f32, 0.0], [0.0, 1.0]]]],
        &Device::Cpu,
    )
    .unwrap(); // (1,1,2,2)
    let k = q.clone();
    let v = Tensor::new(
        &[[[[10.0f32, 20.0], [30.0, 40.0]]]],
        &Device::Cpu,
    )
    .unwrap();

    let causal = backend.attention(&q, &k, &v, 1.0, true).unwrap();
    let non_causal = backend.attention(&q, &k, &v, 1.0, false).unwrap();

    let c_vals: Vec<f32> = causal.flatten_all().unwrap().to_vec1().unwrap();
    let nc_vals: Vec<f32> = non_causal.flatten_all().unwrap().to_vec1().unwrap();

    // First position (causal): attends only to pos 0 → output ≈ [10, 20]
    assert!((c_vals[0] - 10.0).abs() < 1.0);
    // First position (non-causal): attends to both → weighted avg of [10,20] and [30,40]
    // Should differ from causal
    assert!((c_vals[0] - nc_vals[0]).abs() > 0.1);
}

#[test]
fn backend_add_rms_norm_residual_is_sum() {
    let backend = CpuBackend::new();
    let a = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();
    let b = Tensor::new(&[[0.5f32, 0.5, 0.5, 0.5]], &Device::Cpu).unwrap();
    let w = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
    // add_rms_norm returns (residual_sum, normed) — residual = a + b
    let (residual, _normed) = backend.add_rms_norm(&a, &b, &w, 1e-6).unwrap();
    let res_vals: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
    // residual = a + b
    assert!((res_vals[0] - 1.5).abs() < 0.01);
    assert!((res_vals[3] - 4.5).abs() < 0.01);
}

// ── Debug trait ────────────────────────────────────────────────

#[test]
fn backend_implements_debug() {
    let backend = CpuBackend::new();
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("CpuBackend"));
}
