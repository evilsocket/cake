//! Tests for the ComputeBackend trait and backend implementations.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use cake_core::backends::{self, ComputeBackend, CpuBackend};

// ── Factory tests ──────────────────────────────────────────────

#[test]
fn create_backend_cpu_returns_valid_backend() {
    let backend = backends::create_backend(&Device::Cpu);
    // With vulkan feature, may return "vulkan" instead of "cpu"
    assert!(backend.name() == "cpu" || backend.name() == "vulkan");
    assert!(backend.device().is_cpu());
}

#[test]
fn create_backend_returns_arc_dyn() {
    let backend: Arc<dyn ComputeBackend> = backends::create_backend(&Device::Cpu);
    let _cloned = backend.clone();
    assert!(backend.name() == "cpu" || backend.name() == "vulkan");
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

// ── Metal vs CPU correctness tests ───────────────────────────
//
// These tests compare Metal backend output against CPU reference to catch
// numerical regressions in MSL kernels. They run only with --features metal
// and are skipped if no Metal device is available (e.g. CI on Linux).

#[cfg(feature = "metal")]
fn try_metal_backend() -> Option<(Arc<dyn ComputeBackend>, Device)> {
    if !candle_core::utils::metal_is_available() {
        return None;
    }
    let dev = Device::new_metal(0).ok()?;
    Some((backends::create_backend(&dev), dev))
}

#[cfg(feature = "metal")]
fn assert_close(cpu: &Tensor, metal: &Tensor, tol: f32, label: &str) {
    let cpu_vals: Vec<f32> = cpu.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let metal_vals: Vec<f32> = metal.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(cpu_vals.len(), metal_vals.len(), "{label}: length mismatch");
    for (i, (c, m)) in cpu_vals.iter().zip(metal_vals.iter()).enumerate() {
        assert!(
            (c - m).abs() < tol,
            "{label}[{i}]: cpu={c} metal={m} diff={}",
            (c - m).abs()
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_silu_mul_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let gate_cpu = super::helpers::make_tensor(&[1, 1, 64], 100);
    let up_cpu = super::helpers::make_tensor(&[1, 1, 64], 101);
    let gate_metal = gate_cpu.to_device(&dev).unwrap();
    let up_metal = up_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.silu_mul(&gate_cpu, &up_cpu).unwrap();
    let metal_out = metal.silu_mul(&gate_metal, &up_metal).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-3, "silu_mul");
}

#[cfg(feature = "metal")]
#[test]
fn metal_stable_softplus_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let x_cpu = super::helpers::make_tensor(&[1, 1, 64], 200);
    let x_metal = x_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.stable_softplus(&x_cpu).unwrap();
    let metal_out = metal.stable_softplus(&x_metal).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-3, "stable_softplus");
}

#[cfg(feature = "metal")]
#[test]
fn metal_add3_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let a_cpu = super::helpers::make_tensor(&[1, 1, 64], 300);
    let b_cpu = super::helpers::make_tensor(&[1, 1, 64], 301);
    let c_cpu = super::helpers::make_tensor(&[1, 1, 64], 302);
    let a_m = a_cpu.to_device(&dev).unwrap();
    let b_m = b_cpu.to_device(&dev).unwrap();
    let c_m = c_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.add3(&a_cpu, &b_cpu, &c_cpu).unwrap();
    let metal_out = metal.add3(&a_m, &b_m, &c_m).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-5, "add3");
}

#[cfg(feature = "metal")]
#[test]
fn metal_exp_mul_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let x_cpu = super::helpers::make_tensor(&[1, 1, 64], 400);
    let y_cpu = super::helpers::make_tensor(&[1, 1, 64], 401);
    let x_m = x_cpu.to_device(&dev).unwrap();
    let y_m = y_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.exp_mul(&x_cpu, &y_cpu).unwrap();
    let metal_out = metal.exp_mul(&x_m, &y_m).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-4, "exp_mul");
}

#[cfg(feature = "metal")]
#[test]
fn metal_sub_mul_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let a_cpu = super::helpers::make_tensor(&[1, 1, 64], 500);
    let b_cpu = super::helpers::make_tensor(&[1, 1, 64], 501);
    let c_cpu = super::helpers::make_tensor(&[1, 1, 64], 502);
    let a_m = a_cpu.to_device(&dev).unwrap();
    let b_m = b_cpu.to_device(&dev).unwrap();
    let c_m = c_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.sub_mul(&a_cpu, &b_cpu, &c_cpu).unwrap();
    let metal_out = metal.sub_mul(&a_m, &b_m, &c_m).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-5, "sub_mul");
}

#[cfg(feature = "metal")]
#[test]
fn metal_rms_norm_gated_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let x_cpu = super::helpers::make_tensor(&[1, 1, 64], 600);
    let z_cpu = super::helpers::make_tensor(&[1, 1, 64], 601);
    let w_cpu = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();
    let x_m = x_cpu.to_device(&dev).unwrap();
    let z_m = z_cpu.to_device(&dev).unwrap();
    let w_m = w_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.rms_norm_gated(&x_cpu, &z_cpu, &w_cpu, 1e-6).unwrap();
    let metal_out = metal.rms_norm_gated(&x_m, &z_m, &w_m, 1e-6).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-3, "rms_norm_gated");
}

#[cfg(feature = "metal")]
#[test]
fn metal_add_rms_norm_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let a_cpu = super::helpers::make_tensor(&[1, 1, 64], 700);
    let b_cpu = super::helpers::make_tensor(&[1, 1, 64], 701);
    let w_cpu = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();
    let a_m = a_cpu.to_device(&dev).unwrap();
    let b_m = b_cpu.to_device(&dev).unwrap();
    let w_m = w_cpu.to_device(&dev).unwrap();

    let (cpu_res, cpu_norm) = cpu.add_rms_norm(&a_cpu, &b_cpu, &w_cpu, 1e-6).unwrap();
    let (metal_res, metal_norm) = metal.add_rms_norm(&a_m, &b_m, &w_m, 1e-6).unwrap();
    assert_close(&cpu_res, &metal_res, 1e-5, "add_rms_norm residual");
    assert_close(&cpu_norm, &metal_norm, 1e-3, "add_rms_norm normed");
}

#[cfg(feature = "metal")]
#[test]
fn metal_rms_norm_gated_large_hidden_matches_cpu() {
    // Test with hidden > 32 (multiple SIMD groups needed for reduction)
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let x_cpu = super::helpers::make_tensor(&[1, 1, 1024], 800);
    let z_cpu = super::helpers::make_tensor(&[1, 1, 1024], 801);
    let w_cpu = Tensor::ones(1024, DType::F32, &Device::Cpu).unwrap();
    let x_m = x_cpu.to_device(&dev).unwrap();
    let z_m = z_cpu.to_device(&dev).unwrap();
    let w_m = w_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.rms_norm_gated(&x_cpu, &z_cpu, &w_cpu, 1e-6).unwrap();
    let metal_out = metal.rms_norm_gated(&x_m, &z_m, &w_m, 1e-6).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-2, "rms_norm_gated_1024");
}

#[cfg(feature = "metal")]
#[test]
fn metal_depthwise_conv1d_silu_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let window_cpu = super::helpers::make_tensor(&[1, 16, 4], 900);
    let weight_cpu = super::helpers::make_tensor(&[16, 4], 901);
    let window_m = window_cpu.to_device(&dev).unwrap();
    let weight_m = weight_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.depthwise_conv1d_silu(&window_cpu, &weight_cpu, 4, 16).unwrap();
    let metal_out = metal.depthwise_conv1d_silu(&window_m, &weight_m, 4, 16).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-3, "depthwise_conv1d_silu");
}

#[cfg(feature = "metal")]
#[test]
fn metal_depthwise_conv1d_bias_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let padded_cpu = super::helpers::make_tensor(&[1, 16, 11], 1000); // t_padded = 11
    let weight_cpu = super::helpers::make_tensor(&[16, 4], 1001); // (channels, kernel_size)
    let bias_cpu = super::helpers::make_tensor(&[16], 1002);
    let padded_m = padded_cpu.to_device(&dev).unwrap();
    let weight_m = weight_cpu.to_device(&dev).unwrap();
    let bias_m = bias_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.depthwise_conv1d_bias(&padded_cpu, &weight_cpu, &bias_cpu, 4, 16).unwrap();
    let metal_out = metal.depthwise_conv1d_bias(&padded_m, &weight_m, &bias_m, 4, 16).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-3, "depthwise_conv1d_bias");
}

#[cfg(feature = "metal")]
#[test]
fn metal_add_scaled_1d_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let a_cpu = super::helpers::make_tensor(&[1, 4, 8], 1100);
    let b_cpu = super::helpers::make_tensor(&[1, 4, 8], 1101);
    let c_cpu = super::helpers::make_tensor(&[4], 1102); // 1D channel scale
    let a_m = a_cpu.to_device(&dev).unwrap();
    let b_m = b_cpu.to_device(&dev).unwrap();
    let c_m = c_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.add_scaled(&a_cpu, &b_cpu, &c_cpu).unwrap();
    let metal_out = metal.add_scaled(&a_m, &b_m, &c_m).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-4, "add_scaled_1d");
}

#[cfg(feature = "metal")]
#[test]
fn metal_rms_norm_channel_matches_cpu() {
    let Some((metal, dev)) = try_metal_backend() else { return };
    let cpu = CpuBackend::new();
    let x_cpu = super::helpers::make_tensor(&[1, 16, 8], 1200);
    let w_cpu = Tensor::ones(16, DType::F32, &Device::Cpu).unwrap();
    let x_m = x_cpu.to_device(&dev).unwrap();
    let w_m = w_cpu.to_device(&dev).unwrap();

    let cpu_out = cpu.rms_norm_channel(&x_cpu, &w_cpu, 1e-6).unwrap();
    let metal_out = metal.rms_norm_channel(&x_m, &w_m, 1e-6).unwrap();
    assert_close(&cpu_out, &metal_out, 1e-3, "rms_norm_channel");
}

/// Test candle's internal Metal ops at F16 precision to find the source
/// of garbage model output. The model uses F16 dtype on Metal.
#[cfg(feature = "metal")]
#[test]
fn metal_candle_f16_rms_norm_correctness() {
    let Some((_, dev)) = try_metal_backend() else { return };
    // F16 rms_norm on Metal vs F32 on CPU
    let x_cpu = super::helpers::make_tensor(&[1, 1, 1024], 2000);
    let x_f16 = x_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let w_f16 = Tensor::ones(1024, DType::F16, &dev).unwrap();
    let w_f32 = Tensor::ones(1024, DType::F32, &Device::Cpu).unwrap();

    let normed_metal = candle_nn::ops::rms_norm(&x_f16, &w_f16, 1e-6).unwrap();
    let normed_cpu = candle_nn::ops::rms_norm(&x_cpu, &w_f32, 1e-6).unwrap();
    // Allow F16 precision loss (mantissa = 10 bits ≈ 3 decimal digits)
    assert_close(&normed_cpu, &normed_metal, 0.01, "candle_rms_norm_f16");
}

#[cfg(feature = "metal")]
#[test]
fn metal_candle_f16_matmul_correctness() {
    let Some((_, dev)) = try_metal_backend() else { return };
    let a_cpu = super::helpers::make_tensor(&[1, 64], 2100);
    let b_cpu = super::helpers::make_tensor(&[64, 128], 2101);
    let a_f16 = a_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let b_f16 = b_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();

    let mm_metal = a_f16.matmul(&b_f16).unwrap();
    let mm_cpu = a_cpu.matmul(&b_cpu).unwrap();
    assert_close(&mm_cpu, &mm_metal, 0.01, "candle_matmul_f16");
}

#[cfg(feature = "metal")]
#[test]
fn metal_candle_f16_silu_correctness() {
    let Some((_, dev)) = try_metal_backend() else { return };
    let x_cpu = super::helpers::make_tensor(&[1, 1, 1024], 2200);
    let x_f16 = x_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();

    let silu_metal = candle_nn::ops::silu(&x_f16).unwrap();
    let silu_cpu = candle_nn::ops::silu(&x_cpu).unwrap();
    assert_close(&silu_cpu, &silu_metal, 0.001, "candle_silu_f16");
}

#[cfg(feature = "metal")]
#[test]
fn metal_candle_f16_to_f32_conversion() {
    let Some((_, dev)) = try_metal_backend() else { return };
    let x_cpu = super::helpers::make_tensor(&[1, 1, 1024], 2300);
    let x_f16_metal = x_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let x_f32_metal = x_f16_metal.to_dtype(DType::F32).unwrap();
    let x_f16_cpu = x_cpu.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
    assert_close(&x_f16_cpu, &x_f32_metal, 1e-6, "f16_to_f32_conversion");
}
