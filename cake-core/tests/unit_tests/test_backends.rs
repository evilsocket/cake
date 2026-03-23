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

/// Test non-contiguous matmul (from transpose) which the GDN recurrent step uses.
/// This is the likely cause of garbage output on Metal.
#[cfg(feature = "metal")]
#[test]
fn metal_candle_noncontiguous_matmul() {
    let Some((_, dev)) = try_metal_backend() else { return };
    // Simulate GDN recurrent: state.transpose(2,3).matmul(k)
    let state_cpu = super::helpers::make_tensor(&[1, 4, 8, 16], 3000); // (batch, heads, key_dim, val_dim)
    let k_cpu = super::helpers::make_tensor(&[1, 4, 8, 1], 3001);      // (batch, heads, key_dim, 1)
    let state_metal = state_cpu.to_dtype(DType::F32).unwrap().to_device(&dev).unwrap();
    let k_metal = k_cpu.to_dtype(DType::F32).unwrap().to_device(&dev).unwrap();

    // Transpose + matmul on CPU (reference)
    let transposed_cpu = state_cpu.transpose(2, 3).unwrap();
    let result_cpu = transposed_cpu.matmul(&k_cpu).unwrap();

    // Same on Metal
    let transposed_metal = state_metal.transpose(2, 3).unwrap();
    let result_metal = transposed_metal.matmul(&k_metal).unwrap();

    assert_close(&result_cpu, &result_metal, 1e-4, "noncontiguous_matmul");
}

/// Test non-contiguous matmul at F16 (the actual model dtype)
#[cfg(feature = "metal")]
#[test]
fn metal_candle_noncontiguous_matmul_f16() {
    let Some((_, dev)) = try_metal_backend() else { return };
    let state_cpu = super::helpers::make_tensor(&[1, 4, 8, 16], 3100);
    let k_cpu = super::helpers::make_tensor(&[1, 4, 8, 1], 3101);

    // CPU F32 reference
    let result_cpu = state_cpu.transpose(2, 3).unwrap().matmul(&k_cpu).unwrap();

    // Metal F16
    let state_f16 = state_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let k_f16 = k_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let result_f16 = state_f16.transpose(2, 3).unwrap().matmul(&k_f16).unwrap();

    assert_close(&result_cpu, &result_f16, 0.01, "noncontiguous_matmul_f16");
}

/// Simulate a full GDN recurrent step on Metal vs CPU to find divergence
#[cfg(feature = "metal")]
#[test]
fn metal_gdn_recurrent_step_matches_cpu() {
    let Some((_, dev)) = try_metal_backend() else { return };
    use candle_core::D;
    // GDN dimensions from Qwen3.5-0.8B: num_heads=16, key_head_dim=4, value_head_dim=16
    let q = super::helpers::make_tensor(&[1, 16, 4], 4000);
    let k = super::helpers::make_tensor(&[1, 16, 4], 4001);
    let v = super::helpers::make_tensor(&[1, 16, 16], 4002);
    let g = super::helpers::make_tensor(&[1, 16], 4003);
    let beta = super::helpers::make_tensor(&[1, 16], 4004);
    let state = Tensor::zeros((1, 16, 4, 16), DType::F32, &Device::Cpu).unwrap();

    // Run on CPU
    let (cpu_out, cpu_state) = {
        let decay = g.unsqueeze(D::Minus1).unwrap().unsqueeze(D::Minus1).unwrap().exp().unwrap();
        let s = state.broadcast_mul(&decay).unwrap();
        let k_4d = k.unsqueeze(D::Minus1).unwrap();
        let retrieved = s.transpose(2, 3).unwrap().matmul(&k_4d).unwrap().squeeze(D::Minus1).unwrap();
        let beta_3d = beta.unsqueeze(D::Minus1).unwrap();
        let delta = (&v - &retrieved).unwrap().broadcast_mul(&beta_3d).unwrap();
        let update = k_4d.matmul(&delta.unsqueeze(2).unwrap()).unwrap();
        let s = (s + update).unwrap();
        let q_4d = q.unsqueeze(D::Minus1).unwrap();
        let output = s.transpose(2, 3).unwrap().matmul(&q_4d).unwrap().squeeze(D::Minus1).unwrap();
        (output, s)
    };

    // Run on Metal (F32)
    let (metal_out, _metal_state) = {
        let q_m = q.to_device(&dev).unwrap();
        let k_m = k.to_device(&dev).unwrap();
        let v_m = v.to_device(&dev).unwrap();
        let g_m = g.to_device(&dev).unwrap();
        let beta_m = beta.to_device(&dev).unwrap();
        let state_m = state.to_device(&dev).unwrap();

        let decay = g_m.unsqueeze(D::Minus1).unwrap().unsqueeze(D::Minus1).unwrap().exp().unwrap();
        let s = state_m.broadcast_mul(&decay).unwrap();
        let k_4d = k_m.unsqueeze(D::Minus1).unwrap();
        let retrieved = s.transpose(2, 3).unwrap().matmul(&k_4d).unwrap().squeeze(D::Minus1).unwrap();
        let beta_3d = beta_m.unsqueeze(D::Minus1).unwrap();
        let delta = (&v_m - &retrieved).unwrap().broadcast_mul(&beta_3d).unwrap();
        let update = k_4d.matmul(&delta.unsqueeze(2).unwrap()).unwrap();
        let s = (s + update).unwrap();
        let q_4d = q_m.unsqueeze(D::Minus1).unwrap();
        let output = s.transpose(2, 3).unwrap().matmul(&q_4d).unwrap().squeeze(D::Minus1).unwrap();
        (output, s)
    };

    assert_close(&cpu_out, &metal_out, 1e-3, "gdn_recurrent_step");
}

/// Test the full in_proj -> F32 -> conv1d -> recurrent pipeline on Metal
#[cfg(feature = "metal")]
#[test]
fn metal_candle_f16_matmul_then_f32_convert() {
    let Some((_, dev)) = try_metal_backend() else { return };
    // Simulate: x_f16 @ weight_f16^T -> to_f32
    // This is what in_proj.forward(x).to_dtype(F32) does
    let x = super::helpers::make_tensor(&[1, 64], 5000);
    let w = super::helpers::make_tensor(&[256, 64], 5001);

    // CPU reference: F16 matmul then F32 (using candle_nn::Linear pattern)
    let x_f16_cpu = x.to_dtype(DType::F16).unwrap();
    let w_f16_cpu = w.to_dtype(DType::F16).unwrap();
    let proj_cpu = x_f16_cpu.broadcast_matmul(&w_f16_cpu.t().unwrap()).unwrap().to_dtype(DType::F32).unwrap();

    // Metal: same ops
    let x_f16_m = x.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let w_f16_m = w.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let proj_metal = x_f16_m.broadcast_matmul(&w_f16_m.t().unwrap()).unwrap().to_dtype(DType::F32).unwrap();

    assert_close(&proj_cpu, &proj_metal, 0.01, "f16_matmul_then_f32");
}

/// Test unfold + broadcast_mul + sum on Metal (used in causal_conv1d_seq)
#[cfg(feature = "metal")]
#[test]
fn metal_candle_unfold_conv() {
    let Some((_, dev)) = try_metal_backend() else { return };
    // Simulate causal_conv1d_seq: padded.unfold(2, kernel_size, 1).broadcast_mul(weight).sum(-1)
    let kernel_size = 4usize;
    let seq_len = 8usize;
    let channels = 16usize;
    let padded_cpu = super::helpers::make_tensor(&[1, channels, seq_len + kernel_size - 1], 5100);
    let weight_cpu = super::helpers::make_tensor(&[channels, kernel_size], 5101);

    // CPU reference
    let unfolded_cpu = padded_cpu.unfold(2, kernel_size, 1).unwrap().contiguous().unwrap();
    let w_cpu = weight_cpu.unsqueeze(0).unwrap().unsqueeze(2).unwrap();
    let conv_cpu = unfolded_cpu.broadcast_mul(&w_cpu).unwrap().sum(candle_core::D::Minus1).unwrap();

    // Metal
    let padded_m = padded_cpu.to_device(&dev).unwrap();
    let weight_m = weight_cpu.to_device(&dev).unwrap();
    let unfolded_m = padded_m.unfold(2, kernel_size, 1).unwrap().contiguous().unwrap();
    let w_m = weight_m.unsqueeze(0).unwrap().unsqueeze(2).unwrap();
    let conv_m = unfolded_m.broadcast_mul(&w_m).unwrap().sum(candle_core::D::Minus1).unwrap();

    assert_close(&conv_cpu, &conv_m, 1e-4, "unfold_conv");
}

/// Test expand + contiguous + reshape on Metal (used in repeat_key_heads)
#[cfg(feature = "metal")]
#[test]
fn metal_candle_expand_contiguous() {
    let Some((_, dev)) = try_metal_backend() else { return };
    let x_cpu = super::helpers::make_tensor(&[1, 1, 2, 1, 4], 5200);
    let expanded_cpu = x_cpu.expand((1, 1, 2, 4, 4)).unwrap().contiguous().unwrap().reshape((1, 1, 8, 4)).unwrap();

    let x_m = x_cpu.to_device(&dev).unwrap();
    let expanded_m = x_m.expand((1, 1, 2, 4, 4)).unwrap().contiguous().unwrap().reshape((1, 1, 8, 4)).unwrap();

    assert_close(&expanded_cpu, &expanded_m, 1e-6, "expand_contiguous");
}

/// Test a chain of operations that mimics the full GDN forward pass
#[cfg(feature = "metal")]
#[test]
fn metal_chained_ops_accumulation() {
    let Some((_, dev)) = try_metal_backend() else { return };
    // Chain: matmul -> to_f32 -> narrow -> reshape -> rms_norm -> exp -> mul -> add
    // Repeat 24 times (like 24 layers) to check for accumulated divergence
    let x_cpu = super::helpers::make_tensor(&[1, 64], 5300);
    let w_cpu = super::helpers::make_tensor(&[64, 64], 5301);
    let norm_w_cpu = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();

    let x_m = x_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let w_m = w_cpu.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
    let norm_w_m = Tensor::ones(64, DType::F32, &dev).unwrap();

    let mut cpu_val = x_cpu.to_dtype(DType::F16).unwrap();
    let mut metal_val = x_m.clone();

    for _ in 0..24 {
        // Linear: x @ w^T
        cpu_val = cpu_val.broadcast_matmul(&w_cpu.to_dtype(DType::F16).unwrap().t().unwrap()).unwrap();
        metal_val = metal_val.broadcast_matmul(&w_m.t().unwrap()).unwrap();

        // To F32 for norm
        let cpu_f32 = cpu_val.to_dtype(DType::F32).unwrap();
        let metal_f32 = metal_val.to_dtype(DType::F32).unwrap();

        // RMS norm
        let cpu_normed = candle_nn::ops::rms_norm(&cpu_f32, &norm_w_cpu, 1e-6).unwrap();
        let metal_normed = candle_nn::ops::rms_norm(&metal_f32, &norm_w_m, 1e-6).unwrap();

        // Back to F16
        cpu_val = cpu_normed.to_dtype(DType::F16).unwrap();
        metal_val = metal_normed.to_dtype(DType::F16).unwrap();
    }

    assert_close(&cpu_val, &metal_val, 0.1, "chained_24_layers");
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
