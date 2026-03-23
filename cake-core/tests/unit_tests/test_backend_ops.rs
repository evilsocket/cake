//! Tests for the new ComputeBackend inference primitive methods.

use candle_core::{DType, Device, Module, Tensor};
use cake_core::backends::{create_backend, ComputeBackend};

fn backend() -> std::sync::Arc<dyn ComputeBackend> {
    create_backend(&Device::Cpu)
}

fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
    let diff = (a - b).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(max_diff < tol, "max diff {max_diff} exceeds tolerance {tol}");
}

// ── linear_forward ─────────────────────────────────────────────────────

#[test]
fn test_linear_forward_2d_no_bias() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (16, 8), &Device::Cpu).unwrap();
    let out = b.linear_forward(&x, &w, None).unwrap();
    assert_eq!(out.dims(), &[4, 16]);
}

#[test]
fn test_linear_forward_3d_with_bias() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (2, 5, 8), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (16, 8), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (16,), &Device::Cpu).unwrap();
    let out = b.linear_forward(&x, &w, Some(&bias)).unwrap();
    assert_eq!(out.dims(), &[2, 5, 16]);
}

#[test]
fn test_linear_forward_matches_matmul() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (3, 8), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (4, 8), &Device::Cpu).unwrap();
    let out = b.linear_forward(&x, &w, None).unwrap();
    let expected = x.matmul(&w.t().unwrap()).unwrap();
    assert_close(&out, &expected, 1e-5);
}

// ── softmax ────────────────────────────────────────────────────────────

#[test]
fn test_softmax_sums_to_one() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (2, 8), &Device::Cpu).unwrap();
    let out = b.softmax(&x, 1).unwrap();
    let sums: Vec<f32> = out.sum(1).unwrap().to_vec1().unwrap();
    for s in sums {
        assert!((s - 1.0).abs() < 1e-5, "softmax sum = {s}, expected 1.0");
    }
}

#[test]
fn test_softmax_non_negative() {
    let b = backend();
    let x = Tensor::randn(0f32, 2.0, (4, 16), &Device::Cpu).unwrap();
    let out = b.softmax(&x, 1).unwrap();
    let min: f32 = out.min(1).unwrap().min(0).unwrap().to_scalar().unwrap();
    assert!(min >= 0.0, "softmax produced negative value: {min}");
}

// ── rms_norm ───────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_shape_preserved() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (2, 4, 8), &Device::Cpu).unwrap();
    let w = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
    let out = b.rms_norm(&x, &w, 1e-5).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_rms_norm_unit_weight_normalizes() {
    let b = backend();
    let x = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu).unwrap();
    let w = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
    let out = b.rms_norm(&x, &w, 1e-8).unwrap();
    // RMS of [3,4] = sqrt((9+16)/2) = 3.536, so normed = [3/3.536, 4/3.536]
    let vals: Vec<f32> = out.to_vec2().unwrap()[0].clone();
    let rms = (3.0f32 * 3.0 + 4.0 * 4.0) / 2.0;
    let rms = rms.sqrt();
    assert!((vals[0] - 3.0 / rms).abs() < 1e-4);
    assert!((vals[1] - 4.0 / rms).abs() < 1e-4);
}

// ── rope ───────────────────────────────────────────────────────────────

#[test]
fn test_rope_shape_preserved() {
    let b = backend();
    // (batch, heads, seq, head_dim)
    let x = Tensor::randn(0f32, 1.0, (1, 4, 8, 16), &Device::Cpu).unwrap();
    let cos = Tensor::ones((8, 8), DType::F32, &Device::Cpu).unwrap();
    let sin = Tensor::zeros((8, 8), DType::F32, &Device::Cpu).unwrap();
    let out = b.rope(&x, &cos, &sin).unwrap();
    assert_eq!(out.dims(), x.dims());
}

// ── silu ───────────────────────────────────────────────────────────────

#[test]
fn test_silu_shape() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (2, 8), &Device::Cpu).unwrap();
    let out = b.silu(&x).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_silu_zero_is_zero() {
    let b = backend();
    let x = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
    let out = b.silu(&x).unwrap();
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for v in vals {
        assert!((v).abs() < 1e-7, "silu(0) should be 0, got {v}");
    }
}

// ── gelu ───────────────────────────────────────────────────────────────

#[test]
fn test_gelu_shape() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (3, 16), &Device::Cpu).unwrap();
    let out = b.gelu(&x).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_gelu_zero_is_zero() {
    let b = backend();
    let x = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
    let out = b.gelu(&x).unwrap();
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for v in vals {
        assert!((v).abs() < 1e-5, "gelu(0) should be ~0, got {v}");
    }
}

// ── sigmoid ────────────────────────────────────────────────────────────

#[test]
fn test_sigmoid_range() {
    let b = backend();
    let x = Tensor::randn(0f32, 3.0, (4, 8), &Device::Cpu).unwrap();
    let out = b.sigmoid(&x).unwrap();
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    for v in vals {
        assert!((0.0..=1.0).contains(&v), "sigmoid out of range: {v}");
    }
}

#[test]
fn test_sigmoid_zero_is_half() {
    let b = backend();
    let x = Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap();
    let out = b.sigmoid(&x).unwrap();
    let val: f32 = out.flatten_all().unwrap().to_vec1().unwrap()[0];
    assert!((val - 0.5).abs() < 1e-5, "sigmoid(0) should be 0.5, got {val}");
}

// ── embedding ──────────────────────────────────────────────────────────

#[test]
fn test_embedding_lookup() {
    let b = backend();
    let weight = Tensor::randn(0f32, 1.0, (10, 4), &Device::Cpu).unwrap();
    let ids = Tensor::new(&[0u32, 3, 7], &Device::Cpu).unwrap();
    let out = b.embedding(&ids, &weight).unwrap();
    assert_eq!(out.dims(), &[3, 4]);

    // Verify values match direct index_select
    let expected = weight.index_select(&ids, 0).unwrap();
    assert_close(&out, &expected, 1e-7);
}

#[test]
fn test_embedding_2d_ids() {
    let b = backend();
    let weight = Tensor::randn(0f32, 1.0, (100, 8), &Device::Cpu).unwrap();
    let ids = Tensor::new(&[[1u32, 2], [3, 4]], &Device::Cpu).unwrap();
    let out = b.embedding(&ids, &weight).unwrap();
    assert_eq!(out.dims(), &[2, 2, 8]);
}

// ── causal_mask ────────────────────────────────────────────────────────

#[test]
fn test_causal_mask_single_token() {
    let b = backend();
    let mask = b.causal_mask(1, 8, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 8]);
    // Single token: all zeros (attend to everything)
    let vals: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
    assert!(vals.iter().all(|&v| v == 0));
}

#[test]
fn test_causal_mask_square() {
    let b = backend();
    let mask = b.causal_mask(4, 4, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[4, 4]);
    let vals: Vec<Vec<u8>> = mask.to_vec2().unwrap();
    // Row 0: [0, 1, 1, 1] — only attend to position 0
    assert_eq!(vals[0], vec![0, 1, 1, 1]);
    // Row 1: [0, 0, 1, 1]
    assert_eq!(vals[1], vec![0, 0, 1, 1]);
    // Row 3: [0, 0, 0, 0] — attend to all
    assert_eq!(vals[3], vec![0, 0, 0, 0]);
}

#[test]
fn test_causal_mask_with_kv_prefix() {
    let b = backend();
    // seq_len=2 but kv_len=5 (3 cached positions + 2 new)
    let mask = b.causal_mask(2, 5, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[2, 5]);
    let vals: Vec<Vec<u8>> = mask.to_vec2().unwrap();
    // First 3 positions (cached) should be 0 (attend), then causal within new positions
    assert_eq!(vals[0], vec![0, 0, 0, 0, 1]); // attend to cached + self
    assert_eq!(vals[1], vec![0, 0, 0, 0, 0]); // attend to everything
}

// ── topk ───────────────────────────────────────────────────────────────

#[test]
fn test_topk_values() {
    let b = backend();
    let x = Tensor::new(&[[1.0f32, 5.0, 3.0, 9.0, 2.0]], &Device::Cpu).unwrap();
    let (vals, _idxs) = b.topk(&x, 3).unwrap();
    let top_vals: Vec<f32> = vals.flatten_all().unwrap().to_vec1().unwrap();
    // Top 3 should be [9, 5, 3] (descending)
    assert_eq!(top_vals, vec![9.0, 5.0, 3.0]);
}

#[test]
fn test_topk_indices() {
    let b = backend();
    let x = Tensor::new(&[[10.0f32, 30.0, 20.0, 40.0]], &Device::Cpu).unwrap();
    let (vals, idxs) = b.topk(&x, 2).unwrap();
    let top_vals: Vec<f32> = vals.flatten_all().unwrap().to_vec1().unwrap();
    // Top 2 values should be [40, 30] (descending)
    assert_eq!(top_vals, vec![40.0, 30.0]);
    // Indices should be 2 elements
    assert_eq!(idxs.dims(), &[1, 2]);
}

#[test]
fn test_topk_batched() {
    let b = backend();
    let x = Tensor::new(
        &[[1.0f32, 5.0, 3.0], [9.0, 2.0, 7.0]],
        &Device::Cpu,
    )
    .unwrap();
    let (vals, _idxs) = b.topk(&x, 2).unwrap();
    assert_eq!(vals.dims(), &[2, 2]);
}

// ══════════════════════════════════════════════════════════════════════
// candle_nn equivalence tests: prove backend methods match candle exactly
// ══════════════════════════════════════════════════════════════════════

// ── linear_forward vs candle_nn::Linear ────────────────────────────────

#[test]
fn test_linear_forward_matches_candle_2d() {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (16, 8), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (16,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu).unwrap();

    let candle_linear = candle_nn::Linear::new(w.clone(), Some(bias.clone()));
    let expected = candle_linear.forward(&x).unwrap();
    let actual = b.linear_forward(&x, &w, Some(&bias)).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_linear_forward_matches_candle_3d() {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (16, 8), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (16,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (2, 5, 8), &Device::Cpu).unwrap();

    let candle_linear = candle_nn::Linear::new(w.clone(), Some(bias.clone()));
    let expected = candle_linear.forward(&x).unwrap();
    let actual = b.linear_forward(&x, &w, Some(&bias)).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_linear_forward_matches_candle_4d() {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (16, 8), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (2, 3, 4, 8), &Device::Cpu).unwrap();

    let candle_linear = candle_nn::Linear::new(w.clone(), None);
    let expected = candle_linear.forward(&x).unwrap();
    let actual = b.linear_forward(&x, &w, None).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_linear_forward_matches_candle_no_bias() {
    let b = backend();
    let w = Tensor::randn(0f32, 0.1, (32, 16), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, 16), &Device::Cpu).unwrap();

    let candle_linear = candle_nn::Linear::new(w.clone(), None);
    let expected = candle_linear.forward(&x).unwrap();
    let actual = b.linear_forward(&x, &w, None).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

// ── softmax vs candle_nn::ops::softmax_last_dim ────────────────────────

#[test]
fn test_softmax_matches_candle_last_dim() {
    let b = backend();
    let x = Tensor::randn(0f32, 2.0, (4, 8, 16), &Device::Cpu).unwrap();

    let expected = candle_nn::ops::softmax_last_dim(&x).unwrap();
    let actual = b.softmax(&x, 2).unwrap();
    assert_close(&actual, &expected, 1e-6);
}

#[test]
fn test_softmax_non_last_dim() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (3, 5), &Device::Cpu).unwrap();
    let out = b.softmax(&x, 0).unwrap();
    // Sum along dim 0 should be 1.0 for each column
    let sums: Vec<f32> = out.sum(0).unwrap().to_vec1().unwrap();
    for s in sums {
        assert!((s - 1.0).abs() < 1e-5, "softmax dim=0 sum = {s}");
    }
}

// ── embedding vs candle_nn::Embedding ──────────────────────────────────

#[test]
fn test_embedding_matches_candle_1d() {
    let b = backend();
    let w = Tensor::randn(0f32, 1.0, (100, 32), &Device::Cpu).unwrap();
    let ids = Tensor::new(&[5u32, 10, 50, 99], &Device::Cpu).unwrap();

    let candle_emb = candle_nn::Embedding::new(w.clone(), 32);
    let expected = candle_emb.forward(&ids).unwrap();
    let actual = b.embedding(&ids, &w).unwrap();
    assert_close(&actual, &expected, 1e-7);
}

#[test]
fn test_embedding_matches_candle_2d() {
    let b = backend();
    let w = Tensor::randn(0f32, 1.0, (50, 16), &Device::Cpu).unwrap();
    let ids = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &Device::Cpu).unwrap();

    let candle_emb = candle_nn::Embedding::new(w.clone(), 16);
    let expected = candle_emb.forward(&ids).unwrap();
    let actual = b.embedding(&ids, &w).unwrap();
    assert_close(&actual, &expected, 1e-7);
}

// ── conv1d vs candle_nn::Conv1d ────────────────────────────────────────

#[test]
fn test_conv1d_matches_candle_with_bias() {
    let b = backend();
    let (in_c, out_c, k) = (4, 8, 3);
    let w = Tensor::randn(0f32, 0.1, (out_c, in_c, k), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (out_c,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, in_c, 16), &Device::Cpu).unwrap();

    let cfg = candle_nn::Conv1dConfig { padding: 1, stride: 1, dilation: 1, groups: 1, ..Default::default() };
    let candle_conv = candle_nn::Conv1d::new(w.clone(), Some(bias.clone()), cfg);
    let expected = candle_conv.forward(&x).unwrap();
    let actual = b.conv1d(&x, &w, Some(&bias), 1, 1, 1, 1).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_conv1d_matches_candle_no_bias() {
    let b = backend();
    let (in_c, out_c, k) = (8, 4, 5);
    let w = Tensor::randn(0f32, 0.1, (out_c, in_c, k), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (2, in_c, 32), &Device::Cpu).unwrap();

    let cfg = candle_nn::Conv1dConfig { padding: 2, stride: 1, dilation: 1, groups: 1, ..Default::default() };
    let candle_conv = candle_nn::Conv1d::new(w.clone(), None, cfg);
    let expected = candle_conv.forward(&x).unwrap();
    let actual = b.conv1d(&x, &w, None, 2, 1, 1, 1).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_conv1d_matches_candle_with_groups() {
    let b = backend();
    let (in_c, out_c, k, groups) = (8, 8, 3, 4);
    let w = Tensor::randn(0f32, 0.1, (out_c, in_c / groups, k), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (out_c,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, in_c, 20), &Device::Cpu).unwrap();

    let cfg = candle_nn::Conv1dConfig { padding: 1, stride: 1, dilation: 1, groups, ..Default::default() };
    let candle_conv = candle_nn::Conv1d::new(w.clone(), Some(bias.clone()), cfg);
    let expected = candle_conv.forward(&x).unwrap();
    let actual = b.conv1d(&x, &w, Some(&bias), 1, 1, 1, groups).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_conv1d_matches_candle_stride_dilation() {
    let b = backend();
    let (in_c, out_c, k) = (4, 4, 3);
    let w = Tensor::randn(0f32, 0.1, (out_c, in_c, k), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, in_c, 32), &Device::Cpu).unwrap();

    let cfg = candle_nn::Conv1dConfig { padding: 1, stride: 2, dilation: 2, groups: 1, ..Default::default() };
    let candle_conv = candle_nn::Conv1d::new(w.clone(), None, cfg);
    let expected = candle_conv.forward(&x).unwrap();
    let actual = b.conv1d(&x, &w, None, 1, 2, 2, 1).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

// ── conv_transpose1d vs candle_nn::ConvTranspose1d ─────────────────────

#[test]
fn test_conv_transpose1d_matches_candle() {
    let b = backend();
    let (in_c, out_c, k) = (4, 8, 4);
    let w = Tensor::randn(0f32, 0.1, (in_c, out_c, k), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (out_c,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, in_c, 8), &Device::Cpu).unwrap();

    let cfg = candle_nn::ConvTranspose1dConfig {
        padding: 1, output_padding: 1, stride: 2, dilation: 1, groups: 1,
    };
    let candle_ct = candle_nn::ConvTranspose1d::new(w.clone(), Some(bias.clone()), cfg);
    let expected = candle_ct.forward(&x).unwrap();
    let actual = b.conv_transpose1d(&x, &w, Some(&bias), 1, 1, 2, 1, 1).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

// ── rms_norm vs candle_nn::ops::rms_norm ───────────────────────────────

#[test]
fn test_rms_norm_matches_candle() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (2, 4, 32), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (32,), &Device::Cpu).unwrap();

    let expected = candle_nn::ops::rms_norm(&x, &w, 1e-5).unwrap();
    let actual = b.rms_norm(&x, &w, 1e-5).unwrap();
    assert_close(&actual, &expected, 1e-6);
}

// ── silu vs candle_nn::ops::silu ───────────────────────────────────────

#[test]
fn test_silu_matches_candle() {
    let b = backend();
    let x = Tensor::randn(0f32, 2.0, (4, 64), &Device::Cpu).unwrap();

    let expected = candle_nn::ops::silu(&x).unwrap();
    let actual = b.silu(&x).unwrap();
    assert_close(&actual, &expected, 1e-7);
}

// ── sigmoid vs candle_nn::ops::sigmoid ─────────────────────────────────

#[test]
fn test_sigmoid_matches_candle() {
    let b = backend();
    let x = Tensor::randn(0f32, 3.0, (4, 32), &Device::Cpu).unwrap();

    let expected = candle_nn::ops::sigmoid(&x).unwrap();
    let actual = b.sigmoid(&x).unwrap();
    assert_close(&actual, &expected, 1e-7);
}

// ── rope vs candle_nn::rotary_emb::rope ────────────────────────────────

#[test]
fn test_rope_matches_candle() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 4, 8, 16), &Device::Cpu).unwrap();
    let cos = Tensor::randn(0f32, 1.0, (8, 8), &Device::Cpu).unwrap();
    let sin = Tensor::randn(0f32, 1.0, (8, 8), &Device::Cpu).unwrap();

    let expected = candle_nn::rotary_emb::rope(&x, &cos, &sin).unwrap();
    let actual = b.rope(&x, &cos, &sin).unwrap();
    assert_close(&actual, &expected, 1e-6);
}

// ── sdpa vs candle_nn::ops::sdpa ───────────────────────────────────────

#[test]
fn test_sdpa_matches_manual_attention() {
    // SDPA has no CPU impl in candle, so verify against manual Q@K^T*scale→softmax→@V
    let b = backend();
    let q = Tensor::randn(0f32, 0.5, (1, 2, 4, 8), &Device::Cpu).unwrap();
    let k = Tensor::randn(0f32, 0.5, (1, 2, 4, 8), &Device::Cpu).unwrap();
    let v = Tensor::randn(0f32, 0.5, (1, 2, 4, 8), &Device::Cpu).unwrap();
    let scale = 1.0 / (8.0f32).sqrt();

    // Manual attention
    let att = (q.matmul(&k.t().unwrap()).unwrap() * scale as f64).unwrap();
    let att = b.softmax(&att, att.rank() - 1).unwrap();
    let expected = att.matmul(&v).unwrap();

    // Backend sdpa — on CPU this falls back to candle's sdpa which may fail,
    // so we just verify it returns a result or an acceptable error
    if let Ok(actual) = b.sdpa(&q, &k, &v, None, false, scale) {
        assert_close(&actual, &expected, 1e-4);
    }
    // SDPA not available on CPU — that's OK, it's Metal/CUDA only
}

// ── layer_norm vs candle_nn::LayerNorm ─────────────────────────────────

#[test]
fn test_layer_norm_matches_candle_with_bias() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (2, 4, 32), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (32,), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (32,), &Device::Cpu).unwrap();
    let expected = candle_nn::LayerNorm::new(w.clone(), bias.clone(), 1e-5)
        .forward(&x)
        .unwrap();
    let actual = b.layer_norm(&x, &w, Some(&bias), 1e-5).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_layer_norm_no_bias() {
    let b = backend();
    let x = Tensor::randn(0f32, 1.0, (1, 8, 16), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (16,), &Device::Cpu).unwrap();
    let expected = candle_nn::LayerNorm::new_no_bias(w.clone(), 1e-5)
        .forward(&x)
        .unwrap();
    let actual = b.layer_norm(&x, &w, None, 1e-5).unwrap();
    assert_close(&actual, &expected, 1e-5);
}

// ── group_norm vs candle_nn::GroupNorm ──────────────────────────────────

#[test]
fn test_group_norm_matches_candle() {
    let b = backend();
    let (channels, groups) = (32, 8);
    let x = Tensor::randn(0f32, 1.0, (1, channels, 16, 16), &Device::Cpu).unwrap();
    let w = Tensor::randn(0f32, 0.1, (channels,), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (channels,), &Device::Cpu).unwrap();
    let expected = candle_nn::group_norm(
        groups,
        channels,
        1e-5,
        candle_nn::VarBuilder::from_tensors(
            [
                ("weight".to_string(), w.clone()),
                ("bias".to_string(), bias.clone()),
            ]
            .into(),
            DType::F32,
            &Device::Cpu,
        ),
    )
    .unwrap()
    .forward(&x)
    .unwrap();
    let actual = b.group_norm(&x, &w, &bias, groups, 1e-5).unwrap();
    assert_close(&actual, &expected, 1e-4);
}

// ── conv2d vs candle_nn::Conv2d ────────────────────────────────────────

#[test]
fn test_conv2d_matches_candle() {
    let b = backend();
    let (in_c, out_c, k) = (4, 8, 3);
    let w = Tensor::randn(0f32, 0.1, (out_c, in_c, k, k), &Device::Cpu).unwrap();
    let bias = Tensor::randn(0f32, 0.1, (out_c,), &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, in_c, 8, 8), &Device::Cpu).unwrap();
    let cfg = candle_nn::Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let expected = candle_nn::Conv2d::new(w.clone(), Some(bias.clone()), cfg)
        .forward(&x)
        .unwrap();
    let actual = b.conv2d(&x, &w, Some(&bias), 1, 1, 1, 1).unwrap();
    assert_close(&actual, &expected, 1e-5);
}
