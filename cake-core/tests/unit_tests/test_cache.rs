use candle_core::{DType, Device, Tensor};
use cake_core::models::common::{Cache, Config};

use super::helpers::*;

#[test]
fn test_cache_new() {
    let cfg = test_config();
    let cache = make_cache(&cfg);
    assert!(cache.with_kv_cache());
}

#[test]
fn test_cos_sin_shape() {
    let cfg = test_config();
    let cache = make_cache(&cfg);
    let cos = cache.cosine(0, 8, &Device::Cpu).unwrap();
    let sin = cache.sine(0, 8, &Device::Cpu).unwrap();
    // Cache returns (seq_len, head_dim/2) — verify both have same shape and are 2D
    assert_eq!(cos.dims().len(), 2);
    assert_eq!(cos.dims(), sin.dims());
    assert_eq!(cos.dims()[0], 8); // seq_len
}

#[test]
fn test_cos_sin_values_at_pos_0() {
    let cfg = test_config();
    let cache = make_cache(&cfg);
    let cos = cache.cosine(0, 1, &Device::Cpu).unwrap();
    // At position 0, cos(0 * freq) = 1.0 for all frequencies
    let cos_vals: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
    for &v in &cos_vals {
        assert!((v - 1.0).abs() < 1e-5, "cos at pos 0 should be 1.0, got {v}");
    }
}

#[test]
fn test_cos_sin_deterministic() {
    let cfg = test_config();
    let cache = make_cache(&cfg);
    let cos1 = cache.cosine(5, 3, &Device::Cpu).unwrap();
    let cos2 = cache.cosine(5, 3, &Device::Cpu).unwrap();
    let v1: Vec<f32> = cos1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = cos2.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2);
}

#[test]
fn test_mask_causal_shape() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let mask = cache.mask(8, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[8, 8]);
}

#[test]
fn test_mask_causal_upper_triangular() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let mask = cache.mask(4, &Device::Cpu).unwrap();
    // Mask may be U8 (boolean) or F32 depending on implementation.
    // Convert to F32 for uniform checking.
    let mask = mask.to_dtype(DType::F32).unwrap();
    let vals: Vec<Vec<f32>> = mask.to_vec2().unwrap();
    // Upper triangular (future) should be non-zero (masked), diagonal+lower should be 0
    for i in 0..4 {
        for j in 0..4 {
            if j > i {
                assert!(vals[i][j] != 0.0, "pos ({i},{j}) should be masked (non-zero)");
            } else {
                assert_eq!(vals[i][j], 0.0, "pos ({i},{j}) should be unmasked (0)");
            }
        }
    }
}

#[test]
fn test_kv_cache_accumulation() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    // First: 4 tokens
    let k1 = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v1 = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out, v_out) = cache.process_kv(0, k1, v1).unwrap();
    assert_eq!(k_out.dim(2).unwrap(), 4);
    assert_eq!(v_out.dim(2).unwrap(), 4);

    // Second: 1 more token (generation step)
    let k2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out, v_out) = cache.process_kv(0, k2, v2).unwrap();
    assert_eq!(k_out.dim(2).unwrap(), 5); // accumulated: 4 + 1
    assert_eq!(v_out.dim(2).unwrap(), 5);
}

#[test]
fn test_kv_cache_windowed() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;
    let window = 4;

    // First prefill: 6 tokens, stored in full
    let k1 = Tensor::zeros((1, num_kv_heads, 6, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v1 = Tensor::zeros((1, num_kv_heads, 6, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out1, _) = cache.process_kv_windowed(0, k1, v1, window).unwrap();
    // First call stores all, windowing applies on subsequent calls
    let initial_len = k_out1.dim(2).unwrap();

    // Second call: 1 more token, window should truncate
    let k2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out2, _) = cache.process_kv_windowed(0, k2, v2, window).unwrap();
    // Should be truncated to window size
    assert!(
        k_out2.dim(2).unwrap() <= window + 1,
        "windowed cache should be bounded: got {}, expected <= {}",
        k_out2.dim(2).unwrap(),
        window + 1
    );
}

#[test]
fn test_kv_cache_clear() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    let k = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    cache.process_kv(0, k, v).unwrap();

    cache.clear();

    // After clear, new kv should start fresh
    let k2 = Tensor::zeros((1, num_kv_heads, 2, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v2 = Tensor::zeros((1, num_kv_heads, 2, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out, _) = cache.process_kv(0, k2, v2).unwrap();
    assert_eq!(k_out.dim(2).unwrap(), 2); // fresh, not 4+2
}

#[test]
fn test_recurrent_state_roundtrip() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let state = make_tensor(&[1, 4, 8, 8], 100);
    cache.set_recurrent_state(0, state.clone());
    let retrieved = cache.get_recurrent_state(0).unwrap();
    let v1: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = retrieved.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2);
}

#[test]
fn test_conv_state_roundtrip() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let state = make_tensor(&[1, 32, 3], 101);
    cache.set_conv_state(0, state.clone());
    let retrieved = cache.get_conv_state(0).unwrap();
    let v1: Vec<f32> = state.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = retrieved.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2);
}
