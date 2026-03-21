use candle_core::{DType, Device, Tensor};

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
    for (i, row) in vals.iter().enumerate().take(4) {
        for (j, &val) in row.iter().enumerate().take(4) {
            if j > i {
                assert!(val != 0.0, "pos ({i},{j}) should be masked (non-zero)");
            } else {
                assert_eq!(val, 0.0, "pos ({i},{j}) should be unmasked (0)");
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
    let _initial_len = k_out1.dim(2).unwrap();

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

#[test]
fn test_cache_new_no_cache() {
    let cache = cake_core::models::common::Cache::new_no_cache(&Device::Cpu).unwrap();
    assert!(!cache.with_kv_cache());
}

#[test]
fn test_cache_as_new_preserves_cos_sin_but_clears_kvs() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    // Store some kv
    let k = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    cache.process_kv(0, k, v).unwrap();

    let new_cache = cache.as_new();
    // cos/sin should work (preserved)
    let cos = new_cache.cosine(0, 4, &Device::Cpu).unwrap();
    assert_eq!(cos.dims()[0], 4);

    // kv should be cleared: new kv starts fresh
    let mut new_cache = new_cache;
    let k2 = Tensor::zeros((1, num_kv_heads, 2, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v2 = Tensor::zeros((1, num_kv_heads, 2, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out, _) = new_cache.process_kv(0, k2, v2).unwrap();
    assert_eq!(k_out.dim(2).unwrap(), 2); // fresh, not 4+2
}

#[test]
fn test_set_kv_resizes_beyond_current_length() {
    let cfg = test_config(); // 4 layers
    let mut cache = make_cache(&cfg);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    let k = Tensor::zeros((1, num_kv_heads, 2, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::zeros((1, num_kv_heads, 2, head_dim), DType::F32, &Device::Cpu).unwrap();

    // set_kv at index 10 (beyond initial 4 layers)
    cache.set_kv(10, k, v);

    // Should now be able to process_kv at block_idx=10
    let k2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out, _) = cache.process_kv(10, k2, v2).unwrap();
    assert_eq!(k_out.dim(2).unwrap(), 3); // 2 (set) + 1 (new)
}

#[test]
fn test_no_cache_returns_inputs_unchanged() {
    let cfg = test_config();
    let mut cache =
        cake_core::models::common::Cache::new(false, DType::F32, &cfg, &Device::Cpu).unwrap();
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    let k = make_tensor(&[1, num_kv_heads, 3, head_dim], 200);
    let v = make_tensor(&[1, num_kv_heads, 3, head_dim], 201);
    let (k_out, _v_out) = cache.process_kv(0, k.clone(), v.clone()).unwrap();
    // Without kv_cache, should return exact same tensors (seq_len=3, not accumulated)
    assert_eq!(k_out.dim(2).unwrap(), 3);

    let k2 = make_tensor(&[1, num_kv_heads, 1, head_dim], 202);
    let v2 = make_tensor(&[1, num_kv_heads, 1, head_dim], 203);
    let (k_out2, _) = cache.process_kv(0, k2, v2).unwrap();
    // Still 1, not 3+1, because no caching
    assert_eq!(k_out2.dim(2).unwrap(), 1);
}

#[test]
fn test_mask_caching_returns_same_tensor() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);

    let mask1 = cache.mask(4, &Device::Cpu).unwrap();
    let mask2 = cache.mask(4, &Device::Cpu).unwrap();
    // Both should have the same shape and values
    assert_eq!(mask1.dims(), mask2.dims());
    let v1: Vec<u8> = mask1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<u8> = mask2.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2);
}

#[test]
fn test_sliding_window_1_truncates() {
    let cfg = test_config();
    let mut cache = make_cache(&cfg);
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    // Prefill with 4 tokens
    let k1 = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v1 = Tensor::zeros((1, num_kv_heads, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
    cache.process_kv_windowed(0, k1, v1, 1).unwrap();

    // Add 1 more token with window=1
    let k2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let v2 = Tensor::zeros((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu).unwrap();
    let (k_out, _) = cache.process_kv_windowed(0, k2, v2, 1).unwrap();
    assert_eq!(k_out.dim(2).unwrap(), 1, "window=1 should truncate to 1 token");
}
