use std::sync::Arc;
use cake_core::backends::CpuBackend;
use cake_core::models::common::CausalSelfAttention;

use super::helpers::*;

fn cpu_backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

#[test]
fn test_attention_prefill_shape() {
    let cfg = test_config();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 8, 64], 60);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 8, 64]);
}

#[test]
fn test_attention_single_token_shape() {
    let cfg = test_config();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let mut cache = make_cache(&cfg);

    // Prefill 4 tokens first
    let x_pre = make_tensor(&[1, 4, 64], 61);
    attn.forward(&x_pre, 0, 0, &mut cache).unwrap();

    // Then generate 1 token
    let x_gen = make_tensor(&[1, 1, 64], 62);
    let y = attn.forward(&x_gen, 4, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[test]
fn test_attention_deterministic() {
    let cfg = test_config();
    let vb1 = make_vb_attention(&cfg);
    let vb2 = make_vb_attention(&cfg);
    let attn1 = CausalSelfAttention::load(vb1, &cfg, cpu_backend()).unwrap();
    let attn2 = CausalSelfAttention::load(vb2, &cfg, cpu_backend()).unwrap();
    let mut cache1 = make_cache(&cfg);
    let mut cache2 = make_cache(&cfg);

    let x = make_tensor(&[1, 4, 64], 63);
    let y1 = attn1.forward(&x, 0, 0, &mut cache1).unwrap();
    let y2 = attn2.forward(&x, 0, 0, &mut cache2).unwrap();

    let v1: Vec<f32> = y1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = y2.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2, "Attention should be deterministic");
}

#[test]
fn test_attention_with_qk_norm() {
    let cfg = test_config_with_qk_norm();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 64);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[test]
fn test_attention_sliding_window() {
    let cfg = test_config_with_sliding_window(4);
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load_custom(
        vb,
        &cfg,
        false,
        Some(4),
        true,
        cpu_backend(),
    )
    .unwrap();
    let mut cache = make_cache(&cfg);

    // Prefill 8 tokens with window=4
    let x = make_tensor(&[1, 8, 64], 65);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 8, 64]);
}

#[test]
fn test_attention_gqa() {
    // Default config already has 4 q heads, 2 kv heads (GQA 2:1)
    let cfg = test_config();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 66);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[test]
fn test_attention_no_rope() {
    // Gemma3 local layers: no RoPE
    let cfg = test_config();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load_custom(
        vb,
        &cfg,
        false,
        None,
        false, // no RoPE
        cpu_backend(),
    )
    .unwrap();
    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 67);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[test]
fn test_attention_with_bias() {
    let cfg = test_config_with_bias();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 68);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[test]
fn test_attention_nonzero_output() {
    let cfg = test_config();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 69);
    let y = attn.forward(&x, 0, 0, &mut cache).unwrap();
    let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    let nonzero = vals.iter().filter(|v| v.abs() > 1e-10).count();
    assert!(nonzero > 0, "Attention output should not be all zeros");
}
