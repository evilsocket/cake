//! Tests for Transformer::load_for_vibevoice and forward_with_cache.

use candle_core::DType;

use super::helpers::*;
use cake_core::models::common::Transformer;

#[test]
fn test_load_for_vibevoice_succeeds() {
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg);
    assert!(block.is_ok(), "load_for_vibevoice should succeed");
}

#[test]
fn test_forward_with_cache_output_shape_prefill() {
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg).unwrap();
    let mut cache = make_cache(&cfg);

    let x = make_tensor(&[1, 4, 64], 50);
    let out = block.forward_with_cache(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(out.dims(), &[1, 4, 64]);
}

#[test]
fn test_forward_with_cache_output_shape_generation() {
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg).unwrap();
    let mut cache = make_cache(&cfg);

    // Prefill
    let x = make_tensor(&[1, 4, 64], 50);
    block.forward_with_cache(&x, 0, 0, &mut cache).unwrap();

    // Generation step: single token
    let x2 = make_tensor(&[1, 1, 64], 51);
    let out = block.forward_with_cache(&x2, 4, 0, &mut cache).unwrap();
    assert_eq!(out.dims(), &[1, 1, 64]);
}

#[test]
fn test_forward_with_cache_output_is_nonzero() {
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg).unwrap();
    let mut cache = make_cache(&cfg);

    let x = make_tensor(&[1, 2, 64], 52);
    let out = block.forward_with_cache(&x, 0, 0, &mut cache).unwrap();
    let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let sum: f32 = vals.iter().map(|v| v.abs()).sum();
    assert!(sum > 1e-6, "output should be non-zero, got sum={sum}");
}

#[test]
fn test_forward_with_cache_output_differs_from_input() {
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg).unwrap();
    let mut cache = make_cache(&cfg);

    let x = make_tensor(&[1, 2, 64], 53);
    let out = block.forward_with_cache(&x, 0, 0, &mut cache).unwrap();
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    let out_vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let diff: f32 = x_vals
        .iter()
        .zip(&out_vals)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "output should differ from input, diff={diff}");
}

#[test]
fn test_forward_with_cache_preserves_dtype() {
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg).unwrap();
    let mut cache = make_cache(&cfg);

    let x = make_tensor(&[1, 1, 64], 54);
    let out = block.forward_with_cache(&x, 0, 0, &mut cache).unwrap();
    assert_eq!(out.dtype(), DType::F32);
}
