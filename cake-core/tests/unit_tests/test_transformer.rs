use std::sync::Arc;

use candle_core::Module;

use cake_core::backends::CpuBackend;

use super::helpers::*;

fn cpu_backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

// Transformer requires a full Context with var_builder, so we test the
// composition of attention + mlp + norm at the component level instead.

#[test]
fn test_transformer_components_compose() {
    use cake_core::models::common::{CausalSelfAttention, MLP};
    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);

    // Load components with correct prefixes
    let attn = CausalSelfAttention::load(vb.pp("self_attn"), &cfg, cpu_backend()).unwrap();
    let mlp = MLP::load(vb.pp("mlp"), &cfg, cpu_backend()).unwrap();

    // Load norms
    let norm1_w = vb.pp("input_layernorm").get(64, "weight").unwrap();
    let norm1 = candle_nn::RmsNorm::new(norm1_w, 1e-6);
    let norm2_w = vb.pp("post_attention_layernorm").get(64, "weight").unwrap();
    let norm2 = candle_nn::RmsNorm::new(norm2_w, 1e-6);

    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 70);

    // Pre-norm → attention → residual
    let normed = norm1.forward(&x).unwrap();
    let attn_out = attn.forward(&normed, 0, 0, &mut cache).unwrap();
    let residual1 = (&x + &attn_out).unwrap();

    // Pre-norm → MLP → residual
    let normed2 = norm2.forward(&residual1).unwrap();
    let mlp_out = mlp.forward(&normed2).unwrap();
    let output = (&residual1 + &mlp_out).unwrap();

    assert_eq!(output.dims(), &[1, 4, 64]);

    // Output should be close to input (small weights) but not identical
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    let out_vals: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    let diff: f32 = x_vals
        .iter()
        .zip(&out_vals)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "Output should differ from input due to perturbation");
}

#[test]
fn test_transformer_residual_preserves_shape() {
    use cake_core::models::common::{CausalSelfAttention, MLP};

    let cfg = test_config();
    let vb = make_vb_transformer_block(&cfg);
    let attn = CausalSelfAttention::load(vb.pp("self_attn"), &cfg, cpu_backend()).unwrap();
    let mlp = MLP::load(vb.pp("mlp"), &cfg, cpu_backend()).unwrap();

    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 1, 64], 71); // single token

    let attn_out = attn.forward(&x, 0, 0, &mut cache).unwrap();
    let residual = (&x + &attn_out).unwrap();
    let mlp_out = mlp.forward(&residual).unwrap();
    let output = (&residual + &mlp_out).unwrap();

    assert_eq!(output.dims(), &[1, 1, 64]);
}

#[test]
fn test_transformer_with_qk_norm() {
    use cake_core::models::common::{CausalSelfAttention, MLP};

    let cfg = test_config_with_qk_norm();
    let vb = make_vb_transformer_block(&cfg);
    let attn = CausalSelfAttention::load(vb.pp("self_attn"), &cfg, cpu_backend()).unwrap();
    let mlp = MLP::load(vb.pp("mlp"), &cfg, cpu_backend()).unwrap();

    let mut cache = make_cache(&cfg);
    let x = make_tensor(&[1, 4, 64], 72);

    let attn_out = attn.forward(&x, 0, 0, &mut cache).unwrap();
    let residual = (&x + &attn_out).unwrap();
    let mlp_out = mlp.forward(&residual).unwrap();
    let output = (&residual + &mlp_out).unwrap();

    assert_eq!(output.dims(), &[1, 4, 64]);
}
