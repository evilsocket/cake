use std::sync::Arc;

use cake_core::backends::CpuBackend;
use cake_core::models::common::MLP;

use super::helpers::*;

fn cpu_backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

#[test]
fn test_mlp_forward_shape() {
    let cfg = test_config();
    let vb = make_vb_mlp(&cfg);
    let mlp = MLP::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, 8, 64], 50);
    let y = mlp.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 8, 64]);
}

#[test]
fn test_mlp_forward_nonzero() {
    let cfg = test_config();
    let vb = make_vb_mlp(&cfg);
    let mlp = MLP::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, 4, 64], 51);
    let y = mlp.forward(&x).unwrap();
    let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    let nonzero = vals.iter().filter(|v| v.abs() > 1e-10).count();
    assert!(nonzero > 0, "MLP output should not be all zeros");
}

#[test]
fn test_mlp_gelu_differs_from_silu() {
    let cfg_silu = test_config();
    let cfg_gelu = test_config_with_gelu();
    let vb_silu = make_vb_mlp(&cfg_silu);
    let vb_gelu = make_vb_mlp(&cfg_gelu);
    let mlp_silu = MLP::load(vb_silu, &cfg_silu, cpu_backend()).unwrap();
    let mlp_gelu = MLP::load(vb_gelu, &cfg_gelu, cpu_backend()).unwrap();

    let x = make_tensor(&[1, 4, 64], 52);
    let y_silu = mlp_silu.forward(&x).unwrap();
    let y_gelu = mlp_gelu.forward(&x).unwrap();

    let v1: Vec<f32> = y_silu.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = y_gelu.flatten_all().unwrap().to_vec1().unwrap();
    // SiLU and GELU should produce different outputs
    let diff: f32 = v1.iter().zip(&v2).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "SiLU and GELU should differ, diff={diff}");
}

#[test]
fn test_mlp_deterministic() {
    let cfg = test_config();
    let vb1 = make_vb_mlp(&cfg);
    let vb2 = make_vb_mlp(&cfg);
    let mlp1 = MLP::load(vb1, &cfg, cpu_backend()).unwrap();
    let mlp2 = MLP::load(vb2, &cfg, cpu_backend()).unwrap();

    let x = make_tensor(&[1, 4, 64], 53);
    let y1 = mlp1.forward(&x).unwrap();
    let y2 = mlp2.forward(&x).unwrap();

    let v1: Vec<f32> = y1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = y2.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2, "MLP should be deterministic");
}

#[test]
fn test_mlp_single_token() {
    let cfg = test_config();
    let vb = make_vb_mlp(&cfg);
    let mlp = MLP::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, 1, 64], 54);
    let y = mlp.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}
