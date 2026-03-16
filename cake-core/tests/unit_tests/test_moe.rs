use cake_core::models::common::Config;
use cake_core::models::qwen3_moe::moe::SparseMoeMlp;

use super::helpers::*;

fn moe_config() -> Config {
    Config {
        moe_intermediate_size: Some(64),
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        ..test_config()
    }
}

fn make_vb_moe(cfg: &Config) -> candle_nn::VarBuilder<'static> {
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size.unwrap();
    let n = cfg.num_experts;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    // Router
    map.insert("gate.weight".into(), make_tensor(&[n, h], 40));

    // Per-expert weights
    for j in 0..n {
        map.insert(
            format!("experts.{j}.gate_proj.weight"),
            make_tensor(&[i, h], 41 + j as u64 * 3),
        );
        map.insert(
            format!("experts.{j}.up_proj.weight"),
            make_tensor(&[i, h], 42 + j as u64 * 3),
        );
        map.insert(
            format!("experts.{j}.down_proj.weight"),
            make_tensor(&[h, i], 43 + j as u64 * 3),
        );
    }

    candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

#[test]
fn test_moe_forward_shape() {
    let cfg = moe_config();
    let vb = make_vb_moe(&cfg);
    let moe = SparseMoeMlp::load(vb, &cfg).unwrap();
    let x = make_tensor(&[1, 4, 64], 80);
    let y = moe.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[test]
fn test_moe_forward_nonzero() {
    let cfg = moe_config();
    let vb = make_vb_moe(&cfg);
    let moe = SparseMoeMlp::load(vb, &cfg).unwrap();
    let x = make_tensor(&[1, 4, 64], 81);
    let y = moe.forward(&x).unwrap();
    let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    let nonzero = vals.iter().filter(|v| v.abs() > 1e-10).count();
    assert!(nonzero > 0, "MoE output should not be all zeros");
}

#[test]
fn test_moe_single_token() {
    let cfg = moe_config();
    let vb = make_vb_moe(&cfg);
    let moe = SparseMoeMlp::load(vb, &cfg).unwrap();
    let x = make_tensor(&[1, 1, 64], 82);
    let y = moe.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[test]
fn test_moe_deterministic() {
    let cfg = moe_config();
    let vb1 = make_vb_moe(&cfg);
    let vb2 = make_vb_moe(&cfg);
    let moe1 = SparseMoeMlp::load(vb1, &cfg).unwrap();
    let moe2 = SparseMoeMlp::load(vb2, &cfg).unwrap();
    let x = make_tensor(&[1, 4, 64], 83);
    let y1 = moe1.forward(&x).unwrap();
    let y2 = moe2.forward(&x).unwrap();
    let v1: Vec<f32> = y1.flatten_all().unwrap().to_vec1().unwrap();
    let v2: Vec<f32> = y2.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(v1, v2, "MoE should be deterministic");
}
