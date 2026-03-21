//! Tests for LuxTTS model components (FeedforwardModule, ConvolutionModule).

use crate::helpers::make_tensor;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

// ─── FeedforwardModule ───────────────────────────────────────────────────────

fn make_vb_feedforward(dim: usize, ff_dim: usize) -> VarBuilder<'static> {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("in_proj.weight".into(), make_tensor(&[ff_dim, dim], 100));
    map.insert("in_proj.bias".into(), make_tensor(&[ff_dim], 101));
    map.insert("out_proj.weight".into(), make_tensor(&[dim, ff_dim], 102));
    map.insert("out_proj.bias".into(), make_tensor(&[dim], 103));
    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

#[test]
fn test_feedforward_load_and_forward() {
    let dim = 32;
    let ff_dim = 64;
    let vb = make_vb_feedforward(dim, ff_dim);
    let ff = cake_core::models::luxtts::feedforward::FeedforwardModule::load(dim, ff_dim, vb).unwrap();

    let x = make_tensor(&[1, 8, dim], 200);
    let out = ff.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 8, dim]);
    assert_eq!(out.dtype(), DType::F32);
}

#[test]
fn test_feedforward_different_batch() {
    let dim = 16;
    let ff_dim = 32;
    let vb = make_vb_feedforward(dim, ff_dim);
    let ff = cake_core::models::luxtts::feedforward::FeedforwardModule::load(dim, ff_dim, vb).unwrap();

    let x = make_tensor(&[4, 3, dim], 201);
    let out = ff.forward(&x).unwrap();
    assert_eq!(out.dims(), &[4, 3, dim]);
}

#[test]
fn test_feedforward_nonzero_output() {
    let dim = 16;
    let ff_dim = 32;
    let vb = make_vb_feedforward(dim, ff_dim);
    let ff = cake_core::models::luxtts::feedforward::FeedforwardModule::load(dim, ff_dim, vb).unwrap();

    let x = Tensor::ones(&[1, 4, dim], DType::F32, &Device::Cpu).unwrap();
    let out = ff.forward(&x).unwrap();
    let sum: f32 = out.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(sum > 0.0, "output should be non-zero");
}

// ─── ConvolutionModule ───────────────────────────────────────────────────────

fn make_vb_convolution(dim: usize, kernel_size: usize) -> VarBuilder<'static> {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    // in_proj: dim -> 2*dim (for GLU)
    map.insert("in_proj.weight".into(), make_tensor(&[2 * dim, dim], 300));
    map.insert("in_proj.bias".into(), make_tensor(&[2 * dim], 301));
    // depthwise conv: (dim, 1, kernel_size)
    map.insert(
        "depthwise_conv.weight".into(),
        make_tensor(&[dim, 1, kernel_size], 302),
    );
    map.insert("depthwise_conv.bias".into(), make_tensor(&[dim], 303));
    // out_proj: dim -> dim
    map.insert("out_proj.weight".into(), make_tensor(&[dim, dim], 304));
    map.insert("out_proj.bias".into(), make_tensor(&[dim], 305));
    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

#[test]
fn test_convolution_module_load_and_forward() {
    let dim = 16;
    let kernel_size = 3;
    let vb = make_vb_convolution(dim, kernel_size);
    let conv =
        cake_core::models::luxtts::convolution_module::ConvolutionModule::load(dim, kernel_size, vb).unwrap();

    let x = make_tensor(&[1, 8, dim], 400);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 8, dim]);
    assert_eq!(out.dtype(), DType::F32);
}

#[test]
fn test_convolution_module_preserves_seq_len() {
    let dim = 16;
    let kernel_size = 5;
    let vb = make_vb_convolution(dim, kernel_size);
    let conv =
        cake_core::models::luxtts::convolution_module::ConvolutionModule::load(dim, kernel_size, vb).unwrap();

    // Convolution with padding should preserve sequence length
    let x = make_tensor(&[2, 12, dim], 401);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 12, dim]);
}

#[test]
fn test_convolution_module_kernel_1() {
    let dim = 8;
    let kernel_size = 1;
    let vb = make_vb_convolution(dim, kernel_size);
    let conv =
        cake_core::models::luxtts::convolution_module::ConvolutionModule::load(dim, kernel_size, vb).unwrap();

    let x = make_tensor(&[1, 4, dim], 402);
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, dim]);
}

#[test]
fn test_convolution_module_nonzero_output() {
    let dim = 8;
    let kernel_size = 3;
    let vb = make_vb_convolution(dim, kernel_size);
    let conv =
        cake_core::models::luxtts::convolution_module::ConvolutionModule::load(dim, kernel_size, vb).unwrap();

    let x = Tensor::ones(&[1, 6, dim], DType::F32, &Device::Cpu).unwrap();
    let out = conv.forward(&x).unwrap();
    let sum: f32 = out.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(sum > 0.0, "output should be non-zero");
}
