//! Benchmarks for FLUX text-to-image model components.

#![cfg(feature = "flux")]

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
    super::bench_helpers::make_tensor(shape, seed)
}

// ── Timestep embedding ───────────────────────────────────────────────

#[divan::bench(args = [64, 128, 256, 3072])]
fn timestep_embedding(bencher: divan::Bencher, dim: usize) {
    let t = Tensor::new(&[0.5f32], &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        cake_core::models::flux::flux2_model::timestep_embedding(&t, dim, DType::F32).unwrap()
    });
}

// ── Fp8Linear forward ────────────────────────────────────────────────

#[divan::bench(args = [768, 3072])]
fn fp8_linear_forward(bencher: divan::Bencher, out_dim: usize) {
    let in_dim = 3072;
    let weight = make_tensor(&[out_dim, in_dim], 100);
    let linear = cake_core::models::flux::flux1_model::Fp8Linear::new(weight, None);
    let x = make_tensor(&[1, 16, in_dim], 101);
    bencher.bench_local(|| linear.forward(&x).unwrap());
}

// ── Flux2PosEmbed ────────────────────────────────────────────────────

#[divan::bench(args = [64, 256, 1024])]
fn flux2_pos_embed(bencher: divan::Bencher, seq_len: usize) {
    let pe = cake_core::models::flux::flux2_model::Flux2PosEmbed::new_pub(2000, vec![32, 32, 32, 32]);
    let ids = Tensor::zeros((seq_len, 4), DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| pe.forward(&ids).unwrap());
}

// ── F8E4M3 dequantization ────────────────────────────────────────────

#[divan::bench(args = [1024*1024, 3072*3072, 3072*9216])]
fn f8_to_f32_dequant(bencher: divan::Bencher, numel: usize) {
    // Create F8 tensor (simulate weight)
    let f32_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.001) % 2.0 - 1.0).collect();
    let t = Tensor::from_vec(f32_data, &[numel], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F8E4M3)
        .unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(numel))
        .bench_local(|| cake_core::backends::f8_dequant::f8e4m3_to_f32(&t).unwrap());
}

#[divan::bench(args = [1024*1024, 3072*3072, 3072*9216])]
fn f8_to_f16_dequant(bencher: divan::Bencher, numel: usize) {
    let f32_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.001) % 2.0 - 1.0).collect();
    let t = Tensor::from_vec(f32_data, &[numel], &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F8E4M3)
        .unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(numel))
        .bench_local(|| cake_core::backends::f8_dequant::f8e4m3_to_f16(&t).unwrap());
}

// ── Fp8Linear at FLUX.1 realistic sizes ──────────────────────────────

#[divan::bench(args = [4096, 1024, 256])]
fn fp8_linear_flux1_qkv(bencher: divan::Bencher, seq_len: usize) {
    // QKV projection: (seq_len, 3072) → (seq_len, 9216) [3*hidden_size]
    let weight = make_tensor(&[9216, 3072], 300)
        .to_dtype(DType::F8E4M3)
        .unwrap();
    let linear = cake_core::models::flux::flux1_model::Fp8Linear::new(weight, None);
    let x = make_tensor(&[1, seq_len, 3072], 301).to_dtype(DType::F16).unwrap();
    bencher.bench_local(|| linear.forward(&x).unwrap());
}

#[divan::bench(args = [4096, 1024, 256])]
fn fp8_linear_flux1_mlp(bencher: divan::Bencher, seq_len: usize) {
    // MLP first layer: (seq_len, 3072) → (seq_len, 12288) [4*hidden_size]
    let weight = make_tensor(&[12288, 3072], 310)
        .to_dtype(DType::F8E4M3)
        .unwrap();
    let linear = cake_core::models::flux::flux1_model::Fp8Linear::new(weight, None);
    let x = make_tensor(&[1, seq_len, 3072], 311).to_dtype(DType::F16).unwrap();
    bencher.bench_local(|| linear.forward(&x).unwrap());
}

// ── Scaled dot-product attention ─────────────────────────────────────

#[divan::bench(args = [256, 1024])]
fn sdpa_flux1(bencher: divan::Bencher, seq_len: usize) {
    // 24 heads, head_dim=128
    let q = make_tensor(&[1, 24, seq_len, 128], 400);
    let k = make_tensor(&[1, 24, seq_len, 128], 401);
    let v = make_tensor(&[1, 24, seq_len, 128], 402);
    bencher.bench_local(|| {
        let attn = (q.matmul(&k.t().unwrap()).unwrap() * (1.0 / 128.0f64.sqrt())).unwrap();
        candle_nn::ops::softmax_last_dim(&attn).unwrap().matmul(&v).unwrap()
    });
}

// ── VAE ResnetBlock2D ────────────────────────────────────────────────

#[divan::bench(args = [8, 16])]
fn vae_resnet_block(bencher: divan::Bencher, spatial: usize) {
    let h = 32;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("norm1.weight".into(), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("norm1.bias".into(), Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("norm2.weight".into(), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("norm2.bias".into(), Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("conv1.weight".into(), make_tensor(&[h, h, 3, 3], 200));
    map.insert("conv1.bias".into(), make_tensor(&[h], 201));
    map.insert("conv2.weight".into(), make_tensor(&[h, h, 3, 3], 202));
    map.insert("conv2.bias".into(), make_tensor(&[h], 203));

    let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let block = cake_core::models::flux::flux2_vae::ResnetBlock2D::load_pub(vb, h, h, 32).unwrap();
    let x = make_tensor(&[1, h, spatial, spatial], 204);
    bencher.bench_local(|| block.forward_pub(&x).unwrap());
}

// ── Timestep Embedding ──────────────────────────────────────────────

#[divan::bench(args = [64, 128, 256])]
fn flux1_timestep_embedding(bencher: divan::Bencher, dim: usize) {
    let t = Tensor::new(&[0.5f32, 0.3], &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        cake_core::models::flux::flux1_model::timestep_embedding(&t, dim, DType::F32).unwrap()
    });
}
