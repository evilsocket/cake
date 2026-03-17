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
    let linear = cake_core::models::flux::flux1_model::Fp8Linear::new_pub(weight, None);
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
