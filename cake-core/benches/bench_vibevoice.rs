//! Benchmarks for VibeVoice TTS model components.

#![cfg(feature = "vibevoice")]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
    super::bench_helpers::make_tensor(shape, seed)
}

// ── Prediction Head ──────────────────────────────────────────────────

fn make_prediction_head(h: usize, latent: usize, layers: usize)
    -> cake_core::models::vibevoice::prediction_head::PredictionHead
{
    let intermediate = (h as f64 * 3.0) as usize;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("t_embedder.mlp.0.weight".into(), make_tensor(&[h, 256], 1));
    map.insert("t_embedder.mlp.2.weight".into(), make_tensor(&[h, h], 2));
    map.insert("noisy_images_proj.weight".into(), make_tensor(&[h, latent], 3));
    map.insert("cond_proj.weight".into(), make_tensor(&[h, h], 4));
    for i in 0..layers {
        let p = format!("layers.{i}");
        map.insert(format!("{p}.norm.weight"), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
        map.insert(format!("{p}.adaLN_modulation.1.weight"), make_tensor(&[3 * h, h], 10 + i as u64));
        map.insert(format!("{p}.ffn.gate_proj.weight"), make_tensor(&[intermediate, h], 20 + i as u64));
        map.insert(format!("{p}.ffn.up_proj.weight"), make_tensor(&[intermediate, h], 30 + i as u64));
        map.insert(format!("{p}.ffn.down_proj.weight"), make_tensor(&[h, intermediate], 40 + i as u64));
    }
    map.insert("final_layer.adaLN_modulation.1.weight".into(), make_tensor(&[2 * h, h], 60));
    map.insert("final_layer.linear.weight".into(), make_tensor(&[latent, h], 61));

    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let cfg = cake_core::models::vibevoice::config::DiffusionHeadConfig {
        ddpm_num_inference_steps: 20,
        ddpm_num_steps: 1000,
        head_layers: layers,
        hidden_size: h,
        latent_size: latent,
        head_ffn_ratio: 3.0,
        prediction_type: "v_prediction".into(),
        rms_norm_eps: 1e-5,
        ddpm_beta_schedule: "cosine".into(),
    };
    let sched = cake_core::models::vibevoice::ddpm::DpmSolverPP::new_cosine(cfg.ddpm_num_steps, cfg.ddpm_num_inference_steps);
    let backend = cake_core::backends::create_backend(&candle_core::Device::Cpu);
    cake_core::models::vibevoice::prediction_head::PredictionHead::load(vb, &cfg, sched.timesteps(), backend).unwrap()
}

#[divan::bench(args = [1, 4])]
fn prediction_head_forward(bencher: divan::Bencher, batch: usize) {
    let head = make_prediction_head(64, 16, 4);
    let x = make_tensor(&[batch, 16], 100);
    let t = Tensor::from_vec(vec![0.5f32; batch], batch, &Device::Cpu).unwrap();
    let cond = make_tensor(&[batch, 64], 101);
    bencher.bench_local(|| head.forward(&x, &t, &cond).unwrap());
}

// ── DDPM scheduler ───────────────────────────────────────────────────

#[divan::bench(args = [10, 20, 50])]
fn ddpm_full_loop(bencher: divan::Bencher, steps: usize) {
    let sched = cake_core::models::vibevoice::ddpm::DdpmScheduler::new_cosine(1000, steps);
    let zero = Tensor::zeros((1, 64), DType::F32, &Device::Cpu).unwrap();
    let noise = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let mut sample = noise.clone();
        let mut x0_buf: Vec<Tensor> = Vec::new();
        let mut ts_buf: Vec<usize> = Vec::new();
        for step_idx in 0..sched.timesteps().len() {
            sample = sched.step(&zero, step_idx, &sample, &mut x0_buf, &mut ts_buf).unwrap();
        }
        sample
    });
}

// ── Acoustic connector ───────────────────────────────────────────────

#[divan::bench]
fn acoustic_connector_forward(bencher: divan::Bencher) {
    let vae_dim = 64;
    let hidden = 896;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("fc1.weight".into(), make_tensor(&[hidden, vae_dim], 1));
    map.insert("fc1.bias".into(), make_tensor(&[hidden], 2));
    map.insert("norm.weight".into(), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
    map.insert("fc2.weight".into(), make_tensor(&[hidden, hidden], 3));
    map.insert("fc2.bias".into(), make_tensor(&[hidden], 4));
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let conn = cake_core::models::vibevoice::acoustic_connector::AcousticConnector::load(vb, vae_dim, hidden, 1e-5, cake_core::backends::create_backend(&Device::Cpu)).unwrap();
    let x = make_tensor(&[1, vae_dim], 10);
    bencher.bench_local(|| conn.forward(&x).unwrap());
}

// ── Acoustic Connector 1.5B (larger hidden) ─────────────────────────

#[divan::bench]
fn acoustic_connector_1_5b_forward(bencher: divan::Bencher) {
    let vae_dim = 64;
    let hidden = 1536; // 1.5B model hidden size
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("fc1.weight".into(), make_tensor(&[hidden, vae_dim], 1));
    map.insert("fc1.bias".into(), make_tensor(&[hidden], 2));
    map.insert("norm.weight".into(), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
    map.insert("fc2.weight".into(), make_tensor(&[hidden, hidden], 3));
    map.insert("fc2.bias".into(), make_tensor(&[hidden], 4));
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let conn = cake_core::models::vibevoice::acoustic_connector::AcousticConnector::load(vb, vae_dim, hidden, 1e-6, cake_core::backends::create_backend(&Device::Cpu)).unwrap();
    let x = make_tensor(&[1, vae_dim], 10);
    bencher.bench_local(|| conn.forward(&x).unwrap());
}

// ── Semantic Connector 1.5B ─────────────────────────────────────────

#[divan::bench]
fn semantic_connector_1_5b_forward(bencher: divan::Bencher) {
    let vae_dim = 128; // semantic tokenizer dim
    let hidden = 1536;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("fc1.weight".into(), make_tensor(&[hidden, vae_dim], 1));
    map.insert("fc1.bias".into(), make_tensor(&[hidden], 2));
    map.insert("norm.weight".into(), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
    map.insert("fc2.weight".into(), make_tensor(&[hidden, hidden], 3));
    map.insert("fc2.bias".into(), make_tensor(&[hidden], 4));
    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    let conn = cake_core::models::vibevoice::acoustic_connector::AcousticConnector::load(vb, vae_dim, hidden, 1e-6, cake_core::backends::create_backend(&Device::Cpu)).unwrap();
    let x = make_tensor(&[1, 1, vae_dim], 10); // (batch, 1 frame, 128)
    bencher.bench_local(|| conn.forward(&x).unwrap());
}

// ── StreamingConvCache ──────────────────────────────────────────────

#[divan::bench]
fn streaming_cache_take_set_cycle(bencher: divan::Bencher) {
    let t = Tensor::zeros((1, 32, 6), DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        let mut cache = cake_core::models::vibevoice::vae_decoder::StreamingConvCache::new(30);
        for _ in 0..26 {
            let (slot, _) = cache.take_slot();
            cache.set(slot, t.clone());
        }
        cache.reset_counter();
        for _ in 0..26 {
            let (slot, _) = cache.take_slot();
            cache.set(slot, t.clone());
        }
    });
}

// ── Prediction Head 1.5B (1536 hidden) ──────────────────────────────

#[divan::bench(args = [1, 4])]
fn prediction_head_1_5b_forward(bencher: divan::Bencher, batch: usize) {
    let head = make_prediction_head(128, 64, 4); // Scaled down for CPU bench
    let x = make_tensor(&[batch, 64], 100);
    let t = Tensor::from_vec(vec![0.5f32; batch], batch, &Device::Cpu).unwrap();
    let cond = make_tensor(&[batch, 128], 101);
    bencher.bench_local(|| head.forward(&x, &t, &cond).unwrap());
}
