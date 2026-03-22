use std::sync::Arc;
use cake_core::backends::CpuBackend;
use cake_core::models::common::{Cache, CausalSelfAttention, Config};
use candle_core::{DType, Device};

use super::bench_helpers::*;

fn cpu_backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

fn load_attn_and_cache(cfg: &Config) -> (CausalSelfAttention, Cache) {
    let vb = make_vb_attention(cfg);
    let attn = CausalSelfAttention::load(vb, cfg, cpu_backend()).unwrap();
    let cache = make_cache(cfg);
    (attn, cache)
}

#[divan::bench(args = [1, 8, 32])]
fn attention_forward(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config();
    let (attn, _) = load_attn_and_cache(&cfg);
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 100);

    bencher.bench_local(move || {
        let mut cache = make_cache(&cfg);
        attn.forward(&x, 0, 0, &mut cache).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn attention_generation_step(bencher: divan::Bencher, prefill_len: usize) {
    let cfg = test_config();
    let (attn, _) = load_attn_and_cache(&cfg);
    // Prefill to populate cache
    let x_prefill = make_tensor(&[1, prefill_len, cfg.hidden_size], 101);
    let mut cache = make_cache(&cfg);
    let _ = attn.forward(&x_prefill, 0, 0, &mut cache).unwrap();

    let x_gen = make_tensor(&[1, 1, cfg.hidden_size], 102);
    bencher.bench_local(move || {
        // Clone cache to avoid accumulating state across iterations
        let mut cache_copy = cache.clone();
        attn.forward(&x_gen, prefill_len, 0, &mut cache_copy).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn attention_with_qk_norm(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config_with_qk_norm();
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 200);

    bencher.bench_local(move || {
        let mut cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();
        attn.forward(&x, 0, 0, &mut cache).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn attention_sliding_window(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config_with_sliding_window(16);
    let vb = make_vb_attention(&cfg);
    let attn = CausalSelfAttention::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 300);

    bencher.bench_local(move || {
        let mut cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();
        attn.forward(&x, 0, 0, &mut cache).unwrap()
    });
}
