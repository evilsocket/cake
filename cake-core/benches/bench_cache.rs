use candle_core::{DType, Device};
use cake_core::models::common::Cache;

use super::bench_helpers::*;

#[divan::bench]
fn cache_new(bencher: divan::Bencher) {
    let cfg = test_config();
    bencher.bench_local(move || {
        Cache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn cache_cosine(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config();
    let cache = make_cache(&cfg);

    bencher.bench_local(move || {
        cache.cosine(0, seq_len, &Device::Cpu).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn cache_mask(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config();

    bencher.bench_local(move || {
        let mut cache = make_cache(&cfg);
        cache.mask(seq_len, &Device::Cpu).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn cache_process_kv(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config();
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    // k, v shape: (batch=1, num_kv_heads, seq_len, head_dim)
    let k = make_tensor(&[1, cfg.num_key_value_heads, seq_len, head_dim], 10);
    let v = make_tensor(&[1, cfg.num_key_value_heads, seq_len, head_dim], 11);

    bencher.bench_local(move || {
        let mut cache = make_cache(&cfg);
        cache.process_kv(0, k.clone(), v.clone()).unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn cache_process_kv_windowed(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config();
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let k = make_tensor(&[1, cfg.num_key_value_heads, seq_len, head_dim], 20);
    let v = make_tensor(&[1, cfg.num_key_value_heads, seq_len, head_dim], 21);
    let window = 16;

    bencher.bench_local(move || {
        let mut cache = make_cache(&cfg);
        cache.process_kv_windowed(0, k.clone(), v.clone(), window).unwrap()
    });
}
