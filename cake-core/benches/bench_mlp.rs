use std::sync::Arc;

use cake_core::backends::CpuBackend;
use cake_core::models::common::MLP;

use super::bench_helpers::*;

fn cpu_backend() -> Arc<CpuBackend> {
    Arc::new(CpuBackend::new())
}

#[divan::bench(args = [1, 8, 64])]
fn mlp_silu(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config();
    let vb = make_vb_mlp(&cfg);
    let mlp = MLP::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 100);

    bencher.bench_local(move || {
        mlp.forward(&x).unwrap()
    });
}

#[divan::bench(args = [1, 8, 64])]
fn mlp_gelu(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config_with_gelu();
    let vb = make_vb_mlp(&cfg);
    let mlp = MLP::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 200);

    bencher.bench_local(move || {
        mlp.forward(&x).unwrap()
    });
}

#[divan::bench(args = [128, 256, 512])]
fn mlp_larger_hidden(bencher: divan::Bencher, hidden_size: usize) {
    use cake_core::models::common::Config;
    let cfg = Config {
        hidden_size,
        intermediate_size: hidden_size * 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        ..test_config()
    };
    let vb = make_vb_mlp(&cfg);
    let mlp = MLP::load(vb, &cfg, cpu_backend()).unwrap();
    let x = make_tensor(&[1, 1, hidden_size], 300);

    bencher.bench_local(move || {
        mlp.forward(&x).unwrap()
    });
}
