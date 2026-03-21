use cake_core::models::common::Config;
use cake_core::models::qwen3_moe::moe::SparseMoeMlp;
use cake_core::models::qwen3_5_moe::moe::Qwen3_5MoeSparseMlp;

use super::bench_helpers::*;

fn moe_config() -> Config {
    Config {
        moe_intermediate_size: Some(64),
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        ..test_config()
    }
}

fn qwen3_5_moe_config() -> Config {
    Config {
        moe_intermediate_size: Some(32),
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        shared_expert_intermediate_size: Some(48),
        ..test_config()
    }
}

#[divan::bench(args = [1, 4, 16])]
fn qwen3_moe_forward(bencher: divan::Bencher, seq_len: usize) {
    let cfg = moe_config();
    let vb = make_vb_moe(&cfg);
    let moe = SparseMoeMlp::load(vb, &cfg, std::sync::Arc::new(cake_core::backends::CpuBackend::new())).unwrap();
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 100);

    bencher.bench_local(move || {
        moe.forward(&x).unwrap()
    });
}

#[divan::bench(args = [1, 4, 16])]
fn qwen3_5_moe_forward(bencher: divan::Bencher, seq_len: usize) {
    let cfg = qwen3_5_moe_config();
    let vb = make_vb_qwen3_5_moe(&cfg);
    let moe = Qwen3_5MoeSparseMlp::load(vb, &cfg, std::sync::Arc::new(cake_core::backends::CpuBackend::new())).unwrap();
    let x = make_tensor(&[1, seq_len, cfg.hidden_size], 200);

    bencher.bench_local(move || {
        moe.forward(&x).unwrap()
    });
}
