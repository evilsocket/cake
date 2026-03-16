use cake_core::cake::Forwarder;
use cake_core::models::qwen3_5::Qwen3_5Block;

use super::bench_helpers::*;

#[divan::bench(args = [1, 4, 16])]
fn gated_delta_net_forward_prefill(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config_gdn();
    let layer_name = "model.language_model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    bencher.bench_local(|| {
        let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, seq_len, cfg.hidden_size], 400);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

#[divan::bench]
fn gated_delta_net_forward_generation(bencher: divan::Bencher) {
    let cfg = test_config_gdn();
    let layer_name = "model.language_model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    bencher.bench_local(|| {
        let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
        // Prefill
        let x_pre = make_tensor(&[1, 4, cfg.hidden_size], 401);
        rt.block_on(block.forward_mut(&x_pre, 0, 0, &mut ctx)).unwrap();
        // Generation step
        let x = make_tensor(&[1, 1, cfg.hidden_size], 402);
        rt.block_on(block.forward_mut(&x, 4, 0, &mut ctx)).unwrap()
    });
}

#[divan::bench(args = [1, 4, 16])]
fn gated_delta_net_full_attention(bencher: divan::Bencher, seq_len: usize) {
    let cfg = test_config_gdn();
    let layer_name = "model.language_model.layers.3"; // full_attention layer
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    bencher.bench_local(|| {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let q_size = num_heads * head_dim * 2;
        let kv_size = num_kv_heads * head_dim;
        let o_size = num_heads * head_dim;
        let i = cfg.intermediate_size;

        let mut map = std::collections::HashMap::new();
        let prefix = layer_name;
        map.insert(
            format!("{prefix}.input_layernorm.weight"),
            candle_core::Tensor::zeros(h, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
        );
        map.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            candle_core::Tensor::zeros(h, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
        );
        let attn_prefix = format!("{prefix}.self_attn");
        map.insert(format!("{attn_prefix}.q_proj.weight"), make_tensor(&[q_size, h], 70));
        map.insert(format!("{attn_prefix}.k_proj.weight"), make_tensor(&[kv_size, h], 71));
        map.insert(format!("{attn_prefix}.v_proj.weight"), make_tensor(&[kv_size, h], 72));
        map.insert(format!("{attn_prefix}.o_proj.weight"), make_tensor(&[h, o_size], 73));
        map.insert(
            format!("{attn_prefix}.q_norm.weight"),
            candle_core::Tensor::ones(head_dim, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
        );
        map.insert(
            format!("{attn_prefix}.k_norm.weight"),
            candle_core::Tensor::ones(head_dim, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
        );
        map.insert(format!("{prefix}.mlp.gate_proj.weight"), make_tensor(&[i, h], 74));
        map.insert(format!("{prefix}.mlp.up_proj.weight"), make_tensor(&[i, h], 75));
        map.insert(format!("{prefix}.mlp.down_proj.weight"), make_tensor(&[h, i], 76));

        let vb = candle_nn::VarBuilder::from_tensors(map, candle_core::DType::F32, &candle_core::Device::Cpu);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, seq_len, cfg.hidden_size], 500);
        rt.block_on(block.forward_mut(&x, 0, 3, &mut ctx)).unwrap()
    });
}

#[divan::bench]
fn gated_delta_net_prefill_then_generate(bencher: divan::Bencher) {
    let cfg = test_config_gdn();
    let layer_name = "model.language_model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    bencher.bench_local(|| {
        let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
        let x_pre = make_tensor(&[1, 8, cfg.hidden_size], 403);
        rt.block_on(block.forward_mut(&x_pre, 0, 0, &mut ctx)).unwrap();
        let x_gen = make_tensor(&[1, 1, cfg.hidden_size], 404);
        rt.block_on(block.forward_mut(&x_gen, 8, 0, &mut ctx)).unwrap()
    });
}
