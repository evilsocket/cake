use cake_core::cake::Forwarder;
use cake_core::models::common::Config;

use super::bench_helpers::*;

fn gemma3_config() -> Config {
    Config {
        residual_rms_norm: true,
        use_qk_norm: true,
        sliding_window: Some(16),
        global_layers: vec![false, false, false, true],
        ..test_config()
    }
}

fn olmo2_config() -> Config {
    Config {
        use_qk_norm: true,
        pre_reshape_qk_norm: true,
        ..test_config()
    }
}

fn exaone4_config() -> Config {
    Config {
        use_qk_norm: true,
        sliding_window: Some(16),
        global_layers: vec![false, false, false, true],
        ..test_config()
    }
}

fn qwen3_moe_config() -> Config {
    Config {
        use_qk_norm: true,
        moe_intermediate_size: Some(64),
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        ..test_config()
    }
}

// -- Transformer block (LLaMA / Qwen2 / Qwen3 / Phi4 / Mistral / Falcon3) --

#[divan::bench]
fn transformer_block(bencher: divan::Bencher) {
    let cfg = test_config();
    let layer_name = "model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_standard_block(&cfg, layer_name, &[], false, false);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::common::Transformer::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 900);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

// -- Gemma3Block --

#[divan::bench]
fn gemma3_block(bencher: divan::Bencher) {
    let cfg = gemma3_config();
    let layer_name = "model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_standard_block(
            &cfg,
            layer_name,
            &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
            true,
            false,
        );
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::gemma3::Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 100);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

// -- OLMo2Block --

#[divan::bench]
fn olmo2_block(bencher: divan::Bencher) {
    let cfg = olmo2_config();
    let layer_name = "model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb =
            make_vb_standard_block(&cfg, layer_name, &["post_feedforward_layernorm"], true, false);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::olmo2::OLMo2Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 200);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

// -- EXAONE4Block --

#[divan::bench]
fn exaone4_block(bencher: divan::Bencher) {
    let cfg = exaone4_config();
    let layer_name = "model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_standard_block(&cfg, layer_name, &[], true, false);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::exaone4::EXAONE4Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 300);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

// -- Qwen3_5Block (linear attention: GatedDeltaNet) --

#[divan::bench]
fn qwen3_5_linear_block(bencher: divan::Bencher) {
    let cfg = test_config_gdn();
    let layer_name = "model.language_model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::qwen3_5::Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 400);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

// -- Qwen3_5Block (full attention) --

#[divan::bench]
fn qwen3_5_full_block(bencher: divan::Bencher) {
    let cfg = test_config_gdn();
    let layer_name = "model.language_model.layers.3";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_qwen3_5_full(&cfg, layer_name);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::qwen3_5::Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 500);
        rt.block_on(block.forward_mut(&x, 0, 3, &mut ctx)).unwrap()
    });
}

// -- Qwen3MoeBlock --

#[divan::bench]
fn qwen3_moe_block(bencher: divan::Bencher) {
    let cfg = qwen3_moe_config();
    let layer_name = "model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_standard_block(&cfg, layer_name, &[], true, true);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::qwen3_moe::Qwen3MoeBlock::load(layer_name.to_string(), &ctx)
                .unwrap();
        let x = make_tensor(&[1, 4, 64], 600);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}

// -- Qwen3_5MoeBlock --

#[divan::bench]
fn qwen3_5_moe_block(bencher: divan::Bencher) {
    let cfg = qwen3_5_moe_config();
    let layer_name = "model.layers.0";
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let vb = make_vb_qwen3_5_moe_linear(&cfg, layer_name);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block =
            *cake_core::models::qwen3_5_moe::block::Qwen3_5MoeBlock::load(
                layer_name.to_string(),
                &ctx,
            )
            .unwrap();
        let x = make_tensor(&[1, 4, 64], 700);
        rt.block_on(block.forward_mut(&x, 0, 0, &mut ctx)).unwrap()
    });
}
