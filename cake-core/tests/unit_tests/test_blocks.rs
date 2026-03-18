//! Unit tests for model-specific transformer blocks.
//!
//! Each test builds a VarBuilder with all required weight tensors,
//! constructs a minimal Context, loads the block via `Forwarder::load`,
//! and verifies that forward produces output with the same shape as input.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use cake_core::cake::{Context, Forwarder, Topology};
use cake_core::models::common::{Cache, Config, LinearAttnConfig};
use cake_core::{Args, TextModelArch};

use super::helpers::*;

// ---------------------------------------------------------------------------
// Helper: build a minimal Context from a Config + VarBuilder
// ---------------------------------------------------------------------------

fn make_context(cfg: Config, vb: VarBuilder<'static>) -> Context {
    let cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();
    Context {
        args: Args::default(),
        dtype: DType::F32,
        topology: Topology::new(),
        data_path: PathBuf::from("."),
        device: Device::Cpu,
        config: Some(cfg),
        cache: Some(cache),
        var_builder: Some(vb),
        text_model_arch: TextModelArch::Auto,
        quant: Arc::new(cake_core::utils::NoQuantization),
        listener_override: Arc::new(Mutex::new(None)),
    }
}

// ---------------------------------------------------------------------------
// Helper: build VarBuilder for a standard block (attn + mlp + norms)
// The layer_name is the prefix key used in the VarBuilder.
// ---------------------------------------------------------------------------

/// Build a VarBuilder for a standard pre-norm block with CausalSelfAttention + MLP.
/// Works for Gemma3, OLMo2, EXAONE4, and Qwen3MoeBlock patterns.
fn make_vb_standard_block(
    cfg: &Config,
    layer_name: &str,
    extra_norms: &[&str],
    with_qk_norm: bool,
    with_moe: bool,
) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
    let size_q = head_dim * cfg.num_attention_heads;
    let size_kv = head_dim * cfg.num_key_value_heads;
    let i = cfg.intermediate_size;

    let mut map: HashMap<String, Tensor> = HashMap::new();
    let prefix = layer_name;

    // Standard norms
    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );

    // Extra norms (Gemma3 has 2 additional sandwich norms)
    for norm_name in extra_norms {
        map.insert(
            format!("{prefix}.{norm_name}.weight"),
            Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
        );
    }

    // Attention weights
    map.insert(
        format!("{prefix}.self_attn.q_proj.weight"),
        make_tensor(&[size_q, h], 30),
    );
    map.insert(
        format!("{prefix}.self_attn.k_proj.weight"),
        make_tensor(&[size_kv, h], 31),
    );
    map.insert(
        format!("{prefix}.self_attn.v_proj.weight"),
        make_tensor(&[size_kv, h], 32),
    );
    map.insert(
        format!("{prefix}.self_attn.o_proj.weight"),
        make_tensor(&[h, size_q], 33),
    );

    if with_qk_norm {
        let norm_dim = if cfg.pre_reshape_qk_norm {
            size_q
        } else {
            head_dim
        };
        let norm_kv_dim = if cfg.pre_reshape_qk_norm {
            size_kv
        } else {
            head_dim
        };
        map.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            make_tensor(&[norm_dim], 34),
        );
        map.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            make_tensor(&[norm_kv_dim], 35),
        );
    }

    if with_moe {
        // MoE weights
        let moe_i = cfg.moe_intermediate_size.unwrap();
        let n = cfg.num_experts;
        map.insert(
            format!("{prefix}.mlp.gate.weight"),
            make_tensor(&[n, h], 40),
        );
        for j in 0..n {
            map.insert(
                format!("{prefix}.mlp.experts.{j}.gate_proj.weight"),
                make_tensor(&[moe_i, h], 41 + j as u64 * 3),
            );
            map.insert(
                format!("{prefix}.mlp.experts.{j}.up_proj.weight"),
                make_tensor(&[moe_i, h], 42 + j as u64 * 3),
            );
            map.insert(
                format!("{prefix}.mlp.experts.{j}.down_proj.weight"),
                make_tensor(&[h, moe_i], 43 + j as u64 * 3),
            );
        }
    } else {
        // Dense MLP weights
        map.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            make_tensor(&[i, h], 36),
        );
        map.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            make_tensor(&[i, h], 37),
        );
        map.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            make_tensor(&[h, i], 38),
        );
    }

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

// ---------------------------------------------------------------------------
// Gemma3Block tests
// ---------------------------------------------------------------------------

fn gemma3_config() -> Config {
    Config {
        residual_rms_norm: true,
        use_qk_norm: true,
        sliding_window: Some(16),
        // Layer 0 = local, layer 3 = global (every 4th)
        global_layers: vec![false, false, false, true],
        ..test_config()
    }
}

#[tokio::test]
async fn test_gemma3_block_local_prefill() {
    use cake_core::models::gemma3::Gemma3Block;

    let cfg = gemma3_config();
    let layer_name = "model.layers.0"; // local layer
    let vb = make_vb_standard_block(
        &cfg,
        layer_name,
        &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
        true,
        false,
    );
    let ctx = make_context(cfg, vb);
    let mut block = *Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 100);
    let mut ctx = ctx;
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_gemma3_block_local_generation() {
    use cake_core::models::gemma3::Gemma3Block;

    let cfg = gemma3_config();
    let layer_name = "model.layers.1";
    let vb = make_vb_standard_block(
        &cfg,
        layer_name,
        &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
        true,
        false,
    );
    let mut ctx = make_context(cfg, vb);
    let mut block = *Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 101);
    let y = block.forward_mut(&x, 0, 1, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_gemma3_block_global_prefill() {
    use cake_core::models::gemma3::Gemma3Block;

    let cfg = gemma3_config();
    let layer_name = "model.layers.3"; // global layer
    let vb = make_vb_standard_block(
        &cfg,
        layer_name,
        &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
        true,
        false,
    );
    let mut ctx = make_context(cfg, vb);
    let mut block = *Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 102);
    let y = block.forward_mut(&x, 0, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_gemma3_block_global_generation() {
    use cake_core::models::gemma3::Gemma3Block;

    let cfg = gemma3_config();
    let layer_name = "model.layers.3";
    let vb = make_vb_standard_block(
        &cfg,
        layer_name,
        &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
        true,
        false,
    );
    let mut ctx = make_context(cfg, vb);
    let mut block = *Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
    // Prefill first to populate KV cache
    let x_prefill = make_tensor(&[1, 3, 64], 103);
    let _ = block.forward_mut(&x_prefill, 0, 3, &mut ctx).await.unwrap();
    // Generation step
    let x = make_tensor(&[1, 1, 64], 104);
    let y = block.forward_mut(&x, 3, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

// ---------------------------------------------------------------------------
// OLMo2Block tests
// ---------------------------------------------------------------------------

fn olmo2_config() -> Config {
    Config {
        use_qk_norm: true,
        pre_reshape_qk_norm: true,
        ..test_config()
    }
}

#[tokio::test]
async fn test_olmo2_block_prefill() {
    use cake_core::models::olmo2::OLMo2Block;

    let cfg = olmo2_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &["post_feedforward_layernorm"], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *OLMo2Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 200);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_olmo2_block_generation() {
    use cake_core::models::olmo2::OLMo2Block;

    let cfg = olmo2_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &["post_feedforward_layernorm"], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *OLMo2Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 201);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_olmo2_block_prefill_then_generate() {
    use cake_core::models::olmo2::OLMo2Block;

    let cfg = olmo2_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &["post_feedforward_layernorm"], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *OLMo2Block::load(layer_name.to_string(), &ctx).unwrap();
    // Prefill
    let x_prefill = make_tensor(&[1, 4, 64], 202);
    let _ = block.forward_mut(&x_prefill, 0, 0, &mut ctx).await.unwrap();
    // Generation
    let x = make_tensor(&[1, 1, 64], 203);
    let y = block.forward_mut(&x, 4, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

// ---------------------------------------------------------------------------
// EXAONE4Block tests
// ---------------------------------------------------------------------------

fn exaone4_config() -> Config {
    Config {
        use_qk_norm: true,
        sliding_window: Some(16),
        // 3:1 local/global pattern: layers 0,1,2 local, layer 3 global
        global_layers: vec![false, false, false, true],
        ..test_config()
    }
}

#[tokio::test]
async fn test_exaone4_block_local_prefill() {
    use cake_core::models::exaone4::EXAONE4Block;

    let cfg = exaone4_config();
    let layer_name = "model.layers.0"; // local: sliding window + RoPE
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *EXAONE4Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 300);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_exaone4_block_local_generation() {
    use cake_core::models::exaone4::EXAONE4Block;

    let cfg = exaone4_config();
    let layer_name = "model.layers.1";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *EXAONE4Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 301);
    let y = block.forward_mut(&x, 0, 1, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_exaone4_block_global_prefill() {
    use cake_core::models::exaone4::EXAONE4Block;

    let cfg = exaone4_config();
    let layer_name = "model.layers.3"; // global: no RoPE, full context
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *EXAONE4Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 302);
    let y = block.forward_mut(&x, 0, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_exaone4_block_global_generation() {
    use cake_core::models::exaone4::EXAONE4Block;

    let cfg = exaone4_config();
    let layer_name = "model.layers.3";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *EXAONE4Block::load(layer_name.to_string(), &ctx).unwrap();
    // Prefill
    let x_prefill = make_tensor(&[1, 3, 64], 303);
    let _ = block.forward_mut(&x_prefill, 0, 3, &mut ctx).await.unwrap();
    // Generation
    let x = make_tensor(&[1, 1, 64], 304);
    let y = block.forward_mut(&x, 3, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

// ---------------------------------------------------------------------------
// Qwen3_5Block (linear variant - GatedDeltaNet) tests
// ---------------------------------------------------------------------------

fn qwen3_5_config() -> Config {
    Config {
        hidden_size: 64,
        intermediate_size: 128,
        vocab_size: 256,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        bos_token_id: Some(1),
        eos_token_id: None,
        rope_scaling: None,
        tie_word_embeddings: false,
        max_seq_len: 64,
        use_qkv_bias: false,
        model_prefix: "model.language_model".to_string(),
        head_dim: Some(16),
        partial_rotary_factor: 0.25,
        linear_attn: Some(LinearAttnConfig {
            layer_types: vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ],
            conv_kernel_dim: 4,
            num_key_heads: 2,
            key_head_dim: 16,
            num_value_heads: 4,
            value_head_dim: 16,
        }),
        residual_rms_norm: true,
        use_qk_norm: false,
        pre_reshape_qk_norm: false,
        sliding_window: None,
        fused_qkv_proj: false,
        fused_gate_up_proj: false,
        global_layers: vec![],
        use_gelu_mlp: false,
        embed_scale: None,
        moe_intermediate_size: None,
        num_experts: 0,
        num_experts_per_tok: 0,
        norm_topk_prob: false,
        shared_expert_intermediate_size: None,
        attn_output_gate: false,
    }
}

/// Build VarBuilder for a Qwen3_5 linear attention (GatedDeltaNet) block.
fn make_vb_qwen3_5_linear(cfg: &Config, layer_name: &str) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let la = cfg.linear_attn.as_ref().unwrap();

    let num_heads = la.num_value_heads; // 4
    let num_key_heads = la.num_key_heads; // 2
    let key_head_dim = la.key_head_dim; // 16
    let value_head_dim = la.value_head_dim; // 16
    let key_dim = num_key_heads * key_head_dim; // 32
    let value_dim = num_heads * value_head_dim; // 64
    let conv_dim = key_dim * 2 + value_dim; // 128

    let prefix = layer_name;
    let mut map: HashMap<String, Tensor> = HashMap::new();

    // Norms (residual rms_norm: stored weight + 1.0 at load time)
    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );

    // GatedDeltaNet weights (under linear_attn prefix)
    let la_prefix = format!("{prefix}.linear_attn");
    map.insert(
        format!("{la_prefix}.in_proj_qkv.weight"),
        make_tensor(&[conv_dim, h], 50),
    );
    map.insert(
        format!("{la_prefix}.in_proj_a.weight"),
        make_tensor(&[num_heads, h], 51),
    );
    map.insert(
        format!("{la_prefix}.in_proj_b.weight"),
        make_tensor(&[num_heads, h], 52),
    );
    map.insert(
        format!("{la_prefix}.in_proj_z.weight"),
        make_tensor(&[value_dim, h], 53),
    );
    map.insert(
        format!("{la_prefix}.dt_bias"),
        make_tensor(&[num_heads], 54),
    );
    map.insert(
        format!("{la_prefix}.out_proj.weight"),
        make_tensor(&[h, value_dim], 55),
    );
    map.insert(
        format!("{la_prefix}.conv1d.weight"),
        make_tensor(&[conv_dim, 1, la.conv_kernel_dim], 56),
    );
    map.insert(
        format!("{la_prefix}.A_log"),
        make_tensor(&[num_heads], 57),
    );
    map.insert(
        format!("{la_prefix}.norm.weight"),
        Tensor::ones(value_head_dim, DType::F32, &Device::Cpu).unwrap(),
    );

    // MLP weights
    map.insert(
        format!("{prefix}.mlp.gate_proj.weight"),
        make_tensor(&[i, h], 60),
    );
    map.insert(
        format!("{prefix}.mlp.up_proj.weight"),
        make_tensor(&[i, h], 61),
    );
    map.insert(
        format!("{prefix}.mlp.down_proj.weight"),
        make_tensor(&[h, i], 62),
    );

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

#[tokio::test]
async fn test_qwen3_5_linear_block_prefill() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.0"; // linear_attention
    let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 400);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_qwen3_5_linear_block_generation() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.1";
    let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 401);
    let y = block.forward_mut(&x, 0, 1, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_qwen3_5_linear_block_prefill_then_generate() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.0";
    let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    // Prefill
    let x_prefill = make_tensor(&[1, 4, 64], 402);
    let _ = block.forward_mut(&x_prefill, 0, 0, &mut ctx).await.unwrap();
    // Generation (uses conv_state + recurrent_state from prefill)
    let x = make_tensor(&[1, 1, 64], 403);
    let y = block.forward_mut(&x, 4, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

// ---------------------------------------------------------------------------
// Qwen3_5Block (full attention variant) tests
// ---------------------------------------------------------------------------

/// Build VarBuilder for a Qwen3_5 full attention block.
fn make_vb_qwen3_5_full(cfg: &Config, layer_name: &str) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
    let num_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    // Full attention: q_proj outputs num_heads * head_dim * 2 (query + gate)
    let q_size = num_heads * head_dim * 2;
    let kv_size = num_kv_heads * head_dim;
    let o_size = num_heads * head_dim;

    let prefix = layer_name;
    let mut map: HashMap<String, Tensor> = HashMap::new();

    // Norms (residual rms_norm)
    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );

    // Full attention weights (under self_attn prefix)
    let attn_prefix = format!("{prefix}.self_attn");
    map.insert(
        format!("{attn_prefix}.q_proj.weight"),
        make_tensor(&[q_size, h], 70),
    );
    map.insert(
        format!("{attn_prefix}.k_proj.weight"),
        make_tensor(&[kv_size, h], 71),
    );
    map.insert(
        format!("{attn_prefix}.v_proj.weight"),
        make_tensor(&[kv_size, h], 72),
    );
    map.insert(
        format!("{attn_prefix}.o_proj.weight"),
        make_tensor(&[h, o_size], 73),
    );
    map.insert(
        format!("{attn_prefix}.q_norm.weight"),
        Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{attn_prefix}.k_norm.weight"),
        Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap(),
    );

    // MLP weights
    map.insert(
        format!("{prefix}.mlp.gate_proj.weight"),
        make_tensor(&[i, h], 74),
    );
    map.insert(
        format!("{prefix}.mlp.up_proj.weight"),
        make_tensor(&[i, h], 75),
    );
    map.insert(
        format!("{prefix}.mlp.down_proj.weight"),
        make_tensor(&[h, i], 76),
    );

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

#[tokio::test]
async fn test_qwen3_5_full_block_prefill() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.3"; // full_attention
    let vb = make_vb_qwen3_5_full(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 500);
    let y = block.forward_mut(&x, 0, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_qwen3_5_full_block_generation() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.3";
    let vb = make_vb_qwen3_5_full(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 501);
    let y = block.forward_mut(&x, 0, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_qwen3_5_full_block_prefill_then_generate() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.3";
    let vb = make_vb_qwen3_5_full(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    // Prefill
    let x_prefill = make_tensor(&[1, 3, 64], 502);
    let _ = block.forward_mut(&x_prefill, 0, 3, &mut ctx).await.unwrap();
    // Generation
    let x = make_tensor(&[1, 1, 64], 503);
    let y = block.forward_mut(&x, 3, 3, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

// ---------------------------------------------------------------------------
// Qwen3MoeBlock tests
// ---------------------------------------------------------------------------

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

#[tokio::test]
async fn test_qwen3_moe_block_prefill() {
    use cake_core::models::qwen3_moe::Qwen3MoeBlock;

    let cfg = qwen3_moe_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, true);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3MoeBlock::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 600);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_qwen3_moe_block_generation() {
    use cake_core::models::qwen3_moe::Qwen3MoeBlock;

    let cfg = qwen3_moe_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, true);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3MoeBlock::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 601);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_qwen3_moe_block_prefill_then_generate() {
    use cake_core::models::qwen3_moe::Qwen3MoeBlock;

    let cfg = qwen3_moe_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, true);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3MoeBlock::load(layer_name.to_string(), &ctx).unwrap();
    // Prefill
    let x_prefill = make_tensor(&[1, 4, 64], 602);
    let _ = block.forward_mut(&x_prefill, 0, 0, &mut ctx).await.unwrap();
    // Generation
    let x = make_tensor(&[1, 1, 64], 603);
    let y = block.forward_mut(&x, 4, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

// ---------------------------------------------------------------------------
// Cross-block output differs from input (sanity check)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_gemma3_block_output_differs() {
    use cake_core::models::gemma3::Gemma3Block;

    let cfg = gemma3_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(
        &cfg,
        layer_name,
        &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
        true,
        false,
    );
    let mut ctx = make_context(cfg, vb);
    let mut block = *Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 700);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    let y_vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    let diff: f32 = x_vals.iter().zip(&y_vals).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "Block output should differ from input");
}

#[tokio::test]
async fn test_olmo2_block_output_differs() {
    use cake_core::models::olmo2::OLMo2Block;

    let cfg = olmo2_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &["post_feedforward_layernorm"], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *OLMo2Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 701);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    let y_vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    let diff: f32 = x_vals.iter().zip(&y_vals).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "Block output should differ from input");
}

#[tokio::test]
async fn test_qwen3_5_linear_block_output_differs() {
    use cake_core::models::qwen3_5::Qwen3_5Block;

    let cfg = qwen3_5_config();
    let layer_name = "model.language_model.layers.0";
    let vb = make_vb_qwen3_5_linear(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5Block::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 702);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    let y_vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
    let diff: f32 = x_vals.iter().zip(&y_vals).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "Block output should differ from input");
}

// ---------------------------------------------------------------------------
// Determinism tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_gemma3_block_deterministic() {
    use cake_core::models::gemma3::Gemma3Block;

    let cfg = gemma3_config();
    let layer_name = "model.layers.0";

    let run = || async {
        let vb = make_vb_standard_block(
            &cfg,
            layer_name,
            &["pre_feedforward_layernorm", "post_feedforward_layernorm"],
            true,
            false,
        );
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block = *Gemma3Block::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 800);
        block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap()
    };

    let y1: Vec<f32> = run().await.flatten_all().unwrap().to_vec1().unwrap();
    let y2: Vec<f32> = run().await.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(y1, y2, "Block should be deterministic");
}

#[tokio::test]
async fn test_qwen3_moe_block_deterministic() {
    use cake_core::models::qwen3_moe::Qwen3MoeBlock;

    let cfg = qwen3_moe_config();
    let layer_name = "model.layers.0";

    let run = || async {
        let vb = make_vb_standard_block(&cfg, layer_name, &[], true, true);
        let mut ctx = make_context(cfg.clone(), vb);
        let mut block = *Qwen3MoeBlock::load(layer_name.to_string(), &ctx).unwrap();
        let x = make_tensor(&[1, 4, 64], 801);
        block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap()
    };

    let y1: Vec<f32> = run().await.flatten_all().unwrap().to_vec1().unwrap();
    let y2: Vec<f32> = run().await.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(y1, y2, "Block should be deterministic");
}

// ---------------------------------------------------------------------------
// Standard Transformer block (used by LLaMA, Qwen2, Qwen3, Phi4, Mistral, Falcon3)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_transformer_block_prefill() {
    use cake_core::models::common::Transformer;

    let cfg = test_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], false, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Transformer::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 900);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_transformer_block_generation() {
    use cake_core::models::common::Transformer;

    let cfg = test_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], false, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Transformer::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 901);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_transformer_block_prefill_then_generate() {
    use cake_core::models::common::Transformer;

    let cfg = test_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], false, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Transformer::load(layer_name.to_string(), &ctx).unwrap();
    let x_pre = make_tensor(&[1, 4, 64], 902);
    let _ = block.forward_mut(&x_pre, 0, 0, &mut ctx).await.unwrap();
    let x_gen = make_tensor(&[1, 1, 64], 903);
    let y = block.forward_mut(&x_gen, 4, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}

#[tokio::test]
async fn test_transformer_block_with_qk_norm() {
    use cake_core::models::common::Transformer;

    let cfg = Config {
        use_qk_norm: true,
        ..test_config()
    };
    let layer_name = "model.layers.0";
    let vb = make_vb_standard_block(&cfg, layer_name, &[], true, false);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Transformer::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 904);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

// ---------------------------------------------------------------------------
// Qwen3_5MoeBlock tests (linear attention + MoE)
// ---------------------------------------------------------------------------

fn qwen3_5_moe_config() -> Config {
    let mut cfg = qwen3_5_config();
    cfg.moe_intermediate_size = Some(32);
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.norm_topk_prob = true;
    cfg.shared_expert_intermediate_size = Some(48);
    cfg
}

fn make_vb_qwen3_5_moe_linear(cfg: &Config, layer_name: &str) -> VarBuilder<'static> {
    // Start from linear attention VarBuilder and add MoE weights
    let h = cfg.hidden_size;
    let moe_i = cfg.moe_intermediate_size.unwrap();
    let si = cfg.shared_expert_intermediate_size.unwrap();
    let n = cfg.num_experts;
    let prefix = layer_name;

    // Get all the linear attention weights first
    let mut base_vb_map: HashMap<String, Tensor> = HashMap::new();

    // Norms (residual_rms_norm = true → stored as-is, +1 applied at load)
    base_vb_map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );
    base_vb_map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );

    // GatedDeltaNet weights
    let la = cfg.linear_attn.as_ref().unwrap();
    // GDN dimensions (match linear_attention.rs exactly)
    let num_heads_gdn = la.num_value_heads;      // = num_attention_heads
    let num_key_heads = la.num_key_heads;
    let key_head_dim = la.key_head_dim;
    let value_head_dim = la.value_head_dim;
    let key_dim_total = num_key_heads * key_head_dim;
    let val_dim_total = num_heads_gdn * value_head_dim;
    let gdn_conv_dim = key_dim_total * 2 + val_dim_total;
    let la_prefix = format!("{prefix}.linear_attn");
    base_vb_map.insert(
        format!("{la_prefix}.in_proj_qkv.weight"),
        make_tensor(&[gdn_conv_dim, h], 500),
    );
    base_vb_map.insert(
        format!("{la_prefix}.in_proj_a.weight"),
        make_tensor(&[num_heads_gdn, h], 505),
    );
    base_vb_map.insert(
        format!("{la_prefix}.in_proj_b.weight"),
        make_tensor(&[num_heads_gdn, h], 506),
    );
    base_vb_map.insert(
        format!("{la_prefix}.in_proj_z.weight"),
        make_tensor(&[val_dim_total, h], 507),
    );
    base_vb_map.insert(
        format!("{la_prefix}.out_proj.weight"),
        make_tensor(&[h, val_dim_total], 501),
    );
    base_vb_map.insert(
        format!("{la_prefix}.conv1d.weight"),
        make_tensor(&[gdn_conv_dim, 1, la.conv_kernel_dim], 502),
    );
    base_vb_map.insert(
        format!("{la_prefix}.norm.weight"),
        Tensor::ones(value_head_dim, DType::F32, &Device::Cpu).unwrap(),
    );
    base_vb_map.insert(
        format!("{la_prefix}.A_log"),
        make_tensor(&[num_heads_gdn], 503),
    );
    base_vb_map.insert(
        format!("{la_prefix}.dt_bias"),
        make_tensor(&[num_heads_gdn], 504),
    );

    // MoE weights
    let moe_prefix = format!("{prefix}.mlp");
    base_vb_map.insert(
        format!("{moe_prefix}.gate.weight"),
        make_tensor(&[n, h], 510),
    );
    for j in 0..n {
        base_vb_map.insert(
            format!("{moe_prefix}.experts.{j}.gate_proj.weight"),
            make_tensor(&[moe_i, h], 511 + j as u64 * 3),
        );
        base_vb_map.insert(
            format!("{moe_prefix}.experts.{j}.up_proj.weight"),
            make_tensor(&[moe_i, h], 512 + j as u64 * 3),
        );
        base_vb_map.insert(
            format!("{moe_prefix}.experts.{j}.down_proj.weight"),
            make_tensor(&[h, moe_i], 513 + j as u64 * 3),
        );
    }
    base_vb_map.insert(
        format!("{moe_prefix}.shared_expert.gate_proj.weight"),
        make_tensor(&[si, h], 530),
    );
    base_vb_map.insert(
        format!("{moe_prefix}.shared_expert.up_proj.weight"),
        make_tensor(&[si, h], 531),
    );
    base_vb_map.insert(
        format!("{moe_prefix}.shared_expert.down_proj.weight"),
        make_tensor(&[h, si], 532),
    );
    base_vb_map.insert(
        format!("{moe_prefix}.shared_expert_gate.weight"),
        make_tensor(&[1, h], 533),
    );

    VarBuilder::from_tensors(base_vb_map, DType::F32, &Device::Cpu)
}

#[tokio::test]
async fn test_qwen3_5_moe_linear_block_prefill() {
    use cake_core::models::qwen3_5_moe::block::Qwen3_5MoeBlock;

    let cfg = qwen3_5_moe_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_qwen3_5_moe_linear(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5MoeBlock::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 4, 64], 600);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 4, 64]);
}

#[tokio::test]
async fn test_qwen3_5_moe_linear_block_generation() {
    use cake_core::models::qwen3_5_moe::block::Qwen3_5MoeBlock;

    let cfg = qwen3_5_moe_config();
    let layer_name = "model.layers.0";
    let vb = make_vb_qwen3_5_moe_linear(&cfg, layer_name);
    let mut ctx = make_context(cfg, vb);
    let mut block = *Qwen3_5MoeBlock::load(layer_name.to_string(), &ctx).unwrap();
    let x = make_tensor(&[1, 1, 64], 601);
    let y = block.forward_mut(&x, 0, 0, &mut ctx).await.unwrap();
    assert_eq!(y.dims(), &[1, 1, 64]);
}
