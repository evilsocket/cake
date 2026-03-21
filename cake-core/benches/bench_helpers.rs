use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use cake_core::cake::{Context, Topology};
use cake_core::models::common::{Cache, Config, LinearAttnConfig};
use cake_core::{Args, TextModelArch};

/// Small Config for benchmarking. hidden=64, heads=4 (GQA: 2 kv heads), head_dim=16, intermediate=128.
pub fn test_config() -> Config {
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
        model_prefix: "model".to_string(),
        head_dim: None,
        partial_rotary_factor: 1.0,
        linear_attn: None,
        residual_rms_norm: false,
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

pub fn test_config_with_qk_norm() -> Config {
    Config {
        use_qk_norm: true,
        ..test_config()
    }
}

pub fn test_config_with_sliding_window(w: usize) -> Config {
    Config {
        sliding_window: Some(w),
        ..test_config()
    }
}

pub fn test_config_with_gelu() -> Config {
    Config {
        use_gelu_mlp: true,
        ..test_config()
    }
}

/// GatedDeltaNet config for Qwen3.5 linear attention benchmarks.
pub fn test_config_gdn() -> Config {
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

/// Create a deterministic F32 tensor on CPU from a seed.
pub fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let numel: usize = shape.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.1..0.1)).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// Build VarBuilder for CausalSelfAttention.
pub fn make_vb_attention(cfg: &Config) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
    let size_q = head_dim * cfg.num_attention_heads;
    let size_kv = head_dim * cfg.num_key_value_heads;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    if cfg.fused_qkv_proj {
        map.insert(
            "qkv_proj.weight".into(),
            make_tensor(&[size_q + 2 * size_kv, h], 10),
        );
    } else {
        map.insert("q_proj.weight".into(), make_tensor(&[size_q, h], 10));
        map.insert("k_proj.weight".into(), make_tensor(&[size_kv, h], 11));
        map.insert("v_proj.weight".into(), make_tensor(&[size_kv, h], 12));
        if cfg.use_qkv_bias {
            map.insert("q_proj.bias".into(), make_tensor(&[size_q], 13));
            map.insert("k_proj.bias".into(), make_tensor(&[size_kv], 14));
            map.insert("v_proj.bias".into(), make_tensor(&[size_kv], 15));
        }
    }
    map.insert("o_proj.weight".into(), make_tensor(&[h, size_q], 16));

    if cfg.use_qk_norm {
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
        map.insert("q_norm.weight".into(), make_tensor(&[norm_dim], 17));
        map.insert("k_norm.weight".into(), make_tensor(&[norm_kv_dim], 18));
    }

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Build VarBuilder for MLP.
pub fn make_vb_mlp(cfg: &Config) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    if cfg.fused_gate_up_proj {
        map.insert("gate_up_proj.weight".into(), make_tensor(&[2 * i, h], 20));
    } else {
        map.insert("gate_proj.weight".into(), make_tensor(&[i, h], 20));
        map.insert("up_proj.weight".into(), make_tensor(&[i, h], 21));
    }
    map.insert("down_proj.weight".into(), make_tensor(&[h, i], 22));

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Create a Cache for benchmarking.
pub fn make_cache(cfg: &Config) -> Cache {
    Cache::new(true, DType::F32, cfg, &Device::Cpu).unwrap()
}

/// Build VarBuilder for a full Transformer block (pre-norm + attention + MLP).
#[allow(dead_code)]
pub fn make_vb_transformer_block(cfg: &Config) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
    let size_q = head_dim * cfg.num_attention_heads;
    let size_kv = head_dim * cfg.num_key_value_heads;
    let i = cfg.intermediate_size;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    map.insert(
        "input_layernorm.weight".into(),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        "post_attention_layernorm.weight".into(),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );

    map.insert(
        "self_attn.q_proj.weight".into(),
        make_tensor(&[size_q, h], 30),
    );
    map.insert(
        "self_attn.k_proj.weight".into(),
        make_tensor(&[size_kv, h], 31),
    );
    map.insert(
        "self_attn.v_proj.weight".into(),
        make_tensor(&[size_kv, h], 32),
    );
    map.insert(
        "self_attn.o_proj.weight".into(),
        make_tensor(&[h, size_q], 33),
    );

    if cfg.use_qk_norm {
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
            "self_attn.q_norm.weight".into(),
            make_tensor(&[norm_dim], 34),
        );
        map.insert(
            "self_attn.k_norm.weight".into(),
            make_tensor(&[norm_kv_dim], 35),
        );
    }

    map.insert("mlp.gate_proj.weight".into(), make_tensor(&[i, h], 36));
    map.insert("mlp.up_proj.weight".into(), make_tensor(&[i, h], 37));
    map.insert("mlp.down_proj.weight".into(), make_tensor(&[h, i], 38));

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Build VarBuilder for a standard block with optional extras.
pub fn make_vb_standard_block(
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

    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );

    for norm_name in extra_norms {
        map.insert(
            format!("{prefix}.{norm_name}.weight"),
            Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
        );
    }

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

/// Build a minimal Context from a Config + VarBuilder.
pub fn make_context(cfg: Config, vb: VarBuilder<'static>) -> Context {
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
        backend: Arc::new(cake_core::backends::CpuBackend::new()),
    }
}

/// Build VarBuilder for MoE (Qwen3 MoE style: gate + stacked expert weights).
pub fn make_vb_moe(cfg: &Config) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let moe_i = cfg.moe_intermediate_size.unwrap();
    let n = cfg.num_experts;

    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("gate.weight".into(), make_tensor(&[n, h], 40));
    for j in 0..n {
        map.insert(
            format!("experts.{j}.gate_proj.weight"),
            make_tensor(&[moe_i, h], 41 + j as u64 * 3),
        );
        map.insert(
            format!("experts.{j}.up_proj.weight"),
            make_tensor(&[moe_i, h], 42 + j as u64 * 3),
        );
        map.insert(
            format!("experts.{j}.down_proj.weight"),
            make_tensor(&[h, moe_i], 43 + j as u64 * 3),
        );
    }

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Build VarBuilder for Qwen3.5 MoE (per-expert linear + shared expert + sigmoid gate).
pub fn make_vb_qwen3_5_moe(cfg: &Config) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let moe_i = cfg.moe_intermediate_size.unwrap();
    let si = cfg.shared_expert_intermediate_size.unwrap();
    let n = cfg.num_experts;

    let mut map: HashMap<String, Tensor> = HashMap::new();
    map.insert("gate.weight".into(), make_tensor(&[n, h], 40));
    for j in 0..n {
        map.insert(
            format!("experts.{j}.gate_proj.weight"),
            make_tensor(&[moe_i, h], 41 + j as u64 * 3),
        );
        map.insert(
            format!("experts.{j}.up_proj.weight"),
            make_tensor(&[moe_i, h], 42 + j as u64 * 3),
        );
        map.insert(
            format!("experts.{j}.down_proj.weight"),
            make_tensor(&[h, moe_i], 43 + j as u64 * 3),
        );
    }
    map.insert(
        "shared_expert.gate_proj.weight".into(),
        make_tensor(&[si, h], 60),
    );
    map.insert(
        "shared_expert.up_proj.weight".into(),
        make_tensor(&[si, h], 61),
    );
    map.insert(
        "shared_expert.down_proj.weight".into(),
        make_tensor(&[h, si], 62),
    );
    map.insert(
        "shared_expert_gate.weight".into(),
        make_tensor(&[1, h], 63),
    );

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Build VarBuilder for a Qwen3_5 linear attention (GatedDeltaNet) block.
pub fn make_vb_qwen3_5_linear(cfg: &Config, layer_name: &str) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let la = cfg.linear_attn.as_ref().unwrap();

    let num_heads = la.num_value_heads;
    let num_key_heads = la.num_key_heads;
    let key_head_dim = la.key_head_dim;
    let value_head_dim = la.value_head_dim;
    let key_dim = num_key_heads * key_head_dim;
    let value_dim = num_heads * value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;

    let prefix = layer_name;
    let mut map: HashMap<String, Tensor> = HashMap::new();

    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );

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

/// Build VarBuilder for a Qwen3_5 full attention block.
pub fn make_vb_qwen3_5_full(cfg: &Config, layer_name: &str) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
    let num_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;

    let q_size = num_heads * head_dim * 2;
    let kv_size = num_kv_heads * head_dim;
    let o_size = num_heads * head_dim;

    let prefix = layer_name;
    let mut map: HashMap<String, Tensor> = HashMap::new();

    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );

    let attn_prefix = format!("{prefix}.self_attn");
    map.insert(format!("{attn_prefix}.q_proj.weight"), make_tensor(&[q_size, h], 70));
    map.insert(format!("{attn_prefix}.k_proj.weight"), make_tensor(&[kv_size, h], 71));
    map.insert(format!("{attn_prefix}.v_proj.weight"), make_tensor(&[kv_size, h], 72));
    map.insert(format!("{attn_prefix}.o_proj.weight"), make_tensor(&[h, o_size], 73));
    map.insert(
        format!("{attn_prefix}.q_norm.weight"),
        Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{attn_prefix}.k_norm.weight"),
        Tensor::ones(head_dim, DType::F32, &Device::Cpu).unwrap(),
    );

    map.insert(format!("{prefix}.mlp.gate_proj.weight"), make_tensor(&[i, h], 74));
    map.insert(format!("{prefix}.mlp.up_proj.weight"), make_tensor(&[i, h], 75));
    map.insert(format!("{prefix}.mlp.down_proj.weight"), make_tensor(&[h, i], 76));

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Build VarBuilder for a Qwen3_5 MoE linear attention (GatedDeltaNet) block.
pub fn make_vb_qwen3_5_moe_linear(cfg: &Config, layer_name: &str) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let moe_i = cfg.moe_intermediate_size.unwrap();
    let si = cfg.shared_expert_intermediate_size.unwrap();
    let n = cfg.num_experts;
    let prefix = layer_name;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    map.insert(
        format!("{prefix}.input_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        format!("{prefix}.post_attention_layernorm.weight"),
        Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap(),
    );

    let la = cfg.linear_attn.as_ref().unwrap();
    let num_heads_gdn = la.num_value_heads;
    let num_key_heads = la.num_key_heads;
    let key_head_dim = la.key_head_dim;
    let value_head_dim = la.value_head_dim;
    let key_dim_total = num_key_heads * key_head_dim;
    let val_dim_total = num_heads_gdn * value_head_dim;
    let gdn_conv_dim = key_dim_total * 2 + val_dim_total;
    let la_prefix = format!("{prefix}.linear_attn");
    map.insert(format!("{la_prefix}.in_proj_qkv.weight"), make_tensor(&[gdn_conv_dim, h], 500));
    map.insert(format!("{la_prefix}.in_proj_a.weight"), make_tensor(&[num_heads_gdn, h], 505));
    map.insert(format!("{la_prefix}.in_proj_b.weight"), make_tensor(&[num_heads_gdn, h], 506));
    map.insert(format!("{la_prefix}.in_proj_z.weight"), make_tensor(&[val_dim_total, h], 507));
    map.insert(format!("{la_prefix}.out_proj.weight"), make_tensor(&[h, val_dim_total], 501));
    map.insert(format!("{la_prefix}.conv1d.weight"), make_tensor(&[gdn_conv_dim, 1, la.conv_kernel_dim], 502));
    map.insert(
        format!("{la_prefix}.norm.weight"),
        Tensor::ones(value_head_dim, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(format!("{la_prefix}.A_log"), make_tensor(&[num_heads_gdn], 503));
    map.insert(format!("{la_prefix}.dt_bias"), make_tensor(&[num_heads_gdn], 504));

    let moe_prefix = format!("{prefix}.mlp");
    map.insert(format!("{moe_prefix}.gate.weight"), make_tensor(&[n, h], 510));
    for j in 0..n {
        map.insert(format!("{moe_prefix}.experts.{j}.gate_proj.weight"), make_tensor(&[moe_i, h], 511 + j as u64 * 3));
        map.insert(format!("{moe_prefix}.experts.{j}.up_proj.weight"), make_tensor(&[moe_i, h], 512 + j as u64 * 3));
        map.insert(format!("{moe_prefix}.experts.{j}.down_proj.weight"), make_tensor(&[h, moe_i], 513 + j as u64 * 3));
    }
    map.insert(format!("{moe_prefix}.shared_expert.gate_proj.weight"), make_tensor(&[si, h], 530));
    map.insert(format!("{moe_prefix}.shared_expert.up_proj.weight"), make_tensor(&[si, h], 531));
    map.insert(format!("{moe_prefix}.shared_expert.down_proj.weight"), make_tensor(&[h, si], 532));
    map.insert(format!("{moe_prefix}.shared_expert_gate.weight"), make_tensor(&[1, h], 533));

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Qwen3.5 MoE config (linear attention + MoE).
pub fn qwen3_5_moe_config() -> Config {
    let mut cfg = test_config_gdn();
    cfg.moe_intermediate_size = Some(32);
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.norm_topk_prob = true;
    cfg.shared_expert_intermediate_size = Some(48);
    cfg
}
