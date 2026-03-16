use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

use cake_core::models::common::{Cache, Config};

/// Small Config for testing. hidden=64, heads=4 (GQA: 2 kv heads), head_dim=16, intermediate=128.
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

pub fn test_config_with_bias() -> Config {
    Config {
        use_qkv_bias: true,
        ..test_config()
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

/// Build VarBuilder for CausalSelfAttention. Called as vb.pp("self_attn") by the load fn.
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

/// Build VarBuilder for a full Transformer block (pre-norm + attention + MLP).
pub fn make_vb_transformer_block(cfg: &Config) -> VarBuilder<'static> {
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim.unwrap_or(h / cfg.num_attention_heads);
    let size_q = head_dim * cfg.num_attention_heads;
    let size_kv = head_dim * cfg.num_key_value_heads;
    let i = cfg.intermediate_size;

    let mut map: HashMap<String, Tensor> = HashMap::new();

    // Norms
    map.insert(
        "input_layernorm.weight".into(),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );
    map.insert(
        "post_attention_layernorm.weight".into(),
        Tensor::ones(h, DType::F32, &Device::Cpu).unwrap(),
    );

    // Attention weights (under self_attn prefix)
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

    // MLP weights (under mlp prefix)
    map.insert("mlp.gate_proj.weight".into(), make_tensor(&[i, h], 36));
    map.insert("mlp.up_proj.weight".into(), make_tensor(&[i, h], 37));
    map.insert("mlp.down_proj.weight".into(), make_tensor(&[h, i], 38));

    VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Create a Cache for testing.
pub fn make_cache(cfg: &Config) -> Cache {
    Cache::new(true, DType::F32, cfg, &Device::Cpu).unwrap()
}
