use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, LinearAttnConfig, RopeScaling};

fn default_rope() -> f32 { 10_000_000.0 }
fn default_max_pos() -> usize { 262144 }
fn default_partial_rotary() -> f32 { 0.25 }
fn default_head_dim() -> usize { 256 }

/// rope_parameters sub-object (same as dense Qwen3.5).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// text_config sub-object for Qwen3.5 MoE.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3_5MoeTextConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    // Linear attention config (same as dense Qwen3.5)
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,

    // MoE config
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    // Attention output gate (full-attention layers only)
    #[serde(default)]
    pub attn_output_gate: bool,
}

/// Top-level Qwen3.5-MoE config.json wrapper.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3_5MoeConfig {
    pub text_config: Qwen3_5MoeTextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen3_5MoeConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Qwen3.5-MoE configuration from {}", path.display());
        let data = std::fs::read(path)
            .map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    pub fn into_config(self) -> Config {
        let tc = &self.text_config;
        let num_kv_heads = tc.num_key_value_heads.unwrap_or(tc.num_attention_heads);

        let (rope_theta, partial_rotary_factor) = if let Some(ref rp) = tc.rope_parameters {
            (rp.rope_theta, rp.partial_rotary_factor)
        } else {
            (default_rope(), default_partial_rotary())
        };

        let linear_attn = if !tc.layer_types.is_empty() {
            Some(LinearAttnConfig {
                layer_types: tc.layer_types.clone(),
                conv_kernel_dim: tc.linear_conv_kernel_dim.unwrap_or(4),
                num_key_heads: tc.linear_num_key_heads.unwrap_or(16),
                key_head_dim: tc.linear_key_head_dim.unwrap_or(128),
                num_value_heads: tc.linear_num_value_heads.unwrap_or(32),
                value_head_dim: tc.linear_value_head_dim.unwrap_or(128),
            })
        } else {
            None
        };

        let tie_word_embeddings = self.tie_word_embeddings || tc.tie_word_embeddings;

        Config {
            hidden_size: tc.hidden_size,
            intermediate_size: 0, // MoE model — no dense FFN
            vocab_size: tc.vocab_size,
            num_hidden_layers: tc.num_hidden_layers,
            num_attention_heads: tc.num_attention_heads,
            num_key_value_heads: num_kv_heads,
            rms_norm_eps: tc.rms_norm_eps,
            rope_theta,
            bos_token_id: None,
            eos_token_id: tc.eos_token_id.clone(),
            rope_scaling: tc.rope_scaling.clone(),
            tie_word_embeddings,
            max_seq_len: tc.max_position_embeddings,
            use_qkv_bias: false,
            model_prefix: "model.language_model".into(),
            head_dim: Some(tc.head_dim),
            partial_rotary_factor,
            linear_attn,
            residual_rms_norm: true,
            use_qk_norm: false,
            pre_reshape_qk_norm: false,
            sliding_window: None,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: None,
            global_layers: vec![],
            moe_intermediate_size: Some(tc.moe_intermediate_size),
            num_experts: tc.num_experts,
            num_experts_per_tok: tc.num_experts_per_tok,
            norm_topk_prob: true,
            shared_expert_intermediate_size: Some(tc.shared_expert_intermediate_size),
            attn_output_gate: tc.attn_output_gate,
        }
    }
}
