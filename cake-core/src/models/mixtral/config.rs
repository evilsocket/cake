use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::models::common::{Config, EosTokenId};

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_rope_theta() -> f64 {
    1e6
}

fn default_sliding_window() -> usize {
    4096
}

fn default_num_experts_per_tok() -> usize {
    2
}

fn default_num_local_experts() -> usize {
    8
}

fn default_false() -> bool {
    false
}

fn default_max_position_embeddings() -> usize {
    32768
}

/// Mixtral-specific configuration (serde deserialization from config.json).
#[derive(Debug, Clone, Deserialize)]
pub struct MixtralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
}

impl MixtralConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Mixtral configuration from {}", path.display());
        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    /// Convert to the generalized Config for TextModelBase.
    pub fn into_config(self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta as f32,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: None,
            tie_word_embeddings: self.tie_word_embeddings,
            max_seq_len: self.max_position_embeddings,
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: None,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
        }
    }

}
