use std::path::Path;

use anyhow::Result;

/// Max supported sequence length.
pub const MAX_SEQ_LEN: usize = 4096;

fn default_rope() -> f32 {
    10_000.0
}

/// LLama specific configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub max_position_embeddings: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub rope_scaling: Option<RopeConfig>,
    pub rope_type: Option<RopeType>,
}

impl LlamaConfig {
    /// Load the configuration from the given path.
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    /// Return the number of kv heads.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Return a generalized Config object.
    pub fn into_config(self) -> Config {
        // TODO: this is retarded
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            max_position_embeddings: self.max_position_embeddings.unwrap_or(2048),
            tie_word_embeddings: self.tie_word_embeddings,
            rope_scaling: self.rope_scaling,
        }
    }
}

/// Rope scaling type.
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

// Rope scaling configuration
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: RopeType,
}

/// Generalized LLama/LLM configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}
