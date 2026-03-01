use std::path::Path;

use anyhow::Result;
use serde::de::{self, Deserializer};

/// Default max sequence length (LLaMA2 era). Overridden by max_position_embeddings from config.
const DEFAULT_MAX_SEQ_LEN: usize = 4096;

fn default_rope() -> f32 {
    500_000.0
}

fn default_max_position_embeddings() -> usize {
    DEFAULT_MAX_SEQ_LEN
}

fn default_false() -> bool {
    false
}

/// EOS token ID(s) — deserializes from either a single u32 or an array of u32.
#[derive(Debug, Clone)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    /// Check if the given token ID is an EOS token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        match self {
            EosTokenId::Single(id) => *id == token_id,
            EosTokenId::Multiple(ids) => ids.contains(&token_id),
        }
    }
}

impl<'de> serde::Deserialize<'de> for EosTokenId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Number(n) => n
                .as_u64()
                .map(|v| EosTokenId::Single(v as u32))
                .ok_or_else(|| de::Error::custom("expected u32 for eos_token_id")),
            serde_json::Value::Array(arr) => {
                let ids: std::result::Result<Vec<u32>, _> = arr
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .map(|n| n as u32)
                            .ok_or_else(|| de::Error::custom("expected u32 in eos_token_id array"))
                    })
                    .collect();
                ids.map(EosTokenId::Multiple)
            }
            _ => Err(de::Error::custom("expected u32 or array for eos_token_id")),
        }
    }
}

/// RoPE scaling configuration for LLaMA 3.1+.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    /// Factor for scaling (typically 8.0 for LLaMA 3.1).
    #[serde(default)]
    pub factor: f32,
    /// High frequency factor (typically 4.0).
    #[serde(default)]
    pub high_freq_factor: f32,
    /// Low frequency factor (typically 1.0).
    #[serde(default)]
    pub low_freq_factor: f32,
    /// Original max position embeddings before scaling (typically 8192).
    #[serde(default)]
    pub original_max_position_embeddings: usize,
    /// RoPE type (e.g. "llama3").
    #[serde(default)]
    pub rope_type: Option<String>,
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
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
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
            rope_scaling: self.rope_scaling,
            tie_word_embeddings: self.tie_word_embeddings,
            max_seq_len: self.max_position_embeddings,
        }
    }
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
    pub eos_token_id: Option<EosTokenId>,
    pub rope_scaling: Option<RopeScaling>,
    pub tie_word_embeddings: bool,
    pub max_seq_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eos_token_id_single() {
        let json = r#"128001"#;
        let eos: EosTokenId = serde_json::from_str(json).unwrap();
        assert!(eos.is_eos(128001));
        assert!(!eos.is_eos(128008));
    }

    #[test]
    fn test_eos_token_id_array() {
        let json = r#"[128001, 128008, 128009]"#;
        let eos: EosTokenId = serde_json::from_str(json).unwrap();
        assert!(eos.is_eos(128001));
        assert!(eos.is_eos(128008));
        assert!(eos.is_eos(128009));
        assert!(!eos.is_eos(0));
    }

    #[test]
    fn test_rope_scaling_deserialization() {
        let json = r#"{
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }"#;
        let scaling: RopeScaling = serde_json::from_str(json).unwrap();
        assert_eq!(scaling.factor, 8.0);
        assert_eq!(scaling.high_freq_factor, 4.0);
        assert_eq!(scaling.low_freq_factor, 1.0);
        assert_eq!(scaling.original_max_position_embeddings, 8192);
        assert_eq!(scaling.rope_type.as_deref(), Some("llama3"));
    }

    #[test]
    fn test_default_values() {
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5
        }"#;
        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.rope_theta, 500_000.0);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.max_position_embeddings, DEFAULT_MAX_SEQ_LEN);
        assert!(config.rope_scaling.is_none());
        assert!(config.eos_token_id.is_none());
    }

    #[test]
    fn test_llama_3_1_config() {
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "max_position_embeddings": 131072,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }"#;
        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.rope_theta, 500_000.0);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.max_position_embeddings, 131072);

        let eos = config.eos_token_id.as_ref().unwrap();
        assert!(eos.is_eos(128001));
        assert!(eos.is_eos(128008));
        assert!(eos.is_eos(128009));
        assert!(!eos.is_eos(0));

        let scaling = config.rope_scaling.as_ref().unwrap();
        assert_eq!(scaling.factor, 8.0);

        let cfg = config.into_config();
        assert_eq!(cfg.max_seq_len, 131072);
        assert!(cfg.tie_word_embeddings);
    }
}
