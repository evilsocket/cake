use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_false() -> bool {
    false
}

fn default_true() -> bool {
    true
}

/// Qwen2-specific configuration (serde deserialization from config.json).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QwenConfig {
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
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    // Qwen2-specific fields (for future sliding window support)
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_false")]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub max_window_layers: Option<usize>,
}

impl QwenConfig {
    /// Load the configuration from the given path.
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Qwen2 configuration from {}", path.display());

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
            use_qkv_bias: true, // Qwen2 always uses QKV bias
            model_prefix: "model".into(),
            head_dim: None,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_defaults() {
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 5504,
            "vocab_size": 151936,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6
        }"#;
        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert!(config.tie_word_embeddings); // default true for Qwen2
        assert_eq!(config.max_position_embeddings, 32768);
        assert!(config.sliding_window.is_none());
        assert!(!config.use_sliding_window);
    }

    #[test]
    fn test_qwen2_5_coder_config() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 5504,
            "vocab_size": 151936,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "max_position_embeddings": 32768,
            "tie_word_embeddings": true,
            "sliding_window": 32768,
            "use_sliding_window": false,
            "max_window_layers": 28
        }"#;
        let config: QwenConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.sliding_window, Some(32768));
        assert!(!config.use_sliding_window);
        assert_eq!(config.max_window_layers, Some(28));

        let eos = config.eos_token_id.as_ref().unwrap();
        assert!(eos.is_eos(151645));
        assert!(eos.is_eos(151643));

        let cfg = config.into_config();
        assert!(cfg.use_qkv_bias);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.max_seq_len, 32768);
    }
}
