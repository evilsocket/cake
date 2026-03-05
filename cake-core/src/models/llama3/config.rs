use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

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

/// LLama specific configuration (serde deserialization from config.json).
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
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: None,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
            use_qk_norm: false,
            pre_reshape_qk_norm: false,
            sliding_window: None,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: None,
            moe_intermediate_size: None,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            global_layers: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!cfg.use_qkv_bias);
    }
}
