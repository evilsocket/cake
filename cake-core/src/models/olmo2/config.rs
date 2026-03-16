use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    500_000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

/// OLMo 2 configuration (flat JSON, `OLMo2ForCausalLM`).
/// Covers OLMo-2-7B and OLMo-2-32B.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OLMo2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
}

impl OLMo2Config {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading OLMo 2 configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    pub fn into_config(self) -> Config {
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: num_kv_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            tie_word_embeddings: self.tie_word_embeddings,
            max_seq_len: self.max_position_embeddings,
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: self.head_dim,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
            use_qk_norm: true, // OLMo2 uses QK-norm
            pre_reshape_qk_norm: true, // OLMo2 QK-norm is applied before head reshape
            sliding_window: None,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: None,
            moe_intermediate_size: None,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            shared_expert_intermediate_size: None,
            attn_output_gate: false,
            global_layers: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "vocab_size": 100278,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "bos_token_id": 100257,
            "eos_token_id": 100257,
            "tie_word_embeddings": false,
            "max_position_embeddings": 4096
        }"#
    }

    #[test]
    fn test_olmo2_deserialize() {
        let cfg: OLMo2Config = serde_json::from_str(sample_json()).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_key_value_heads, Some(8));
        assert_eq!(cfg.rms_norm_eps, 1e-5);
    }

    #[test]
    fn test_olmo2_into_config() {
        let cfg: OLMo2Config = serde_json::from_str(sample_json()).unwrap();
        let c = cfg.into_config();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.max_seq_len, 4096);
        assert!(c.use_qk_norm, "OLMo2 must enable QK-norm");
        assert!(c.pre_reshape_qk_norm, "OLMo2 QK-norm is pre-reshape");
        assert!(!c.use_qkv_bias);
        assert_eq!(c.model_prefix, "model");
    }

    #[test]
    fn test_olmo2_defaults() {
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "vocab_size": 100278,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-5
        }"#;
        let cfg: OLMo2Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.rope_theta, 500_000.0);
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!(cfg.num_key_value_heads.is_none());
        let c = cfg.into_config();
        assert_eq!(c.num_key_value_heads, 16);
    }
}
