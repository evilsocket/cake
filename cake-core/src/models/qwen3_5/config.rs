use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, LinearAttnConfig, RopeScaling};

fn default_rope() -> f32 {
    10_000_000.0
}

fn default_max_position_embeddings() -> usize {
    262144
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

fn default_head_dim() -> usize {
    256
}

/// Nested rope_parameters within text_config.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// The text_config sub-object from Qwen3.5's config.json.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3_5TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
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
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    // Linear attention config
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
}

/// Top-level Qwen3.5 config.json wrapper.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3_5Config {
    pub text_config: Qwen3_5TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen3_5Config {
    /// Load the configuration from the given path.
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Qwen3.5 configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    /// Return a generalized Config object.
    pub fn into_config(self) -> Config {
        let tc = &self.text_config;
        let num_kv_heads = tc.num_key_value_heads.unwrap_or(tc.num_attention_heads);

        // Extract rope parameters
        let (rope_theta, partial_rotary_factor) = if let Some(ref rp) = tc.rope_parameters {
            (rp.rope_theta, rp.partial_rotary_factor)
        } else {
            (default_rope(), default_partial_rotary_factor())
        };

        // Build linear attention config if layer_types is present
        let linear_attn = if !tc.layer_types.is_empty() {
            Some(LinearAttnConfig {
                layer_types: tc.layer_types.clone(),
                conv_kernel_dim: tc.linear_conv_kernel_dim.unwrap_or(4),
                num_key_heads: tc.linear_num_key_heads.unwrap_or(16),
                key_head_dim: tc.linear_key_head_dim.unwrap_or(128),
                num_value_heads: tc.linear_num_value_heads.unwrap_or(16),
                value_head_dim: tc.linear_value_head_dim.unwrap_or(128),
            })
        } else {
            None
        };

        // Use top-level tie_word_embeddings if set, else text_config level
        let tie_word_embeddings = self.tie_word_embeddings || tc.tie_word_embeddings;

        Config {
            hidden_size: tc.hidden_size,
            intermediate_size: tc.intermediate_size,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_5_config() {
        let json = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "vocab_size": 248320,
                "num_hidden_layers": 24,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "eos_token_id": 248044,
                "tie_word_embeddings": true,
                "max_position_embeddings": 262144,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "default"
                }
            }
        }"#;
        let config: Qwen3_5Config = serde_json::from_str(json).unwrap();
        let cfg = config.into_config();

        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, Some(256));
        assert_eq!(cfg.model_prefix, "model.language_model");
        assert_eq!(cfg.partial_rotary_factor, 0.25);
        assert!(cfg.tie_word_embeddings);

        let la = cfg.linear_attn.unwrap();
        assert_eq!(la.layer_types.len(), 4);
        assert_eq!(la.conv_kernel_dim, 4);
        assert_eq!(la.num_key_heads, 16);
        assert_eq!(la.key_head_dim, 128);
    }
}
