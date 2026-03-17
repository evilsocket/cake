//! VibeVoice-1.5B (non-streaming) configuration.
//!
//! Maps the HuggingFace config.json to cake's common Config struct.
//! Single Qwen2.5-1.5B LM (28 layers, 1536 hidden) + semantic tokenizer.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

use crate::models::common::Config;

// Reuse shared config types from the 0.5B config
pub use super::config::{AcousticTokenizerConfig, DecoderConfig, DiffusionHeadConfig};

/// Semantic tokenizer configuration (encoder-only, no VAE).
#[derive(Debug, Deserialize)]
pub struct SemanticTokenizerConfig {
    pub vae_dim: usize,
    pub encoder_n_filters: usize,
    pub encoder_ratios: Vec<usize>,
    pub encoder_depths: Option<String>,
    pub layernorm: String,
    pub layernorm_eps: f64,
    #[serde(default)]
    pub causal: bool,
    #[serde(default)]
    pub fix_std: f64,
    #[serde(default = "default_std_dist_type")]
    pub std_dist_type: String,
}

fn default_std_dist_type() -> String {
    "none".to_string()
}

/// Top-level VibeVoice-1.5B config.json structure.
#[derive(Debug, Deserialize)]
pub struct VibeVoice1_5BConfig {
    pub acoustic_vae_dim: usize,
    pub semantic_vae_dim: usize,
    pub decoder_config: DecoderConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,
    pub semantic_tokenizer_config: SemanticTokenizerConfig,
    pub model_type: String,
}

/// Special token IDs for VibeVoice-1.5B (from Qwen2.5 vocabulary).
pub const SPEECH_START_ID: u32 = 151652; // <|vision_start|>
pub const SPEECH_END_ID: u32 = 151653; // <|vision_end|>
pub const SPEECH_DIFFUSION_ID: u32 = 151654; // <|vision_pad|>
pub const EOS_ID: u32 = 151643; // <|endoftext|>
pub const IMAGE_PAD_ID: u32 = 151655; // <|image_pad|> used as pad

impl VibeVoice1_5BConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&data)?;
        Ok(cfg)
    }

    /// Convert the LLM backbone config to cake's common Config.
    pub fn into_config(&self) -> Config {
        let dc = &self.decoder_config;
        Config {
            hidden_size: dc.hidden_size,
            intermediate_size: dc.intermediate_size,
            vocab_size: dc.vocab_size,
            num_hidden_layers: dc.num_hidden_layers,
            num_attention_heads: dc.num_attention_heads,
            num_key_value_heads: dc.num_key_value_heads,
            rms_norm_eps: dc.rms_norm_eps,
            rope_theta: dc.rope_theta as f32,
            bos_token_id: None,
            eos_token_id: Some(crate::models::common::EosTokenId::Single(EOS_ID)),
            rope_scaling: None,
            tie_word_embeddings: dc.tie_word_embeddings,
            max_seq_len: dc.max_position_embeddings,
            use_qkv_bias: true, // Qwen2.5 uses QKV bias
            model_prefix: "model.language_model".into(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_1_5b_config() {
        let config_path = std::path::Path::new(
            &std::env::var("HOME").unwrap_or_default()
        ).join(".cache/huggingface/hub/models--microsoft--VibeVoice-1.5B/snapshots/c00898d257e6b46004e3e2866a47534085fb685a/config.json");
        if !config_path.exists() {
            return; // Skip if model not downloaded
        }
        let cfg = VibeVoice1_5BConfig::from_path(&config_path).unwrap();
        assert_eq!(cfg.model_type, "vibevoice");
        assert_eq!(cfg.acoustic_vae_dim, 64);
        assert_eq!(cfg.semantic_vae_dim, 128);
        assert_eq!(cfg.decoder_config.hidden_size, 1536);
        assert_eq!(cfg.decoder_config.num_hidden_layers, 28);
        assert_eq!(cfg.decoder_config.num_attention_heads, 12);
        assert_eq!(cfg.decoder_config.num_key_value_heads, 2);
        assert_eq!(cfg.diffusion_head_config.hidden_size, 1536);
        assert_eq!(cfg.semantic_tokenizer_config.vae_dim, 128);
        let common = cfg.into_config();
        assert_eq!(common.hidden_size, 1536);
    }
}
