//! VibeVoice-Realtime-0.5B configuration.
//!
//! Maps the HuggingFace config.json to cake's common Config struct.
//! The LLM backbone is Qwen2.5-0.5B (24 layers, 896 hidden).

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

use crate::models::common::Config;

/// Diffusion head configuration.
#[derive(Debug, Deserialize)]
pub struct DiffusionHeadConfig {
    pub ddpm_num_inference_steps: usize,
    pub ddpm_num_steps: usize,
    pub head_layers: usize,
    pub hidden_size: usize,
    pub latent_size: usize,
    pub head_ffn_ratio: f64,
    pub prediction_type: String,
    pub rms_norm_eps: f64,
    #[serde(default = "default_ddpm_beta_schedule")]
    pub ddpm_beta_schedule: String,
}

fn default_ddpm_beta_schedule() -> String {
    "cosine".to_string()
}

/// Acoustic tokenizer configuration.
#[derive(Debug, Deserialize)]
pub struct AcousticTokenizerConfig {
    pub vae_dim: usize,
    pub encoder_n_filters: usize,
    pub decoder_n_filters: Option<usize>,
    pub encoder_ratios: Vec<usize>,
    pub decoder_ratios: Option<Vec<usize>>,
    pub encoder_depths: Option<String>,
    pub decoder_depths: Option<String>,
    pub layernorm: String,
    pub layernorm_eps: f64,
    #[serde(default)]
    pub causal: bool,
}

/// Qwen2.5 decoder (LLM backbone) configuration.
#[derive(Debug, Deserialize)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Top-level VibeVoice config.json structure.
#[derive(Debug, Deserialize)]
pub struct VibeVoiceConfig {
    pub acoustic_vae_dim: usize,
    pub decoder_config: DecoderConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,
    pub tts_backbone_num_hidden_layers: usize,
}

impl VibeVoiceConfig {
    /// Load from a config.json file path.
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
            eos_token_id: None,
            rope_scaling: None,
            tie_word_embeddings: dc.tie_word_embeddings,
            max_seq_len: dc.max_position_embeddings,
            use_qkv_bias: true, // Qwen2.5 uses QKV bias
            model_prefix: "model.tts_language_model".into(),
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

    const SAMPLE_CONFIG: &str = r#"{
        "acoustic_vae_dim": 64,
        "acoustic_tokenizer_config": {
            "causal": true,
            "channels": 1,
            "conv_bias": true,
            "conv_norm": "none",
            "decoder_n_filters": 32,
            "decoder_ratios": [8, 5, 5, 4, 2, 2],
            "encoder_depths": "3-3-3-3-3-3-8",
            "encoder_n_filters": 32,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "fix_std": 0.5,
            "layernorm": "RMSNorm",
            "layernorm_eps": 1e-05,
            "mixer_layer": "depthwise_conv",
            "model_type": "vibevoice_acoustic_tokenizer",
            "vae_dim": 64
        },
        "architectures": ["VibeVoiceStreamingForConditionalGenerationInference"],
        "decoder_config": {
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 896,
            "initializer_range": 0.02,
            "intermediate_size": 4864,
            "max_position_embeddings": 8192,
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "vocab_size": 151936
        },
        "diffusion_head_config": {
            "ddpm_batch_mul": 4,
            "ddpm_beta_schedule": "cosine",
            "ddpm_num_inference_steps": 20,
            "ddpm_num_steps": 1000,
            "diffusion_type": "ddpm",
            "head_ffn_ratio": 3.0,
            "head_layers": 4,
            "hidden_size": 896,
            "latent_size": 64,
            "model_type": "vibevoice_diffusion_head",
            "prediction_type": "v_prediction",
            "rms_norm_eps": 1e-05,
            "speech_vae_dim": 64
        },
        "model_type": "vibevoice_streaming",
        "torch_dtype": "bfloat16",
        "tts_backbone_num_hidden_layers": 20
    }"#;

    #[test]
    fn test_vibevoice_config_deserialize() {
        let cfg: VibeVoiceConfig = serde_json::from_str(SAMPLE_CONFIG).unwrap();
        assert_eq!(cfg.acoustic_vae_dim, 64);
        assert_eq!(cfg.tts_backbone_num_hidden_layers, 20);
        assert_eq!(cfg.decoder_config.hidden_size, 896);
        assert_eq!(cfg.decoder_config.num_attention_heads, 14);
        assert_eq!(cfg.decoder_config.num_key_value_heads, 2);
        assert_eq!(cfg.decoder_config.num_hidden_layers, 24);
        assert_eq!(cfg.decoder_config.vocab_size, 151936);
        assert_eq!(cfg.diffusion_head_config.head_layers, 4);
        assert_eq!(cfg.diffusion_head_config.latent_size, 64);
        assert_eq!(cfg.diffusion_head_config.hidden_size, 896);
        assert_eq!(cfg.diffusion_head_config.ddpm_num_inference_steps, 20);
        assert_eq!(cfg.diffusion_head_config.prediction_type, "v_prediction");
        assert_eq!(cfg.acoustic_tokenizer_config.vae_dim, 64);
        assert_eq!(cfg.acoustic_tokenizer_config.decoder_ratios, Some(vec![8, 5, 5, 4, 2, 2]));
    }

    #[test]
    fn test_vibevoice_into_config() {
        let cfg: VibeVoiceConfig = serde_json::from_str(SAMPLE_CONFIG).unwrap();
        let common = cfg.into_config();
        assert_eq!(common.hidden_size, 896);
        assert_eq!(common.intermediate_size, 4864);
        assert_eq!(common.num_attention_heads, 14);
        assert_eq!(common.num_key_value_heads, 2);
        assert_eq!(common.num_hidden_layers, 24);
        assert_eq!(common.vocab_size, 151936);
        assert!(common.use_qkv_bias);
        assert_eq!(common.rope_theta, 1_000_000.0);
        assert_eq!(common.model_prefix, "model.tts_language_model");
    }
}
