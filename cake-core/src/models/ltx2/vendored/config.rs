//! LTX-2 model configuration.

use serde::{Deserialize, Serialize};

/// Which modalities the model processes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ltx2ModelType {
    AudioVideo,
    VideoOnly,
    AudioOnly,
}

impl Ltx2ModelType {
    pub fn is_video_enabled(self) -> bool {
        matches!(self, Self::AudioVideo | Self::VideoOnly)
    }
    pub fn is_audio_enabled(self) -> bool {
        matches!(self, Self::AudioVideo | Self::AudioOnly)
    }
}

/// Full transformer configuration for LTX-2.
///
/// Can be loaded from the HF `transformer/config.json` via serde with aliases.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Ltx2TransformerConfig {
    #[serde(default = "default_video_only")]
    pub model_type: Ltx2ModelType,

    // Video stream
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub cross_attention_dim: usize,

    // Audio stream
    #[serde(default = "default_32")]
    pub audio_num_attention_heads: usize,
    #[serde(default = "default_64")]
    pub audio_attention_head_dim: usize,
    #[serde(default = "default_128")]
    pub audio_in_channels: usize,
    #[serde(default = "default_128")]
    pub audio_out_channels: usize,
    #[serde(default = "default_2048")]
    pub audio_cross_attention_dim: usize,

    // Shared
    pub num_layers: usize,
    pub norm_eps: f64,
    pub activation_fn: String,
    pub attention_bias: bool,
    #[serde(alias = "timestep_scale_multiplier")]
    pub timestep_scale_multiplier: f32,

    // RoPE — HF config uses rope_theta, we map it
    #[serde(alias = "rope_theta")]
    pub positional_embedding_theta: f32,
    #[serde(default = "default_max_pos")]
    pub positional_embedding_max_pos: Vec<usize>,
    #[serde(default = "default_audio_max_pos")]
    pub audio_positional_embedding_max_pos: Vec<usize>,

    // AdaLN
    #[serde(default)]
    pub cross_attention_adaln: bool,

    // Caption projection
    pub caption_channels: usize,
    #[serde(default = "default_2048")]
    pub audio_caption_channels: usize,

    // LTX-2.3 features
    /// Whether attention blocks use learned per-head gating (to_gate_logits).
    #[serde(default)]
    pub gated_attention: bool,
    /// Whether blocks have prompt-specific AdaLN modulation (prompt_scale_shift_table).
    #[serde(default)]
    pub prompt_modulation: bool,
}

fn default_video_only() -> Ltx2ModelType { Ltx2ModelType::VideoOnly }
fn default_32() -> usize { 32 }
fn default_64() -> usize { 64 }
fn default_128() -> usize { 128 }
fn default_2048() -> usize { 2048 }
fn default_max_pos() -> Vec<usize> { vec![20, 2048, 2048] }
fn default_audio_max_pos() -> Vec<usize> { vec![20] }

impl Default for Ltx2TransformerConfig {
    fn default() -> Self {
        Self {
            model_type: Ltx2ModelType::VideoOnly,

            num_attention_heads: 32,
            attention_head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            cross_attention_dim: 4096,

            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_cross_attention_dim: 2048,

            num_layers: 48,
            norm_eps: 1e-6,
            activation_fn: "gelu-approximate".to_string(),
            attention_bias: true,
            timestep_scale_multiplier: 1000.0,

            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: vec![20, 2048, 2048],
            audio_positional_embedding_max_pos: vec![20],

            cross_attention_adaln: false,

            // Gemma-3 outputs 3840-dim embeddings (not 4096)
            caption_channels: 3840,
            audio_caption_channels: 2048,

            gated_attention: false,
            prompt_modulation: false,
        }
    }
}

impl Ltx2TransformerConfig {
    /// Video inner dimension.
    pub fn video_inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio inner dimension.
    pub fn audio_inner_dim(&self) -> usize {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    /// Number of AdaLN parameters per block.
    /// 6 base (shift+scale+gate for self-attn and MLP) + 3 if cross_attention_adaln.
    pub fn adaln_params(&self) -> usize {
        6 + if self.cross_attention_adaln { 3 } else { 0 }
    }

    /// Number of prompt AdaLN parameters per block (LTX-2.3).
    /// 2 params: shift + scale (no gate) for prompt modulation.
    pub fn prompt_adaln_params(&self) -> usize {
        if self.prompt_modulation { 2 } else { 0 }
    }
}

/// LTX-2 scheduler config (separate from the flow-match scheduler used by LTX-Video).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ltx2SchedulerConfig {
    pub base_shift: f32,
    pub max_shift: f32,
    pub power: f32,
    pub stretch_terminal: Option<f32>,
}

impl Default for Ltx2SchedulerConfig {
    fn default() -> Self {
        Self {
            base_shift: 0.95,
            max_shift: 2.05,
            power: 1.0,
            stretch_terminal: Some(0.1),
        }
    }
}

/// LTX-2 text connectors config (Gemma → transformer embedding projection).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Ltx2ConnectorConfig {
    pub caption_channels: usize,
    pub video_connector_num_layers: usize,
    pub video_connector_num_attention_heads: usize,
    pub video_connector_attention_head_dim: usize,
    pub video_connector_num_learnable_registers: usize,
    pub audio_connector_num_layers: usize,
    pub audio_connector_num_attention_heads: usize,
    pub audio_connector_attention_head_dim: usize,
    pub audio_connector_num_learnable_registers: usize,
    pub text_proj_in_factor: usize,
    pub rope_theta: f32,
    pub connector_rope_base_seq_len: usize,
    /// Whether connector uses gated attention (LTX-2.3).
    pub gated_attention: bool,
    /// Whether a separate feature_extractor is used instead of text_proj_in (LTX-2.3).
    pub has_feature_extractor: bool,
    /// Output dim for the feature extractor (LTX-2.3: 4096 = transformer cross_attention_dim).
    /// Only used when has_feature_extractor is true. Defaults to 0 (use video_inner_dim).
    pub feature_extractor_out_dim: usize,
}

impl Default for Ltx2ConnectorConfig {
    fn default() -> Self {
        Self {
            caption_channels: 3840,
            video_connector_num_layers: 2,
            video_connector_num_attention_heads: 30,
            video_connector_attention_head_dim: 128,
            video_connector_num_learnable_registers: 128,
            audio_connector_num_layers: 2,
            audio_connector_num_attention_heads: 30,
            audio_connector_attention_head_dim: 128,
            audio_connector_num_learnable_registers: 128,
            text_proj_in_factor: 49,
            rope_theta: 10000.0,
            connector_rope_base_seq_len: 4096,
            gated_attention: false,
            has_feature_extractor: false,
            feature_extractor_out_dim: 0,
        }
    }
}

impl Ltx2ConnectorConfig {
    /// Config for LTX-2.3 (8 connector blocks, 32 heads, gated attention, feature extractor).
    pub fn for_ltx23() -> Self {
        Self {
            video_connector_num_layers: 8,
            video_connector_num_attention_heads: 32,
            audio_connector_num_layers: 8,
            audio_connector_num_attention_heads: 32,
            gated_attention: true,
            has_feature_extractor: true,
            feature_extractor_out_dim: 4096,
            ..Default::default()
        }
    }

    pub fn video_inner_dim(&self) -> usize {
        self.video_connector_num_attention_heads * self.video_connector_attention_head_dim
    }

    pub fn audio_inner_dim(&self) -> usize {
        self.audio_connector_num_attention_heads * self.audio_connector_attention_head_dim
    }
}

/// VAE config shared with LTX-Video.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ltx2VaeConfig {
    pub latent_channels: usize,
    pub temporal_compression_ratio: usize,
    pub spatial_compression_ratio: usize,
    pub scaling_factor: f32,
    pub timestep_conditioning: bool,
    /// Per-channel mean for latent normalization (128 channels).
    pub latents_mean: Vec<f32>,
    /// Per-channel std for latent normalization (128 channels).
    pub latents_std: Vec<f32>,
}

impl Default for Ltx2VaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 128,
            temporal_compression_ratio: 8,
            spatial_compression_ratio: 32,
            scaling_factor: 1.0,
            // LTX-2 VAE does NOT use timestep conditioning (unlike LTX-Video 0.9.x)
            timestep_conditioning: false,
            // Default: zero mean, unit std (no normalization effect)
            // These should be overridden from the model's config.json
            latents_mean: vec![0.0; 128],
            latents_std: vec![1.0; 128],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_transformer_config() {
        let config = Ltx2TransformerConfig::default();
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.attention_head_dim, 128);
        assert_eq!(config.video_inner_dim(), 4096);
        assert_eq!(config.cross_attention_dim, 4096);
        assert_eq!(config.caption_channels, 3840); // Gemma-3 output dim
        assert_eq!(config.adaln_params(), 6); // no cross_attention_adaln
        assert!(config.model_type.is_video_enabled());
        assert!(!config.model_type.is_audio_enabled());
    }

    #[test]
    fn test_parse_hf_transformer_config() {
        let json = r#"{
            "_class_name": "LTX2VideoTransformer3DModel",
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "in_channels": 128,
            "out_channels": 128,
            "cross_attention_dim": 4096,
            "num_layers": 48,
            "norm_eps": 1e-06,
            "activation_fn": "gelu-approximate",
            "attention_bias": true,
            "caption_channels": 3840,
            "rope_theta": 10000.0,
            "timestep_scale_multiplier": 1000
        }"#;
        let config: Ltx2TransformerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.caption_channels, 3840);
        assert_eq!(config.positional_embedding_theta, 10000.0); // via alias
        assert_eq!(config.timestep_scale_multiplier, 1000.0);
        assert_eq!(config.video_inner_dim(), 4096);
    }

    #[test]
    fn test_default_scheduler_config() {
        let config = Ltx2SchedulerConfig::default();
        assert!((config.base_shift - 0.95).abs() < 1e-6);
        assert!((config.max_shift - 2.05).abs() < 1e-6);
        assert_eq!(config.stretch_terminal, Some(0.1));
    }

    #[test]
    fn test_default_vae_config() {
        let config = Ltx2VaeConfig::default();
        assert_eq!(config.latent_channels, 128);
        assert_eq!(config.temporal_compression_ratio, 8);
        assert_eq!(config.spatial_compression_ratio, 32);
        assert!(!config.timestep_conditioning); // LTX-2 VAE: no timestep conditioning
        assert_eq!(config.latents_mean.len(), 128);
        assert_eq!(config.latents_std.len(), 128);
    }

    #[test]
    fn test_default_connector_config() {
        let config = Ltx2ConnectorConfig::default();
        assert_eq!(config.caption_channels, 3840);
        assert_eq!(config.video_connector_num_layers, 2);
        assert_eq!(config.video_connector_num_learnable_registers, 128);
        assert_eq!(config.video_inner_dim(), 3840); // 30 * 128
    }

    #[test]
    fn test_audio_video_model_type() {
        let av = Ltx2ModelType::AudioVideo;
        assert!(av.is_video_enabled());
        assert!(av.is_audio_enabled());

        let vo = Ltx2ModelType::VideoOnly;
        assert!(vo.is_video_enabled());
        assert!(!vo.is_audio_enabled());

        let ao = Ltx2ModelType::AudioOnly;
        assert!(!ao.is_video_enabled());
        assert!(ao.is_audio_enabled());
    }
}
