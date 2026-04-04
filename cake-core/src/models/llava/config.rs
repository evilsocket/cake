use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    500_000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

fn default_false() -> bool {
    false
}

fn default_mm_projector_type() -> String {
    "mlp2x_gelu".to_string()
}

fn default_mm_vision_select_layer() -> isize {
    -2
}

fn default_mm_vision_select_feature() -> String {
    "patch".to_string()
}

fn default_image_token_index() -> u32 {
    32000
}

/// Raw LLaVA text (LLM backbone) config from config.json, matching HF format.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlavaTextConfig {
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

/// Raw LLaVA vision (CLIP) config from config.json.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlavaVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

/// LLaVA-specific configuration (serde deserialization from config.json).
///
/// Supports both the HuggingFace LlavaForConditionalGeneration format
/// (nested `text_config` / `vision_config`) and the original LLaVA format
/// (flat config with `mm_*` fields).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlavaConfig {
    /// Nested text (LLM) config — HF format.
    pub text_config: Option<LlavaTextConfig>,
    /// Nested vision (CLIP) config — HF format.
    pub vision_config: Option<LlavaVisionConfig>,

    // Flat LLM fields — original LLaVA format (used when text_config is absent).
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: Option<f64>,
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

    // Multi-modal fields.
    /// Hidden size of CLIP vision tower output.
    #[serde(default)]
    pub mm_hidden_size: Option<usize>,
    /// Projector type (e.g. "mlp2x_gelu").
    #[serde(default = "default_mm_projector_type")]
    pub mm_projector_type: String,
    /// Which vision encoder layer to extract features from (typically -2).
    #[serde(default = "default_mm_vision_select_layer")]
    pub mm_vision_select_layer: isize,
    /// Which feature to use: "patch" (no CLS token) or "cls_patch" (with CLS).
    #[serde(default = "default_mm_vision_select_feature")]
    pub mm_vision_select_feature: String,
    /// Token ID used as `<image>` placeholder in text.
    #[serde(default = "default_image_token_index")]
    pub image_token_index: u32,
}

impl LlavaConfig {
    /// Load the configuration from the given path.
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading LLaVA configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    /// Whether the config uses the HF nested format.
    fn is_hf_format(&self) -> bool {
        self.text_config.is_some()
    }

    /// Return a generalized Config object for the LLM backbone.
    pub fn into_config(&self) -> Config {
        if let Some(ref tc) = self.text_config {
            // HF format: LlavaForConditionalGeneration
            Config {
                hidden_size: tc.hidden_size,
                intermediate_size: tc.intermediate_size,
                vocab_size: tc.vocab_size,
                num_hidden_layers: tc.num_hidden_layers,
                num_attention_heads: tc.num_attention_heads,
                num_key_value_heads: tc.num_key_value_heads.unwrap_or(tc.num_attention_heads),
                rms_norm_eps: tc.rms_norm_eps,
                rope_theta: tc.rope_theta,
                bos_token_id: tc.bos_token_id,
                eos_token_id: tc.eos_token_id.clone(),
                rope_scaling: tc.rope_scaling.clone(),
                tie_word_embeddings: tc.tie_word_embeddings,
                max_seq_len: tc.max_position_embeddings,
                use_qkv_bias: false,
                model_prefix: "language_model.model".into(),
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
                shared_expert_intermediate_size: None,
                attn_output_gate: false,
                global_layers: vec![],
            }
        } else {
            // Original LLaVA format (flat config)
            Config {
                hidden_size: self.hidden_size.unwrap_or(4096),
                intermediate_size: self.intermediate_size.unwrap_or(11008),
                vocab_size: self.vocab_size.unwrap_or(32064),
                num_hidden_layers: self.num_hidden_layers.unwrap_or(32),
                num_attention_heads: self.num_attention_heads.unwrap_or(32),
                num_key_value_heads: self
                    .num_key_value_heads
                    .unwrap_or(self.num_attention_heads.unwrap_or(32)),
                rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
                rope_theta: self.rope_theta,
                bos_token_id: self.bos_token_id,
                eos_token_id: self.eos_token_id.clone(),
                rope_scaling: self.rope_scaling.clone(),
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
                shared_expert_intermediate_size: None,
                attn_output_gate: false,
                global_layers: vec![],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llava_hf_config() {
        let json = r#"{
            "architectures": ["LlavaForConditionalGeneration"],
            "text_config": {
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "vocab_size": 32064,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "rms_norm_eps": 1e-5
            },
            "vision_config": {
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "image_size": 336,
                "patch_size": 14
            },
            "mm_projector_type": "mlp2x_gelu",
            "mm_vision_select_layer": -2,
            "mm_vision_select_feature": "patch",
            "image_token_index": 32000
        }"#;
        let config: LlavaConfig = serde_json::from_str(json).unwrap();
        assert!(config.is_hf_format());
        assert_eq!(config.mm_vision_select_layer, -2);
        assert_eq!(config.image_token_index, 32000);

        let cfg = config.into_config();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.model_prefix, "language_model.model");
    }

    #[test]
    fn test_llava_original_config() {
        let json = r#"{
            "architectures": ["LlavaLlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "vocab_size": 32064,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-5,
            "mm_hidden_size": 1024,
            "mm_projector_type": "mlp2x_gelu",
            "image_token_index": 32000
        }"#;
        let config: LlavaConfig = serde_json::from_str(json).unwrap();
        assert!(!config.is_hf_format());
        assert_eq!(config.mm_hidden_size, Some(1024));

        let cfg = config.into_config();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.model_prefix, "model");
    }
}
