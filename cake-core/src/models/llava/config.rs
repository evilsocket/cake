use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId};

fn default_hf() -> bool {
    false
}

fn default_image_token_index() -> isize {
    -200
}

fn default_mm_patch_merge_type() -> String {
    "flat".to_string()
}

fn default_image_aspect_ratio() -> String {
    "square".to_string()
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

fn default_false() -> bool {
    false
}

/// LLaVA-specific configuration (serde deserialization from config.json).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlavaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    // Vision/multimodal fields
    #[serde(default = "default_image_aspect_ratio")]
    pub image_aspect_ratio: String,
    #[serde(default)]
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    #[serde(default)]
    pub mm_hidden_size: Option<usize>,
    #[serde(default = "default_mm_patch_merge_type")]
    pub mm_patch_merge_type: String,
    #[serde(default)]
    pub mm_projector_type: Option<String>,
    #[serde(default)]
    pub mm_vision_select_feature: Option<String>,
    #[serde(default)]
    pub mm_vision_select_layer: Option<isize>,
    #[serde(default)]
    pub mm_vision_tower: Option<String>,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: isize,
    #[serde(default = "default_hf")]
    pub hf: bool,

    // HuggingFace-format fields (llava-hf models)
    #[serde(default)]
    pub vision_config: Option<HfVisionConfig>,
    #[serde(default)]
    pub text_config: Option<HfTextConfig>,
    #[serde(default)]
    pub vision_feature_layer: Option<isize>,
    #[serde(default)]
    pub vision_feature_select_strategy: Option<String>,
    #[serde(default)]
    pub projector_hidden_act: Option<String>,
}

/// HF-format vision config (nested in config.json for llava-hf models).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HfVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    #[serde(default)]
    pub projection_dim: Option<usize>,
}

/// HF-format text config (nested in config.json for llava-hf models).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HfTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    pub vocab_size: usize,
}

impl LlavaConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading LLaVA configuration from {}", path.display());
        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    pub fn num_key_value_heads(&self) -> usize {
        if let Some(tc) = &self.text_config {
            tc.num_key_value_heads
        } else {
            self.num_key_value_heads.unwrap_or(self.num_attention_heads)
        }
    }

    /// Effective number of LLM layers.
    pub fn effective_num_hidden_layers(&self) -> usize {
        if let Some(tc) = &self.text_config {
            tc.num_hidden_layers
        } else {
            self.num_hidden_layers
        }
    }

    /// Effective hidden size.
    pub fn effective_hidden_size(&self) -> usize {
        if let Some(tc) = &self.text_config {
            tc.hidden_size
        } else {
            self.hidden_size
        }
    }

    /// Effective intermediate size.
    pub fn effective_intermediate_size(&self) -> usize {
        if let Some(tc) = &self.text_config {
            tc.intermediate_size
        } else {
            self.intermediate_size
        }
    }

    /// Effective vocab size.
    pub fn effective_vocab_size(&self) -> usize {
        if let Some(tc) = &self.text_config {
            tc.vocab_size
        } else {
            self.vocab_size
        }
    }

    /// Convert to the generalized Config for TextModelBase.
    pub fn into_config(self) -> Config {
        let hidden_size = self.effective_hidden_size();
        let intermediate_size = self.effective_intermediate_size();
        let vocab_size = self.effective_vocab_size();
        let num_hidden_layers = self.effective_num_hidden_layers();
        let num_attention_heads = if let Some(tc) = &self.text_config {
            tc.num_attention_heads
        } else {
            self.num_attention_heads
        };
        let num_key_value_heads = self.num_key_value_heads();
        let rms_norm_eps = if let Some(tc) = &self.text_config {
            tc.rms_norm_eps.unwrap_or(self.rms_norm_eps)
        } else {
            self.rms_norm_eps
        };
        let rope_theta = if let Some(tc) = &self.text_config {
            tc.rope_theta.unwrap_or(self.rope_theta)
        } else {
            self.rope_theta
        };
        let max_seq_len = if let Some(tc) = &self.text_config {
            tc.max_position_embeddings
        } else {
            self.max_position_embeddings
        };

        // HF-format LLaVA uses "language_model" prefix, original uses "model"
        let model_prefix = if self.hf || self.text_config.is_some() {
            "language_model.model".into()
        } else {
            "model".into()
        };

        Config {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: None,
            tie_word_embeddings: self.tie_word_embeddings,
            max_seq_len,
            use_qkv_bias: false,
            model_prefix,
            head_dim: None,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
        }
    }

    /// Get the mm_hidden_size (vision tower output dim).
    pub fn effective_mm_hidden_size(&self) -> usize {
        if let Some(vc) = &self.vision_config {
            vc.hidden_size
        } else {
            self.mm_hidden_size.unwrap_or(1024)
        }
    }

    /// Get the vision select layer.
    pub fn effective_vision_select_layer(&self) -> isize {
        self.vision_feature_layer
            .or(self.mm_vision_select_layer)
            .unwrap_or(-2)
    }

    /// Get the vision select feature method.
    pub fn effective_vision_select_feature(&self) -> String {
        if let Some(ref strategy) = self.vision_feature_select_strategy {
            if strategy == "default" {
                "patch".to_string()
            } else {
                strategy.clone()
            }
        } else {
            self.mm_vision_select_feature
                .clone()
                .unwrap_or_else(|| "patch".to_string())
        }
    }

    /// Get the projector type.
    pub fn effective_projector_type(&self) -> String {
        if let Some(ref act) = self.projector_hidden_act {
            if act == "gelu" {
                "mlp2x_gelu".to_string()
            } else {
                act.clone()
            }
        } else {
            self.mm_projector_type
                .clone()
                .unwrap_or_else(|| "mlp2x_gelu".to_string())
        }
    }

    /// Build the candle-transformers LLaVAConfig for loading the upstream model.
    pub fn to_candle_llava_config(&self) -> candle_transformers::models::llava::config::LLaVAConfig {
        let is_hf = self.hf || self.text_config.is_some();
        candle_transformers::models::llava::config::LLaVAConfig {
            architectures: vec!["LlavaForConditionalGeneration".to_string()],
            bos_token_id: self.bos_token_id.unwrap_or(1) as usize,
            eos_token_id: match &self.eos_token_id {
                Some(EosTokenId::Single(id)) => *id as usize,
                _ => 2,
            },
            hidden_size: self.effective_hidden_size(),
            image_aspect_ratio: self.image_aspect_ratio.clone(),
            image_crop_resolution: 224,
            image_grid_pinpoints: if self.image_grid_pinpoints.is_empty() {
                vec![(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]
            } else {
                self.image_grid_pinpoints.clone()
            },
            image_split_resolution: 224,
            intermediate_size: self.effective_intermediate_size(),
            max_position_embeddings: if let Some(tc) = &self.text_config {
                tc.max_position_embeddings
            } else {
                self.max_position_embeddings
            },
            mm_hidden_size: self.effective_mm_hidden_size(),
            mm_patch_merge_type: self.mm_patch_merge_type.clone(),
            mm_projector_type: self.effective_projector_type(),
            mm_use_im_start_end: false,
            mm_vision_select_feature: self.effective_vision_select_feature(),
            mm_vision_select_layer: self.effective_vision_select_layer(),
            mm_vision_tower: self.mm_vision_tower.clone(),
            model_type: "llava".to_string(),
            num_attention_heads: if let Some(tc) = &self.text_config {
                tc.num_attention_heads
            } else {
                self.num_attention_heads
            },
            num_hidden_layers: self.effective_num_hidden_layers(),
            num_key_value_heads: self.num_key_value_heads(),
            pad_token_id: 0,
            rms_norm_eps: if let Some(tc) = &self.text_config {
                tc.rms_norm_eps.unwrap_or(self.rms_norm_eps) as f32
            } else {
                self.rms_norm_eps as f32
            },
            rope_theta: if let Some(tc) = &self.text_config {
                tc.rope_theta.unwrap_or(self.rope_theta)
            } else {
                self.rope_theta
            },
            tokenizer_model_max_length: None,
            torch_dtype: "float16".to_string(),
            use_cache: true,
            vocab_size: self.effective_vocab_size(),
            image_token_index: self.image_token_index,
            hf: is_hf,
            tie_word_embeddings: Some(self.tie_word_embeddings),
        }
    }
}
