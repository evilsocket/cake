use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    10_000.0
}

fn default_sliding_window() -> usize {
    1024
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_sliding_window_pattern() -> usize {
    6
}

/// Gemma 3 configuration (flat JSON, `Gemma3ForCausalLM`).
///
/// Gemma 3 uses interleaved local (sliding window, no RoPE) and global (full RoPE) attention.
/// The default pattern is every 6th layer (0-indexed) is global, the rest are local.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3Config {
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
    /// Head dimension (e.g. 256 for Gemma 3).
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Sliding window size for local layers (default 1024).
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// How often a global layer occurs: every Nth layer is global (default 6).
    /// E.g. 6 means layers 5, 11, 17, ... are global (0-indexed).
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    /// Optional per-layer schedule (true=global, false=local). Overrides pattern if present.
    #[serde(default)]
    pub sliding_window_attention_schedule: Vec<bool>,
    /// Attention logit scaling factor (default = head_dim^(-0.5), but Gemma3 may override).
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f64>,
}

impl Gemma3Config {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Gemma 3 configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    /// Returns true if `layer_idx` is a global (full attention) layer.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        if !self.sliding_window_attention_schedule.is_empty() {
            // Explicit schedule: true means global
            self.sliding_window_attention_schedule
                .get(layer_idx)
                .copied()
                .unwrap_or(false)
        } else {
            // Pattern: every `sliding_window_pattern`-th layer (0-indexed) is global.
            // E.g. pattern=6 → layer 5, 11, 17, ... are global.
            (layer_idx + 1) % self.sliding_window_pattern == 0
        }
    }

    pub fn into_config(self) -> Config {
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        // Build per-layer global schedule for use in Gemma3Block
        let global_layers: Vec<bool> = (0..self.num_hidden_layers)
            .map(|i| self.is_global_layer(i))
            .collect();
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
            tie_word_embeddings: true, // Gemma3 always ties lm_head to embed_tokens
            max_seq_len: self.max_position_embeddings,
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: self.head_dim,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: true, // Gemma3RMSNorm: weight stored as delta from 0, forward = (1+weight)*norm(x)
            use_qk_norm: true,
            pre_reshape_qk_norm: false, // Both local and global layers use QK-norm
            sliding_window: Some(self.sliding_window), // local-layer window size
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            global_layers,
            use_gelu_mlp: true, // Gemma3 uses GELU-tanh (gelu_pytorch_tanh), not SiLU
            embed_scale: Some((self.hidden_size as f32).sqrt()),
            moe_intermediate_size: None,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma3_pattern() {
        let json = r#"{
            "architectures": ["Gemma3ForCausalLM"],
            "hidden_size": 1152,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "head_dim": 256
        }"#;
        let config: Gemma3Config = serde_json::from_str(json).unwrap();

        // layers 5, 11, 17, 23 are global (0-indexed, every 6th)
        assert!(!config.is_global_layer(0));
        assert!(!config.is_global_layer(4));
        assert!(config.is_global_layer(5));
        assert!(!config.is_global_layer(6));
        assert!(config.is_global_layer(11));
        assert!(config.is_global_layer(17));
        assert!(config.is_global_layer(23));
        assert!(!config.is_global_layer(24));
        assert!(!config.is_global_layer(25));

        let cfg = config.into_config();
        assert!(cfg.use_qk_norm);
        assert_eq!(cfg.sliding_window, Some(1024));
    }
}
