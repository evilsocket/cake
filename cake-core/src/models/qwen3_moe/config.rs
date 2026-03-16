use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    131072
}

/// Qwen3 MoE model configuration (`Qwen3MoeForCausalLM`).
/// Covers Qwen3-30B-A3B and Qwen3-235B-A22B (and Qwen3-Coder MoE variants).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3MoeConfig {
    pub hidden_size: usize,
    /// Dense MLP intermediate size (used only for `mlp_only_layers`, typically unused).
    pub intermediate_size: usize,
    /// Per-expert FFN intermediate size (the MoE-active dimension).
    pub moe_intermediate_size: usize,
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
    /// Optional explicit head dimension.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Total number of experts in the pool.
    pub num_experts: usize,
    /// Number of experts activated per token (top-K).
    pub num_experts_per_tok: usize,
    /// Re-normalise top-K weights to sum to 1.0 after selection.
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
}

fn default_true() -> bool {
    true
}

impl Qwen3MoeConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Qwen3 MoE configuration from {}", path.display());
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
            use_qk_norm: true,   // Qwen3 MoE uses QK-norm (same as dense Qwen3)
            pre_reshape_qk_norm: false,
            sliding_window: None,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: None,
            global_layers: vec![],
            moe_intermediate_size: Some(self.moe_intermediate_size),
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            norm_topk_prob: self.norm_topk_prob,
            shared_expert_intermediate_size: None,
            attn_output_gate: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "moe_intermediate_size": 1024,
            "vocab_size": 151936,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "norm_topk_prob": true,
            "max_position_embeddings": 131072
        }"#
    }

    #[test]
    fn test_qwen3_moe_deserialize() {
        let cfg: Qwen3MoeConfig = serde_json::from_str(sample_json()).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.moe_intermediate_size, 1024);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert!(cfg.norm_topk_prob);
    }

    #[test]
    fn test_qwen3_moe_into_config() {
        let cfg: Qwen3MoeConfig = serde_json::from_str(sample_json()).unwrap();
        let c = cfg.into_config();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.intermediate_size, 8192); // dense intermediate preserved
        assert_eq!(c.moe_intermediate_size, Some(1024));
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.num_experts_per_tok, 8);
        assert!(c.norm_topk_prob);
        assert!(c.use_qk_norm, "Qwen3 MoE uses QK-norm");
        assert!(!c.pre_reshape_qk_norm);
        assert_eq!(c.num_key_value_heads, 4);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert!(c.shared_expert_intermediate_size.is_none());
    }

    #[test]
    fn test_qwen3_moe_defaults() {
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "moe_intermediate_size": 1024,
            "vocab_size": 151936,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-6,
            "num_experts": 64,
            "num_experts_per_tok": 4
        }"#;
        let cfg: Qwen3MoeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.rope_theta, 1_000_000.0); // default
        assert_eq!(cfg.max_position_embeddings, 131072); // default
        assert!(cfg.norm_topk_prob); // default_true
        assert!(cfg.num_key_value_heads.is_none());
        let c = cfg.into_config();
        assert_eq!(c.num_key_value_heads, 16); // falls back to num_attention_heads
    }
}
