use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

/// Mixtral MoE configuration (`MixtralForCausalLM`).
/// Covers Mixtral-8x7B-Instruct and Mixtral-8x22B-Instruct variants.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MixtralConfig {
    pub hidden_size: usize,
    /// Per-expert FFN intermediate size. In Mixtral config.json this is called
    /// `intermediate_size` but it represents the MoE expert dimension.
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
    /// Optional explicit head dimension.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Total number of experts in the pool (e.g. 8).
    pub num_local_experts: usize,
    /// Number of experts activated per token (top-K, e.g. 2).
    pub num_experts_per_tok: usize,
    /// Sliding window size (e.g. 4096 for 8x7B, absent for 8x22B).
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl MixtralConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Mixtral configuration from {}", path.display());
        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    pub fn into_config(self) -> Config {
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        Config {
            hidden_size: self.hidden_size,
            // All Mixtral layers are MoE; set dense intermediate_size to the expert size.
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
            use_qk_norm: false,
            pre_reshape_qk_norm: false,
            sliding_window: self.sliding_window,
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: None,
            global_layers: vec![],
            // Mixtral's `intermediate_size` IS the per-expert MLP size.
            moe_intermediate_size: Some(self.intermediate_size),
            num_experts: self.num_local_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            norm_topk_prob: true, // Mixtral normalises routing weights
            shared_expert_intermediate_size: None,
            attn_output_gate: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixtral_8x7b_config() {
        let json = r#"{
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "sliding_window": 4096,
            "max_position_embeddings": 32768
        }"#;
        let config: MixtralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_local_experts, 8);
        assert_eq!(config.num_experts_per_tok, 2);
        assert_eq!(config.sliding_window, Some(4096));

        let cfg = config.into_config();
        assert_eq!(cfg.moe_intermediate_size, Some(14336));
        assert_eq!(cfg.num_experts, 8);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert!(cfg.norm_topk_prob);
        assert_eq!(cfg.sliding_window, Some(4096));
        assert!(!cfg.use_qk_norm);
    }

    #[test]
    fn test_mixtral_8x22b_config() {
        let json = r#"{
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 6144,
            "intermediate_size": 16384,
            "vocab_size": 32768,
            "num_hidden_layers": 56,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 65536
        }"#;
        let config: MixtralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, None);
        let cfg = config.into_config();
        assert_eq!(cfg.sliding_window, None);
        assert_eq!(cfg.num_experts, 8);
    }
}
