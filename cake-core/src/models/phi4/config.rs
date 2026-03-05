use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    1_000_000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_max_position_embeddings() -> usize {
    131072
}

/// Phi-3 / Phi-4-mini / Phi-4 configuration (flat JSON, `Phi3ForCausalLM` or `Phi4ForCausalLM`).
///
/// These models use pre-fused `qkv_proj` and `gate_up_proj` weight tensors instead of
/// separate `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Phi4Config {
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
    /// Explicit head dimension (e.g. 96 for Phi-4-mini).
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Fraction of head dims that get RoPE applied (Phi-4-mini = 0.75).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
}

impl Phi4Config {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Phi-4 configuration from {}", path.display());

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
            partial_rotary_factor: self.partial_rotary_factor,
            linear_attn: None,
            residual_rms_norm: false,
            use_qk_norm: false,
            pre_reshape_qk_norm: false,
            sliding_window: None,
            fused_qkv_proj: true,
            fused_gate_up_proj: true,
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

    #[test]
    fn test_phi4_mini_config() {
        let json = r#"{
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "vocab_size": 200064,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 128000
        }"#;
        let config: Phi4Config = serde_json::from_str(json).unwrap();
        let cfg = config.into_config();

        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(cfg.fused_qkv_proj);
        assert!(cfg.fused_gate_up_proj);
        assert!(!cfg.use_qk_norm);
    }
}
