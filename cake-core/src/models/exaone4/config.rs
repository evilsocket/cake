use std::path::Path;

use anyhow::Result;

use crate::models::common::{Config, EosTokenId, RopeScaling};

fn default_rope() -> f32 {
    500_000.0
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_sliding_window() -> usize {
    4096
}

/// EXAONE 4.0 configuration (flat JSON, `ExaoneForCausalLM`).
///
/// EXAONE 4.0 uses a 3:1 local/global hybrid pattern:
/// - 3 local layers (sliding window + RoPE) then 1 global layer (full context, **no RoPE**)
/// The global attention layers skip positional embeddings entirely.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EXAONE4Config {
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
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Sliding window size for local layers (default 4096).
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// Global layer period: every Nth layer is global (default 4 → 3 local + 1 global).
    #[serde(default)]
    pub global_layer_period: Option<usize>,
}

impl EXAONE4Config {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading EXAONE 4.0 configuration from {}", path.display());

        let data =
            std::fs::read(path).map_err(|e| anyhow!("can't read {}: {:?}", path.display(), e))?;
        serde_json::from_slice(&data)
            .map_err(|e| anyhow!("can't parse {}: {:?}", path.display(), e))
    }

    /// Returns true if `layer_idx` is a global (no-RoPE, full context) layer.
    /// Default: every 4th layer (0-indexed: 3, 7, 11, ...) is global.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        let period = self.global_layer_period.unwrap_or(4);
        (layer_idx + 1) % period == 0
    }

    pub fn into_config(self) -> Config {
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
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
            tie_word_embeddings: self.tie_word_embeddings,
            max_seq_len: self.max_position_embeddings,
            use_qkv_bias: false,
            model_prefix: "model".into(),
            head_dim: self.head_dim,
            partial_rotary_factor: 1.0,
            linear_attn: None,
            residual_rms_norm: false,
            use_qk_norm: true,
            pre_reshape_qk_norm: false, // EXAONE 4.0 uses QK-norm
            sliding_window: Some(self.sliding_window), // local layer window
            fused_qkv_proj: false,
            fused_gate_up_proj: false,
            use_gelu_mlp: false,
            embed_scale: None,
            moe_intermediate_size: None,
            num_experts: 0,
            num_experts_per_tok: 0,
            norm_topk_prob: false,
            global_layers,
        }
    }
}
