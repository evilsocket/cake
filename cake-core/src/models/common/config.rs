use anyhow::Result;
use serde::de::{self, Deserializer};

/// EOS token ID(s) — deserializes from either a single u32 or an array of u32.
#[derive(Debug, Clone)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    /// Check if the given token ID is an EOS token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        match self {
            EosTokenId::Single(id) => *id == token_id,
            EosTokenId::Multiple(ids) => ids.contains(&token_id),
        }
    }
}

impl<'de> serde::Deserialize<'de> for EosTokenId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Number(n) => n
                .as_u64()
                .map(|v| EosTokenId::Single(v as u32))
                .ok_or_else(|| de::Error::custom("expected u32 for eos_token_id")),
            serde_json::Value::Array(arr) => {
                let ids: std::result::Result<Vec<u32>, _> = arr
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .map(|n| n as u32)
                            .ok_or_else(|| de::Error::custom("expected u32 in eos_token_id array"))
                    })
                    .collect();
                ids.map(EosTokenId::Multiple)
            }
            _ => Err(de::Error::custom("expected u32 or array for eos_token_id")),
        }
    }
}

/// RoPE scaling configuration for models with extended context (e.g. LLaMA 3.1+).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    /// Factor for scaling (typically 8.0 for LLaMA 3.1).
    #[serde(default)]
    pub factor: f32,
    /// High frequency factor (typically 4.0).
    #[serde(default)]
    pub high_freq_factor: f32,
    /// Low frequency factor (typically 1.0).
    #[serde(default)]
    pub low_freq_factor: f32,
    /// Original max position embeddings before scaling (typically 8192).
    #[serde(default)]
    pub original_max_position_embeddings: usize,
    /// RoPE type (e.g. "llama3").
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Configuration for linear (recurrent) attention layers (e.g. Gated DeltaNet in Qwen3.5).
#[derive(Debug, Clone)]
pub struct LinearAttnConfig {
    /// Per-layer type: "linear_attention" or "full_attention".
    pub layer_types: Vec<String>,
    /// Conv1d kernel size for short convolution preprocessing.
    pub conv_kernel_dim: usize,
    /// Number of key heads in linear attention.
    pub num_key_heads: usize,
    /// Per-head key dimension in linear attention.
    pub key_head_dim: usize,
    /// Number of value heads in linear attention.
    pub num_value_heads: usize,
    /// Per-head value dimension in linear attention.
    pub value_head_dim: usize,
}

/// Generalized LLM configuration shared by all decoder-only text models.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<EosTokenId>,
    pub rope_scaling: Option<RopeScaling>,
    pub tie_word_embeddings: bool,
    pub max_seq_len: usize,
    /// Whether Q/K/V projections use bias (true for Qwen2, false for LLaMA).
    pub use_qkv_bias: bool,
    /// Weight tensor prefix for the transformer stack (e.g. "model" or "model.language_model").
    pub model_prefix: String,
    /// Explicit head dimension when it differs from hidden_size / num_attention_heads.
    pub head_dim: Option<usize>,
    /// Fraction of head dims to apply rotary embeddings to (1.0 = all, 0.25 = first quarter).
    pub partial_rotary_factor: f32,
    /// Linear attention configuration (None for pure softmax-attention models).
    pub linear_attn: Option<LinearAttnConfig>,
    /// Whether RMS norm uses residual weight: `(1 + weight) * norm(x)` instead of `weight * norm(x)`.
    /// True for Qwen3.5 whose norm weights are initialized to zero with +1 applied at runtime.
    pub residual_rms_norm: bool,
    /// Whether Q and K projections are normalized via RmsNorm after reshape (Qwen3, Gemma3).
    pub use_qk_norm: bool,
    /// Whether QK-norm is applied BEFORE the head reshape on the full projection output (OLMo2).
    /// When true, norm weights have size `num_heads * head_dim` instead of `head_dim`.
    pub pre_reshape_qk_norm: bool,
    /// Model-level sliding window size for local attention (None = full context).
    /// Per-layer overrides are handled in model-specific blocks (e.g. Gemma3).
    pub sliding_window: Option<usize>,
    /// Whether QKV weights are stored as a single pre-fused 'qkv_proj' tensor (Phi-3/4 style).
    /// If false (default), weights are separate 'q_proj'/'k_proj'/'v_proj'.
    pub fused_qkv_proj: bool,
    /// Whether MLP gate+up weights are stored as a single pre-fused 'gate_up_proj' tensor (Phi-3/4 style).
    /// If false (default), weights are stored as 'gate_proj' + 'up_proj'.
    pub fused_gate_up_proj: bool,
    /// Per-layer attention type: true = global (full RoPE, full context),
    /// false = local (sliding window, no RoPE for Gemma3).
    /// Empty for models where all layers share the same config.
    pub global_layers: Vec<bool>,
    /// Use GELU-tanh (approximate GELU) activation in MLP instead of SiLU (Gemma3).
    pub use_gelu_mlp: bool,
    /// Scale embedding outputs by this factor before the transformer blocks.
    /// Gemma models scale by sqrt(hidden_size); None means no scaling.
    pub embed_scale: Option<f32>,
    /// MoE: FFN intermediate dimension per expert (None = dense model).
    pub moe_intermediate_size: Option<usize>,
    /// MoE: total number of experts in the pool (0 = dense model).
    pub num_experts: usize,
    /// MoE: number of experts activated per token (top-K).
    pub num_experts_per_tok: usize,
    /// MoE: re-normalise top-K routing weights to sum to 1.0 after selection.
    pub norm_topk_prob: bool,
    /// MoE: intermediate size of the always-active shared expert (None = no shared expert).
    pub shared_expert_intermediate_size: Option<usize>,
    /// Whether the attention output gate is enabled (Qwen3.5 MoE full-attention layers):
    /// q_proj outputs 2× heads; the second half gates the attention output via sigmoid.
    pub attn_output_gate: bool,
}

/// Load an RMS norm weight tensor, optionally applying the residual pattern `(1 + weight)`.
/// When `residual` is true (Qwen3.5), the stored weight is treated as a residual and 1.0 is added.
/// Returns the raw weight tensor — callers use `backend.rms_norm(x, &weight, eps)` at forward time.
pub fn load_rms_norm_weight(
    size: usize,
    residual: bool,
    vb: candle_nn::VarBuilder,
) -> candle_core::Result<candle_core::Tensor> {
    let weight = vb.get(size, "weight")?;
    if residual {
        // Residual RMS norm: forward = (1 + weight) * rms_norm(x).
        // Weights are stored as deltas from 0, so add 1.0 at load time.
        Ok((weight + 1.0)?)
    } else {
        Ok(weight)
    }
}

/// Auto-detect text model architecture from config.json's "architectures" field.
pub fn detect_text_model_arch(config_path: &std::path::Path) -> Result<String> {
    let data = std::fs::read(config_path)
        .map_err(|e| anyhow!("can't read {}: {:?}", config_path.display(), e))?;
    let json: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|e| anyhow!("can't parse {}: {:?}", config_path.display(), e))?;

    if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
        for arch in archs {
            if let Some(s) = arch.as_str() {
                return Ok(s.to_string());
            }
        }
    }

    Ok(String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eos_token_id_single() {
        let json = r#"128001"#;
        let eos: EosTokenId = serde_json::from_str(json).unwrap();
        assert!(eos.is_eos(128001));
        assert!(!eos.is_eos(128008));
    }

    #[test]
    fn test_eos_token_id_array() {
        let json = r#"[128001, 128008, 128009]"#;
        let eos: EosTokenId = serde_json::from_str(json).unwrap();
        assert!(eos.is_eos(128001));
        assert!(eos.is_eos(128008));
        assert!(eos.is_eos(128009));
        assert!(!eos.is_eos(0));
    }

    #[test]
    fn test_rope_scaling_deserialization() {
        let json = r#"{
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }"#;
        let scaling: RopeScaling = serde_json::from_str(json).unwrap();
        assert_eq!(scaling.factor, 8.0);
        assert_eq!(scaling.high_freq_factor, 4.0);
        assert_eq!(scaling.low_freq_factor, 1.0);
        assert_eq!(scaling.original_max_position_embeddings, 8192);
        assert_eq!(scaling.rope_type.as_deref(), Some("llama3"));
    }
}
