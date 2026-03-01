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
