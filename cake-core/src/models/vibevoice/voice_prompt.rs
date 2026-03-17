//! Voice prompt loading for VibeVoice.
//!
//! Loads pre-computed KV caches from a voice reference sample.
//! These caches provide speaker identity for voice cloning.
//!
//! Format: safetensors with keys:
//! - `{lm,tts_lm,neg_lm,neg_tts_lm}.last_hidden_state` → (1, seq, 896)
//! - `{lm,tts_lm,neg_lm,neg_tts_lm}.kv.{layer}.{key,value}` → (1, kv_heads, seq, head_dim)

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

/// Pre-computed KV cache for one LM (base or TTS).
pub struct LmCache {
    /// Last hidden state from the voice prompt.
    pub last_hidden_state: Tensor,
    /// KV cache: Vec of (key, value) per layer.
    pub kv_cache: Vec<(Tensor, Tensor)>,
    /// Sequence length of the cached prompt.
    pub seq_len: usize,
}

/// Complete voice prompt with 4 caches (for CFG).
pub struct VoicePrompt {
    /// Base LM cache (4 layers, positive condition).
    pub lm: LmCache,
    /// TTS LM cache (20 layers, positive condition).
    pub tts_lm: LmCache,
    /// Base LM cache (4 layers, negative/unconditional).
    pub neg_lm: LmCache,
    /// TTS LM cache (20 layers, negative/unconditional).
    pub neg_tts_lm: LmCache,
}

impl VoicePrompt {
    /// Load a voice prompt in F32 (matching model dtype for numerical stability).
    pub fn load_f32(path: &Path, device: &Device) -> Result<Self> {
        Self::load(path, device, DType::F32)
    }

    /// Load a voice prompt from a safetensors file.
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.to_path_buf()], dtype, device)?
        };

        let load_cache = |prefix: &str, num_layers: usize| -> Result<LmCache> {
            let hidden = vb.get_unchecked(&format!("{prefix}.last_hidden_state"))?;
            let seq_len = hidden.dim(1)?;
            let mut kv_cache = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let key = vb.get_unchecked(&format!("{prefix}.kv.{i}.key"))?;
                let value = vb.get_unchecked(&format!("{prefix}.kv.{i}.value"))?;
                kv_cache.push((key, value));
            }
            Ok(LmCache {
                last_hidden_state: hidden,
                kv_cache,
                seq_len,
            })
        };

        Ok(Self {
            lm: load_cache("lm", 4)?,
            tts_lm: load_cache("tts_lm", 20)?,
            neg_lm: load_cache("neg_lm", 4)?,
            neg_tts_lm: load_cache("neg_tts_lm", 20)?,
        })
    }
}

/// Inject a pre-computed KV cache into a cake Cache.
/// This sets the KV pairs for each layer so that subsequent forward passes
/// continue from where the voice prompt left off.
pub fn inject_kv_cache(
    cache: &mut crate::models::common::Cache,
    lm_cache: &LmCache,
) {
    for (block_idx, (key, value)) in lm_cache.kv_cache.iter().enumerate() {
        cache.set_kv(block_idx, key.clone(), value.clone());
    }
}
