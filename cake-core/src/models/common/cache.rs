use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;

use super::Config;

/// Abstraction over cosine and sine tables, kv-caching and attention masking.
#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,

    masks: HashMap<usize, Tensor>,
    use_kv_cache: bool,
    kvs: Vec<KvCache>,

    /// Recurrent state matrices for linear attention layers (Gated DeltaNet).
    /// Shape per entry: (batch=1, num_heads, key_dim, value_dim).
    recurrent_states: Vec<Option<Tensor>>,
    /// Conv1d history for linear attention layers.
    /// Shape per entry: (batch=1, channels, kernel_size-1).
    conv_states: Vec<Option<Tensor>>,

    device: Device,
}

impl Cache {
    /// Creates a new cache instance with the provided configuration.
    /// Set `use_kv_cache` to false to disable kv-caching.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // Compute rotary dimension, respecting partial_rotary_factor
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);
        let rotary_dim = (head_dim as f32 * config.partial_rotary_factor) as usize;
        let max_seq_len = config.max_seq_len;

        log::debug!("cache::head_dim = {head_dim}");
        log::debug!("cache::rotary_dim = {rotary_dim}");
        log::debug!("cache::max_seq_len = {max_seq_len}");

        let mut theta: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / rotary_dim as f32))
            .collect();

        // Apply LLaMA3 RoPE frequency scaling if configured
        if let Some(ref rope_scaling) = config.rope_scaling {
            let is_llama3 = rope_scaling
                .rope_type
                .as_deref()
                .map_or(false, |t| t == "llama3");

            if is_llama3 && rope_scaling.original_max_position_embeddings > 0 {
                let factor = rope_scaling.factor;
                let low_freq_factor = rope_scaling.low_freq_factor;
                let high_freq_factor = rope_scaling.high_freq_factor;
                let old_context_len = rope_scaling.original_max_position_embeddings as f32;

                let low_freq_wavelen = old_context_len / low_freq_factor;
                let high_freq_wavelen = old_context_len / high_freq_factor;

                for freq in theta.iter_mut() {
                    let wavelen = 2.0 * std::f32::consts::PI / *freq;
                    if wavelen < high_freq_wavelen {
                        // High frequency: keep as-is
                    } else if wavelen > low_freq_wavelen {
                        // Low frequency: scale down by factor
                        *freq /= factor;
                    } else {
                        // Medium frequency: smooth interpolation
                        let smooth = (old_context_len / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        *freq = (1.0 - smooth) * (*freq / factor) + smooth * *freq;
                    }
                }

                log::debug!("cache: applied llama3 rope scaling (factor={factor}, low_freq={low_freq_factor}, high_freq={high_freq_factor})");
            }
        }

        let theta = Tensor::new(theta.as_slice(), device)?;

        log::debug!("cache::theta = {}", &theta);

        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        log::debug!("cache::idx_theta = {}", &idx_theta);

        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        log::debug!("cache::cos = {}", &cos);
        log::debug!("cache::sin = {}", &sin);

        let num_layers = config.num_hidden_layers;

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: (0..num_layers).map(|_| KvCache::new(2, max_seq_len)).collect(),
            recurrent_states: vec![None; num_layers],
            conv_states: vec![None; num_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Return true if kv-caching is enabled.
    pub fn with_kv_cache(&self) -> bool {
        self.use_kv_cache
    }

    /// Return the cached cosine value for the given position and sequence length.
    /// When `device` differs from the cache's own device, the result is copied
    /// to that device (enables multi-GPU workers).
    pub fn cosine(&self, index_pos: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
        self.cos.narrow(0, index_pos, seq_len)?.to_device(device)
    }

    /// Return the cached sine value for the given position and sequence length.
    pub fn sine(&self, index_pos: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
        self.sin.narrow(0, index_pos, seq_len)?.to_device(device)
    }

    /// Get the attention mask for the given sequence length.
    pub fn mask(&mut self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Always create/cache on self.device, then copy to target if needed
        if !self.masks.contains_key(&seq_len) {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask);
        }
        self.masks.get(&seq_len).unwrap().clone().to_device(device)
    }

    /// Process the input k and v using pre-allocated KV cache.
    ///
    /// Uses candle-nn's KvCache with `slice_set` for O(1) per-token append
    /// instead of O(N) concatenation, making total generation O(N) instead of O(N²).
    pub fn process_kv(
        &mut self,
        block_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.use_kv_cache {
            self.kvs[block_idx].append(&k, &v)
        } else {
            Ok((k, v))
        }
    }

    /// Like `process_kv` but caps the KV cache to `window` tokens (sliding window attention).
    pub fn process_kv_windowed(
        &mut self,
        block_idx: usize,
        k: Tensor,
        v: Tensor,
        window: usize,
    ) -> Result<(Tensor, Tensor)> {
        if self.use_kv_cache {
            let (k_out, v_out) = self.kvs[block_idx].append(&k, &v)?;
            let k_seq_len = k_out.dims()[2];
            if k_seq_len > window {
                let k_trimmed = k_out.narrow(2, k_seq_len - window, window)?.contiguous()?;
                let v_trimmed = v_out.narrow(2, k_seq_len - window, window)?.contiguous()?;
                Ok((k_trimmed, v_trimmed))
            } else {
                Ok((k_out, v_out))
            }
        } else {
            Ok((k, v))
        }
    }

    /// Get the recurrent state for a linear attention layer.
    pub fn get_recurrent_state(&self, block_idx: usize) -> Option<&Tensor> {
        self.recurrent_states[block_idx].as_ref()
    }

    /// Set the recurrent state for a linear attention layer.
    pub fn set_recurrent_state(&mut self, block_idx: usize, state: Tensor) {
        self.recurrent_states[block_idx] = Some(state);
    }

    /// Get the conv state for a linear attention layer.
    pub fn get_conv_state(&self, block_idx: usize) -> Option<&Tensor> {
        self.conv_states[block_idx].as_ref()
    }

    /// Set the conv state for a linear attention layer.
    pub fn set_conv_state(&mut self, block_idx: usize, state: Tensor) {
        self.conv_states[block_idx] = Some(state);
    }

    /// Return a copy of this cache with the same state but new kv table.
    pub fn as_new(&self) -> Self {
        let mut copy = self.clone();
        copy.clear();
        copy
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.masks.clear();
        for kv in &mut self.kvs {
            kv.reset();
        }
        self.recurrent_states = vec![None; self.recurrent_states.len()];
        self.conv_states = vec![None; self.conv_states.len()];
    }
}
