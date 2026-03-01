use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::Config;

/// Abstraction over cosine and sine tables, kv-caching and attention masking.
#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,

    masks: HashMap<usize, Tensor>,
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    max_seq_len: usize,

    device: Device,
}

impl Cache {
    /// Creates a new cache instance with the provided configuration.
    /// Set `use_kv_cache` to false to disable kv-caching.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_seq_len;

        log::debug!("cache::n_elem = {n_elem}");
        log::debug!("cache::max_seq_len = {max_seq_len}");

        let mut theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
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

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            max_seq_len,
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
    pub fn cosine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.cos.narrow(0, index_pos, seq_len)
    }

    /// Return the cached sine value for the given position and sequence length.
    pub fn sine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.sin.narrow(0, index_pos, seq_len)
    }

    /// Get the attention mask for the given sequence length.
    pub fn mask(&mut self, seq_len: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&seq_len) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask.clone());
            Ok(mask)
        }
    }

    /// Process the input k and v by either generating their cache entry or applying a previously cached one.
    pub fn process_kv(
        &mut self,
        block_idx: usize,
        mut k: Tensor,
        mut v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.use_kv_cache {
            // if this block_idx in cache
            if let Some((cache_k, cache_v)) = &self.kvs[block_idx] {
                // update cache entry: concatenate on dim 2 (seq_len)
                // tensor shape is (batch, num_heads, seq_len, head_dim)
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;

                // truncate on dim 2 (seq_len) if over limit
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_seq_len {
                    k = k
                        .narrow(2, k_seq_len - self.max_seq_len, self.max_seq_len)?
                        .contiguous()?;
                }
                let v_seq_len = v.dims()[2];
                if v_seq_len > self.max_seq_len {
                    v = v
                        .narrow(2, v_seq_len - self.max_seq_len, self.max_seq_len)?
                        .contiguous()?;
                }
            }
            // set entry for this block
            self.kvs[block_idx] = Some((k.clone(), v.clone()))
        }
        Ok((k, v))
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
        self.kvs = vec![None; self.kvs.len()];
    }
}
