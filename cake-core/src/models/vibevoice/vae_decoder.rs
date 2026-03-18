//! σ-VAE acoustic decoder for VibeVoice.
//!
//! Converts 64-dim acoustic latents to 24kHz audio waveform.
//! Architecture: 7-stage Conv1d decoder with depthwise conv mixers,
//! RMSNorm, FFN blocks, and ConvTranspose1d upsampling.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// Streaming cache for Conv1d context between frames.
///
/// Each conv layer gets a sequential slot index (assigned by `take_slot()`).
/// The counter is reset at the start of each decode/encode call via `reset_counter()`.
pub struct StreamingConvCache {
    states: Vec<Option<Tensor>>,
    counter: usize,
}

impl StreamingConvCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            states: vec![None; capacity],
            counter: 0,
        }
    }

    pub fn reset_counter(&mut self) {
        self.counter = 0;
    }

    /// Take the next cache slot, returning (index, is_first_use).
    pub fn take_slot(&mut self) -> (usize, bool) {
        let idx = self.counter;
        self.counter += 1;
        if idx >= self.states.len() {
            self.states.resize(idx + 1, None);
        }
        let is_first = self.states[idx].is_none();
        (idx, is_first)
    }

    pub fn get(&self, idx: usize) -> Option<&Tensor> {
        self.states.get(idx).and_then(|s| s.as_ref())
    }

    pub fn set(&mut self, idx: usize, state: Tensor) {
        if idx >= self.states.len() {
            self.states.resize(idx + 1, None);
        }
        self.states[idx] = Some(state);
    }

    pub fn clear(&mut self) {
        for s in &mut self.states {
            *s = None;
        }
    }
}

/// A single decoder block with depthwise conv mixer + FFN.
#[derive(Debug, Clone)]
struct DecoderBlock {
    norm_weight: Tensor,
    ffn_norm_weight: Tensor,
    eps: f32,
    gamma: Tensor,
    /// Depthwise conv weight: (channels, 1, 7) — stored as (channels, 7) for manual impl
    mixer_weight: Tensor,
    mixer_bias: Tensor,
    ffn_gamma: Tensor,
    ffn_linear1: candle_nn::Linear,
    ffn_linear2: candle_nn::Linear,
}

/// Manual depthwise conv1d using broadcast_mul + sum.
/// This avoids candle's pathologically slow grouped Conv1d CUDA kernel.
/// Input: (batch, channels, time+pad), weight: (channels, kernel), bias: (channels,)
/// Output: (batch, channels, time)
pub fn depthwise_conv1d_manual(x: &Tensor, weight: &Tensor, bias: &Tensor, kernel_size: usize) -> Result<Tensor> {
    let (_b, _c, t_padded) = x.dims3()?;
    let out_len = t_padded - kernel_size + 1;
    // Collect windowed slices and sum with weights
    let mut acc: Option<Tensor> = None;
    for k in 0..kernel_size {
        let slice = x.narrow(2, k, out_len)?; // (b, c, out_len)
        let w_k = weight.narrow(1, k, 1)?; // (c, 1)
        let w_k = w_k.unsqueeze(0)?; // (1, c, 1)
        let term = slice.broadcast_mul(&w_k)?;
        acc = Some(match acc {
            None => term,
            Some(a) => (a + term)?,
        });
    }
    let out = acc.unwrap();
    // Add bias: (channels,) → (1, channels, 1)
    let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
    out.broadcast_add(&bias)
}

impl DecoderBlock {
    fn load(vb: VarBuilder, channels: usize, eps: f64) -> Result<Self> {
        let norm_weight = vb.pp("norm").get(channels, "weight")?;
        let gamma = vb.get(channels, "gamma")?;

        let conv_vb = vb.pp("mixer").pp("conv").pp("conv").pp("conv");
        let mixer_weight = conv_vb.get((channels, 1, 7), "weight")?.squeeze(1)?;
        let mixer_bias = conv_vb.get(channels, "bias")?;

        let ffn_norm_weight = vb.pp("ffn_norm").get(channels, "weight")?;
        let ffn_gamma = vb.get(channels, "ffn_gamma")?;
        let ffn_linear1 = candle_nn::linear(channels, channels * 4, vb.pp("ffn").pp("linear1"))?;
        let ffn_linear2 = candle_nn::linear(channels * 4, channels, vb.pp("ffn").pp("linear2"))?;

        Ok(Self {
            norm_weight,
            ffn_norm_weight,
            eps: eps as f32,
            gamma,
            mixer_weight,
            mixer_bias,
            ffn_gamma,
            ffn_linear1,
            ffn_linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use crate::utils::fused_ops;
        let channels = x.dim(1)?;

        let h = fused_ops::rms_norm_channel(x, &self.norm_weight, self.eps)?;
        let zeros = Tensor::zeros((h.dim(0)?, channels, 6), h.dtype(), h.device())?;
        let h = fused_ops::depthwise_conv1d_bias_ctx(
            &zeros, &h, &self.mixer_weight, &self.mixer_bias, 7, channels,
        )?;
        let x = fused_ops::add_scaled(x, &h, &self.gamma)?;

        let h = fused_ops::rms_norm_channel(&x, &self.ffn_norm_weight, self.eps)?;
        let h = h.transpose(1, 2)?;
        let h = self.ffn_linear1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ffn_linear2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        fused_ops::add_scaled(&x, &h, &self.ffn_gamma)
    }

    /// Forward with streaming cache: uses cached context instead of zero-padding.
    fn forward_cached(&self, x: &Tensor, cache: &mut StreamingConvCache) -> Result<Tensor> {
        use crate::utils::fused_ops;
        let channels = x.dim(1)?;

        let h = fused_ops::rms_norm_channel(x, &self.norm_weight, self.eps)?;
        let (slot, is_first) = cache.take_slot();
        let context = if is_first {
            Tensor::zeros((h.dim(0)?, channels, 6), h.dtype(), h.device())?
        } else {
            cache.get(slot).unwrap().clone()
        };

        // Update cache: last 6 samples of [context, h]
        let h_len = h.dim(2)?;
        if h_len >= 6 {
            cache.set(slot, h.narrow(2, h_len - 6, 6)?);
        } else {
            // h shorter than 6: take from context + h
            let ctx_take = 6 - h_len;
            let ctx_part = context.narrow(2, 6 - ctx_take, ctx_take)?;
            cache.set(slot, Tensor::cat(&[&ctx_part, &h], 2)?);
        }

        // Fused conv reads from [context, h] virtually — no cat allocation
        let h = fused_ops::depthwise_conv1d_bias_ctx(
            &context, &h, &self.mixer_weight, &self.mixer_bias, 7, channels,
        )?;
        let x = fused_ops::add_scaled(x, &h, &self.gamma)?;

        let h = fused_ops::rms_norm_channel(&x, &self.ffn_norm_weight, self.eps)?;
        let h = h.transpose(1, 2)?;
        let h = self.ffn_linear1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ffn_linear2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        fused_ops::add_scaled(&x, &h, &self.ffn_gamma)
    }
}

/// One decoder stage: upsample + N blocks.
#[derive(Debug, Clone)]
struct DecoderStage {
    blocks: Vec<DecoderBlock>,
}

impl DecoderStage {
    fn load(vb: VarBuilder, channels: usize, num_blocks: usize, eps: f64) -> Result<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(DecoderBlock::load(vb.pp(i), channels, eps)?);
        }
        Ok(Self { blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    fn forward_cached(&self, x: &Tensor, cache: &mut StreamingConvCache) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward_cached(&x, cache)?;
        }
        Ok(x)
    }
}

/// Upsample layer: either Conv1d (first) or ConvTranspose1d (rest).
#[derive(Debug, Clone)]
enum UpsampleLayer {
    Conv(Conv1d),
    ConvTranspose(ConvTranspose1d),
}

impl UpsampleLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(c) => c.forward(x),
            Self::ConvTranspose(ct) => ct.forward(x),
        }
    }
}

/// Complete acoustic VAE decoder.
#[derive(Debug, Clone)]
pub struct AcousticVaeDecoder {
    upsample_layers: Vec<UpsampleLayer>,
    stages: Vec<DecoderStage>,
    head_conv: Conv1d,
    /// Upsample ratios per stage (0 for first Conv1d stage).
    ratios: Vec<usize>,
}

impl AcousticVaeDecoder {
    pub fn load(vb: VarBuilder, cfg: &super::config::AcousticTokenizerConfig) -> Result<Self> {
        let eps = cfg.layernorm_eps;
        let n_filters = cfg.decoder_n_filters.unwrap_or(cfg.encoder_n_filters);
        let ratios = cfg.decoder_ratios.as_ref().unwrap_or(&cfg.encoder_ratios);

        let num_stages = ratios.len() + 1; // 7 stages for 6 ratios

        // Channel progression: n_filters * 2^(num_stages-1) down to n_filters
        // From weights: 2048, 1024, 512, 256, 128, 64, 32 for n_filters=32, 7 stages
        let mut channels = Vec::with_capacity(num_stages);
        let first_ch = n_filters * (1 << (num_stages - 1)); // 32 * 64 = 2048
        channels.push(first_ch);
        for _ in 0..ratios.len() {
            channels.push(channels.last().unwrap() / 2);
        }

        // Upsample layers
        let mut upsample_layers = Vec::with_capacity(num_stages);

        // First upsample: Conv1d (vae_dim → first_channels, kernel=7, NO padding — causal model)
        let first_up = candle_nn::conv1d(
            cfg.vae_dim,
            channels[0],
            7,
            Conv1dConfig::default(), // padding=0, stride=1
            vb.pp("upsample_layers").pp("0").pp("0").pp("conv").pp("conv"),
        )?;
        upsample_layers.push(UpsampleLayer::Conv(first_up));

        // Remaining upsamples: ConvTranspose1d
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_ch = channels[i];
            let out_ch = channels[i + 1];
            let kernel = ratio * 2;
            // ConvTranspose1d with NO padding (causal model handles padding externally)
            let ct = candle_nn::conv_transpose1d(
                in_ch,
                out_ch,
                kernel,
                ConvTranspose1dConfig {
                    stride: ratio,
                    ..Default::default() // padding=0, output_padding=0
                },
                vb.pp("upsample_layers").pp(i + 1).pp("0").pp("convtr").pp("convtr"),
            )?;
            upsample_layers.push(UpsampleLayer::ConvTranspose(ct));
        }

        // Stages (each follows its corresponding upsample)
        let depths = Self::parse_depths(cfg, num_stages);
        let mut stages = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let stage = DecoderStage::load(
                vb.pp("stages").pp(i),
                channels[i],
                depths[i],
                eps,
            )?;
            stages.push(stage);
        }

        // Head conv: final channels → 1 (mono audio), NO padding (causal)
        let head_conv = candle_nn::conv1d(
            channels[num_stages - 1],
            1,
            7,
            Conv1dConfig::default(), // padding=0
            vb.pp("head").pp("conv").pp("conv"),
        )?;

        // Build ratios vec: 0 for first Conv1d, then the actual ratios
        let mut ratios_vec = vec![0usize];
        ratios_vec.extend_from_slice(ratios);

        Ok(Self { upsample_layers, stages, head_conv, ratios: ratios_vec })
    }

    fn parse_depths(cfg: &super::config::AcousticTokenizerConfig, num_stages: usize) -> Vec<usize> {
        // decoder_depths may be specified as "3-3-3-3-3-3-8" string
        if let Some(ref dd) = cfg.decoder_depths {
            dd.split('-')
                .map(|s| s.parse().unwrap_or(3))
                .collect()
        } else if let Some(ref ed) = cfg.encoder_depths {
            // Mirror encoder depths
            let mut d: Vec<usize> = ed.split('-').map(|s| s.parse().unwrap_or(3)).collect();
            d.reverse();
            d.resize(num_stages, 3);
            d
        } else {
            vec![3; num_stages]
        }
    }

    /// Apply causal left-padding to a tensor: zero-pad `amount` on the left of dim 2.
    fn causal_pad(x: &Tensor, amount: usize) -> Result<Tensor> {
        if amount == 0 {
            return Ok(x.clone());
        }
        let pad = Tensor::zeros((x.dim(0)?, x.dim(1)?, amount), x.dtype(), x.device())?;
        Tensor::cat(&[&pad, x], 2)
    }

    /// Trim causal ConvTranspose1d output to remove extra samples.
    /// ConvTranspose1d with kernel=2*stride, stride=S, padding=0 produces:
    /// out_len = (in_len - 1) * stride + kernel = in_len * stride + stride
    /// We want: in_len * stride, so trim `stride` from the right.
    fn causal_trim(x: &Tensor, trim: usize) -> Result<Tensor> {
        if trim == 0 {
            return Ok(x.clone());
        }
        let len = x.dim(2)?;
        if len > trim {
            x.narrow(2, 0, len - trim)
        } else {
            Ok(x.clone())
        }
    }

    /// Decode acoustic latents to audio waveform.
    /// Input: (batch, vae_dim, frames) or (batch, frames, vae_dim)
    /// Output: (batch, 1, samples)
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let x = if latents.dim(1)? == 64 {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };

        let mut h = x;

        for (i, (upsample, stage)) in self.upsample_layers.iter().zip(self.stages.iter()).enumerate() {
            if i == 0 {
                // First layer: Conv1d kernel=7, causal left-pad 6
                h = Self::causal_pad(&h, 6)?;
            }
            h = upsample.forward(&h)?;
            if i > 0 {
                // ConvTranspose1d: trim extra samples from right
                h = Self::causal_trim(&h, self.ratios[i])?;
            }
            h = stage.forward(&h)?;
        }

        // Head conv: kernel=7, causal left-pad 6
        h = Self::causal_pad(&h, 6)?;
        self.head_conv.forward(&h)
    }

    /// Streaming decode: uses cache for correct context between frames.
    /// Each call processes a single latent frame and produces audio samples.
    pub fn decode_streaming(&self, latents: &Tensor, cache: &mut StreamingConvCache) -> Result<Tensor> {
        let x = if latents.dim(1)? == 64 {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };

        cache.reset_counter();
        let mut h = x;
        let profile = log::log_enabled!(log::Level::Trace);
        let mut stage_times: Vec<(f64, f64, usize)> = Vec::new(); // (upsample_ms, blocks_ms, seq_len)

        for (i, (upsample, stage)) in self.upsample_layers.iter().zip(self.stages.iter()).enumerate() {
            let t_stage = std::time::Instant::now();
            if i == 0 {
                // Conv1d k=7: use cached context (6 samples) instead of zero-pad
                let (slot, is_first) = cache.take_slot();
                let ctx = if is_first {
                    Tensor::zeros((h.dim(0)?, h.dim(1)?, 6), h.dtype(), h.device())?
                } else {
                    cache.get(slot).unwrap().clone()
                };
                let padded = Tensor::cat(&[&ctx, &h], 2)?;
                let plen = padded.dim(2)?;
                cache.set(slot, padded.narrow(2, plen.saturating_sub(6), 6.min(plen))?);
                h = upsample.forward(&padded)?;
            } else {
                // ConvTranspose1d: cache input history for context
                let kernel_size = self.ratios[i] * 2;
                let ctx_size = kernel_size - 1;
                let stride = self.ratios[i];
                let new_len = h.dim(2)?;

                let (slot, is_first) = cache.take_slot();

                let full_input = if is_first {
                    h.clone()
                } else {
                    let cached = cache.get(slot).unwrap();
                    Tensor::cat(&[cached, &h], 2)?
                };

                let full_output = upsample.forward(&full_input)?;
                let full_output = Self::causal_trim(&full_output, stride)?;

                h = if is_first {
                    full_output
                } else {
                    let out_len = full_output.dim(2)?;
                    let new_out = new_len * stride;
                    full_output.narrow(2, out_len - new_out, new_out)?
                };

                // Update cache: last ctx_size samples of full_input
                let fi_len = full_input.dim(2)?;
                if fi_len > ctx_size {
                    cache.set(slot, full_input.narrow(2, fi_len - ctx_size, ctx_size)?);
                } else {
                    cache.set(slot, full_input);
                }
            }
            let t_up = t_stage.elapsed();
            let seq_len = h.dim(2)?;
            let t_blocks_start = std::time::Instant::now();
            h = stage.forward_cached(&h, cache)?;
            if profile {
                stage_times.push((
                    t_up.as_secs_f64() * 1000.0,
                    t_blocks_start.elapsed().as_secs_f64() * 1000.0,
                    seq_len,
                ));
            }
        }

        if profile {
            for (i, (up_ms, blk_ms, slen)) in stage_times.iter().enumerate() {
                log::trace!("    dec stage {i}: up={up_ms:.2}ms blocks={blk_ms:.2}ms seq={slen}");
            }
        }

        // Head conv: use cached context
        let (slot, is_first) = cache.take_slot();
        let ctx = if is_first {
            Tensor::zeros((h.dim(0)?, h.dim(1)?, 6), h.dtype(), h.device())?
        } else {
            cache.get(slot).unwrap().clone()
        };
        let padded = Tensor::cat(&[&ctx, &h], 2)?;
        let plen = padded.dim(2)?;
        cache.set(slot, padded.narrow(2, plen.saturating_sub(6), 6.min(plen))?);
        self.head_conv.forward(&padded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp};

    #[test]
    fn test_decoder_block_forward() {
        let ch = 32;
        let mut map = std::collections::HashMap::new();
        let dev = Device::Cpu;

        fn mt(shape: &[usize], seed: u64) -> Tensor {
            use rand::{Rng, SeedableRng};
            let n: usize = shape.iter().product();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let d: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.01..0.01)).collect();
            Tensor::from_vec(d, shape, &Device::Cpu).unwrap()
        }

        map.insert("norm.weight".into(), Tensor::ones(ch, DType::F32, &dev).unwrap());
        map.insert("gamma".into(), Tensor::ones(ch, DType::F32, &dev).unwrap());
        map.insert("mixer.conv.conv.conv.weight".into(), mt(&[ch, 1, 7], 1));
        map.insert("mixer.conv.conv.conv.bias".into(), mt(&[ch], 2));
        map.insert("ffn_norm.weight".into(), Tensor::ones(ch, DType::F32, &dev).unwrap());
        map.insert("ffn_gamma".into(), Tensor::ones(ch, DType::F32, &dev).unwrap());
        map.insert("ffn.linear1.weight".into(), mt(&[ch * 4, ch], 3));
        map.insert("ffn.linear1.bias".into(), mt(&[ch * 4], 4));
        map.insert("ffn.linear2.weight".into(), mt(&[ch, ch * 4], 5));
        map.insert("ffn.linear2.bias".into(), mt(&[ch], 6));

        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        let block = DecoderBlock::load(vb, ch, 1e-5).unwrap();

        // Input: (batch=1, channels=32, seq=16)
        let x = mt(&[1, ch, 16], 10);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, ch, 16]);
    }

    // --- StreamingConvCache ---

    #[test]
    fn test_streaming_cache_new() {
        let cache = StreamingConvCache::new(10);
        assert_eq!(cache.states.len(), 10);
        assert!(cache.states.iter().all(|s| s.is_none()));
    }

    #[test]
    fn test_streaming_cache_take_slot_sequential() {
        let mut cache = StreamingConvCache::new(5);
        for i in 0..5 {
            let (idx, is_first) = cache.take_slot();
            assert_eq!(idx, i);
            assert!(is_first, "slot {i} should be first-use");
        }
    }

    #[test]
    fn test_streaming_cache_take_slot_after_set() {
        let mut cache = StreamingConvCache::new(3);
        let t = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();
        cache.set(0, t);

        cache.reset_counter();
        let (idx, is_first) = cache.take_slot();
        assert_eq!(idx, 0);
        assert!(!is_first, "slot 0 was set, should not be first-use");
    }

    #[test]
    fn test_streaming_cache_set_and_get() {
        let mut cache = StreamingConvCache::new(3);
        let t = Tensor::ones((1, 4, 6), DType::F32, &Device::Cpu).unwrap();
        cache.set(1, t.clone());

        assert!(cache.get(0).is_none());
        let got = cache.get(1).unwrap();
        assert_eq!(got.dims(), &[1, 4, 6]);
    }

    #[test]
    fn test_streaming_cache_auto_resize() {
        let mut cache = StreamingConvCache::new(2);
        let t = Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
        cache.set(5, t); // beyond initial capacity
        assert!(cache.states.len() >= 6);
        assert!(cache.get(5).is_some());
    }

    #[test]
    fn test_streaming_cache_clear() {
        let mut cache = StreamingConvCache::new(3);
        let t = Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
        cache.set(0, t.clone());
        cache.set(1, t.clone());
        cache.set(2, t);
        cache.clear();
        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_none());
    }

    #[test]
    fn test_streaming_cache_reset_counter() {
        let mut cache = StreamingConvCache::new(3);
        cache.take_slot();
        cache.take_slot();
        assert_eq!(cache.counter, 2);
        cache.reset_counter();
        assert_eq!(cache.counter, 0);
    }

    // --- causal_pad / causal_trim ---

    #[test]
    fn test_causal_pad_zero() {
        let x = Tensor::ones(&[1, 4, 10], DType::F32, &Device::Cpu).unwrap();
        let padded = AcousticVaeDecoder::causal_pad(&x, 0).unwrap();
        assert_eq!(padded.dims(), &[1, 4, 10]);
    }

    #[test]
    fn test_causal_pad_nonzero() {
        let x = Tensor::ones(&[1, 4, 10], DType::F32, &Device::Cpu).unwrap();
        let padded = AcousticVaeDecoder::causal_pad(&x, 6).unwrap();
        assert_eq!(padded.dims(), &[1, 4, 16]);
        // First 6 values should be zero
        let vals: Vec<f32> = padded.i((0, 0, ..6)).unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| *v == 0.0));
        // Last 10 values should be ones
        let vals: Vec<f32> = padded.i((0, 0, 6..)).unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn test_causal_trim_zero() {
        let x = Tensor::ones(&[1, 4, 10], DType::F32, &Device::Cpu).unwrap();
        let trimmed = AcousticVaeDecoder::causal_trim(&x, 0).unwrap();
        assert_eq!(trimmed.dims(), &[1, 4, 10]);
    }

    #[test]
    fn test_causal_trim_nonzero() {
        let x = Tensor::ones(&[1, 4, 10], DType::F32, &Device::Cpu).unwrap();
        let trimmed = AcousticVaeDecoder::causal_trim(&x, 3).unwrap();
        assert_eq!(trimmed.dims(), &[1, 4, 7]);
    }

    #[test]
    fn test_causal_trim_all() {
        let x = Tensor::ones(&[1, 4, 5], DType::F32, &Device::Cpu).unwrap();
        // Trim more than length — should return clone
        let trimmed = AcousticVaeDecoder::causal_trim(&x, 10).unwrap();
        assert_eq!(trimmed.dims(), &[1, 4, 5]);
    }
}
