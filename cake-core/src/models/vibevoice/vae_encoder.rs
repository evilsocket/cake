//! σ-VAE encoder for VibeVoice tokenizers.
//!
//! Converts 24kHz audio waveform to latent representations.
//! Architecture: 7-stage Conv1d encoder with depthwise conv mixers,
//! RMSNorm, FFN blocks, and stride-based downsampling.
//!
//! Used by both acoustic tokenizer (vae_dim=64, Gaussian sampling)
//! and semantic tokenizer (vae_dim=128, no sampling).

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, RmsNorm, VarBuilder};

/// A single encoder block with depthwise conv mixer + FFN.
/// Identical architecture to DecoderBlock in vae_decoder.rs.
#[derive(Debug, Clone)]
struct EncoderBlock {
    norm: RmsNorm,
    gamma: Tensor,
    /// Depthwise conv weight: (channels, 7) for manual broadcast_mul implementation
    mixer_weight: Tensor,
    mixer_bias: Tensor,
    ffn_norm: RmsNorm,
    ffn_gamma: Tensor,
    ffn_linear1: candle_nn::Linear,
    ffn_linear2: candle_nn::Linear,
}

impl EncoderBlock {
    fn load(vb: VarBuilder, channels: usize, eps: f64) -> Result<Self> {
        let norm = candle_nn::rms_norm(channels, eps, vb.pp("norm"))?;
        let gamma = vb.get(channels, "gamma")?;

        // Load depthwise conv weights manually: (channels, 1, 7) → (channels, 7)
        let conv_vb = vb.pp("mixer").pp("conv").pp("conv").pp("conv");
        let mixer_weight = conv_vb.get((channels, 1, 7), "weight")?.squeeze(1)?;
        let mixer_bias = conv_vb.get(channels, "bias")?;

        let ffn_norm = candle_nn::rms_norm(channels, eps, vb.pp("ffn_norm"))?;
        let ffn_gamma = vb.get(channels, "ffn_gamma")?;
        let ffn_linear1 = candle_nn::linear(channels, channels * 4, vb.pp("ffn").pp("linear1"))?;
        let ffn_linear2 = candle_nn::linear(channels * 4, channels, vb.pp("ffn").pp("linear2"))?;

        Ok(Self {
            norm,
            gamma,
            mixer_weight,
            mixer_bias,
            ffn_norm,
            ffn_gamma,
            ffn_linear1,
            ffn_linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mixer: norm → causal left-pad → fused depthwise conv+bias → scale by gamma
        let residual = x;
        let h = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let channels = h.dim(1)?;
        // Causal left-padding: pad (kernel_size - 1) zeros on the left
        let h = Tensor::cat(
            &[
                &Tensor::zeros((h.dim(0)?, channels, 6), h.dtype(), h.device())?,
                &h,
            ],
            2,
        )?;
        // Fused depthwise conv1d + bias (1 kernel instead of 14)
        let h = crate::utils::fused_ops::depthwise_conv1d_bias(
            &h, &self.mixer_weight, &self.mixer_bias, 7, channels,
        )?;
        let gamma = self.gamma.unsqueeze(0)?.unsqueeze(2)?;
        let x = (residual + h.broadcast_mul(&gamma)?)?;

        // FFN: norm → linear1 → gelu → linear2 → scale by gamma
        let residual = &x;
        let h = self
            .ffn_norm
            .forward(&x.transpose(1, 2)?)?
            .transpose(1, 2)?;
        let h = h.transpose(1, 2)?;
        let h = self.ffn_linear1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ffn_linear2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let ffn_gamma = self.ffn_gamma.unsqueeze(0)?.unsqueeze(2)?;
        residual + h.broadcast_mul(&ffn_gamma)?
    }

    /// Forward with streaming cache: uses cached context instead of zero-padding.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut super::vae_decoder::StreamingConvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let channels = h.dim(1)?;

        let (slot, is_first) = cache.take_slot();
        let context = if is_first {
            Tensor::zeros((h.dim(0)?, channels, 6), h.dtype(), h.device())?
        } else {
            cache.get(slot).unwrap().clone()
        };
        let padded = Tensor::cat(&[&context, &h], 2)?;
        let plen = padded.dim(2)?;
        let start = plen.saturating_sub(6);
        cache.set(slot, padded.narrow(2, start, plen - start)?);

        let h = crate::utils::fused_ops::depthwise_conv1d_bias(
            &padded, &self.mixer_weight, &self.mixer_bias, 7, channels,
        )?;
        let gamma = self.gamma.unsqueeze(0)?.unsqueeze(2)?;
        let x = (residual + h.broadcast_mul(&gamma)?)?;

        let residual = &x;
        let h = self
            .ffn_norm
            .forward(&x.transpose(1, 2)?)?
            .transpose(1, 2)?;
        let h = h.transpose(1, 2)?;
        let h = self.ffn_linear1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ffn_linear2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let ffn_gamma = self.ffn_gamma.unsqueeze(0)?.unsqueeze(2)?;
        residual + h.broadcast_mul(&ffn_gamma)?
    }
}

/// One encoder stage: N blocks (applied after its corresponding downsample).
#[derive(Debug, Clone)]
struct EncoderStage {
    blocks: Vec<EncoderBlock>,
}

impl EncoderStage {
    fn load(vb: VarBuilder, channels: usize, num_blocks: usize, eps: f64) -> Result<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(EncoderBlock::load(vb.pp(i), channels, eps)?);
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

    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut super::vae_decoder::StreamingConvCache,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward_cached(&x, cache)?;
        }
        Ok(x)
    }
}

/// Complete tokenizer encoder.
/// Mirrors the decoder: downsample stages then head projection.
#[derive(Debug, Clone)]
pub struct TokenizerEncoder {
    /// Downsample layers: [stem Conv1d, stride Conv1d * N_ratios]
    downsample_convs: Vec<Conv1d>,
    /// Stages: each contains N EncoderBlocks at corresponding channel width
    stages: Vec<EncoderStage>,
    /// Causal padding amounts for each downsample layer
    downsample_paddings: Vec<usize>,
    /// Stride for each downsample layer (1 for stem, ratio for others)
    downsample_strides: Vec<usize>,
    /// Head conv: final channels → vae_dim
    head_conv: Conv1d,
    /// Channel count at each stage (for understanding the architecture)
    #[allow(dead_code)]
    channels: Vec<usize>,
}

impl TokenizerEncoder {
    pub fn load(vb: VarBuilder, cfg: &super::config::AcousticTokenizerConfig) -> Result<Self> {
        Self::load_with_vae_dim(vb, cfg, cfg.vae_dim)
    }

    /// Load encoder with explicit vae_dim (for semantic tokenizer which has different dim).
    pub fn load_with_vae_dim(
        vb: VarBuilder,
        cfg: &super::config::AcousticTokenizerConfig,
        vae_dim: usize,
    ) -> Result<Self> {
        let eps = cfg.layernorm_eps;
        let n_filters = cfg.encoder_n_filters;
        // Encoder ratios are REVERSED from config (config has decoder order)
        let ratios: Vec<usize> = cfg.encoder_ratios.iter().rev().copied().collect();

        let num_stages = ratios.len() + 1; // 7 stages for 6 ratios

        // Channel progression: n_filters, n_filters*2, n_filters*4, ..., n_filters*2^6
        // For n_filters=32: [32, 64, 128, 256, 512, 1024, 2048]
        let mut channels = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            channels.push(n_filters * (1 << i));
        }

        // Downsample layers
        let mut downsample_convs = Vec::with_capacity(num_stages);
        let mut downsample_paddings = Vec::with_capacity(num_stages);
        let mut downsample_strides = Vec::with_capacity(num_stages);

        // Stem: Conv1d(1 → n_filters, kernel=7, stride=1)
        // Causal padding = (kernel-1)*dilation - (stride-1) = 6
        let stem = candle_nn::conv1d(
            1, // mono audio input
            channels[0],
            7,
            Conv1dConfig::default(),
            vb.pp("downsample_layers").pp("0").pp("0").pp("conv").pp("conv"),
        )?;
        downsample_convs.push(stem);
        downsample_paddings.push(6); // (7-1) - (1-1) = 6
        downsample_strides.push(1);

        // Downsampling Conv1d layers with stride
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_ch = channels[i];
            let out_ch = channels[i + 1];
            let kernel = ratio * 2;
            // Causal padding = (kernel-1) - (stride-1) = kernel - stride = 2*ratio - ratio = ratio
            let conv = candle_nn::conv1d(
                in_ch,
                out_ch,
                kernel,
                Conv1dConfig {
                    stride: ratio,
                    ..Default::default()
                },
                vb.pp("downsample_layers")
                    .pp(i + 1)
                    .pp("0")
                    .pp("conv")
                    .pp("conv"),
            )?;
            downsample_convs.push(conv);
            downsample_paddings.push(kernel - ratio); // (2*ratio - 1) - (ratio - 1) = ratio
            downsample_strides.push(ratio);
        }

        // Stages (blocks at each channel width)
        let depths = Self::parse_depths(cfg, num_stages);
        let mut stages = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let stage = EncoderStage::load(vb.pp("stages").pp(i), channels[i], depths[i], eps)?;
            stages.push(stage);
        }

        // Head conv: last_channels → vae_dim, kernel=7
        // Causal padding = 6
        let head_conv = candle_nn::conv1d(
            channels[num_stages - 1],
            vae_dim,
            7,
            Conv1dConfig::default(),
            vb.pp("head").pp("conv").pp("conv"),
        )?;

        Ok(Self {
            downsample_convs,
            stages,
            downsample_paddings,
            downsample_strides,
            head_conv,
            channels,
        })
    }

    fn parse_depths(
        cfg: &super::config::AcousticTokenizerConfig,
        num_stages: usize,
    ) -> Vec<usize> {
        if let Some(ref ed) = cfg.encoder_depths {
            ed.split('-')
                .map(|s| s.parse().unwrap_or(3))
                .collect()
        } else {
            vec![3; num_stages]
        }
    }

    /// Causal left-padding for Conv1d.
    fn causal_pad(x: &Tensor, amount: usize) -> Result<Tensor> {
        if amount == 0 {
            return Ok(x.clone());
        }
        let pad = Tensor::zeros((x.dim(0)?, x.dim(1)?, amount), x.dtype(), x.device())?;
        Tensor::cat(&[&pad, x], 2)
    }

    /// Encode audio waveform to latent representation.
    /// Input: (batch, 1, samples) or (batch, samples)
    /// Output: (batch, frames, vae_dim)
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        // Ensure input is (batch, channels=1, samples)
        let x = if audio.rank() == 2 {
            audio.unsqueeze(1)?
        } else {
            audio.clone()
        };

        let mut h = x;

        // Interleave: downsample → stage blocks
        for (i, (conv, stage)) in self
            .downsample_convs
            .iter()
            .zip(self.stages.iter())
            .enumerate()
        {
            // Causal left-pad then conv
            h = Self::causal_pad(&h, self.downsample_paddings[i])?;

            // For strided convolutions, add extra padding for alignment
            if self.downsample_strides[i] > 1 {
                let kernel = conv.weight().dim(2)?;
                let stride = self.downsample_strides[i];
                let length = h.dim(2)?;
                let n_frames = (length - kernel) / stride + 1;
                let ideal_length = n_frames * stride + kernel;
                if ideal_length < length {
                    // This shouldn't happen with proper causal padding
                } else if ideal_length > length {
                    let extra = ideal_length - length;
                    h = Tensor::cat(
                        &[
                            &h,
                            &Tensor::zeros(
                                (h.dim(0)?, h.dim(1)?, extra),
                                h.dtype(),
                                h.device(),
                            )?,
                        ],
                        2,
                    )?;
                }
            }

            h = conv.forward(&h)?;
            h = stage.forward(&h)?;
        }

        // Head conv: causal left-pad 6
        h = Self::causal_pad(&h, 6)?;
        h = self.head_conv.forward(&h)?;

        // Return as (batch, frames, vae_dim)
        h.transpose(1, 2)
    }

    /// Streaming encode: uses cache for correct context between frames.
    pub fn encode_streaming(
        &self,
        audio: &Tensor,
        cache: &mut super::vae_decoder::StreamingConvCache,
    ) -> Result<Tensor> {
        let x = if audio.rank() == 2 {
            audio.unsqueeze(1)?
        } else {
            audio.clone()
        };

        cache.reset_counter();
        let mut h = x;

        for (i, (conv, stage)) in self
            .downsample_convs
            .iter()
            .zip(self.stages.iter())
            .enumerate()
        {
            // Streaming: use cached context instead of zero-padding
            let ctx_size = self.downsample_paddings[i];
            let (slot, is_first) = cache.take_slot();
            let context = if is_first {
                Tensor::zeros((h.dim(0)?, h.dim(1)?, ctx_size), h.dtype(), h.device())?
            } else {
                cache.get(slot).unwrap().clone()
            };
            let padded = Tensor::cat(&[&context, &h], 2)?;

            // Update cache: last ctx_size samples of padded
            let plen = padded.dim(2)?;
            let start = plen.saturating_sub(ctx_size);
            cache.set(slot, padded.narrow(2, start, plen - start)?);

            h = conv.forward(&padded)?;
            h = stage.forward_cached(&h, cache)?;
        }

        // Head conv: streaming context
        let (slot, is_first) = cache.take_slot();
        let ctx = if is_first {
            Tensor::zeros((h.dim(0)?, h.dim(1)?, 6), h.dtype(), h.device())?
        } else {
            cache.get(slot).unwrap().clone()
        };
        let padded = Tensor::cat(&[&ctx, &h], 2)?;
        let plen = padded.dim(2)?;
        cache.set(slot, padded.narrow(2, plen.saturating_sub(6), 6.min(plen))?);
        h = self.head_conv.forward(&padded)?;

        h.transpose(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp};

    fn mt(shape: &[usize], seed: u64) -> Tensor {
        use rand::{Rng, SeedableRng};
        let n: usize = shape.iter().product();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let d: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.01..0.01)).collect();
        Tensor::from_vec(d, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_encoder_block_forward() {
        let ch = 32;
        let mut map = std::collections::HashMap::new();
        let dev = Device::Cpu;

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

        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &dev);
        let block = EncoderBlock::load(vb, ch, 1e-5).unwrap();

        // Input: (batch=1, channels=32, seq=16)
        let x = mt(&[1, ch, 16], 10);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, ch, 16]);
    }

    #[test]
    fn test_encoder_block_residual() {
        let ch = 16;
        let mut map = std::collections::HashMap::new();
        let dev = Device::Cpu;

        // Zero weights → output should equal input (residual connection)
        map.insert("norm.weight".into(), Tensor::ones(ch, DType::F32, &dev).unwrap());
        map.insert("gamma".into(), Tensor::zeros(ch, DType::F32, &dev).unwrap());
        map.insert("mixer.conv.conv.conv.weight".into(), Tensor::zeros(&[ch, 1, 7], DType::F32, &dev).unwrap());
        map.insert("mixer.conv.conv.conv.bias".into(), Tensor::zeros(ch, DType::F32, &dev).unwrap());
        map.insert("ffn_norm.weight".into(), Tensor::ones(ch, DType::F32, &dev).unwrap());
        map.insert("ffn_gamma".into(), Tensor::zeros(ch, DType::F32, &dev).unwrap());
        map.insert("ffn.linear1.weight".into(), Tensor::zeros(&[ch * 4, ch], DType::F32, &dev).unwrap());
        map.insert("ffn.linear1.bias".into(), Tensor::zeros(ch * 4, DType::F32, &dev).unwrap());
        map.insert("ffn.linear2.weight".into(), Tensor::zeros(&[ch, ch * 4], DType::F32, &dev).unwrap());
        map.insert("ffn.linear2.bias".into(), Tensor::zeros(ch, DType::F32, &dev).unwrap());

        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &dev);
        let block = EncoderBlock::load(vb, ch, 1e-5).unwrap();

        let x = mt(&[1, ch, 8], 42);
        let y = block.forward(&x).unwrap();
        // With zero gamma, both mixer and FFN contribute nothing → y ≈ x
        let diff: f32 = (y - x).unwrap().abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(diff < 1e-5, "residual should be identity with zero gamma, got diff={diff}");
    }

    #[test]
    fn test_causal_pad() {
        let x = Tensor::ones(&[1, 4, 10], DType::F32, &Device::Cpu).unwrap();
        let padded = TokenizerEncoder::causal_pad(&x, 3).unwrap();
        assert_eq!(padded.dims(), &[1, 4, 13]);
        // First 3 samples should be zeros
        let first: Vec<f32> = padded.i((0, 0, ..3)).unwrap().to_vec1().unwrap();
        assert!(first.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_causal_pad_zero() {
        let x = Tensor::ones(&[1, 4, 10], DType::F32, &Device::Cpu).unwrap();
        let padded = TokenizerEncoder::causal_pad(&x, 0).unwrap();
        assert_eq!(padded.dims(), &[1, 4, 10]);
    }
}
