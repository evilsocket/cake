//! σ-VAE acoustic decoder for VibeVoice.
//!
//! Converts 64-dim acoustic latents to 24kHz audio waveform.
//! Architecture: 7-stage Conv1d decoder with depthwise conv mixers,
//! RMSNorm, FFN blocks, and ConvTranspose1d upsampling.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, RmsNorm, VarBuilder};

/// A single decoder block with depthwise conv mixer + FFN.
#[derive(Debug, Clone)]
struct DecoderBlock {
    norm: RmsNorm,
    gamma: Tensor,
    mixer_conv: Conv1d,
    ffn_norm: RmsNorm,
    ffn_gamma: Tensor,
    ffn_linear1: candle_nn::Linear,
    ffn_linear2: candle_nn::Linear,
}

impl DecoderBlock {
    fn load(vb: VarBuilder, channels: usize, eps: f64) -> Result<Self> {
        let norm = candle_nn::rms_norm(channels, eps, vb.pp("norm"))?;
        let gamma = vb.get(channels, "gamma")?;

        // Depthwise conv: groups=channels, kernel=7, NO padding (causal model)
        let mixer_conv = candle_nn::conv1d(
            channels,
            channels,
            7,
            Conv1dConfig {
                groups: channels,
                ..Default::default() // padding=0
            },
            vb.pp("mixer").pp("conv").pp("conv").pp("conv"),
        )?;

        let ffn_norm = candle_nn::rms_norm(channels, eps, vb.pp("ffn_norm"))?;
        let ffn_gamma = vb.get(channels, "ffn_gamma")?;
        let ffn_linear1 = candle_nn::linear(channels, channels * 4, vb.pp("ffn").pp("linear1"))?;
        let ffn_linear2 = candle_nn::linear(channels * 4, channels, vb.pp("ffn").pp("linear2"))?;

        Ok(Self {
            norm,
            gamma,
            mixer_conv,
            ffn_norm,
            ffn_gamma,
            ffn_linear1,
            ffn_linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mixer: norm → causal left-pad → depthwise conv → scale by gamma
        let residual = x;
        let h = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        // Causal left-padding: pad (kernel_size - 1) zeros on the left
        let h = Tensor::cat(&[
            &Tensor::zeros((h.dim(0)?, h.dim(1)?, 6), h.dtype(), h.device())?,
            &h,
        ], 2)?;
        let h = self.mixer_conv.forward(&h)?;
        let gamma = self.gamma.unsqueeze(0)?.unsqueeze(2)?;
        let x = (residual + h.broadcast_mul(&gamma)?)?;

        // FFN: norm → linear1 → gelu → linear2 → scale by gamma
        let residual = &x;
        let h = self.ffn_norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        // FFN operates on last dim, so transpose to (batch, seq, channels)
        let h = h.transpose(1, 2)?;
        let h = self.ffn_linear1.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ffn_linear2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        let ffn_gamma = self.ffn_gamma.unsqueeze(0)?.unsqueeze(2)?;
        residual + h.broadcast_mul(&ffn_gamma)?
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

        Ok(Self { upsample_layers, stages, head_conv })
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

    /// Decode acoustic latents to audio waveform.
    /// Input: (batch, vae_dim, frames) or (batch, frames, vae_dim)
    /// Output: (batch, 1, samples)
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        // Ensure (batch, channels, seq) layout
        let x = if latents.dim(1)? == 64 {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };

        // Process through upsample + stage pairs
        let mut h = x;
        for (upsample, stage) in self.upsample_layers.iter().zip(self.stages.iter()) {
            h = upsample.forward(&h)?;
            h = stage.forward(&h)?;
        }

        // Final conv to mono
        self.head_conv.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

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
}
