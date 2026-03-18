//! Vocos vocoder -- ISTFT-based neural vocoder with ConvNeXt backbone.
//!
//! Weights:
//! - `backbone.embed` [512, 100, 7] -- Conv1d input embedding
//! - `backbone.norm` -- LayerNorm after embedding
//! - `backbone.convnext.{i}` -- 8 ConvNeXt blocks (dwconv, gamma, norm, pwconv1, pwconv2)
//! - `backbone.final_layer_norm` -- final LayerNorm
//! - `head.out` [1026, 512] -- ISTFT head (513 mag + 513 phase)
//! - `head.istft.window` [n_fft] -- ISTFT window

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use rustfft::{num_complex::Complex, FftPlanner};

const NUM_CONVNEXT_LAYERS: usize = 8;

/// A single ConvNeXt block.
#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv_weight: Tensor, // [dim, 1, kernel_size]
    dwconv_bias: Tensor,   // [dim]
    gamma: Tensor,         // [dim] -- learned scale
    norm_weight: Tensor,   // [dim]
    norm_bias: Tensor,     // [dim]
    pwconv1: Linear,       // [ff_dim, dim]
    pwconv2: Linear,       // [dim, ff_dim]
    kernel_size: usize,
    #[allow(dead_code)]
    dim: usize,
}

impl ConvNeXtBlock {
    fn load(dim: usize, ff_mult: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let ff_dim = dim * ff_mult;
        let dwconv_weight = vb.get((dim, 1, kernel_size), "dwconv.weight")?;
        let dwconv_bias = vb.get(dim, "dwconv.bias")?;
        let gamma = vb.get(dim, "gamma")?;
        let norm_weight = vb.get(dim, "norm.weight")?;
        let norm_bias = vb.get(dim, "norm.bias")?;
        let pwconv1 = candle_nn::linear(dim, ff_dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(ff_dim, dim, vb.pp("pwconv2"))?;
        Ok(Self {
            dwconv_weight,
            dwconv_bias,
            gamma,
            norm_weight,
            norm_bias,
            pwconv1,
            pwconv2,
            kernel_size,
            dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, dim, seq]
        let residual = x.clone();

        // Depthwise conv
        let x = self.depthwise_conv1d(x)?;

        // Transpose to [batch, seq, dim] for pointwise operations
        let x = x.transpose(1, 2)?;

        // LayerNorm
        let x = self.layer_norm(&x)?;

        // Pointwise convolutions with GELU activation
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu_erf()?;
        let x = self.pwconv2.forward(&x)?;

        // Apply gamma (channel-wise scale)
        let x = x.broadcast_mul(&self.gamma)?;

        // Transpose back to [batch, dim, seq]
        let x = x.transpose(1, 2)?;

        // Residual connection
        Ok((&x + &residual)?)
    }

    fn layer_norm(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq, dim]
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + 1e-5)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;
        let scaled = normalized.broadcast_mul(&self.norm_weight)?;
        Ok(scaled.broadcast_add(&self.norm_bias)?)
    }

    fn depthwise_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, channels, seq_len) = x.dims3()?;
        let pad = self.kernel_size / 2;

        let x_padded = if pad > 0 {
            x.pad_with_zeros(2, pad, pad)?
        } else {
            x.clone()
        };

        let w = self.dwconv_weight.squeeze(1)?;

        let mut outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let slice = x_padded.narrow(2, i, self.kernel_size)?;
            let prod = slice.broadcast_mul(&w)?;
            let summed = prod.sum(candle_core::D::Minus1)?;
            outputs.push(summed);
        }
        let result = Tensor::stack(&outputs, 2)?;
        let bias = self.dwconv_bias.reshape((1, channels, 1))?;
        Ok(result.broadcast_add(&bias)?)
    }
}

#[derive(Debug, Clone)]
pub struct Vocos {
    // Input embedding: Conv1d [backbone_dim, feat_dim, 7]
    embed_weight: Tensor,
    embed_bias: Tensor,
    embed_kernel: usize,
    // Input norm (LayerNorm)
    norm_weight: Tensor,
    norm_bias: Tensor,
    // ConvNeXt backbone
    convnext: Vec<ConvNeXtBlock>,
    // Final layer norm
    final_norm_weight: Tensor,
    final_norm_bias: Tensor,
    // ISTFT head: single Linear [n_freq*2, backbone_dim]
    head_out: Linear,
    // ISTFT window
    istft_window: Vec<f32>,
    // Config
    n_fft: usize,
    hop_length: usize,
    backbone_dim: usize,
}

impl Vocos {
    pub fn load(
        feat_dim: usize,
        backbone_dim: usize,
        n_fft: usize,
        hop_length: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n_freq = n_fft / 2 + 1;
        let embed_kernel = 7;

        // backbone.embed: Conv1d [backbone_dim, feat_dim, 7]
        let embed_weight = vb.get(
            (backbone_dim, feat_dim, embed_kernel),
            "backbone.embed.weight",
        )?;
        let embed_bias = vb.get(backbone_dim, "backbone.embed.bias")?;

        // backbone.norm: LayerNorm
        let norm_weight = vb.get(backbone_dim, "backbone.norm.weight")?;
        let norm_bias = vb.get(backbone_dim, "backbone.norm.bias")?;

        // ConvNeXt blocks
        let mut convnext = Vec::with_capacity(NUM_CONVNEXT_LAYERS);
        for i in 0..NUM_CONVNEXT_LAYERS {
            let block = ConvNeXtBlock::load(
                backbone_dim,
                3, // ff_mult
                7, // kernel_size
                vb.pp(format!("backbone.convnext.{i}")),
            )?;
            convnext.push(block);
        }

        // backbone.final_layer_norm
        let final_norm_weight = vb.get(backbone_dim, "backbone.final_layer_norm.weight")?;
        let final_norm_bias = vb.get(backbone_dim, "backbone.final_layer_norm.bias")?;

        // head.out: Linear [n_freq*2, backbone_dim]
        let head_out = candle_nn::linear(backbone_dim, n_freq * 2, vb.pp("head.out"))?;

        // head.istft.window
        let window_tensor = vb.get(n_fft, "head.istft.window")?;
        let istft_window = window_tensor
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;

        Ok(Self {
            embed_weight,
            embed_bias,
            embed_kernel,
            norm_weight,
            norm_bias,
            convnext,
            final_norm_weight,
            final_norm_bias,
            head_out,
            istft_window,
            n_fft,
            hop_length,
            backbone_dim,
        })
    }

    /// Forward: mel features [batch, feat_dim, time] -> waveform samples.
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<f32>> {
        // Input embedding: Conv1d
        let x = self.embed_conv1d(mel)?; // [batch, backbone_dim, time]

        // Input LayerNorm (transpose to [batch, time, dim])
        let x = x.transpose(1, 2)?;
        let x = self.layer_norm(&x, &self.norm_weight, &self.norm_bias)?;
        let x = x.transpose(1, 2)?; // back to [batch, dim, time]

        // ConvNeXt backbone
        let mut x = x;
        for block in &self.convnext {
            x = block.forward(&x)?;
        }

        // Final LayerNorm
        let x = x.transpose(1, 2)?;
        let x = self.layer_norm(&x, &self.final_norm_weight, &self.final_norm_bias)?;

        // Head: predict mag and phase via single linear
        let out = self.head_out.forward(&x)?; // [batch, time, n_freq*2]
        let n_freq = self.n_fft / 2 + 1;
        let mag = out.narrow(candle_core::D::Minus1, 0, n_freq)?;
        let phase = out.narrow(candle_core::D::Minus1, n_freq, n_freq)?;

        // Convert to CPU for ISTFT
        let mag = mag
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let phase = phase
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;

        let mag_data = mag.to_vec2::<f32>()?;
        let phase_data = phase.to_vec2::<f32>()?;

        let samples = self.istft(&mag_data, &phase_data);
        Ok(samples)
    }

    /// Conv1d for input embedding.
    fn embed_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, in_channels, seq]
        let (_batch, _in_ch, seq_len) = x.dims3()?;
        let pad = self.embed_kernel / 2;
        let x_padded = x.pad_with_zeros(2, pad, pad)?;

        // embed_weight: [out_channels, in_channels, kernel_size]
        // Standard conv1d (not depthwise)
        let mut outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let slice = x_padded.narrow(2, i, self.embed_kernel)?; // [batch, in_ch, kernel]
            // For each output position: sum over (in_ch, kernel) dimensions
            // slice: [batch, in_ch, kernel], weight: [out_ch, in_ch, kernel]
            // output[pos] = einsum('bik,oik->bo', slice, weight)
            let slice_expanded = slice.unsqueeze(1)?; // [batch, 1, in_ch, kernel]
            let weight_expanded = self.embed_weight.unsqueeze(0)?; // [1, out_ch, in_ch, kernel]
            let prod = slice_expanded.broadcast_mul(&weight_expanded)?; // [batch, out_ch, in_ch, kernel]
            let summed = prod.sum(candle_core::D::Minus1)?.sum(candle_core::D::Minus1)?; // [batch, out_ch]
            outputs.push(summed);
        }
        let result = Tensor::stack(&outputs, 2)?; // [batch, out_ch, seq]
        let bias = self.embed_bias.reshape((1, self.backbone_dim, 1))?;
        Ok(result.broadcast_add(&bias)?)
    }

    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + 1e-5)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;
        let scaled = normalized.broadcast_mul(weight)?;
        Ok(scaled.broadcast_add(bias)?)
    }

    /// Inverse STFT matching Vocos's ISTFT implementation.
    /// Uses irfft semantics (half-spectrum input, norm="backward") with "same" padding.
    fn istft(&self, mag: &[Vec<f32>], phase: &[Vec<f32>]) -> Vec<f32> {
        let n_frames = mag.len();
        let n_freq = self.n_fft / 2 + 1;
        let output_len = (n_frames - 1) * self.hop_length + self.n_fft;
        let pad = (self.n_fft - self.hop_length) / 2; // "same" padding trim

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(self.n_fft);

        let mut output = vec![0.0f32; output_len];
        let mut window_envelope = vec![0.0f32; output_len];

        for frame_idx in 0..n_frames {
            // Build complex spectrum from mag (log domain) and phase
            let mut spectrum: Vec<Complex<f32>> = Vec::with_capacity(self.n_fft);

            for i in 0..n_freq {
                let m = mag[frame_idx][i].exp().min(100.0); // clip max=1e2
                let p = phase[frame_idx][i];
                spectrum.push(Complex::new(m * p.cos(), m * p.sin()));
            }

            // Mirror for negative frequencies (irfft does this internally,
            // but rustfft needs the full spectrum)
            for i in (1..n_freq - 1).rev() {
                spectrum.push(spectrum[i].conj());
            }
            spectrum.resize(self.n_fft, Complex::new(0.0, 0.0));

            // rustfft inverse is unnormalized; torch irfft(norm="backward") divides by N
            ifft.process(&mut spectrum);

            // Apply window and overlap-add (with 1/N normalization)
            let start = frame_idx * self.hop_length;
            let inv_n = 1.0 / self.n_fft as f32;
            for i in 0..self.n_fft {
                if start + i < output_len {
                    let w = self.istft_window[i];
                    output[start + i] += spectrum[i].re * inv_n * w;
                    window_envelope[start + i] += w * w;
                }
            }
        }

        // Normalize by window envelope
        for i in 0..output_len {
            if window_envelope[i] > 1e-11 {
                output[i] /= window_envelope[i];
            }
        }

        // Apply "same" padding: trim pad from each side
        if pad > 0 && output_len > 2 * pad {
            output[pad..output_len - pad].to_vec()
        } else {
            output
        }
    }
}

/// Upsample audio from source_rate to target_rate using linear interpolation.
pub fn upsample(samples: &[f32], source_rate: usize, target_rate: usize) -> Vec<f32> {
    if source_rate == target_rate {
        return samples.to_vec();
    }
    let ratio = target_rate as f64 / source_rate as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        let s0 = samples.get(idx).copied().unwrap_or(0.0);
        let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
        output.push(s0 + frac * (s1 - s0));
    }
    output
}

/// Save audio samples as WAV file (16-bit PCM).
///
/// Re-exported from [`crate::utils::wav::save_wav`].
pub use crate::utils::wav::save_wav;
