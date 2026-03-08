// Rust 2024
// Скомпилируется как crate (lib). Структура: один файл src/lib.rs.
//
// Реализовано:
// - CausalConv3d через сумму Conv2d по временной оси (kt срезов веса)
// - RMSNorm "channels-first" как permute -> rmsnorm(last-dim) -> permute back
// - ResnetBlock3d, Down/Up blocks, Encoder3d, Decoder3d
// - DiagonalGaussianDistribution (sample/mode)
// - AutoencoderKLLTXVideo: encode/decode/forward + slicing + tiling + temporal tiling (API как в python)
//
// Примечание: gradient_checkpointing флаги сохранены как поля, но без checkpoint логики.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use super::t2v_pipeline::{VaeConfig, VaeLtxVideo};
use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{
    Activation, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, RmsNorm, VarBuilder, ops,
};

use serde::{Deserialize, Serialize};


#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoencoderKLLtxVideoConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub decoder_block_out_channels: Vec<usize>,
    #[serde(alias = "spatio_temporal_scaling")]
    pub spatiotemporal_scaling: Vec<bool>,
    #[serde(alias = "decoder_spatio_temporal_scaling")]
    pub decoder_spatiotemporal_scaling: Vec<bool>,
    pub layers_per_block: Vec<usize>,
    pub decoder_layers_per_block: Vec<usize>,
    pub patch_size: usize,
    pub patch_size_t: usize,
    #[serde(alias = "resnet_norm_eps")]
    pub resnet_eps: f64,
    pub scaling_factor: f64,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
    pub decoder_inject_noise: Vec<bool>,
    #[serde(alias = "upsample_residual")]
    pub decoder_upsample_residual: Vec<bool>,
    #[serde(alias = "upsample_factor")]
    pub decoder_upsample_factor: Vec<usize>,
    pub timestep_conditioning: bool,
    /// Per-block upsampler strides (t, h, w). If empty, derived from decoder_spatiotemporal_scaling.
    #[serde(default)]
    pub decoder_strides: Vec<(usize, usize, usize)>,
    #[serde(default)]
    pub latents_mean: Vec<f32>,
    #[serde(default)]
    pub latents_std: Vec<f32>,
    #[serde(alias = "downsample_type")]
    pub downsample_types: Vec<String>,
    #[serde(alias = "encoder_causal")]
    pub is_causal: bool,
    pub decoder_causal: bool,
}

impl Default for AutoencoderKLLtxVideoConfig {
    fn default() -> Self {
        // Values from official LTX-Video 0.9.5 VAE config.json
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            block_out_channels: vec![128, 256, 512, 1024, 2048],
            decoder_block_out_channels: vec![256, 512, 1024],
            spatiotemporal_scaling: vec![true, true, true, true],
            decoder_spatiotemporal_scaling: vec![true, true, true],
            layers_per_block: vec![4, 6, 6, 2, 2],
            decoder_layers_per_block: vec![5, 5, 5, 5],
            patch_size: 4,
            patch_size_t: 1,
            resnet_eps: 1e-6,
            scaling_factor: 1.0,
            spatial_compression_ratio: 32,
            temporal_compression_ratio: 8,
            decoder_inject_noise: vec![false, false, false, false],
            decoder_upsample_residual: vec![true, true, true],
            decoder_upsample_factor: vec![2, 2, 2],
            timestep_conditioning: true,
            decoder_strides: vec![],
            latents_mean: vec![0.0; 128],
            latents_std: vec![1.0; 128],
            downsample_types: vec![
                "spatial".into(),
                "temporal".into(),
                "spatiotemporal".into(),
                "spatiotemporal".into(),
            ],
            is_causal: true,
            decoder_causal: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecoderOutput {
    pub sample: Tensor,
}

#[derive(Clone, Debug)]
pub struct AutoencoderKLOutput {
    pub latent_dist: DiagonalGaussianDistribution,
}

/// Аналог diffusers.models.vae.DiagonalGaussianDistribution.
#[derive(Clone, Debug)]
pub struct DiagonalGaussianDistribution {
    pub mean: Tensor,
    pub logvar: Tensor,
}

impl DiagonalGaussianDistribution {
    pub fn new(moments: &Tensor) -> Result<Self> {
        // moments: (B, 2*C, T, H, W) -> mean/logvar split по каналу.
        let (_b, ch2, _t, _h, _w) = moments.dims5()?;
        if ch2 % 2 != 0 {
            candle_core::bail!("moments channels must be even, got {}", ch2)
        }
        let ch = ch2 / 2;
        let mean = moments.i((.., 0..ch, .., .., ..))?;
        let logvar = moments.i((.., ch..(2 * ch), .., .., ..))?;
        Ok(Self { mean, logvar })
    }

    pub fn mode(&self) -> Result<Tensor> {
        Ok(self.mean.clone())
    }

    pub fn sample(&self) -> Result<Tensor> {
        // eps ~ N(0,1), z = mean + exp(0.5*logvar)*eps
        let eps = Tensor::randn(0f32, 1f32, self.mean.shape(), self.mean.device())?
            .to_dtype(self.mean.dtype())?;
        let std = (self.logvar.affine(0.5, 0.)?).exp()?;
        self.mean.add(&std.mul(&eps)?) // mean + std*eps
    }
}

fn rmsnorm_channels_first(norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
    // (B,C,T,H,W) -> (B,T,H,W,C) -> norm -> back
    x.permute((0, 2, 3, 4, 1))?
        .apply(norm)?
        .permute((0, 4, 1, 2, 3))
}

fn layernorm_channels_first(norm: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    x.permute((0, 2, 3, 4, 1))?
        .apply(norm)?
        .permute((0, 4, 1, 2, 3))
}

fn silu(x: &Tensor) -> Result<Tensor> {
    ops::silu(x)
}

fn cat_dim(xs: &[Tensor], dim: usize) -> Result<Tensor> {
    let refs: Vec<&Tensor> = xs.iter().collect();
    Tensor::cat(&refs, dim)
}

/// Sinusoidal timestep embeddings (like Timesteps in diffusers)
/// Parameters match PixArtAlphaCombinedTimestepSizeEmbeddings: flip_sin_to_cos=True, downscale_freq_shift=0
fn get_timestep_embedding(timesteps: &Tensor, embedding_dim: usize) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let device = timesteps.device();
    let dtype = timesteps.dtype();

    // Python: exponent = -math.log(max_period) * torch.arange(0, half_dim) / (half_dim - downscale_freq_shift)
    // With downscale_freq_shift=0: exponent / half_dim (not half_dim - 1!)
    let max_period = 10000f64;
    let downscale_freq_shift = 0.0; // PixArtAlphaCombinedTimestepSizeEmbeddings uses 0

    let exponent_coef = -(max_period.ln()) / (half_dim as f64 - downscale_freq_shift);
    let emb = (Tensor::arange(0u32, half_dim as u32, device)?
        .to_dtype(DType::F32)?
        .affine(exponent_coef, 0.0))?
    .exp()?;

    // timesteps: (B,) -> (B, 1) * emb -> (B, half_dim)
    let timesteps_f = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;
    let emb = timesteps_f.broadcast_mul(&emb.unsqueeze(0)?)?;

    // Python: [sin, cos] then flip -> [cos, sin] if flip_sin_to_cos=True
    // PixArtAlphaCombinedTimestepSizeEmbeddings uses flip_sin_to_cos=True
    let sin_emb = emb.sin()?;
    let cos_emb = emb.cos()?;
    // flip_sin_to_cos=True means [cos, sin] order
    Tensor::cat(&[&cos_emb, &sin_emb], 1)?.to_dtype(dtype)
}

/// TimestepEmbedder: MLP that embeds timesteps (like TimestepEmbedding in diffusers)
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = candle_nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // Debug: print weight info
        let h = self.linear_1.forward(t)?;
        let h = silu(&h)?;
        self.linear_2.forward(&h)
    }
}

/// Combined timestep embedder (like PixArtAlphaCombinedTimestepSizeEmbeddings)
#[derive(Debug, Clone)]
pub struct CombinedTimestepEmbedder {
    timestep_embedder: TimestepEmbedder,
}

impl CombinedTimestepEmbedder {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder =
            TimestepEmbedder::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self { timestep_embedder })
    }

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: DType) -> Result<Tensor> {
        // timestep -> sinusoidal -> MLP
        let timesteps_proj = get_timestep_embedding(timestep, 256)?;


        self.timestep_embedder
            .forward(&timesteps_proj.to_dtype(hidden_dtype)?)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Conv3dLikeConfig {
    pub stride_t: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dil_t: usize,
    pub dil_h: usize,
    pub dil_w: usize,
    pub groups: usize,
    pub padding_mode_zeros: bool,
    pub is_causal: bool,
}

impl Default for Conv3dLikeConfig {
    fn default() -> Self {
        Self {
            stride_t: 1,
            stride_h: 1,
            stride_w: 1,
            dil_t: 1,
            dil_h: 1,
            dil_w: 1,
            groups: 1,
            padding_mode_zeros: true,
            is_causal: true,
        }
    }
}

/// Эквивалент LTXVideoCausalConv3d из python, но реализованный через kt Conv2d по времени.
#[derive(Debug, Clone)]
pub struct LtxVideoCausalConv3d {
    kt: usize,
    pub _kh: usize,
    pub _kw: usize,
    cfg: Conv3dLikeConfig,
    conv2d_slices: Vec<Conv2d>, // длина kt
    bias: Option<Tensor>,       // (out_channels)
}

impl LtxVideoCausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        is_causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let (st, sh, sw) = stride;
        let (dt, dh, dw) = dilation;

        // In diffusers, LtxVideoCausalConv3d has an inner `conv` module
        let conv_vb = vb.pp("conv");
        // вес как у conv3d: (out, in/groups, kt, kh, kw)
        let w = conv_vb.get((out_channels, in_channels / groups, kt, kh, kw), "weight")?;

        // Wait. Python Conv3d default bias=True.
        // Are there cases where bias is disabled?
        // LTX uses LayerNorm/RMSNorm. Sometimes Conv bias is removed.
        // But Diffusers code initialized Conv3d with defaults (bias=True).
        // Exceptions?
        // Code snippet 4772 line 66: padding_mode passed. bias NOT passed (so True).
        // So ALL CausalConv3d layers MUST have bias.
        // So I should remove .ok().
        let b = conv_vb.get(out_channels, "bias")?;

        let hpad = kh / 2;
        let _wpad = kw / 2;

        let mut conv2d_slices = Vec::with_capacity(kt);
        for ti in 0..kt {
            let w2 = w.i((.., .., ti, .., ..))?.contiguous()?;
            let c2cfg = Conv2dConfig {
                padding: hpad,
                stride: sh,
                dilation: dh,
                groups,
                ..Default::default()
            };
            // bias добавим один раз после суммы.
            conv2d_slices.push(Conv2d::new(w2, None, c2cfg));
        }

        Ok(Self {
            kt,
            _kh: kh,
            _kw: kw,
            cfg: Conv3dLikeConfig {
                stride_t: st,
                stride_h: sh,
                stride_w: sw,
                dil_t: dt,
                dil_h: dh,
                dil_w: dw,
                groups,
                padding_mode_zeros: true,
                is_causal,
            },
            conv2d_slices,
            bias: Some(b),
        })
    }

    fn pad_time_replicate(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B,C,T,H,W)
        let (_, _, t, _, _) = x.dims5()?;
        let kt = self.kt;

        if kt <= 1 {
            return Ok(x.clone());
        }

        if self.cfg.is_causal {
            let left = kt - 1;
            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let pad_left = first.repeat((1, 1, left, 1, 1))?;
            cat_dim(&[pad_left, x.clone()], 2)
        } else {
            let left = (kt - 1) / 2;
            let right = (kt - 1) / 2;

            let first = x.i((.., .., 0, .., ..))?.unsqueeze(2)?;
            let last = x.i((.., .., t - 1, .., ..))?.unsqueeze(2)?;

            let pad_left = if left == 0 {
                None
            } else {
                Some(first.repeat((1, 1, left, 1, 1))?)
            };
            let pad_right = if right == 0 {
                None
            } else {
                Some(last.repeat((1, 1, right, 1, 1))?)
            };

            match (pad_left, pad_right) {
                (None, None) => Ok(x.clone()),
                (Some(pl), None) => cat_dim(&[pl, x.clone()], 2),
                (None, Some(pr)) => cat_dim(&[x.clone(), pr], 2),
                (Some(pl), Some(pr)) => cat_dim(&[pl, x.clone(), pr], 2),
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Реализация свертки по времени: y[t_out] = sum_k Conv2d(x[t_in + k*dil_t])
        // с temporal padding replicate и temporal stride.
        let x = self.pad_time_replicate(x)?;
        let (_b, _c, t_pad, _h, _w) = x.dims5()?;

        let kt = self.kt;
        let dt = self.cfg.dil_t;
        let st = self.cfg.stride_t;

        // t_out: сколько можно сдвигов без выхода за границу.
        let needed = (kt - 1) * dt + 1;
        if t_pad < needed {
            candle_core::bail!(
                "time dim too small after padding: t_pad={}, needed={}",
                t_pad,
                needed
            )
        }
        let t_out = (t_pad - needed) / st + 1;

        let mut ys: Vec<Tensor> = Vec::with_capacity(t_out);

        for to in 0..t_out {
            let base_t = to * st;

            let mut acc: Option<Tensor> = None;
            for ki in 0..kt {
                let ti = base_t + ki * dt;
                let xt = x.i((.., .., ti, .., ..))?; // (B,C,H,W)
                let yt = xt.apply(&self.conv2d_slices[ki])?; // (B,Out,H',W')
                acc = Some(match acc {
                    None => yt,
                    Some(prev) => prev.add(&yt)?,
                });
            }

            let yt = acc.expect("kt>=1 so acc is Some");
            ys.push(yt.unsqueeze(2)?); // (B,Out,1,H',W')
        }

        let y = cat_dim(&ys, 2)?; // (B,Out,T_out,H',W')

        if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, bias.dims1()?, 1, 1, 1))?;
            y.broadcast_add(&bias)
        } else {
            Ok(y)
        }
    }
}

/// Downsample type for LTX-Video 0.9.5 VAE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DownsampleType {
    Conv,           // stride (2,2,2) direct conv
    Spatial,        // stride (1,2,2) pixel unshuffle
    Temporal,       // stride (2,1,1) pixel unshuffle
    Spatiotemporal, // stride (2,2,2) pixel unshuffle
}

impl DownsampleType {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "spatial" => Self::Spatial,
            "temporal" => Self::Temporal,
            "spatiotemporal" => Self::Spatiotemporal,
            _ => Self::Conv,
        }
    }

    pub fn stride(&self) -> (usize, usize, usize) {
        match self {
            Self::Conv => (2, 2, 2),
            Self::Spatial => (1, 2, 2),
            Self::Temporal => (2, 1, 1),
            Self::Spatiotemporal => (2, 2, 2),
        }
    }
}

/// Pixel unshuffle downsampler for LTX-Video 0.9.5
#[derive(Debug, Clone)]
pub struct LtxVideoDownsampler3d {
    stride: (usize, usize, usize),
    group_size: usize,
    conv: LtxVideoCausalConv3d,
}

impl LtxVideoDownsampler3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        is_causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (st, sh, sw) = stride;
        let group_size = (in_channels * st * sh * sw) / out_channels;
        let conv_out_channels = out_channels / (st * sh * sw);

        let conv = LtxVideoCausalConv3d::new(
            in_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv"),
        )?;

        Ok(Self {
            stride,
            group_size,
            conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (st, sh, sw) = self.stride;
        let (b, c, _t, _h, _w) = x.dims5()?;

        // Pad temporal dimension: cat(x[:,:,:st-1], x, dim=2)
        let padded = if st > 1 {
            let pad_slice = x.i((.., .., ..(st - 1), .., ..))?;
            Tensor::cat(&[&pad_slice, x], 2)?
        } else {
            x.clone()
        };
        let (_, _, t_pad, h_pad, w_pad) = padded.dims5()?;

        // Compute new dimensions after pixel unshuffle
        let t_new = t_pad / st;
        let h_new = h_pad / sh;
        let w_new = w_pad / sw;

        // === Residual path: pixel unshuffle + mean ===
        // Shape: (B, C, T, H, W) -> (B, C, T', st, H', sh, W', sw)
        let residual = padded
            .reshape(&[b, c, t_new, st, h_new, sh, w_new, sw])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])? // (B, C, st, sh, sw, T', H', W')
            .reshape((b, c * st * sh * sw, t_new, h_new, w_new))?;

        // Group and average: unflatten(1, (-1, group_size)).mean(dim=2)
        let residual = residual
            .reshape(&[
                b,
                c * st * sh * sw / self.group_size,
                self.group_size,
                t_new,
                h_new,
                w_new,
            ])?
            .mean(2)?;

        // === Conv path: same pixel unshuffle ===
        let conv_out = self.conv.forward(&padded)?;
        let (_, c_conv, _, _, _) = conv_out.dims5()?;

        let hidden = conv_out
            .reshape(&[b, c_conv, t_new, st, h_new, sh, w_new, sw])?
            .permute(vec![0, 1, 3, 5, 7, 2, 4, 6])?
            .reshape((b, c_conv * st * sh * sw, t_new, h_new, w_new))?;

        hidden.add(&residual)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoResnetBlock3d {
    norm1: Option<RmsNorm>,
    conv1: LtxVideoCausalConv3d,
    norm2: Option<RmsNorm>,
    _dropout: f64,
    conv2: LtxVideoCausalConv3d,

    // shortcut при смене каналов
    norm3: Option<LayerNorm>,
    conv_shortcut: Option<LtxVideoCausalConv3d>,

    // noise injection
    per_channel_scale1: Option<Tensor>, // (C,1,1)
    per_channel_scale2: Option<Tensor>,

    // timestep conditioning
    scale_shift_table: Option<Tensor>, // (4, C)
}

impl LtxVideoResnetBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        dropout: f64,
        eps: f64,
        elementwise_affine: bool,
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // LTX-Video resnet blocks may not have norm layers - make them optional
        // Helper to load RmsNorm or fallback to default (ones) if affine is false or loading fails
        let load_norm = |name: &str, size: usize| -> Result<RmsNorm> {
            if elementwise_affine {
                let norm_res = candle_nn::rms_norm(size, 1e-8, vb.pp(name));
                if let Ok(norm) = norm_res {
                    return Ok(norm);
                }
            }
            // Fallback: create RmsNorm with ones (representing no affine scaling)
            let ones = Tensor::ones((size,), vb.dtype(), vb.device())?;
            Ok(RmsNorm::new(ones, 1e-8))
        };

        let norm1 = Some(load_norm("norm1", in_channels)?);
        let conv1 = LtxVideoCausalConv3d::new(
            in_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv1"),
        )?;

        let norm2 = Some(load_norm("norm2", out_channels)?);
        let conv2 = LtxVideoCausalConv3d::new(
            out_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv2"),
        )?;

        let (norm3, conv_shortcut) = if in_channels != out_channels {
            let lncfg = LayerNormConfig {
                eps,
                affine: elementwise_affine,
                ..Default::default()
            };
            let norm3 = candle_nn::layer_norm(in_channels, lncfg, vb.pp("norm3")).ok();
            let conv_shortcut = LtxVideoCausalConv3d::new(
                in_channels,
                out_channels,
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                1,
                is_causal,
                vb.pp("conv_shortcut"),
            )?;
            (norm3, Some(conv_shortcut))
        } else {
            (None, None)
        };

        let per_channel_scale1 = if inject_noise {
            vb.pp("per_channel_scale1")
                .get((in_channels, 1, 1), "weight")
                .ok()
        } else {
            None
        };
        let per_channel_scale2 = if inject_noise {
            vb.pp("per_channel_scale2")
                .get((in_channels, 1, 1), "weight")
                .ok()
        } else {
            None
        };

        let scale_shift_table = if timestep_conditioning {
            vb.get((4, in_channels), "scale_shift_table").ok()
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            _dropout: dropout,
            conv2,
            norm3,
            conv_shortcut,
            per_channel_scale1,
            per_channel_scale2,
            scale_shift_table,
        })
    }

    fn maybe_apply_scale_shift(
        &self,
        x: Tensor,
        temb: Option<&Tensor>,
        stage: usize, // 0: (shift1,scale1), 1: (shift2,scale2)
    ) -> Result<Tensor> {
        let Some(tbl) = &self.scale_shift_table else {
            return Ok(x);
        };
        let Some(temb) = temb else {
            return Ok(x);
        };

        // temb: (B, 4*C, 1, 1, 1) -> unflatten dim1 to (B, 4, C, 1, 1, 1)
        let (b, temb_dim, _, _, _) = temb.dims5()?;
        let c = tbl.dims2()?.1;
        if temb_dim != 4 * c {
            candle_core::bail!("temb dim mismatch: got {}, expected {}", temb_dim, 4 * c)
        }
        let temb = temb
            .reshape((b, 4, c, 1, 1, 1))?
            .broadcast_add(&tbl.unsqueeze(0)?.unsqueeze(3)?.unsqueeze(4)?.unsqueeze(5)?)?;

        let shift = temb.i((.., stage * 2, .., .., .., ..))?; // (B,C,1,1,1)
        let scale = temb.i((.., stage * 2 + 1, .., .., .., ..))?;
        // x * (1 + scale) + shift
        x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)
    }

    fn maybe_inject_noise(&self, x: Tensor, pcs: &Option<Tensor>) -> Result<Tensor> {
        let Some(scale) = pcs else {
            return Ok(x);
        };
        // spatialshape = (H,W) как в python-коде.
        let (_b, _c, _t, h, w) = x.dims5()?;
        let noise = Tensor::randn(0f32, 1f32, (h, w), x.device())?.to_dtype(x.dtype())?;
        // (H,W) -> (1,1,1,H,W)
        let noise = noise.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?;
        // scale: (C,1,1) -> (1,C,1,1,1)
        let scale = scale.unsqueeze(0)?.unsqueeze(2)?;
        x.add(&(noise.broadcast_mul(&scale)?))
    }

    pub fn forward(&self, inputs: &Tensor, temb: Option<&Tensor>, _train: bool) -> Result<Tensor> {
        let mut h = inputs.clone();

        // Only apply norm if it exists
        if let Some(ref norm1) = self.norm1 {
            h = rmsnorm_channels_first(norm1, &h)?;
        }

        h = self.maybe_apply_scale_shift(h, temb, 0)?;

        h = silu(&h)?;

        h = self.conv1.forward(&h)?;

        h = self.maybe_inject_noise(h, &self.per_channel_scale1)?;

        if let Some(ref norm2) = self.norm2 {
            h = rmsnorm_channels_first(norm2, &h)?;
        }
        h = self.maybe_apply_scale_shift(h, temb, 1)?;
        h = silu(&h)?;

        // dropout unused in inference

        h = self.conv2.forward(&h)?;

        // if let Ok(vals) = h.flatten_all()?.to_vec1::<f32>() {
        //      println!("[DEBUG] Resnet output conv2 mean: {:.4}", vals.iter().sum::<f32>() / vals.len() as f32);
        // }

        h = self.maybe_inject_noise(h, &self.per_channel_scale2)?;

        let mut x = inputs.clone();
        if let Some(n3) = &self.norm3 {
            x = layernorm_channels_first(n3, &x)?;
        }
        if let Some(cs) = &self.conv_shortcut {
            x = cs.forward(&x)?;
        }
        let result = h.add(&x)?;

        Ok(result)
    }
}

/// Wrapper for different downsampler types
#[derive(Debug, Clone)]
pub enum Downsampler {
    Conv(LtxVideoCausalConv3d),
    PixelUnshuffle(LtxVideoDownsampler3d),
}

impl Downsampler {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Downsampler::Conv(c) => c.forward(x),
            Downsampler::PixelUnshuffle(p) => p.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoDownBlock3d {
    resnets: Vec<LtxVideoResnetBlock3d>,
    downsamplers: Option<Vec<Downsampler>>,
    conv_out: Option<LtxVideoResnetBlock3d>,
    pub gradient_checkpointing: bool,
}

impl LtxVideoDownBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        spatiotemporal_scale: bool,
        is_causal: bool,
        downsample_type: DownsampleType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(LtxVideoResnetBlock3d::new(
                in_channels,
                in_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                false,
                false,
                vb.pp(format!("resnets.{i}")),
            )?);
        }

        let downsamplers = if spatiotemporal_scale {
            let ds = match downsample_type {
                DownsampleType::Conv => {
                    // Direct stride conv
                    Downsampler::Conv(LtxVideoCausalConv3d::new(
                        in_channels,
                        in_channels,
                        (3, 3, 3),
                        (2, 2, 2),
                        (1, 1, 1),
                        1,
                        is_causal,
                        vb.pp("downsamplers.0").pp("conv"),
                    )?)
                }
                _ => {
                    // Pixel unshuffle types (spatial/temporal/spatiotemporal)
                    let stride = downsample_type.stride();
                    Downsampler::PixelUnshuffle(LtxVideoDownsampler3d::new(
                        in_channels,
                        out_channels,
                        stride,
                        is_causal,
                        vb.pp("downsamplers.0"),
                    )?)
                }
            };
            Some(vec![ds])
        } else {
            None
        };

        // conv_out only needed for channel change in some configs
        let conv_out = if in_channels != out_channels && downsample_type == DownsampleType::Conv {
            LtxVideoResnetBlock3d::new(
                in_channels,
                out_channels,
                dropout,
                resnet_eps,
                true,
                is_causal,
                false,
                false,
                vb.pp("conv_out"),
            )
            .ok()
        } else {
            None
        };

        Ok(Self {
            resnets,
            downsamplers,
            conv_out,
            gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();
        for r in self.resnets.iter() {
            h = r.forward(&h, temb, train)?;
        }
        if let Some(ds) = &self.downsamplers {
            for d in ds.iter() {
                h = d.forward(&h)?;
            }
        }
        if let Some(co) = &self.conv_out {
            h = co.forward(&h, temb, train)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoMidBlock3d {
    resnets: Vec<LtxVideoResnetBlock3d>,
    pub gradient_checkpointing: bool,
    time_embedder: Option<CombinedTimestepEmbedder>,
}

impl LtxVideoMidBlock3d {
    pub fn new(
        in_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            resnets.push(LtxVideoResnetBlock3d::new(
                in_channels,
                in_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                inject_noise,
                timestep_conditioning,
                vb.pp(format!("resnets.{i}")),
            )?);
        }

        let time_embedder = if timestep_conditioning {
            // Block channels * 4 for scale_shift_table compatibility
            let emb_dim = in_channels * 4;
            CombinedTimestepEmbedder::new(emb_dim, vb.pp("time_embedder")).ok()
        } else {
            None
        };

        Ok(Self {
            resnets,
            time_embedder,
            gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();

        // Apply time embedding if present
        let temb_proj = if let (Some(te), Some(t)) = (&self.time_embedder, temb) {
            // Python: temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
            let emb = te.forward(t, h.dtype())?;
            let batch_size = h.dims5()?.0;
            let emb_dim = emb.dims2()?.1;
            Some(emb.reshape((batch_size, emb_dim, 1, 1, 1))?)
        } else {
            None
        };

        for r in self.resnets.iter() {
            h = r.forward(&h, temb_proj.as_ref(), train)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoUpsampler3d {
    stride_t: usize,
    stride_h: usize,
    stride_w: usize,
    residual: bool,

    channel_repeats: usize,
    conv: LtxVideoCausalConv3d,
}

impl LtxVideoUpsampler3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize, usize),
        is_causal: bool,
        residual: bool,
        _upscale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (st, sh, sw) = stride;
        let stride_product = st * sh * sw;
        // Conv output needs to be such that after shuffle we get out_channels.
        // Shuffle reduces channels by stride_product.
        // So conv_out = out_channels * stride_product.
        let conv_out_channels = out_channels * stride_product;
        // For residual: must match main path channels after shuffle
        // main_channels = conv_out_channels / stride_product = out_channels
        // residual_channels = in_channels / stride_product
        // channel_repeats = main_channels / residual_channels = (out_channels * stride_product / in_channels)
        let channel_repeats = conv_out_channels / in_channels;

        let conv = LtxVideoCausalConv3d::new(
            in_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv"),
        )?;
        Ok(Self {
            stride_t: st,
            stride_h: sh,
            stride_w: sw,
            residual,
            channel_repeats,
            conv,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python: permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        let (b, _c, t, h, w) = x.dims5()?;
        let st = self.stride_t;
        let sh = self.stride_h;
        let sw = self.stride_w;


        let residual = if self.residual {
            let cprime = x.dims5()?.1;
            let c_out = cprime / (st * sh * sw);

            // reshape to [B, C', st, sh, sw, T, H, W]
            let x2 = x.reshape(&[b, c_out, st, sh, sw, t, h, w])?;
            // permute(0, 1, 5, 2, 6, 3, 7, 4) -> [B, C', T, st, H, sh, W, sw]
            let x2 = x2.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?.contiguous()?;
            // flatten(6, 7) -> [B, C', T, st, H, sh, W*sw]
            let x2 = x2.reshape(&[b, c_out, t, st, h, sh, w * sw])?;
            // flatten(4, 5) -> [B, C', T, st, H*sh, W*sw]
            let x2 = x2.reshape(&[b, c_out, t, st, h * sh, w * sw])?;
            // flatten(2, 3) -> [B, C', T*st, H*sh, W*sw]
            let x2 = x2.reshape(&[b, c_out, t * st, h * sh, w * sw])?;

            // repeat channels if needed
            let x2 = if self.channel_repeats > 1 {
                x2.repeat((1, self.channel_repeats, 1, 1, 1))?
            } else {
                x2
            };
            // slice: [:, :, st-1:]
            let x2 = x2.i((.., .., (st - 1).., .., ..))?;
            Some(x2)
        } else {
            None
        };

        let h0 = self.conv.forward(x)?;
        let h0 = h0.contiguous()?;

        let (_b2, c2, t2, h2, w2) = h0.dims5()?;
        let c_out = c2 / (st * sh * sw);

        // reshape to [B, C', st, sh, sw, T, H, W]
        let h1 = h0.reshape(&[b, c_out, st, sh, sw, t2, h2, w2])?;

        // permute(0, 1, 5, 2, 6, 3, 7, 4) -> [B, C', T, st, H, sh, W, sw]
        let h1 = h1.permute(vec![0, 1, 5, 2, 6, 3, 7, 4])?.contiguous()?;

        // flatten(6, 7) -> [B, C', T, st, H, sh, W*sw]
        let h1 = h1.reshape(&[b, c_out, t2, st, h2, sh, w2 * sw])?;
        // flatten(4, 5) -> [B, C', T, st, H*sh, W*sw]
        let h1 = h1.reshape(&[b, c_out, t2, st, h2 * sh, w2 * sw])?;
        // flatten(2, 3) -> [B, C', T*st, H*sh, W*sw]
        let h1 = h1.reshape(&[b, c_out, t2 * st, h2 * sh, w2 * sw])?;

        // slice: [:, :, st-1:]
        let h1 = h1.i((.., .., (st - 1).., .., ..))?;

        let h1 = if let Some(r) = residual {
            h1.add(&r)?
        } else {
            h1
        };
        Ok(h1)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoUpBlock3d {
    conv_in: Option<LtxVideoResnetBlock3d>,
    upsamplers: Option<Vec<LtxVideoUpsampler3d>>,
    resnets: Vec<LtxVideoResnetBlock3d>,
    time_embedder: Option<CombinedTimestepEmbedder>,
    pub gradient_checkpointing: bool,
}

impl LtxVideoUpBlock3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        spatiotemporal_scale: bool,
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        upsampler_residual: bool,
        up_scale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let stride = if spatiotemporal_scale {
            (2, 2, 2)
        } else {
            (1, 2, 2)
        };
        Self::new_with_stride(
            in_channels, out_channels, num_layers, dropout, resnet_eps,
            stride, is_causal, inject_noise, timestep_conditioning,
            upsampler_residual, up_scale_factor, vb,
        )
    }

    /// Create an up block with an explicit upsampler stride (t, h, w).
    pub fn new_with_stride(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        dropout: f64,
        resnet_eps: f64,
        stride: (usize, usize, usize),
        is_causal: bool,
        inject_noise: bool,
        timestep_conditioning: bool,
        upsampler_residual: bool,
        up_scale_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // conv_in may not exist in some VAE configs (e.g. official 0.9.5)
        let conv_in = if in_channels != out_channels {
            Some(LtxVideoResnetBlock3d::new(
                in_channels,
                out_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                inject_noise,
                timestep_conditioning,
                vb.pp("conv_in"),
            )?)
        } else {
            None
        };

        let upsamplers = Some(vec![LtxVideoUpsampler3d::new(
            out_channels * up_scale_factor,
            out_channels,
            stride,
            is_causal,
            upsampler_residual,
            up_scale_factor,
            vb.pp("upsamplers.0"),
        )?]);

        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // If upsampler exists (which is always true in default config/logic above), it changes channels to out_channels.
            // If explicit None case handled later, logic changes.
            // Assuming upsamplers always exist for now (based on diffusers 'if i > 0' and reversed decoder logic starting at block 2).
            // But wait, what if `spatiotemporal_scale` logic creates None? NO, I removed the `else { None }` block above and added spatial fallback.
            // So upsampler ALWAYS outputs `out_channels`.
            let in_c = out_channels;
            resnets.push(LtxVideoResnetBlock3d::new(
                in_c,
                out_channels,
                dropout,
                resnet_eps,
                false,
                is_causal,
                inject_noise,
                timestep_conditioning,
                vb.pp(format!("resnets.{i}")),
            )?);
        }

        let time_embedder = if timestep_conditioning {
            // Block channels * 4 for scale_shift_table compatibility
            let emb_dim = out_channels * 4;
            CombinedTimestepEmbedder::new(emb_dim, vb.pp("time_embedder")).ok()
        } else {
            None
        };

        Ok(Self {
            conv_in,
            upsamplers,
            resnets,
            time_embedder,
            gradient_checkpointing: false,
        })
    }

    pub fn forward(&self, x: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let mut h = x.clone();

        // Python order:
        // 1. conv_in with RAW temb (if exists)
        // 2. time_embedder to get temb_proj
        // 3. upsamplers
        // 4. resnets with temb_proj

        // 1. conv_in uses RAW temb (before time_embedder transformation)
        //    Note: conv_in's internal scale_shift_table expects 4*C dimensional temb
        //    But the raw temb passed from decoder is just a scalar, so conv_in won't apply scale_shift
        if let Some(ci) = &self.conv_in {
            h = ci.forward(&h, None, train)?; // conv_in doesn't use temb in 0.9.5
        }

        // 2. Apply time_embedder AFTER conv_in (matches Python order)
        let temb_proj = if let (Some(te), Some(t)) = (&self.time_embedder, temb) {
            let emb = te.forward(t, h.dtype())?;
            let batch_size = h.dims5()?.0;
            let emb_dim = emb.dims2()?.1;
            Some(emb.reshape((batch_size, emb_dim, 1, 1, 1))?)
        } else {
            None
        };

        // 3. upsamplers
        if let Some(us) = &self.upsamplers {
            for u in us.iter() {
                h = u.forward(&h)?;
            }
        }

        // 4. resnets use the transformed temb_proj
        for r in self.resnets.iter() {
            h = r.forward(&h, temb_proj.as_ref(), train)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoEncoder3d {
    patch_size: usize,
    patch_size_t: usize,
    conv_in: LtxVideoCausalConv3d,
    down_blocks: Vec<LtxVideoDownBlock3d>,
    mid_block: LtxVideoMidBlock3d,
    norm_out: Option<RmsNorm>,
    conv_act: Activation,
    conv_out: LtxVideoCausalConv3d,
    pub gradient_checkpointing: bool,
}

impl LtxVideoEncoder3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        block_out_channels: &[usize],
        spatiotemporal_scaling: &[bool],
        layers_per_block: &[usize],
        downsample_types: &[DownsampleType],
        patch_size: usize,
        patch_size_t: usize,
        resnet_eps: f64,
        is_causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channels_patched = in_channels * patch_size * patch_size * patch_size_t;
        let conv_in = LtxVideoCausalConv3d::new(
            in_channels_patched,
            block_out_channels[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::new();
        let n = block_out_channels.len() - 1;
        let mut current = block_out_channels[0];

        for i in 0..n {
            // For pixel unshuffle downsamplers, out_channels is the NEXT block's channels
            let outc = block_out_channels[i + 1];

            // Use downsample_type from config, default to Conv if not provided
            let ds_type = downsample_types
                .get(i)
                .copied()
                .unwrap_or(DownsampleType::Conv);

            let db = LtxVideoDownBlock3d::new(
                current,
                outc,
                layers_per_block[i],
                0.0,
                resnet_eps,
                spatiotemporal_scaling[i],
                is_causal,
                ds_type,
                vb.pp(format!("down_blocks.{i}")),
            )?;
            down_blocks.push(db);
            current = outc;
        }

        let mid_layers = *layers_per_block.last().unwrap_or(&1);
        let mid_block = LtxVideoMidBlock3d::new(
            current,
            mid_layers.saturating_sub(1),
            0.0,
            resnet_eps,
            is_causal,
            false,
            false,
            vb.pp("mid_block"),
        )?;

        let norm_out = if let Ok(norm) = candle_nn::rms_norm(current, 1e-8, vb.pp("norm_out")) {
            Some(norm)
        } else {
            let ones = Tensor::ones((current,), vb.dtype(), vb.device())?;
            Some(RmsNorm::new(ones, 1e-8))
        };
        let conv_act = Activation::Silu;
        let conv_out = LtxVideoCausalConv3d::new(
            current,
            out_channels + 1,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            patch_size,
            patch_size_t,
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_act,
            conv_out,
            gradient_checkpointing: false,
        })
    }

    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        // (B,C,F,H,W) -> (B, C*pt*p*p, F/pt, H/p, W/p) с тем же порядком, что в python.
        let p = self.patch_size;
        let pt = self.patch_size_t;
        let (b, c, f, h, w) = x.dims5()?;
        if f % pt != 0 || h % p != 0 || w % p != 0 {
            candle_core::bail!("input not divisible by patch sizes")
        }
        let post_f = f / pt;
        let post_h = h / p;
        let post_w = w / p;

        let x = x.reshape(&[b, c, post_f, pt, post_h, p, post_w, p])?;
        let x = x
            .permute(vec![0, 1, 3, 7, 5, 2, 4, 6])?
            .contiguous()?
            .reshape((b, c * pt * p * p, post_f, post_h, post_w))?;
        Ok(x)
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut h = self.patchify(x)?;
        h = self.conv_in.forward(&h)?;
        for db in self.down_blocks.iter() {
            h = db.forward(&h, None, train)?;
        }
        h = self.mid_block.forward(&h, None, train)?;

        // Apply norm_out only if it exists
        if let Some(ref norm) = self.norm_out {
            h = rmsnorm_channels_first(norm, &h)?;
        }

        h = h.apply(&self.conv_act)?;
        h = self.conv_out.forward(&h)?;
        // println!("[DEBUG] conv_out final min/max: {:.4}/{:.4}", h.flatten_all()?.to_vec1::<f32>()?.iter().cloned().fold(f32::INFINITY, f32::min), h.flatten_all()?.to_vec1::<f32>()?.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        // last channel replication trick (как в python)
        let (_b, ch, _t, _h, _w) = h.dims5()?;
        let last = h.i((.., (ch - 1), .., .., ..))?.unsqueeze(1)?; // (B,1,T,H,W)
        let rep = last.repeat((1, ch.saturating_sub(2), 1, 1, 1))?;
        cat_dim(&[h, rep], 1)
    }
}

#[derive(Debug, Clone)]
pub struct LtxVideoDecoder3d {
    patch_size: usize,
    patch_size_t: usize,
    pub conv_in: LtxVideoCausalConv3d,
    pub mid_block: LtxVideoMidBlock3d,
    pub up_blocks: Vec<LtxVideoUpBlock3d>,
    pub norm_out: Option<RmsNorm>,
    pub conv_act: Activation,
    pub conv_out: LtxVideoCausalConv3d,
    // Timestep conditioning
    pub time_embedder: Option<CombinedTimestepEmbedder>,
    pub scale_shift_table: Option<Tensor>,
    pub timestep_scale_multiplier: Option<Tensor>,
    pub gradient_checkpointing: bool,
}

impl LtxVideoDecoder3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        block_out_channels: &[usize],
        spatiotemporal_scaling: &[bool],
        layers_per_block: &[usize],
        patch_size: usize,
        patch_size_t: usize,
        resnet_eps: f64,
        is_causal: bool,
        inject_noise: &[bool],
        timestep_conditioning: bool,
        upsampler_residual: &[bool],
        upsample_factor: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        // Derive strides from spatiotemporal_scaling
        let strides: Vec<(usize, usize, usize)> = spatiotemporal_scaling
            .iter()
            .map(|&s| if s { (2, 2, 2) } else { (1, 2, 2) })
            .collect();
        Self::new_with_strides(
            in_channels, out_channels, block_out_channels, &strides,
            layers_per_block, patch_size, patch_size_t, resnet_eps,
            is_causal, inject_noise, timestep_conditioning,
            upsampler_residual, upsample_factor, vb,
        )
    }

    /// Create a decoder with explicit per-block upsampler strides.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_strides(
        in_channels: usize,
        out_channels: usize,
        block_out_channels: &[usize],
        strides: &[(usize, usize, usize)],
        layers_per_block: &[usize],
        patch_size: usize,
        patch_size_t: usize,
        resnet_eps: f64,
        is_causal: bool,
        inject_noise: &[bool],
        timestep_conditioning: bool,
        upsampler_residual: &[bool],
        upsample_factor: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        // decoder uses reversed lists
        let mut boc = block_out_channels.to_vec();
        boc.reverse();
        let mut strides_rev = strides.to_vec();
        strides_rev.reverse();
        let mut lpb = layers_per_block.to_vec();
        lpb.reverse();

        let mut inj = inject_noise.to_vec();
        inj.reverse();
        let mut upr = upsampler_residual.to_vec();
        upr.reverse();
        let mut upf = upsample_factor.to_vec();
        upf.reverse();

        let conv_in = LtxVideoCausalConv3d::new(
            in_channels,
            boc[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_in"),
        )?;

        let mid_block = LtxVideoMidBlock3d::new(
            boc[0],
            lpb[0],
            0.0,
            resnet_eps,
            is_causal,
            inj[0],
            timestep_conditioning,
            vb.pp("mid_block"),
        )?;

        let mut up_blocks = Vec::new();
        let n = boc.len();
        let mut current_channels = boc[0];

        for i in 0..n {
            let output_channel = boc[i] / upf[i];
            let input_channel = output_channel;

            let ub = LtxVideoUpBlock3d::new_with_stride(
                input_channel,
                output_channel,
                lpb[i + 1],
                0.0,
                resnet_eps,
                strides_rev[i],
                is_causal,
                inj[i + 1],
                timestep_conditioning,
                upr[i],
                upf[i],
                vb.pp(format!("up_blocks.{i}")),
            )?;
            up_blocks.push(ub);
            current_channels = output_channel;
        }

        // norm_out has elementwise_affine=False in Python, so no weights in safetensors.
        // We must create a default LayerNorm (ones weight, zeros bias) if loading fails.
        // Note: Python uses eps=1e-6. We use resnet_eps (which is 1e-6 in config).
        // Python hardcodes eps=1e-8 for norm_out and elementwise_affine=False
        // We use RmsNorm with fallback to ones (no weights in safetensors)
        let norm_out =
            if let Ok(norm) = candle_nn::rms_norm(current_channels, 1e-8, vb.pp("norm_out")) {
                Some(norm)
            } else {
                let ones = Tensor::ones((current_channels,), vb.dtype(), vb.device())?;
                Some(RmsNorm::new(ones, 1e-8))
            };
        let conv_act = Activation::Silu;

        let conv_out_channels = out_channels * patch_size * patch_size;
        let conv_out = LtxVideoCausalConv3d::new(
            current_channels,
            conv_out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            1,
            is_causal,
            vb.pp("conv_out"),
        )?;

        // Timestep conditioning (0.9.5 has timestep_conditioning=true)
        // Global decoder-level time embedder and scale_shift_table
        let (time_embedder, scale_shift_table, timestep_scale_multiplier) = if timestep_conditioning
        {
            // time_embedder output is 256 = 2 * 128 (for shift and scale)
            let emb_dim = current_channels * 2; // 128 * 2 = 256
            let te = CombinedTimestepEmbedder::new(emb_dim, vb.pp("time_embedder")).ok();
            // scale_shift_table: [2, 128]
            let sst = vb.get((2, current_channels), "scale_shift_table").ok();
            let tsm = vb.get((), "timestep_scale_multiplier").ok();
            (te, sst, tsm)
        } else {
            (None, None, None)
        };

        Ok(Self {
            patch_size,
            patch_size_t,
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_act,
            conv_out,
            time_embedder,
            scale_shift_table,
            timestep_scale_multiplier,
            gradient_checkpointing: false,
        })
    }

    pub fn unpatchify(&self, x: &Tensor) -> Result<Tensor> {
        // Python: reshape(batch, -1, p_t, p, p, num_frames, height, width)
        //         permute(0, 1, 5, 2, 6, 4, 7, 3)
        //         flatten(6, 7).flatten(4, 5).flatten(2, 3)
        let (b, c, f, h, w) = x.dims5()?;
        let p = self.patch_size; // 4
        let pt = self.patch_size_t; // 1
        let out_c = c / (pt * p * p); // 48 / 16 = 3
        let x = x.reshape(&[b, out_c, pt, p, p, f, h, w])?;

        // permute(0, 1, 5, 2, 6, 4, 7, 3) -> [B, C, F, pt, H, p_h, W, p_w]
        let x = x.permute(vec![0, 1, 5, 2, 6, 4, 7, 3])?;
        let x = x.contiguous()?;

        // After permute shape: [B, C, F, pt, H, p, W, p]
        // Python flattens: flatten(6, 7).flatten(4, 5).flatten(2, 3)
        // We must do this step-by-step to match Python's memory layout

        // flatten(6, 7): merge dimensions 6 and 7 -> [B, C, F, pt, H, p, W*p]
        let x = x.reshape(&[b, out_c, f, pt, h, p, w * p])?;

        // flatten(4, 5): merge dimensions 4 and 5 -> [B, C, F, pt, H*p, W*p]
        let x = x.reshape(&[b, out_c, f, pt, h * p, w * p])?;

        // flatten(2, 3): merge dimensions 2 and 3 -> [B, C, F*pt, H*p, W*p]
        let x = x.reshape(&[b, out_c, f * pt, h * p, w * p])?;

        Ok(x)
    }

    pub fn forward(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        let model_dtype = self.conv_in.conv2d_slices[0].weight().dtype();
        let z = z.to_dtype(model_dtype)?;
        let temb = match temb {
            Some(t) => Some(t.to_dtype(model_dtype)?),
            None => None,
        };

        let mut h = self.conv_in.forward(&z)?;

        // CRITICAL: Python applies timestep_scale_multiplier at the START of decoder.forward(),
        // BEFORE passing to mid_block and up_blocks. Each block's internal time_embedder
        // then receives the SCALED temb value.
        let temb_scaled =
            if let (Some(tsm), Some(t)) = (&self.timestep_scale_multiplier, temb.as_ref()) {
                let t_flat = t.flatten_all()?;
                Some(t_flat.broadcast_mul(tsm)?)
            } else if let Some(t) = temb.as_ref() {
                Some(t.flatten_all()?)
            } else {
                None
            };
        let temb_for_blocks_ref = temb_scaled.as_ref();

        h = self.mid_block.forward(&h, temb_for_blocks_ref, train)?;

        for ub in self.up_blocks.iter() {
            h = ub.forward(&h, temb_for_blocks_ref, train)?;
        }

        // Apply norm_out only if it exists
        if let Some(ref norm) = self.norm_out {
            h = rmsnorm_channels_first(norm, &h)?;
        }

        // Apply global time_embedder + scale_shift_table if present
        // NOTE: temb_scaled already has timestep_scale_multiplier applied from earlier
        if let (Some(te), Some(sst), Some(temb_s)) =
            (&self.time_embedder, &self.scale_shift_table, &temb_scaled)
        {
            let temb_proj = te.forward(temb_s, h.dtype())?;

            // temb_proj: (B, 256) = (B, 2*128)
            // reshape to (B, 2, 128) and add scale_shift_table (2, 128)
            let batch_size = h.dims5()?.0;
            let c = sst.dims2()?.1; // 128
            let temb_shaped = temb_proj
                .reshape((batch_size, 2, c))?
                .broadcast_add(&sst.unsqueeze(0)?)? // (B, 2, C)
                .unsqueeze(3)? // (B, 2, C, 1)
                .unsqueeze(4)? // (B, 2, C, 1, 1)
                .unsqueeze(5)?; // (B, 2, C, 1, 1, 1)

            // shift = temb_shaped[:, 0], scale = temb_shaped[:, 1]
            let shift = temb_shaped.i((.., 0, .., .., .., ..))?.squeeze(1)?;
            let scale = temb_shaped.i((.., 1, .., .., .., ..))?.squeeze(1)?;

            // Python: h = h * (1 + scale) + shift
            let h_shape = h.shape();
            let scale_b = scale.broadcast_as(h_shape)?;
            let shift_b = shift.broadcast_as(h_shape)?;

            h = h
                .broadcast_mul(&scale_b.affine(1.0, 1.0)?)?
                .broadcast_add(&shift_b)?;
        }

        h = h.apply(&self.conv_act)?;
        h = self.conv_out.forward(&h)?;
        self.unpatchify(&h)
    }
}

#[derive(Debug, Clone)]
pub struct AutoencoderKLLtxVideo {
    pub encoder: LtxVideoEncoder3d,
    pub decoder: LtxVideoDecoder3d,
    pub quant_conv: Option<LtxVideoCausalConv3d>,
    pub post_quant_conv: Option<LtxVideoCausalConv3d>,

    pub latents_mean: Tensor, // (C,)
    pub latents_std: Tensor,  // (C,)

    pub scaling_factor: f64,

    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,

    pub use_slicing: bool,
    pub use_tiling: bool,
    pub use_framewise_encoding: bool,
    pub use_framewise_decoding: bool,

    pub num_sample_frames_batch_size: usize,
    pub num_latent_frames_batch_size: usize,

    pub tile_sample_min_height: usize,
    pub tile_sample_min_width: usize,
    pub tile_sample_min_num_frames: usize,

    pub tile_sample_stride_height: usize,
    pub tile_sample_stride_width: usize,
    pub tile_sample_stride_num_frames: usize,

    pub config: AutoencoderKLLtxVideoConfig,
    pub vae_config: VaeConfig,
}

impl AutoencoderKLLtxVideo {
    pub fn new(config: AutoencoderKLLtxVideoConfig, vb: VarBuilder) -> Result<Self> {
        let ds_types: Vec<DownsampleType> = config
            .downsample_types
            .iter()
            .map(|s| DownsampleType::parse(s))
            .collect();

        let encoder = LtxVideoEncoder3d::new(
            config.in_channels,
            config.latent_channels,
            &config.block_out_channels,
            &config.spatiotemporal_scaling,
            &config.layers_per_block,
            &ds_types,
            config.patch_size,
            config.patch_size_t,
            config.resnet_eps,
            config.is_causal,
            vb.pp("encoder"),
        )?;

        let quant_conv = LtxVideoCausalConv3d::new(
            config.latent_channels * 2,
            config.latent_channels * 2,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            config.is_causal,
            vb.pp("quant_conv"),
        )
        .ok();

        let post_quant_conv = LtxVideoCausalConv3d::new(
            config.latent_channels,
            config.latent_channels,
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            config.is_causal,
            vb.pp("post_quant_conv"),
        )
        .ok();

        let decoder = LtxVideoDecoder3d::new(
            config.latent_channels,
            config.out_channels,
            &config.decoder_block_out_channels,
            &config.decoder_spatiotemporal_scaling,
            &config.decoder_layers_per_block,
            config.patch_size,
            config.patch_size_t,
            config.resnet_eps,
            config.decoder_causal,
            &config.decoder_inject_noise,
            config.timestep_conditioning,
            &config.decoder_upsample_residual,
            &config.decoder_upsample_factor,
            vb.pp("decoder"),
        )?;

        let latents_mean = if vb.contains_tensor("latents_mean") {
            println!("Loading latents_mean from weights");
            vb.get(config.latent_channels, "latents_mean")?
        } else {
            Tensor::new(config.latents_mean.as_slice(), vb.device())?.to_dtype(vb.dtype())?
        };
        let latents_std = if vb.contains_tensor("latents_std") {
            println!("Loading latents_std from weights");
            vb.get(config.latent_channels, "latents_std")?
        } else {
            Tensor::new(config.latents_std.as_slice(), vb.device())?.to_dtype(vb.dtype())?
        };
        let vae_config = VaeConfig {
            scaling_factor: config.scaling_factor as f32,
            timestep_conditioning: config.timestep_conditioning,
        };

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            tile_sample_min_height: 512,
            tile_sample_min_width: 512,
            tile_sample_min_num_frames: 16,
            tile_sample_stride_height: 384,
            tile_sample_stride_width: 384,
            tile_sample_stride_num_frames: 8,
            scaling_factor: config.scaling_factor,
            spatial_compression_ratio: config.spatial_compression_ratio,
            temporal_compression_ratio: config.temporal_compression_ratio,
            use_slicing: false,
            use_tiling: true,
            use_framewise_encoding: false,
            use_framewise_decoding: true,
            num_sample_frames_batch_size: 1,
            num_latent_frames_batch_size: 1,
            config,
            latents_mean,
            latents_std,
            vae_config,
        })
    }

    pub fn enable_tiling(
        &mut self,
        tile_sample_min_height: Option<usize>,
        tile_sample_min_width: Option<usize>,
        tile_sample_min_num_frames: Option<usize>,
        tile_sample_stride_height: Option<usize>,
        tile_sample_stride_width: Option<usize>,
        tile_sample_stride_num_frames: Option<usize>,
    ) {
        self.use_tiling = true;
        if let Some(h) = tile_sample_min_height {
            self.tile_sample_min_height = h;
        }
        if let Some(w) = tile_sample_min_width {
            self.tile_sample_min_width = w;
        }
        if let Some(f) = tile_sample_min_num_frames {
            self.tile_sample_min_num_frames = f;
        }
        if let Some(h) = tile_sample_stride_height {
            self.tile_sample_stride_height = h;
        }
        if let Some(w) = tile_sample_stride_width {
            self.tile_sample_stride_width = w;
        }
        if let Some(f) = tile_sample_stride_num_frames {
            self.tile_sample_stride_num_frames = f;
        }
    }

    pub fn latents_mean(&self) -> &Tensor {
        &self.latents_mean
    }

    pub fn latents_std(&self) -> &Tensor {
        &self.latents_std
    }

    pub fn config(&self) -> &AutoencoderKLLtxVideoConfig {
        &self.config
    }

    pub fn vae_config(&self) -> &VaeConfig {
        &self.vae_config
    }

    fn split_batch_5d(x: &Tensor) -> Result<Vec<Tensor>> {
        let b = x.dims5()?.0;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            out.push(x.i((i..(i + 1), .., .., .., ..))?);
        }
        Ok(out)
    }

    /// Blend по ширине W (dim=4): b[..., :blend] = lerp(a[..., -blend:], b[..., :blend], w)
    fn blend_h(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python: for x in range(blend): b[..., x] = a[..., -blend+x]*(1-x/blend) + b[..., x]*(x/blend) [file:1]
        let blend = blend_extent.min(a.dims5()?.4).min(b.dims5()?.4);
        if blend == 0 {
            return Ok(b.clone());
        }

        // w: (blend,) from 0..blend-1 divided by blend
        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, 1, 1, blend))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        // b_head: первые blend столбцов, b_tail: остаток
        let b_head = b.i((.., .., .., .., 0..blend))?;
        let b_tail = b.i((.., .., .., .., blend..))?;

        // a_tail: последние blend столбцов
        let aw = a.dims5()?.4;
        let a_tail = a.i((.., .., .., .., (aw - blend)..aw))?;

        // mixed = a_tail*(1-w) + b_head*w
        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 4)
    }

    /// Blend по высоте H (dim=3)
    fn blend_v(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python: for y in range(blend): b[..., y, :] = a[..., -blend+y, :]*(1-y/blend) + b[..., y, :]*(y/blend) [file:1]
        let blend = blend_extent.min(a.dims5()?.3).min(b.dims5()?.3);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, 1, blend, 1))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., .., 0..blend, ..))?;
        let b_tail = b.i((.., .., .., blend.., ..))?;

        let ah = a.dims5()?.3;
        let a_tail = a.i((.., .., .., (ah - blend)..ah, ..))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 3)
    }

    /// Blend по времени T (dim=2)
    fn blend_t(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python: for x in range(blend): b[..., x, :, :] = a[..., -blend+x, :, :]*(1-x/blend) + b[..., x, :, :]*(x/blend) [file:1]
        let blend = blend_extent.min(a.dims5()?.2).min(b.dims5()?.2);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, blend, 1, 1))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., 0..blend, .., ..))?;
        let b_tail = b.i((.., .., blend.., .., ..))?;

        let at = a.dims5()?.2;
        let a_tail = a.i((.., .., (at - blend)..at, .., ..))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 2)
    }

    fn split_batch_2d(x: &Tensor) -> Result<Vec<Tensor>> {
        let (b, _d) = x.dims2()?;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            out.push(x.i((i..(i + 1), ..))?);
        }
        Ok(out)
    }

    fn encode_z(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let tile_sample_min_num_frames = self.tile_sample_min_num_frames;
        if self.use_framewise_encoding && x.dims5()?.2 > tile_sample_min_num_frames {
            return self.temporal_tiled_encode(x, train);
        }

        if self.use_tiling
            && (x.dims5()?.3 > self.tile_sample_min_height
                || x.dims5()?.4 > self.tile_sample_min_width)
        {
            return self.tiled_encode(x, train);
        }

        let mut h = self.encoder.forward(x, train)?;
        if let Some(ref qc) = self.quant_conv {
            h = qc.forward(&h)?;
        }
        Ok(h)
    }

    fn decode_z(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        // Python LTX VAE _decode does NOT use post_quant_conv or latents_mean/std
        // It directly calls decoder(z, temb)

        // Convert inputs to model dtype (BF16) if needed
        let model_dtype = self.decoder.conv_in.conv2d_slices[0].weight().dtype();
        let z = z.to_dtype(model_dtype)?;
        let temb_converted = match temb {
            Some(t) => Some(t.to_dtype(model_dtype)?),
            None => None,
        };

        let (_b, _c, t, h, w) = z.dims5()?;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;
        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;

        if self.use_framewise_decoding && t > tile_latent_min_t {
            let out = self.temporal_tiled_decode(&z, temb_converted.as_ref(), train)?;
            return Ok(out);
        }

        if self.use_tiling && (w > tile_latent_min_w || h > tile_latent_min_h) {
            let out = self.tiled_decode(&z, temb_converted.as_ref(), train)?;
            return Ok(out);
        }

        self.decoder.forward(&z, temb_converted.as_ref(), train)
    }

    // ===== public API =====

    pub fn encode(
        &self,
        x: &Tensor,
        return_dict: bool,
        train: bool,
    ) -> Result<(Option<AutoencoderKLOutput>, DiagonalGaussianDistribution)> {
        // python: if useslicing and batch>1: encode each slice then cat [file:1]
        let h = if self.use_slicing && x.dims5()?.0 > 1 {
            let xs = Self::split_batch_5d(x)?;
            let mut encs = Vec::with_capacity(xs.len());
            for xs_i in xs.iter() {
                encs.push(self.encode_z(xs_i, train)?);
            }
            cat_dim(&encs, 0)?
        } else {
            self.encode_z(x, train)?
        };

        let posterior = DiagonalGaussianDistribution::new(&h)?;
        if return_dict {
            Ok((
                Some(AutoencoderKLOutput {
                    latent_dist: posterior.clone(),
                }),
                posterior,
            ))
        } else {
            Ok((None, posterior))
        }
    }

    pub fn decode(
        &self,
        z: &Tensor,
        temb: Option<&Tensor>,
        return_dict: bool,
        train: bool,
    ) -> Result<(Option<DecoderOutput>, Tensor)> {
        // python: if useslicing and batch>1: decode each slice then cat [file:1]
        let decoded = if self.use_slicing && z.dims5()?.0 > 1 {
            let zs = Self::split_batch_5d(z)?;
            let ts = match temb {
                None => None,
                Some(t) => Some(Self::split_batch_2d(t)?),
            };

            let mut outs = Vec::with_capacity(zs.len());
            for (idx, z_i) in zs.iter().enumerate() {
                let t_i = ts.as_ref().map(|v| v[idx].as_ref());
                outs.push(self.decode_z(z_i, t_i, train)?);
            }
            cat_dim(&outs, 0)?
        } else {
            self.decode_z(z, temb, train)?
        };

        if return_dict {
            Ok((
                Some(DecoderOutput {
                    sample: decoded.clone(),
                }),
                decoded,
            ))
        } else {
            Ok((None, decoded))
        }
    }

    /// python forward(sample, temb=None, sample_posterior=False, return_dict=True) [file:1]
    pub fn forward(
        &self,
        sample: &Tensor,
        temb: Option<&Tensor>,
        sample_posterior: bool,
        return_dict: bool,
        train: bool,
    ) -> Result<(Option<DecoderOutput>, Tensor)> {
        let (_out, posterior) = self.encode(sample, true, train)?;
        let z = if sample_posterior {
            posterior.sample()?
        } else {
            posterior.mode()?
        };
        self.decode(&z, temb, return_dict, train)
    }

    // ===== spatial tiling =====

    fn tiled_encode(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // python tiled_encode: loops in sample-space, blends in latent-space [file:1]
        let (_b, _c, _t, height, width) = x.dims5()?;

        let latent_height = height / self.spatial_compression_ratio;
        let latent_width = width / self.spatial_compression_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_stride_h = self.tile_sample_stride_height / self.spatial_compression_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.spatial_compression_ratio;

        let blend_h = tile_latent_min_h.saturating_sub(tile_latent_stride_h);
        let blend_w = tile_latent_min_w.saturating_sub(tile_latent_stride_w);

        // rows[i][j] = encoder(tile)
        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(self.tile_sample_stride_height) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(self.tile_sample_stride_width) {
                let h_end = (i + self.tile_sample_min_height).min(height);
                let w_end = (j + self.tile_sample_min_width).min(width);
                let tile = x.i((.., .., .., i..h_end, j..w_end))?;
                let mut enc = self.encoder.forward(&tile, train)?;
                if let Some(ref qc) = self.quant_conv {
                    enc = qc.forward(&enc)?;
                }
                row.push(enc);
            }
            rows.push(row);
        }

        let mut prev_row_blended: Vec<Tensor> = Vec::new();
        let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
            let mut curr_row_blended: Vec<Tensor> = Vec::with_capacity(row.len());
            for (cj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();

                if ri > 0 {
                    let above = &prev_row_blended[cj];
                    tile = self.blend_v(above, &tile, blend_h)?;
                }
                if cj > 0 {
                    let left = &curr_row_blended[cj - 1];
                    tile = self.blend_h(left, &tile, blend_w)?;
                }

                // Store fully blended tile for future neighbors
                curr_row_blended.push(tile.clone());

                // Keep only the non-overlapping part for concatenation
                let h_slice = tile_latent_stride_h.min(tile.dim(3)?);
                let w_slice = tile_latent_stride_w.min(tile.dim(4)?);
                let sliced_tile = tile.i((.., .., .., 0..h_slice, 0..w_slice))?;
                result_row.push(sliced_tile);
            }
            result_rows.push(cat_dim(&result_row, 4)?);
            prev_row_blended = curr_row_blended;
        }

        let enc = cat_dim(&result_rows, 3)?;
        enc.i((.., .., .., 0..latent_height, 0..latent_width))
    }

    fn tiled_decode(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        // python tiled_decode: loops in latent-space, blends/crops in sample-space [file:1]
        let (_b, _c, _t, height, width) = z.dims5()?;

        let sample_height = height * self.spatial_compression_ratio;
        let sample_width = width * self.spatial_compression_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_stride_h = self.tile_sample_stride_height / self.spatial_compression_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.spatial_compression_ratio;

        let blend_h = self
            .tile_sample_min_height
            .saturating_sub(self.tile_sample_stride_height);
        let blend_w = self
            .tile_sample_min_width
            .saturating_sub(self.tile_sample_stride_width);

        // rows[i][j] = decoder(tile)
        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(tile_latent_stride_h) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(tile_latent_stride_w) {
                let h_end = (i + tile_latent_min_h).min(height);
                let w_end = (j + tile_latent_min_w).min(width);
                let tile = z.i((.., .., .., i..h_end, j..w_end))?;
                let dec = self.decoder.forward(&tile, temb, train)?;
                row.push(dec);
            }
            rows.push(row);
        }

        let mut prev_row_blended: Vec<Tensor> = Vec::new();
        let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
            let mut curr_row_blended: Vec<Tensor> = Vec::with_capacity(row.len());
            for (cj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();

                if ri > 0 {
                    let above = &prev_row_blended[cj];
                    tile = self.blend_v(above, &tile, blend_h)?;
                }
                if cj > 0 {
                    let left = &curr_row_blended[cj - 1];
                    tile = self.blend_h(left, &tile, blend_w)?;
                }

                // Store fully blended tile for future neighbors
                curr_row_blended.push(tile.clone());

                let h_slice = self.tile_sample_stride_height.min(tile.dim(3)?);
                let w_slice = self.tile_sample_stride_width.min(tile.dim(4)?);
                let sliced_tile = tile.i((.., .., .., 0..h_slice, 0..w_slice))?;
                result_row.push(sliced_tile);
            }
            result_rows.push(cat_dim(&result_row, 4)?);
            prev_row_blended = curr_row_blended;
        }

        let dec = cat_dim(&result_rows, 3)?;
        dec.i((.., .., .., 0..sample_height, 0..sample_width))
    }

    // ===== temporal tiling =====

    fn temporal_tiled_encode(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // python temporal_tiled_encode (stride in sample frames), blends in latent time [file:1]
        let (_b, _c, num_frames, _h, _w) = x.dims5()?;

        let latent_num_frames = (num_frames - 1) / self.temporal_compression_ratio + 1; // python formula [file:1]

        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;
        let tile_latent_stride_t =
            self.tile_sample_stride_num_frames / self.temporal_compression_ratio;
        let blend_t = tile_latent_min_t.saturating_sub(tile_latent_stride_t);

        let mut row: Vec<Tensor> = Vec::new();
        for i in (0..num_frames).step_by(self.tile_sample_stride_num_frames) {
            let t_end = (i + self.tile_sample_min_num_frames + 1).min(num_frames);
            let tile = x.i((.., .., i..t_end, .., ..))?;

            let tile = if self.use_tiling
                && (tile.dims5()?.3 > self.tile_sample_min_height
                    || tile.dims5()?.4 > self.tile_sample_min_width)
            {
                self.tiled_encode(&tile, train)?
            } else {
                let mut h = self.encoder.forward(&tile, train)?;
                if let Some(ref qc) = self.quant_conv {
                    h = qc.forward(&h)?;
                }
                h
            };

            // python: if i == 0: tile = tile[:, :, 1:] [file:1]
            let tile = if i == 0 {
                tile.i((.., .., 1.., .., ..))?
            } else {
                tile
            };
            row.push(tile);
        }

        // Python logic:
        // for i, tile in enumerate(row):
        //     if i > 0:
        //         tile = self.blend_t(row[i - 1], tile, blend_num_frames)
        //         result_row.append(tile[:, :, :stride, :, :])  # Take FIRST stride frames
        //     else:
        //         result_row.append(tile[:, :, :stride+1, :, :])  # First tile: stride+1 frames
        let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
        for (idx, tile) in row.iter().enumerate() {
            let tile = if idx > 0 {
                let blended = self.blend_t(&row[idx - 1], tile, blend_t)?;
                // Take FIRST stride frames (not last!)
                let end = tile_latent_stride_t.min(blended.dim(2)?);
                blended.i((.., .., 0..end, .., ..))?
            } else {
                // First tile: keep stride + 1 frames
                let end = (tile_latent_stride_t + 1).min(tile.dim(2)?);
                tile.i((.., .., 0..end, .., ..))?
            };
            result_row.push(tile);
        }

        let enc = cat_dim(&result_row, 2)?;
        enc.i((.., .., 0..latent_num_frames, .., ..))
    }

    fn temporal_tiled_decode(
        &self,
        z: &Tensor,
        temb: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        // python temporal_tiled_decode: stride in latent time, blends in sample time [file:1]
        let (_b, _c, num_frames, _h, _w) = z.dims5()?;

        let num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1; // python formula [file:1]

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;
        let tile_latent_stride_t =
            self.tile_sample_stride_num_frames / self.temporal_compression_ratio;

        // Python: blend_num_frames = tile_sample_min - tile_sample_stride = 16 - 8 = 8
        let blend_t_sample = self
            .tile_sample_min_num_frames
            .saturating_sub(self.tile_sample_stride_num_frames);

        let mut row: Vec<Tensor> = Vec::new();
        for (loop_idx, i) in (0..num_frames).step_by(tile_latent_stride_t).enumerate() {
            let t_end = (i + tile_latent_min_t + 1).min(num_frames);
            let tile = z.i((.., .., i..t_end, .., ..))?;

            let decoded = if self.use_tiling
                && (tile.dims5()?.3 > tile_latent_min_h || tile.dims5()?.4 > tile_latent_min_w)
            {
                self.tiled_decode(&tile, temb, train)?
            } else {
                self.decoder.forward(&tile, temb, train)?
            };

            // Python: if i > 0: decoded = decoded[:, :, :-1, :, :]
            // Remove last sample frame from all tiles except first
            let decoded = if loop_idx > 0 {
                let t = decoded.dim(2)?;
                if t > 1 {
                    decoded.i((.., .., 0..(t - 1), .., ..))?
                } else {
                    decoded
                }
            } else {
                decoded
            };

            row.push(decoded);
        }

        // Python logic:
        // for i, tile in enumerate(row):
        //     if i > 0:
        //         tile = self.blend_t(row[i - 1], tile, blend_num_frames)
        //         tile = tile[:, :, :stride, :, :]  # Take FIRST stride frames
        //     else:
        //         tile = tile[:, :, :stride+1, :, :]  # First tile: stride+1 frames
        let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
        for (idx, tile) in row.iter().enumerate() {
            let tile = if idx > 0 {
                let blended = self.blend_t(&row[idx - 1], tile, blend_t_sample)?;
                // Take FIRST stride frames (not last!)
                let end = self.tile_sample_stride_num_frames.min(blended.dim(2)?);
                blended.i((.., .., 0..end, .., ..))?
            } else {
                // First tile: keep stride + 1 frames
                let end = (self.tile_sample_stride_num_frames + 1).min(tile.dim(2)?);
                tile.i((.., .., 0..end, .., ..))?
            };
            result_row.push(tile);
        }

        let dec = cat_dim(&result_row, 2)?;
        dec.i((.., .., 0..num_sample_frames, .., ..))
    }
}

impl VaeLtxVideo for AutoencoderKLLtxVideo {
    fn dtype(&self) -> DType {
        // Return actual weight dtype (e.g., DType::BF16)
        self.decoder.conv_in.conv2d_slices[0].weight().dtype()
    }
    fn spatial_compression_ratio(&self) -> usize {
        self.config.spatial_compression_ratio
    }
    fn temporal_compression_ratio(&self) -> usize {
        self.config.temporal_compression_ratio
    }
    fn config(&self) -> &VaeConfig {
        &self.vae_config
    }

    fn latents_mean(&self) -> &Tensor {
        &self.latents_mean
    }
    fn latents_std(&self) -> &Tensor {
        &self.latents_std
    }

    fn decode(&self, latents: &Tensor, timestep: Option<&Tensor>) -> Result<Tensor> {
        let (_, decoded) = self.decode(latents, timestep, false, false)?;
        Ok(decoded)
    }
}
