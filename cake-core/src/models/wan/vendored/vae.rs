// Wan 3D Causal VAE Decoder with chunked temporal decoding.
//
// Architecture: CausalConv3d-based decoder with temporal+spatial upsampling.
// Compression: 4x temporal, 8x spatial (stride 4,8,8).
// Config: base_dim=96, z_dim=16, dim_mult=[1,2,4,4], temporal_downsample=[F,T,T].
//
// Decoding processes ONE latent frame at a time, using a feature cache
// (Vec<Option<Tensor>>) so each CausalConv3d can access the previous CACHE_T=2
// frames from its input as causal padding instead of zeros.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use super::config::WanVaeConfig;

/// Number of frames each CausalConv3d caches from its input for the next chunk.
const CACHE_T: usize = 2;

/// Feature cache: one slot per CausalConv3d in the decoder.
/// Each slot stores the last CACHE_T frames of that conv's input.
pub type FeatureCache = Vec<Option<Tensor>>;

/// L2-normalize RMSNorm used by Wan VAE.
/// Computes: F.normalize(x, dim=1) * scale * gamma + bias
/// where scale = sqrt(dim).
#[derive(Debug, Clone)]
struct WanVaeRmsNorm {
    gamma: Tensor,
    bias: Tensor,
    scale: f64,
}

impl WanVaeRmsNorm {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Try flat [dim] first (original format), then squeeze from [dim, 1, 1, 1] (diffusers)
        let gamma = match vb.get(dim, "gamma") {
            Ok(g) => g,
            Err(_) => vb.get((dim, 1, 1, 1), "gamma")?.flatten_all()?,
        };
        // Diffusers format may not have beta — default to zeros
        let bias = match vb.get(dim, "beta") {
            Ok(b) => b,
            Err(_) => Tensor::zeros(dim, gamma.dtype(), gamma.device())?,
        };
        Ok(Self {
            gamma,
            bias,
            scale: (dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // L2-normalize along channel dim (dim=1 for [B, C, ...])
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let x_norm = x.broadcast_div(&(norm + 1e-8)?)?;
        // Scale and apply affine
        let gamma = self.gamma.reshape((1, self.gamma.elem_count(), 1, 1, 1))?;
        let bias = self.bias.reshape((1, self.bias.elem_count(), 1, 1, 1))?;
        Ok((x_norm * self.scale)?.broadcast_mul(&gamma)?.broadcast_add(&bias)?)
    }
}

/// Causal 3D convolution: pads only past frames (causal in time).
/// When cache is provided, uses cached frames as temporal padding instead of zeros.
#[derive(Debug, Clone)]
struct CausalConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    kernel_t: usize,
}

impl CausalConv3d {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_ch, in_ch, kernel.0, kernel.1, kernel.2),
            "weight",
        )?;
        let bias = vb.get(out_ch, "bias").ok();
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            kernel_t: kernel.0,
        })
    }

    /// Forward with optional feature cache.
    ///
    /// If `cache_x` is Some, prepend the cached frames instead of zero-padding.
    /// Returns the output tensor.
    ///
    /// The caller is responsible for saving/restoring cache entries.
    fn forward_cached(&self, x: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        let causal_pad = 2 * self.padding.0;

        let x = if causal_pad > 0 {
            match cache_x {
                Some(cached) => {
                    // Use cached frames as causal padding.
                    // If cache has fewer frames than causal_pad, pad remaining with first cached frame.
                    let cached_t = cached.dims()[2];
                    if cached_t >= causal_pad {
                        Tensor::cat(&[&cached.narrow(2, cached_t - causal_pad, causal_pad)?, x], 2)?
                    } else {
                        // Pad: replicate first available frame to fill the gap
                        let first = cached.narrow(2, 0, 1)?;
                        let extra = first.repeat((1, 1, causal_pad - cached_t, 1, 1))?;
                        Tensor::cat(&[&extra, cached, x], 2)?
                    }
                }
                None => {
                    // No cache: zero-pad temporally (matching Python's F.pad)
                    x.pad_with_zeros(2, causal_pad, 0)?
                }
            }
        } else {
            x.clone()
        };

        self.conv3d_impl(&x)
    }

    /// Raw conv3d implementation (no padding — assumes input already padded).
    fn conv3d_impl(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, f, h, w) = x.dims5()?;
        let (c_out, _, kt, kh, kw) = self.weight.dims5()?;

        if kt == 1 && self.stride.0 == 1 {
            // Temporal kernel=1: apply 2D conv per frame
            let w2d = self.weight.squeeze(2)?; // [c_out, c_in, kh, kw]
            let x_flat = x.permute((0, 2, 1, 3, 4))?.reshape((b * f, c_in, h, w))?;
            let cfg = candle_nn::Conv2dConfig {
                padding: self.padding.1,
                stride: self.stride.1,
                ..Default::default()
            };
            let y = x_flat.conv2d(&w2d, cfg.padding, cfg.stride, 1, 1)?;
            let (_, c_o, h_o, w_o) = y.dims4()?;
            let y = y.reshape((b, f, c_o, h_o, w_o))?.permute((0, 2, 1, 3, 4))?.contiguous()?;

            if let Some(ref bias) = self.bias {
                let bias = bias.reshape((1, c_out, 1, 1, 1))?;
                Ok(y.broadcast_add(&bias)?)
            } else {
                Ok(y)
            }
        } else {
            // General 3D conv: loop over output temporal positions.
            // IMPORTANT: Apply spatial padding BEFORE temporal slicing to match
            // Python's F.pad() which pads all dims at once before Conv3d.
            if f < kt {
                anyhow::bail!("conv3d_impl: temporal dim f={f} < kernel kt={kt} (stride={}, padding={:?})", self.stride.0, self.padding);
            }
            let sp = self.padding.1; // spatial padding
            let x = if sp > 0 {
                // Pad spatially BEFORE temporal slicing (matches Python's F.pad)
                x.pad_with_zeros(3, sp, sp)?  // pad H
                 .pad_with_zeros(4, sp, sp)?  // pad W
            } else {
                x.clone()
            };

            let (b, c_in, f, h, w) = x.dims5()?;
            let f_out = (f - kt) / self.stride.0 + 1;
            let w2d = self.weight.reshape((c_out, c_in * kt, kh, kw))?;

            let mut frames_out = Vec::with_capacity(f_out);
            for t in 0..f_out {
                let t_start = t * self.stride.0;
                let temporal_slice = x.narrow(2, t_start, kt)?;
                let temporal_flat = temporal_slice.reshape((b, c_in * kt, h, w))?;
                // NO spatial padding in conv2d — already padded above
                let y_frame = temporal_flat.conv2d(
                    &w2d, 0, self.stride.1, 1, 1,
                )?;
                frames_out.push(y_frame);
            }
            let y = Tensor::stack(&frames_out.iter().collect::<Vec<_>>(), 2)?;

            if let Some(ref bias) = self.bias {
                let bias = bias.reshape((1, c_out, 1, 1, 1))?;
                Ok(y.broadcast_add(&bias)?)
            } else {
                Ok(y)
            }
        }
    }
}

/// Residual block with two CausalConv3d layers.
/// Uses 2 cache slots: one for conv1, one for conv2.
/// If there's a shortcut conv (1x1x1, no temporal padding), it doesn't need a cache slot.
#[derive(Debug, Clone)]
struct WanResBlock {
    norm1: WanVaeRmsNorm,
    conv1: CausalConv3d,
    norm2: WanVaeRmsNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl WanResBlock {
    fn load(in_ch: usize, out_ch: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = WanVaeRmsNorm::load(in_ch, vb.pp("norm1"))?;
        let conv1 = CausalConv3d::load(in_ch, out_ch, (3, 3, 3), (1, 1, 1), (1, 1, 1), vb.pp("conv1"))?;
        let norm2 = WanVaeRmsNorm::load(out_ch, vb.pp("norm2"))?;
        let conv2 = CausalConv3d::load(out_ch, out_ch, (3, 3, 3), (1, 1, 1), (1, 1, 1), vb.pp("conv2"))?;

        let shortcut = if in_ch != out_ch {
            Some(CausalConv3d::load(in_ch, out_ch, (1, 1, 1), (1, 1, 1), (0, 0, 0), vb.pp("shortcut"))?)
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, shortcut })
    }

    /// Number of cache slots this block uses (2 for conv1 + conv2).
    fn num_cache_slots(&self) -> usize {
        2
    }

    /// Forward with feature cache. `cache_offset` is the index into the cache vec
    /// for this block's first slot.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut FeatureCache,
        cache_offset: usize,
    ) -> Result<Tensor> {
        let residual = if let Some(ref sc) = self.shortcut {
            sc.forward_cached(x, None)?
        } else {
            x.clone()
        };

        let h = self.norm1.forward(x)?;
        let h = h.silu()?;

        // conv1: Python order: build new cache from input, then use OLD cache for padding
        let h = {
            let new_cache = build_cache_entry(&h, &cache[cache_offset])?;
            let old_cache = cache[cache_offset].take();
            cache[cache_offset] = Some(new_cache);
            self.conv1.forward_cached(&h, old_cache.as_ref())?
        };

        let h = self.norm2.forward(&h)?;
        let h = h.silu()?;

        // conv2: same pattern
        let h = {
            let new_cache = build_cache_entry(&h, &cache[cache_offset + 1])?;
            let old_cache = cache[cache_offset + 1].take();
            cache[cache_offset + 1] = Some(new_cache);
            self.conv2.forward_cached(&h, old_cache.as_ref())?
        };

        Ok((h + residual)?)
    }
}

/// Temporal+spatial upsampling with feature cache.
///
/// Python Resample behavior:
/// - time_conv is a CausalConv3d that doubles channels: dim -> 2*dim
/// - After time_conv, reshape (b,2,c,t,h,w) then interleave -> (b,c,2t,h,w)
/// - BUT: on first_chunk (cache=None for time_conv), skip temporal upsample
///   (just take first channel group, no doubling of frames)
/// - On subsequent chunks, do the full temporal interleave
/// - Then spatial: nearest 2x + conv2d
#[derive(Debug, Clone)]
struct Upsample3d {
    time_conv: CausalConv3d,
    spatial_conv: Conv2d,
    dim: usize,
}

impl Upsample3d {
    fn load(dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let time_conv = CausalConv3d::load(
            dim, dim * 2, (3, 1, 1), (1, 1, 1), (1, 0, 0),
            vb.pp("time_conv"),
        )?;
        let spatial_conv = candle_nn::conv2d(
            dim, out_dim,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("spatial_conv"),
        )?;
        Ok(Self { time_conv, spatial_conv, dim })
    }

    /// Number of cache slots: 1 for time_conv.
    fn num_cache_slots(&self) -> usize {
        1
    }

    /// Forward with cache. Returns output tensor.
    ///
    /// On first chunk (cache slot is None): time_conv produces 2C channels from 1 frame,
    /// but we DON'T do temporal upsample — just take the first C channels (1 frame out).
    /// On subsequent chunks: full temporal interleave (1 input frame -> 2 output frames).
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: &mut FeatureCache,
        cache_offset: usize,
    ) -> Result<Tensor> {
        let (b, c, _f, h, w) = x.dims5()?;

        // time_conv with cache
        let is_first = cache[cache_offset].is_none();

        let x = if is_first {
            // First chunk: Python skips time_conv entirely, just marks cache as "Rep"
            // and goes straight to spatial upsample with original input
            cache[cache_offset] = Some(Tensor::zeros(
                (1,), candle_core::DType::F32, x.device()
            )?); // placeholder marker (like Python's "Rep" string)
            x.clone()
        } else {
            // Subsequent chunks: run time_conv with cache, then temporal interleave
            let is_rep = cache[cache_offset].as_ref()
                .map(|t| t.dims().len() == 1) // our "Rep" marker is 1D
                .unwrap_or(false);

            // Build proper cache for time_conv
            let new_cache = build_cache_entry(x, &if is_rep { None } else { cache[cache_offset].clone() })?;
            let old_cache = if is_rep {
                None // "Rep" means run time_conv without cache (fresh)
            } else {
                cache[cache_offset].take()
            };
            cache[cache_offset] = Some(new_cache);
            let x = self.time_conv.forward_cached(x, old_cache.as_ref())?;

            // Temporal interleave: [B, 2*C, F, H, W] → [B, C, 2*F, H, W]
            let (b, _c2, f, h, w) = x.dims5()?;
            let x = x.reshape((b, 2, c, f, h, w))?;
            let x = x.permute((0, 2, 3, 1, 4, 5))?;
            x.reshape((b, c, f * 2, h, w))?
        };

        // Spatial upsample: nearest 2x
        let (b, c, f, h, w) = x.dims5()?;
        let x = x.reshape((b * c, f, h, w))?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        let x = x.reshape((b, c, f, h * 2, w * 2))?;

        // Apply 2D conv per frame
        let (b, c_in, f, h, w) = x.dims5()?;
        let x = x.permute((0, 2, 1, 3, 4))?.reshape((b * f, c_in, h, w))?;
        let x = self.spatial_conv.forward(&x)?;
        let c_out = x.dims()[1];
        let x = x.reshape((b, f, c_out, h, w))?.permute((0, 2, 1, 3, 4))?;

        Ok(x)
    }
}

/// Spatial-only upsampling (2x nearest + conv2d). No cache needed.
#[derive(Debug, Clone)]
struct Upsample2d {
    conv: Conv2d,
}

impl Upsample2d {
    fn load(dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv = candle_nn::conv2d(
            dim, out_dim,
            3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c_in, f, h, w) = x.dims5()?;
        let x = x.permute((0, 2, 1, 3, 4))?.reshape((b * f, _c_in, h, w))?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        let x = self.conv.forward(&x)?;
        let c_out = x.dims()[1];
        let x = x.reshape((b, f, c_out, h * 2, w * 2))?.permute((0, 2, 1, 3, 4))?;
        Ok(x)
    }
}

/// Mid-block self-attention (per-frame 2D attention). No temporal cache needed.
#[derive(Debug, Clone)]
struct WanVaeAttention {
    norm: WanVaeRmsNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
    dim: usize,
}

impl WanVaeAttention {
    fn load(dim: usize, vb: candle_nn::VarBuilder) -> Result<Self> {
        let norm = WanVaeRmsNorm::load(dim, vb.pp("norm"))?;
        let to_qkv = candle_nn::conv2d(dim, dim * 3, 1, Conv2dConfig::default(), vb.pp("to_qkv"))?;
        let proj = candle_nn::conv2d(dim, dim, 1, Conv2dConfig::default(), vb.pp("proj"))?;
        Ok(Self { norm, to_qkv, proj, dim })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, c, f, h, w) = x.dims5()?;

        let x = self.norm.forward(x)?;

        let x = x.permute((0, 2, 1, 3, 4))?.reshape((b * f, c, h, w))?;
        let qkv = self.to_qkv.forward(&x)?; // [B*F, 3*C, H, W]
        // Python: reshape(b*t, 1, c*3, hw).permute(0,1,3,2).chunk(3, dim=-1)
        // → q,k,v each (b*t, 1, hw, c) — attention over spatial positions
        let hw = h * w;
        let qkv = qkv.reshape((b * f, 1, c * 3, hw))?
            .permute((0, 1, 3, 2))?; // [B*F, 1, HW, 3*C]
        let q = qkv.narrow(3, 0, c)?;     // [B*F, 1, HW, C]
        let k = qkv.narrow(3, c, c)?;     // [B*F, 1, HW, C]
        let v = qkv.narrow(3, 2 * c, c)?; // [B*F, 1, HW, C]

        // Scaled dot-product attention over spatial positions
        let scale = (c as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?; // [B*F, 1, HW, HW]
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [B*F, 1, HW, C]

        // Back to spatial layout
        let out = out.squeeze(1)?.permute((0, 2, 1))?.reshape((b * f, c, h, w))?;
        let out = self.proj.forward(&out)?;
        let out = out.reshape((b, f, c, h, w))?.permute((0, 2, 1, 3, 4))?;

        Ok((out + residual)?)
    }
}

#[derive(Debug, Clone)]
enum UpBlock {
    Spatial(Upsample2d),
    SpatioTemporal(Upsample3d),
}

impl UpBlock {
    fn num_cache_slots(&self) -> usize {
        match self {
            UpBlock::Spatial(_) => 0,
            UpBlock::SpatioTemporal(u) => u.num_cache_slots(),
        }
    }
}

/// Wan 3D VAE Decoder with chunked temporal decoding support.
#[derive(Debug)]
pub struct WanVaeDecoder {
    conv_in: CausalConv3d,
    mid_block_0: WanResBlock,
    mid_attn: WanVaeAttention,
    mid_block_2: WanResBlock,
    // Up blocks: list of (Vec<ResBlock>, Option<Upsample>)
    up_blocks: Vec<(Vec<WanResBlock>, Option<UpBlock>)>,
    norm_out: WanVaeRmsNorm,
    conv_out: CausalConv3d,
    /// Total number of cache slots needed for the entire decoder.
    total_cache_slots: usize,
}

/// Build a cache entry from the current tensor `x` and the previous cache.
/// Matches Python: `cache_x = x[:,:,-CACHE_T:]`, then if short, prepend last frame from prev cache.
fn build_cache_entry(x: &Tensor, prev_cache: &Option<Tensor>) -> Result<Tensor> {
    let t = x.dims()[2];
    let cache_x = x.narrow(2, t.saturating_sub(CACHE_T), t.min(CACHE_T))?.contiguous()?;

    if cache_x.dims()[2] < CACHE_T {
        if let Some(ref prev) = prev_cache {
            let prev_t = prev.dims()[2];
            let last_prev = prev.narrow(2, prev_t - 1, 1)?;
            Ok(Tensor::cat(&[&last_prev, &cache_x], 2)?)
        } else {
            // No previous cache — replicate with zeros (first chunk)
            let zeros = Tensor::zeros_like(&cache_x)?;
            Ok(Tensor::cat(&[&zeros, &cache_x], 2)?)
        }
    } else {
        Ok(cache_x)
    }
}

impl WanVaeDecoder {
    pub fn load(vb: VarBuilder, cfg: &WanVaeConfig) -> Result<Self> {
        let channels: Vec<usize> = cfg.dim_mult.iter().map(|m| cfg.base_dim * m).collect();
        let last_ch = *channels.last().unwrap();

        // conv_in: z_dim -> last_channel
        let conv_in = CausalConv3d::load(
            cfg.z_dim, last_ch, (3, 3, 3), (1, 1, 1), (1, 1, 1),
            vb.pp("decoder").pp("conv1"),
        )?;

        // Mid block: ResBlock + Attention + ResBlock
        let mid_vb = vb.pp("decoder").pp("middle");
        let mid_block_0 = WanResBlock::load(last_ch, last_ch, mid_vb.pp("0"))?;
        let mid_attn = WanVaeAttention::load(last_ch, mid_vb.pp("1"))?;
        let mid_block_2 = WanResBlock::load(last_ch, last_ch, mid_vb.pp("2"))?;

        // Up blocks (reversed channel order)
        let up_vb = vb.pp("decoder").pp("upsamples");
        let mut up_blocks = Vec::new();
        let num_stages = channels.len();
        let temporal_up: Vec<bool> = {
            let mut t = cfg.temporal_downsample.clone();
            t.reverse();
            t
        };

        let mut prev_upsample_out = last_ch;
        for stage in 0..num_stages {
            let in_ch = prev_upsample_out;
            let out_ch = channels[num_stages - 1 - stage];

            let stage_vb = up_vb.pp(stage);
            let num_blocks = cfg.num_res_blocks + 1;
            let mut res_blocks = Vec::new();
            for b in 0..num_blocks {
                let block_in = if b == 0 { in_ch } else { out_ch };
                res_blocks.push(WanResBlock::load(block_in, out_ch, stage_vb.pp(format!("block.{b}")))?);
            }

            let upsample = if stage < num_stages - 1 {
                let upsample_out = out_ch / 2;
                if stage < temporal_up.len() && temporal_up[stage] {
                    Some(UpBlock::SpatioTemporal(
                        Upsample3d::load(out_ch, upsample_out, stage_vb.pp("upsample"))?,
                    ))
                } else {
                    Some(UpBlock::Spatial(
                        Upsample2d::load(out_ch, upsample_out, stage_vb.pp("upsample"))?,
                    ))
                }
            } else {
                None
            };

            prev_upsample_out = if stage < num_stages - 1 { out_ch / 2 } else { out_ch };
            up_blocks.push((res_blocks, upsample));
        }

        let norm_out = WanVaeRmsNorm::load(channels[0], vb.pp("decoder").pp("head").pp("0"))?;
        let conv_out = CausalConv3d::load(
            channels[0], 3, (3, 3, 3), (1, 1, 1), (1, 1, 1),
            vb.pp("decoder").pp("head").pp("2"),
        )?;

        // Count total cache slots:
        // conv_in: 1
        // mid_block_0: 2, mid_block_2: 2
        // each res block in up_blocks: 2
        // each SpatioTemporal upsample: 1
        // conv_out: 1
        let mut total = 0;
        total += 1; // conv_in
        total += mid_block_0.num_cache_slots(); // 2
        total += mid_block_2.num_cache_slots(); // 2
        for (res_blocks, upsample) in &up_blocks {
            for rb in res_blocks {
                total += rb.num_cache_slots();
            }
            if let Some(up) = upsample {
                total += up.num_cache_slots();
            }
        }
        total += 1; // conv_out

        Ok(Self {
            conv_in,
            mid_attn,
            mid_block_0,
            mid_block_2,
            up_blocks,
            norm_out,
            conv_out,
            total_cache_slots: total,
        })
    }

    /// Create a fresh (empty) feature cache for chunked decoding.
    fn new_cache(&self) -> FeatureCache {
        vec![None; self.total_cache_slots]
    }

    /// Decode a single temporal chunk through the full decoder.
    /// `z_chunk`: [B, z_dim, 1, H, W] (single latent frame)
    /// `cache`: mutable feature cache, updated in-place
    /// `first_chunk`: true for the very first frame (affects Resample behavior)
    fn decode_chunk(
        &self,
        z_chunk: &Tensor,
        cache: &mut FeatureCache,
        _first_chunk: bool,
    ) -> Result<Tensor> {
        let mut slot = 0;

        // conv_in
        let new_cache = build_cache_entry(z_chunk, &cache[slot])?;
        let old_cache = cache[slot].take();
        cache[slot] = Some(new_cache);
        let mut h = self.conv_in.forward_cached(z_chunk, old_cache.as_ref())?;
        slot += 1;

        // Mid block: ResBlock + Attention + ResBlock

        h = self.mid_block_0.forward_cached(&h, cache, slot)?;
        slot += self.mid_block_0.num_cache_slots();


        h = self.mid_attn.forward(&h)?;


        h = self.mid_block_2.forward_cached(&h, cache, slot)?;
        slot += self.mid_block_2.num_cache_slots();


        // Up blocks
        for (stage_idx, (res_blocks, upsample)) in self.up_blocks.iter().enumerate() {
            for (bi, block) in res_blocks.iter().enumerate() {
                h = block.forward_cached(&h, cache, slot)?;
                slot += block.num_cache_slots();
            }
            if let Some(up) = upsample {
                h = match up {
                    UpBlock::Spatial(u) => u.forward(&h)?,
                    UpBlock::SpatioTemporal(u) => {
                        let result = u.forward_cached(&h, cache, slot)?;
                        slot += u.num_cache_slots();
                        result
                    }
                };
            }
        }

        // Output head: norm + silu + conv_out
        h = self.norm_out.forward(&h)?;
        h = h.silu()?;

        let new_cache = build_cache_entry(&h, &cache[slot])?;
        let old_cache = cache[slot].take();
        cache[slot] = Some(new_cache);
        h = self.conv_out.forward_cached(&h, old_cache.as_ref())?;
        // slot += 1; // last slot

        Ok(h)
    }

    /// Decode latents using chunked temporal decoding (one latent frame at a time).
    ///
    /// z: [B, z_dim, F, H, W] where F is the number of latent frames.
    ///
    /// Processes each frame sequentially, using the feature cache so that
    /// CausalConv3d layers have access to previous frames' activations.
    pub fn decode(&mut self, z: &Tensor) -> Result<Tensor> {
        let num_latent_frames = z.dims()[2];

        // Create fresh cache
        let mut cache = self.new_cache();

        let mut output: Option<Tensor> = None;

        for i in 0..num_latent_frames {
            let z_chunk = z.narrow(2, i, 1)?; // [B, z_dim, 1, H, W]
            let first_chunk = i == 0;

            let decoded = self.decode_chunk(&z_chunk, &mut cache, first_chunk)?;

            output = Some(match output {
                None => decoded,
                Some(prev) => Tensor::cat(&[&prev, &decoded], 2)?,
            });
        }

        let output = output.ok_or_else(|| anyhow::anyhow!("no latent frames to decode"))?;

        // Clamp to [-1, 1]
        Ok(output.clamp(-1f32, 1f32)?)
    }
}
