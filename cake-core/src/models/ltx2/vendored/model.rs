//! LTXModel — the full LTX-2 transformer.
//!
//! Wraps N `BasicAVTransformerBlock` layers with input/output projections,
//! AdaLN timestep embedding, caption projection, and RoPE.
//!
//! Supports block-range sharding: load only blocks N..M for distributed inference.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::adaln::{AdaLayerNormSingle, TextProjection};
use super::attention::rms_norm;
use super::config::Ltx2TransformerConfig;
use super::rope::precompute_freqs_cis;
use super::transformer_block::BasicAVTransformerBlock;

/// Velocity-to-denoised conversion: denoised = sample - sigma * velocity.
pub fn to_denoised(sample: &Tensor, sigma: &Tensor, velocity: &Tensor) -> Result<Tensor> {
    // sigma needs to broadcast to sample shape
    let sigma = sigma.unsqueeze(1)?.unsqueeze(2)?; // [B, 1, 1]
    sample.broadcast_sub(&sigma.broadcast_mul(velocity)?)
}

/// Full LTX-2 transformer model (video-only path).
///
/// Supports partial block loading via `new_block_range()` for distributed inference.
/// When loaded with a block range, only those blocks are in memory. The setup
/// (proj_in, adaln, caption_projection) and finalize (scale_shift_table, proj_out)
/// are only loaded when block_start == 0 or block_end == num_layers respectively.
#[derive(Debug)]
pub struct LTXModel {
    config: Ltx2TransformerConfig,

    // Video components (None when not needed for this block range)
    proj_in: Option<Linear>,
    adaln_single: Option<AdaLayerNormSingle>,
    caption_projection: Option<TextProjection>,
    scale_shift_table: Option<Tensor>, // [2, video_inner_dim] — final output modulation

    // Transformer blocks (may be a subset)
    blocks: Vec<BasicAVTransformerBlock>,
    /// First block index (0 for full model or first shard)
    block_start: usize,

    // Output
    proj_out: Option<Linear>,
}

impl LTXModel {
    /// Load the full model (all blocks + setup + finalize).
    pub fn new(config: Ltx2TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_block_range(config, vb, 0, None)
    }

    /// Load a range of blocks [block_start, block_end).
    ///
    /// - Setup (proj_in, adaln, caption_projection) is loaded only when block_start == 0.
    /// - Finalize (scale_shift_table, proj_out) is loaded only when block_end == num_layers.
    /// - For workers that only run a middle range, neither setup nor finalize is loaded.
    pub fn new_block_range(
        config: Ltx2TransformerConfig,
        vb: VarBuilder,
        block_start: usize,
        block_end: Option<usize>,
    ) -> Result<Self> {
        let has_video = config.model_type.is_video_enabled();
        let video_dim = config.video_inner_dim();
        let adaln_params = config.adaln_params();
        let num_layers = config.num_layers;
        let block_end = block_end.unwrap_or(num_layers);

        let is_first = block_start == 0;
        let is_last = block_end >= num_layers;

        log::info!(
            "Loading LTX-2 transformer blocks {}-{} of {} (setup={}, finalize={})",
            block_start, block_end - 1, num_layers, is_first, is_last
        );

        // Setup: only load for the first shard
        let (proj_in, adaln_single, caption_projection) = if has_video && is_first {
            let proj_in = candle_nn::linear(config.in_channels, video_dim, vb.pp("proj_in"))?;
            let adaln = AdaLayerNormSingle::new(video_dim, adaln_params, vb.pp("time_embed"))?;
            let caption = TextProjection::new(
                config.caption_channels,
                video_dim,
                vb.pp("caption_projection"),
            )?;
            (Some(proj_in), Some(adaln), Some(caption))
        } else {
            (None, None, None)
        };

        // Finalize: only load for the last shard
        let (sst, proj_out) = if has_video && is_last {
            let sst = vb.get((2, video_dim), "scale_shift_table")?;
            let proj_out = candle_nn::linear(video_dim, config.out_channels, vb.pp("proj_out"))?;
            (Some(sst), Some(proj_out))
        } else {
            (None, None)
        };

        // Load only the blocks in range
        let mut blocks = Vec::with_capacity(block_end - block_start);
        for i in block_start..block_end {
            let block = BasicAVTransformerBlock::new(
                i,
                &config,
                vb.pp(format!("transformer_blocks.{i}")),
            )?;
            blocks.push(block);
        }

        log::info!("Loaded {} transformer blocks ({}-{})", blocks.len(), block_start, block_end - 1);

        Ok(Self {
            config,
            proj_in,
            adaln_single,
            caption_projection,
            scale_shift_table: sst,
            blocks,
            block_start,
            proj_out,
        })
    }

    pub fn config(&self) -> &Ltx2TransformerConfig {
        &self.config
    }

    /// Whether this model shard includes the setup components (proj_in, adaln, caption).
    pub fn has_setup(&self) -> bool {
        self.proj_in.is_some()
    }

    /// Whether this model shard includes the finalize components (scale_shift_table, proj_out).
    pub fn has_finalize(&self) -> bool {
        self.proj_out.is_some()
    }

    /// Run setup: proj_in + adaln + caption_projection + RoPE.
    ///
    /// Returns (hidden, temb, embedded_ts, pe, context_projected).
    pub fn forward_setup(
        &self,
        video_latent: &Tensor,
        timesteps: &Tensor,
        positions: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, (Tensor, Tensor), Tensor)> {
        let proj_in = self.proj_in.as_ref().expect("forward_setup requires proj_in");
        let adaln = self.adaln_single.as_ref().expect("forward_setup requires adaln");
        let caption_proj = self.caption_projection.as_ref().expect("forward_setup requires caption_projection");

        let video_dim = self.config.video_inner_dim();
        let adaln_params = self.config.adaln_params();

        // 1. Project input
        let hidden = proj_in.forward(video_latent)?;

        // 2. Timestep embedding → AdaLN params
        let scaled_ts = timesteps.affine(self.config.timestep_scale_multiplier as f64, 0.0)?;
        let (temb, embedded_ts) = adaln.forward(&scaled_ts)?;

        let (b, _) = temb.dims2()?;
        let temb = temb.reshape((b, 1, adaln_params, video_dim))?;
        let embedded_ts = embedded_ts.reshape((b, 1, video_dim))?;

        // 3. Caption projection
        let context = caption_proj.forward(context)?;

        // 4. Compute RoPE
        let pe = precompute_freqs_cis(
            positions,
            self.config.num_attention_heads * self.config.attention_head_dim,
            self.config.positional_embedding_theta,
            &self.config.positional_embedding_max_pos,
            self.config.num_attention_heads,
            hidden.dtype(),
        )?;

        Ok((hidden, temb, embedded_ts, pe, context))
    }

    /// Run transformer blocks on pre-setup hidden states.
    ///
    /// `hidden`: [B, T, video_dim] — output of proj_in or previous block range
    /// `temb`: [B, 1, adaln_params, video_dim]
    /// `pe`: (cos, sin) RoPE
    /// `context`: [B, L, video_dim] — already through caption projection
    /// `context_mask`: [B, L]
    pub fn forward_blocks(
        &self,
        hidden: &Tensor,
        temb: &Tensor,
        pe: &(Tensor, Tensor),
        context: &Tensor,
        context_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = hidden.clone();
        for block in self.blocks.iter() {
            x = block.forward_video_only(&x, temb, Some(pe), context, context_mask)?;
        }
        Ok(x)
    }

    /// Run finalize: final AdaLN modulation + proj_out.
    pub fn forward_finalize(
        &self,
        x: &Tensor,
        embedded_ts: &Tensor,
    ) -> Result<Tensor> {
        let sst = self.scale_shift_table.as_ref().expect("forward_finalize requires scale_shift_table");
        let proj_out = self.proj_out.as_ref().expect("forward_finalize requires proj_out");

        let sst_4d = sst.unsqueeze(0)?.unsqueeze(0)?;
        let et_4d = embedded_ts.unsqueeze(2)?;
        let scale_shift = sst_4d
            .to_dtype(et_4d.dtype())?
            .broadcast_add(&et_4d)?;
        let shift = scale_shift.narrow(2, 0, 1)?.squeeze(2)?;
        let scale = scale_shift.narrow(2, 1, 1)?.squeeze(2)?;

        let x = rms_norm(x, self.config.norm_eps)?;
        let x = x
            .broadcast_mul(&scale.broadcast_add(&Tensor::ones_like(&scale)?)?)?
            .broadcast_add(&shift)?;

        proj_out.forward(&x)
    }

    /// Full forward pass (video-only mode). Convenience method that calls
    /// forward_setup + forward_blocks + forward_finalize.
    pub fn forward_video(
        &self,
        video_latent: &Tensor,
        _sigma: &Tensor,
        timesteps: &Tensor,
        positions: &Tensor,
        context: &Tensor,
        context_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t0 = std::time::Instant::now();

        log::info!(
            "Transformer input shapes: video_latent={:?} timesteps={:?} positions={:?} context={:?} dtype={:?} device={:?}",
            video_latent.shape(), timesteps.shape(), positions.shape(), context.shape(),
            video_latent.dtype(), video_latent.device(),
        );

        let (hidden, temb, embedded_ts, pe, context) =
            self.forward_setup(video_latent, timesteps, positions, context)?;

        log::info!("Transformer setup: {}ms", t0.elapsed().as_millis());

        let x = self.forward_blocks(&hidden, &temb, &pe, &context, context_mask)?;
        let x = self.forward_finalize(&x, &embedded_ts)?;

        log::info!("Transformer forward total: {}ms ({} blocks)", t0.elapsed().as_millis(), self.blocks.len());

        Ok(x)
    }

    /// Forward pass for block-range workers.
    ///
    /// Input: pre-setup hidden states + metadata (no raw latents).
    /// Output: hidden states after running this shard's blocks.
    /// If this shard includes finalize, output is the final velocity prediction.
    pub fn forward_blocks_only(
        &self,
        hidden: &Tensor,
        temb: &Tensor,
        pe: &(Tensor, Tensor),
        context: &Tensor,
        context_mask: Option<&Tensor>,
        embedded_ts: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = self.forward_blocks(hidden, temb, pe, context, context_mask)?;

        if self.has_finalize() {
            let ets = embedded_ts.expect("forward_blocks_only with finalize needs embedded_ts");
            self.forward_finalize(&x, ets)
        } else {
            Ok(x)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn small_config() -> Ltx2TransformerConfig {
        Ltx2TransformerConfig {
            num_attention_heads: 2,
            attention_head_dim: 8,
            in_channels: 16,
            out_channels: 16,
            cross_attention_dim: 16,
            num_layers: 4,
            caption_channels: 32,
            ..Default::default()
        }
    }

    #[test]
    fn test_ltx_model_video_forward_shape() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = LTXModel::new(config.clone(), vb).unwrap();

        let b = 1;
        let seq = 8;

        let video_latent =
            Tensor::randn(0f32, 1f32, (b, seq, config.in_channels), &device).unwrap();
        let sigma = Tensor::full(0.5f32, (b,), &device).unwrap();
        let timestep = Tensor::full(0.5f32, (b,), &device).unwrap();
        let positions = Tensor::randn(0f32, 1f32, (b, 3, seq), &device).unwrap();
        let context =
            Tensor::randn(0f32, 1f32, (b, 4, config.caption_channels), &device).unwrap();

        let out = model
            .forward_video(&video_latent, &sigma, &timestep, &positions, &context, None)
            .unwrap();
        assert_eq!(out.dims(), &[b, seq, config.out_channels]);
    }

    #[test]
    fn test_block_range_split() {
        let device = Device::Cpu;
        let config = small_config();

        // Full model
        let vb_full = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let full_model = LTXModel::new(config.clone(), vb_full).unwrap();

        // Split: first half (blocks 0-1) with setup
        let vb1 = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let first_half = LTXModel::new_block_range(config.clone(), vb1, 0, Some(2)).unwrap();
        assert!(first_half.has_setup());
        assert!(!first_half.has_finalize());
        assert_eq!(first_half.blocks.len(), 2);

        // Split: second half (blocks 2-3) with finalize
        let vb2 = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let second_half = LTXModel::new_block_range(config.clone(), vb2, 2, Some(4)).unwrap();
        assert!(!second_half.has_setup());
        assert!(second_half.has_finalize());
        assert_eq!(second_half.blocks.len(), 2);

        // Run full model
        let b = 1;
        let seq = 8;
        let video_latent =
            Tensor::randn(0f32, 1f32, (b, seq, config.in_channels), &device).unwrap();
        let sigma = Tensor::full(0.5f32, (b,), &device).unwrap();
        let timestep = Tensor::full(0.5f32, (b,), &device).unwrap();
        let positions = Tensor::randn(0f32, 1f32, (b, 3, seq), &device).unwrap();
        let context =
            Tensor::randn(0f32, 1f32, (b, 4, config.caption_channels), &device).unwrap();

        let full_out = full_model
            .forward_video(&video_latent, &sigma, &timestep, &positions, &context, None)
            .unwrap();

        // Run split pipeline
        let (hidden, temb, embedded_ts, pe, ctx) =
            first_half.forward_setup(&video_latent, &timestep, &positions, &context).unwrap();
        let x = first_half.forward_blocks(&hidden, &temb, &pe, &ctx, None).unwrap();
        let x = second_half.forward_blocks(&x, &temb, &pe, &ctx, None).unwrap();
        let split_out = second_half.forward_finalize(&x, &embedded_ts).unwrap();

        // Results should match (both use zeros weights)
        assert_eq!(full_out.dims(), split_out.dims());
    }

    #[test]
    fn test_to_denoised() {
        let device = Device::Cpu;
        let sample = Tensor::new(&[1.0f32, 2.0, 3.0], &device)
            .unwrap()
            .reshape((1, 1, 3))
            .unwrap();
        let sigma = Tensor::new(&[0.5f32], &device).unwrap();
        let velocity = Tensor::new(&[0.1f32, 0.2, 0.3], &device)
            .unwrap()
            .reshape((1, 1, 3))
            .unwrap();

        let denoised = to_denoised(&sample, &sigma, &velocity).unwrap();
        let vals: Vec<f32> = denoised.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 0.95).abs() < 1e-5);
        assert!((vals[1] - 1.9).abs() < 1e-5);
        assert!((vals[2] - 2.85).abs() < 1e-5);
    }
}
