//! LTXModel — the full LTX-2 transformer.
//!
//! Wraps N `BasicAVTransformerBlock` layers with input/output projections,
//! AdaLN timestep embedding, caption projection, and RoPE.

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
#[derive(Debug)]
pub struct LTXModel {
    config: Ltx2TransformerConfig,

    // Video components
    proj_in: Option<Linear>,
    adaln_single: Option<AdaLayerNormSingle>,
    caption_projection: Option<TextProjection>,
    scale_shift_table: Option<Tensor>, // [2, video_inner_dim] — final output modulation

    // Transformer blocks
    blocks: Vec<BasicAVTransformerBlock>,

    // Output
    proj_out: Option<Linear>,
}

impl LTXModel {
    pub fn new(config: Ltx2TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let has_video = config.model_type.is_video_enabled();
        let video_dim = config.video_inner_dim();
        let adaln_params = config.adaln_params();

        // Video components
        let (proj_in, adaln_single, caption_projection, sst, proj_out) = if has_video {
            let proj_in = candle_nn::linear(config.in_channels, video_dim, vb.pp("proj_in"))?;
            let adaln = AdaLayerNormSingle::new(video_dim, adaln_params, vb.pp("time_embed"))?;
            let caption = TextProjection::new(
                config.caption_channels,
                video_dim,
                vb.pp("caption_projection"),
            )?;
            let sst = vb.get((2, video_dim), "scale_shift_table")?;
            let proj_out = candle_nn::linear(video_dim, config.out_channels, vb.pp("proj_out"))?;
            (
                Some(proj_in),
                Some(adaln),
                Some(caption),
                Some(sst),
                Some(proj_out),
            )
        } else {
            (None, None, None, None, None)
        };

        // Blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = BasicAVTransformerBlock::new(
                i,
                &config,
                vb.pp(format!("transformer_blocks.{i}")),
            )?;
            blocks.push(block);
        }

        Ok(Self {
            config,
            proj_in,
            adaln_single,
            caption_projection,
            scale_shift_table: sst,
            blocks,
            proj_out,
        })
    }

    pub fn config(&self) -> &Ltx2TransformerConfig {
        &self.config
    }

    /// Forward pass (video-only mode).
    ///
    /// `video_latent`: patchified video tokens, `[B, T, in_channels]`
    /// `sigma`: noise level per sample, `[B]`
    /// `timesteps`: scalar timestep per sample, `[B]`
    /// `positions`: positional coordinates, `[B, n_dims, T]` (3 for video: t,h,w)
    /// `context`: text embeddings from Gemma connector, `[B, L, cross_attention_dim]`
    /// `context_mask`: binary mask for text, `[B, L]`
    ///
    /// Returns velocity prediction, same shape as `video_latent`.
    pub fn forward_video(
        &self,
        video_latent: &Tensor,
        _sigma: &Tensor,
        timesteps: &Tensor,
        positions: &Tensor,
        context: &Tensor,
        context_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let proj_in = self.proj_in.as_ref().expect("video proj_in");
        let adaln = self.adaln_single.as_ref().expect("video adaln");
        let caption_proj = self.caption_projection.as_ref().expect("video caption_proj");
        let sst = self.scale_shift_table.as_ref().expect("video scale_shift_table");
        let proj_out = self.proj_out.as_ref().expect("video proj_out");

        let video_dim = self.config.video_inner_dim();
        let adaln_params = self.config.adaln_params();

        // 1. Project input
        let hidden = proj_in.forward(video_latent)?;

        // 2. Timestep embedding → AdaLN params
        // Python: timestep.flatten() — ensure [B]
        let scaled_ts = timesteps.affine(self.config.timestep_scale_multiplier as f64, 0.0)?;
        let (temb, embedded_ts) = adaln.forward(&scaled_ts)?;

        // temb: [B, adaln_params * dim] -> [B, 1, adaln_params, dim]
        let (b, _) = temb.dims2()?;
        let temb = temb.reshape((b, 1, adaln_params, video_dim))?;
        // embedded_ts: [B, dim] -> [B, 1, dim] (for output layer modulation)
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

        // 5. Run through transformer blocks
        let mut x = hidden;
        for block in &self.blocks {
            x = block.forward_video_only(&x, &temb, Some(&pe), &context, context_mask)?;
        }

        // 6. Final output with AdaLN modulation
        // Python: scale_shift_values = sst[None,None] + embedded_timestep[:,:,None]
        // sst: [2, dim] -> [1, 1, 2, dim]
        // embedded_ts: [B, 1, dim] -> [B, 1, 1, dim]
        // sum: [B, 1, 2, dim], then shift=[:,:,0], scale=[:,:,1]
        let sst_4d = sst.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 2, dim]
        let et_4d = embedded_ts.unsqueeze(2)?; // [B, 1, 1, dim]
        let scale_shift = sst_4d
            .to_dtype(et_4d.dtype())?
            .broadcast_add(&et_4d)?; // [B, 1, 2, dim]
        let shift = scale_shift.narrow(2, 0, 1)?.squeeze(2)?; // [B, 1, dim]
        let scale = scale_shift.narrow(2, 1, 1)?.squeeze(2)?; // [B, 1, dim]

        let x = rms_norm(&x, self.config.norm_eps)?;
        let x = x
            .broadcast_mul(&scale.broadcast_add(&Tensor::ones_like(&scale)?)?)?
            .broadcast_add(&shift)?;

        let x = proj_out.forward(&x)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn small_config() -> Ltx2TransformerConfig {
        // cross_attention_dim must equal video_inner_dim (heads * d_head)
        // because caption_projection maps caption_channels -> video_dim,
        // and attn2 expects context of size cross_attention_dim.
        Ltx2TransformerConfig {
            num_attention_heads: 2,
            attention_head_dim: 8,
            in_channels: 16,
            out_channels: 16,
            cross_attention_dim: 16, // = 2 * 8 = video_inner_dim
            num_layers: 1,
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
        let video_dim = config.video_inner_dim();

        let video_latent =
            Tensor::randn(0f32, 1f32, (b, seq, config.in_channels), &device).unwrap();
        let sigma = Tensor::full(0.5f32, (b,), &device).unwrap();
        let timestep = Tensor::full(0.5f32, (b,), &device).unwrap();
        let positions = Tensor::randn(0f32, 1f32, (b, 3, seq), &device).unwrap();
        // context has caption_channels dim (goes through caption_projection first)
        let context =
            Tensor::randn(0f32, 1f32, (b, 4, config.caption_channels), &device).unwrap();

        let out = model
            .forward_video(&video_latent, &sigma, &timestep, &positions, &context, None)
            .unwrap();
        assert_eq!(out.dims(), &[b, seq, config.out_channels]);
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
        // denoised = sample - sigma * velocity
        assert!((vals[0] - 0.95).abs() < 1e-5);
        assert!((vals[1] - 1.9).abs() < 1e-5);
        assert!((vals[2] - 2.85).abs() < 1e-5);
    }
}
