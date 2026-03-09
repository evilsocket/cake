use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::path::PathBuf;

use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};

// LTX-2 VAE decoder reuses the same building blocks as LTX-Video,
// but the encoder is architecturally different (AutoencoderKLLTX2Video).
// We only need the decoder for generation, so we load it directly.
use crate::models::ltx_video::vendored::vae::{AutoencoderKLLtxVideoConfig, LtxVideoDecoder3d};

/// LTX-2 Video VAE Forwarder (decoder-only).
///
/// Layer name: `"ltx2-vae"`
///
/// The LTX-2 VAE (`AutoencoderKLLTX2Video`) has a different encoder architecture
/// from LTX-Video, but shares the same decoder building blocks. Since video
/// generation only needs decode (latents → pixels), we skip the encoder entirely.
#[derive(Debug)]
pub struct Ltx2Vae {
    name: String,
    decoder: LtxVideoDecoder3d,
    /// Per-channel latent normalization mean (loaded from safetensors).
    pub latents_mean: Vec<f32>,
    /// Per-channel latent normalization std (loaded from safetensors).
    pub latents_std: Vec<f32>,
}

impl std::fmt::Display for Ltx2Vae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl Ltx2Vae {
    fn vae_config(is_ltx23: bool) -> AutoencoderKLLtxVideoConfig {
        if is_ltx23 {
            // LTX-2.3 VAE: 4 up_blocks with different channel dims and strides
            AutoencoderKLLtxVideoConfig {
                block_out_channels: vec![256, 512, 1024, 2048],
                decoder_block_out_channels: vec![256, 512, 512, 1024],
                layers_per_block: vec![4, 6, 6, 2, 2],
                decoder_layers_per_block: vec![4, 6, 4, 2, 2],
                latent_channels: 128,
                patch_size: 4,
                patch_size_t: 1,
                timestep_conditioning: false,
                decoder_spatiotemporal_scaling: vec![true, true, true, true],
                decoder_inject_noise: vec![false, false, false, false, false],
                decoder_upsample_residual: vec![true, true, true, true],
                decoder_upsample_factor: vec![2, 2, 1, 2],
                // Per-block strides (un-reversed, matching decoder_block_out_channels order):
                // After reversal: block0=(2,2,2), block1=(2,2,2), block2=(2,1,1), block3=(1,2,2)
                decoder_strides: vec![(1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)],
                ..Default::default()
            }
        } else {
            // LTX-2 VAE: 3 up_blocks, same as LTX-Video structure
            AutoencoderKLLtxVideoConfig {
                block_out_channels: vec![256, 512, 1024, 2048],
                decoder_block_out_channels: vec![256, 512, 1024],
                layers_per_block: vec![4, 6, 6, 2, 2],
                decoder_layers_per_block: vec![5, 5, 5, 5],
                latent_channels: 128,
                patch_size: 4,
                patch_size_t: 1,
                timestep_conditioning: false,
                ..Default::default()
            }
        }
    }

    fn resolve_weights(ctx: &Context) -> Result<PathBuf> {
        let ltx_args = &ctx.args.ltx_args;
        if let Some(ref p) = ltx_args.ltx_vae {
            return Ok(PathBuf::from(p));
        }

        // Try direct path: --model points to directory containing vae/
        let model_dir = PathBuf::from(&ctx.args.model);
        let direct = model_dir.join("vae/diffusion_pytorch_model.safetensors");
        if direct.exists() {
            return Ok(direct);
        }

        // Fall back to HF cache
        let repo = ltx_args.ltx_repo();
        let cache_path = model_dir.join("hub");
        let api = if cache_path.is_dir() {
            ApiBuilder::from_cache(Cache::new(cache_path)).build()?
        } else {
            let mut builder = ApiBuilder::new();
            if let Ok(token) = std::env::var("HF_TOKEN") {
                builder = builder.with_token(Some(token));
            }
            builder.build()?
        };
        let model_api = api.model(repo);
        Ok(model_api.get("vae/diffusion_pytorch_model.safetensors")?)
    }

    fn load_inner(name: String, ctx: &Context) -> Result<Self> {
        let weights_path = Self::resolve_weights(ctx)?;
        let is_ltx23 = ctx.args.ltx_args.is_ltx23();
        info!("Loading LTX-2{} VAE (decoder-only) from {:?}...",
            if is_ltx23 { ".3" } else { "" }, weights_path);

        // LTX-2 VAE weights are BF16 — load as BF16 to avoid conversion artifacts
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                DType::BF16,
                &ctx.device,
            )?
        };

        let config = Self::vae_config(is_ltx23);

        // Load latents_mean and latents_std from safetensors (registered buffers)
        let latents_mean: Vec<f32> = vb
            .get(config.latent_channels, "latents_mean")?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        let latents_std: Vec<f32> = vb
            .get(config.latent_channels, "latents_std")?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        info!(
            "VAE latents_mean range: [{:.4}, {:.4}], latents_std range: [{:.4}, {:.4}]",
            latents_mean.iter().cloned().fold(f32::INFINITY, f32::min),
            latents_mean.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            latents_std.iter().cloned().fold(f32::INFINITY, f32::min),
            latents_std.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );

        // Load decoder directly — skip encoder (different architecture in LTX-2)
        let decoder = if !config.decoder_strides.is_empty() {
            LtxVideoDecoder3d::new_with_strides(
                config.latent_channels,
                config.out_channels,
                &config.decoder_block_out_channels,
                &config.decoder_strides,
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
            )?
        } else {
            LtxVideoDecoder3d::new(
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
            )?
        };

        info!("LTX-2 VAE decoder loaded!");

        Ok(Self {
            name,
            decoder,
            latents_mean,
            latents_std,
        })
    }

    pub fn load_model(ctx: &Context) -> Result<Box<dyn Forwarder>> {
        Ok(Box::new(Self::load_inner("ltx2-vae".to_string(), ctx)?))
    }

    /// Load VAE and return (forwarder, latents_mean, latents_std).
    pub fn load_with_stats(ctx: &Context) -> Result<(Box<dyn Forwarder>, Vec<f32>, Vec<f32>)> {
        let vae = Self::load_inner("ltx2-vae".to_string(), ctx)?;
        let mean = vae.latents_mean.clone();
        let std = vae.latents_std.clone();
        Ok((Box::new(vae), mean, std))
    }

    /// Decode latents through the VAE (no timestep conditioning for LTX-2).
    pub async fn decode(
        forwarder: &mut Box<dyn Forwarder>,
        latents: Tensor,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let tensors = vec![
            Tensor::from_slice(&[0f32], 1, &ctx.device)?, // direction: 0.0 = decode
            latents,
        ];
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }
}

#[async_trait]
impl Forwarder for Ltx2Vae {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        Ok(Box::new(Self::load_inner(name, ctx)?))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        let direction_vec: Vec<f32> = unpacked[0].to_vec1()?;
        let direction = direction_vec[0];
        // VAE weights are BF16 — convert input to match
        let input = unpacked[1].to_dtype(DType::BF16)?;

        if direction == 1.0 {
            anyhow::bail!(
                "LTX-2 VAE encoding not supported — encoder architecture differs from decoder. \
                 Use LTX-Video VAE for encoding."
            );
        }

        let timestep = if unpacked.len() > 2 {
            Some(unpacked[2].to_dtype(DType::BF16)?)
        } else {
            None
        };
        let decoded = self.decoder.forward(&input, timestep.as_ref(), false)?;
        Ok(decoded)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}
