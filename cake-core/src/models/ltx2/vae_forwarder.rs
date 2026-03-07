use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::path::PathBuf;

use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};

// LTX-2 reuses the same VAE architecture as LTX-Video
use crate::models::ltx_video::vendored::vae::{AutoencoderKLLtxVideo, AutoencoderKLLtxVideoConfig};

/// LTX-2 Video VAE Forwarder.
///
/// Layer name: `"ltx2-vae"`
///
/// Reuses the LTX-Video VAE architecture (same decoder, 128 latent channels).
#[derive(Debug)]
pub struct Ltx2Vae {
    name: String,
    model: AutoencoderKLLtxVideo,
}

impl std::fmt::Display for Ltx2Vae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl Ltx2Vae {
    fn vae_config() -> AutoencoderKLLtxVideoConfig {
        // LTX-2 uses AutoencoderKLLTX2Video — different from LTX-Video 0.9.x
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
        let mut cache_path = model_dir;
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let model_api = api.model(repo);
        Ok(model_api.get("vae/diffusion_pytorch_model.safetensors")?)
    }

    fn load_inner(name: String, ctx: &Context) -> Result<Self> {
        let weights_path = Self::resolve_weights(ctx)?;
        info!("Loading LTX-2 VAE from {:?}...", weights_path);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                ctx.dtype,
                &ctx.device,
            )?
        };

        let model = AutoencoderKLLtxVideo::new(Self::vae_config(), vb)?;
        info!("LTX-2 VAE loaded!");

        Ok(Self { name, model })
    }

    pub fn load_model(ctx: &Context) -> Result<Box<dyn Forwarder>> {
        Ok(Box::new(Self::load_inner("ltx2-vae".to_string(), ctx)?))
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
        let input = unpacked[1].to_dtype(ctx.dtype)?;

        if direction == 1.0 {
            let encoded = self.model.encoder.forward(&input, false)?;
            let dist =
                crate::models::ltx_video::vendored::vae::DiagonalGaussianDistribution::new(
                    &encoded,
                )?;
            Ok(dist.mode()?)
        } else {
            let timestep = if unpacked.len() > 2 {
                Some(unpacked[2].to_dtype(ctx.dtype)?)
            } else {
                None
            };
            let decoded = self.model.decoder.forward(&input, timestep.as_ref(), false)?;
            Ok(decoded)
        }
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
