//! FLUX.2 VAE Forwarder.
//!
//! Uses a custom decoder matching `AutoencoderKLFlux2` weight layout,
//! with batch norm denormalization for the patchified latent space.

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{pack_tensors, unpack_tensors};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use log::{debug, info};
use std::fmt::{Debug, Display, Formatter};

use super::config::FluxModelFile;
use super::flux2_vae::Flux2VAE;

#[derive(Debug)]
pub struct FluxVAE {
    pub(crate) model: Flux2VAE,
    /// Batch norm running mean: (128,) — patchified latent channels
    pub bn_running_mean: Tensor,
    /// Batch norm running var: (128,)
    pub bn_running_var: Tensor,
}

impl Display for FluxVAE {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FluxVAE (local)")
    }
}

#[async_trait]
impl Forwarder for FluxVAE {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(&ctx.device, ctx.dtype, &ctx.args.model)
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let unpacked = unpack_tensors(x)?;

        // New interface: [dims(h_half, w_half), patchified_latents]
        // Old interface: [direction_scalar, latent_image] — direction=0 means decode
        if unpacked[0].dims().len() == 1 && unpacked[0].dim(0)? == 2 && unpacked.len() == 2 {
            // New packed format: BN denorm + unpatchify + decode
            let dims = unpacked[0].to_vec1::<f32>()?;
            let h_half = dims[0] as usize;
            let w_half = dims[1] as usize;
            let img = unpacked[1].to_dtype(DType::F32)?;

            // BN denormalization
            let bn_eps = 0.0001_f64;
            let bn_std = self.bn_running_var
                .to_dtype(DType::F32)?
                .broadcast_add(&Tensor::new(&[bn_eps as f32], self.bn_running_var.device())?)?
                .sqrt()?;
            let img = img
                .broadcast_mul(&bn_std.unsqueeze(0)?.unsqueeze(0)?)?
                .broadcast_add(&self.bn_running_mean.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?)?;

            // Unpatchify: (b, h*w, 128) → (b, 32, h*2, w*2)
            let img = img
                .reshape((1, h_half, w_half, 128))?
                .permute((0, 3, 1, 2))?;
            let img = img
                .reshape((1, 32, 2, 2, h_half, w_half))?
                .permute((0, 1, 4, 2, 5, 3))?
                .reshape((1, 32, h_half * 2, w_half * 2))?;

            Ok(self.model.decode(&img)?)
        } else {
            // Legacy interface: [direction, input]
            let direction_vec = unpacked[0].to_vec1::<f32>()?;
            let direction = direction_vec[0];
            let input = &unpacked[1].to_dtype(DType::F32)?;

            debug!("FluxVAE forwarding (direction={direction})...");

            if direction == 1.0 {
                anyhow::bail!("FluxVAE encode not implemented")
            } else {
                Ok(self.model.decode(input)?)
            }
        }
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        "flux_vae"
    }
}

impl FluxVAE {
    pub fn load_model(
        device: &Device,
        _dtype: DType,
        model_repo: &str,
    ) -> anyhow::Result<Box<Self>> {
        // VAE always runs in F32 for numerical stability
        let dtype = DType::F32;

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .to_string_lossy()
            .to_string();

        let weights_path = FluxModelFile::Vae.get(model_repo, &cache_dir)?;
        info!("loading FLUX VAE from {}", weights_path.display());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
        };

        // Load batch norm running statistics
        let bn_running_mean = vb.get(128, "bn.running_mean")?.to_dtype(DType::F32)?;
        let bn_running_var = vb.get(128, "bn.running_var")?.to_dtype(DType::F32)?;

        let model = Flux2VAE::load(vb)?;
        info!("FLUX VAE loaded (custom decoder)");

        Ok(Box::new(Self {
            model,
            bn_running_mean,
            bn_running_var,
        }))
    }

    /// Decode latents to image (legacy interface, used by SD-style callers).
    #[allow(dead_code)]
    pub async fn decode(
        forwarder: &mut Box<dyn Forwarder>,
        latents: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = vec![
            Tensor::from_slice(&[0f32], 1, &ctx.device)?,
            latents,
        ];
        let combined = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&combined, 0, 0, ctx).await
    }
}
