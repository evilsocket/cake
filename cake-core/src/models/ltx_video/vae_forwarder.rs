use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};
use async_trait::async_trait;
use candle_core::Tensor;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::fmt::{Debug, Display, Formatter};
use std::path::PathBuf;

use super::vendored::configs::get_config_by_version;
use super::vendored::vae::AutoencoderKLLtxVideo;

#[derive(Debug)]
pub struct LtxVae {
    model: AutoencoderKLLtxVideo,
}

impl Display for LtxVae {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ltx-vae (local)")
    }
}

#[async_trait]
impl Forwarder for LtxVae {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(ctx)
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("LTX VAE forwarding...");

        let unpacked = unpack_tensors(x)?;

        // Protocol: [direction, data, optional_timestep]
        // direction: 1.0 = encode, 0.0 = decode
        let direction_vec: Vec<f32> = unpacked[0].to_vec1()?;
        let direction = *direction_vec.first().expect("Error retrieving direction");

        let input = unpacked[1].to_dtype(ctx.dtype)?;

        if direction == 1.0 {
            // Encode
            let encoded = self.model.encoder.forward(&input, false)?;
            let dist =
                super::vendored::vae::DiagonalGaussianDistribution::new(&encoded)?;
            Ok(dist.mode()?)
        } else {
            // Decode
            let timestep = if unpacked.len() > 2 {
                Some(unpacked[2].to_dtype(ctx.dtype)?)
            } else {
                None
            };

            let decoded = self.model.decoder.forward(
                &input,
                timestep.as_ref(),
                false,
            )?;
            Ok(decoded)
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
        "ltx-vae"
    }
}

impl LtxVae {
    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let ltx_args = &ctx.args.ltx_args;
        let version = &ltx_args.ltx_version;
        let config = get_config_by_version(version);

        let weights_path = if let Some(ref p) = ltx_args.ltx_vae {
            PathBuf::from(p)
        } else {
            let repo = ltx_args.ltx_repo();
            let mut cache_path = PathBuf::from(&ctx.args.model);
            cache_path.push("hub");
            let cache = Cache::new(cache_path);
            let api = ApiBuilder::from_cache(cache).build()?;
            let model_api = api.model(repo);
            model_api.get("vae/diffusion_pytorch_model.safetensors")?
        };

        info!("Loading LTX VAE from {:?}...", weights_path);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                ctx.dtype,
                &ctx.device,
            )?
        };
        let model = AutoencoderKLLtxVideo::new(config.vae, vb)?;

        info!("LTX VAE loaded!");

        Ok(Box::new(Self { model }))
    }

    pub async fn decode(
        forwarder: &mut Box<dyn Forwarder>,
        latents: Tensor,
        timestep: Option<Tensor>,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let mut tensors = vec![
            Tensor::from_slice(&[0f32], 1, &ctx.device)?,
            latents,
        ];
        if let Some(t) = timestep {
            tensors.push(t);
        }
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }
}
