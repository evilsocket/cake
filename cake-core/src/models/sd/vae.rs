use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{get_sd_config, pack_tensors, unpack_tensors};
use crate::models::sd::ModelFile;
use crate::StableDiffusionVersion;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use log::{debug, info};
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct VAE {
    vae_model: AutoEncoderKL,
}

impl Display for VAE {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "VAE (local)")
    }
}

#[async_trait]
impl Forwarder for VAE {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let sd_config = get_sd_config(ctx)?;

        Self::load_model(
            ctx.args.sd_args.vae.clone(),
            ctx.args.sd_args.sd_version,
            ctx.args.sd_args.use_f16,
            &ctx.device,
            ctx.dtype,
            ctx.args.model.clone(),
            &sd_config,
        )
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("VAE model forwarding...");

        let unpacked_tensors = unpack_tensors(x)?;

        let direction_tensor = &unpacked_tensors[0];
        let direction_vec = direction_tensor.to_vec1()?;
        let direction_f32: f32 = *direction_vec
            .get(0)
            .expect("Error retrieving direction info");

        let input = &unpacked_tensors[1].to_dtype(ctx.dtype)?;

        debug!("VAE tensors decoded.");

        if direction_f32 == 1.0 {
            let dist = self.vae_model.encode(&input)?;
            Ok(dist.sample()?)
        } else {
            Ok(self.vae_model.decode(&input)?)
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
        "vae"
    }
}

impl VAE {
    pub fn load_model(
        name: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
        device: &Device,
        dtype: DType,
        cache_dir: String,
        config: &StableDiffusionConfig,
    ) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let vae_weights = ModelFile::Vae.get(name, version, use_f16, cache_dir)?;
        let vae_model = config.build_vae(vae_weights, device, dtype)?;

        info!("Loading VAE model...");

        Ok(Box::new(Self { vae_model }))
    }

    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        image: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = Vec::from([Tensor::from_slice(&[1f32], 1, &ctx.device)?, image]);

        let combined_tensor = pack_tensors(tensors, &ctx.device)?;

        Ok(forwarder.forward_mut(&combined_tensor, 0, 0, ctx).await?)
    }

    pub async fn decode(
        forwarder: &mut Box<dyn Forwarder>,
        latents: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = Vec::from([Tensor::from_slice(&[0f32], 1, &ctx.device)?, latents]);

        let combined_tensor = pack_tensors(tensors, &ctx.device)?;

        let result = forwarder.forward_mut(&combined_tensor, 0, 0, ctx).await?;
        Ok(result)
    }
}
