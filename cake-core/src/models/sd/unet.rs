use std::fmt::{Debug, Display, Formatter};
use async_trait::async_trait;
use candle_core::{Device, DType, Tensor};
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;
use crate::cake::{Context, Forwarder};
use crate::models::sd::ModelFile;
use crate::models::sd::util::{get_device, get_sd_config, pack_tensors, unpack_tensors};
use crate::StableDiffusionVersion;
use log::info;

#[derive(Debug)]
pub struct UNet {
    unet_model: UNet2DConditionModel
}

impl Display for UNet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UNet (local)")
    }
}

#[async_trait]
impl Forwarder for UNet {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        let dtype = if ctx.args.sd_args.use_f16 { DType::F16 } else { DType::F32 };
        let device = get_device(ctx.args.cpu)?;
        let sd_config = get_sd_config(ctx)?;

        Self::load_model(
            ctx.args.sd_args.unet.clone(),
            ctx.args.sd_args.use_flash_attention,
            ctx.args.sd_args.sd_version,
            ctx.args.sd_args.use_f16,
            &device,
            dtype,
            ctx.args.model.clone(),
            &sd_config,
        )
    }

    async fn forward(&self, x: &Tensor, _index_pos: usize, _block_idx: usize, ctx: &mut Context) -> anyhow::Result<Tensor> {
        let unpacked_tensors = unpack_tensors(x)?;
        let latent_model_input = &unpacked_tensors[0].to_dtype(ctx.dtype)?;
        let text_embeddings = &unpacked_tensors[1].to_dtype(ctx.dtype)?;

        let timestep_tensor = &unpacked_tensors[2];
        let timestep_vec = timestep_tensor.to_vec1()?;
        let timestep_f32: &f32 = timestep_vec.get(0).expect("Error retrieving timestep");

        info!("UNet model forwarding...");
        
        Ok(self.unet_model.forward(latent_model_input, *timestep_f32 as f64, text_embeddings).expect("Error running UNet forward"))
    }

    async fn forward_mut(&mut self, x: &Tensor, index_pos: usize, block_idx: usize, ctx: &mut Context) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        "unet"
    }
}

impl UNet {
    pub fn load_model(name: Option<String>, use_flash_attn: bool, version: StableDiffusionVersion, use_f16: bool, device: &Device, dtype: DType, cache_dir: String, config: &StableDiffusionConfig) -> anyhow::Result<Box<Self>>
    where
        Self: Sized {

        let unet_weights = ModelFile::Unet.get(name, version, use_f16, cache_dir)?;
        let unet = config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;

        info!("Loading UNet model...");
        
        Ok(Box::new(Self{
            unet_model: unet,
        }))
    }

    pub async fn forward_unpacked(
        forwarder: &mut Box<dyn Forwarder>,
        latent_model_input: Tensor,
        text_embeddings: Tensor,
        timestep: usize,
        ctx: &mut Context
    ) -> anyhow::Result<Tensor> {

        // Pack the tensors to be sent into one
        let timestep_tensor = Tensor::from_slice(&[timestep as f32], 1, &ctx.device)?;

        let tensors = Vec::from([
            latent_model_input,
            text_embeddings,
            timestep_tensor
        ]);

        let combined_tensor = pack_tensors(tensors, &ctx.device)?;
        Ok(forwarder.forward_mut(&combined_tensor, 0, 0, ctx).await?)
    }
}
