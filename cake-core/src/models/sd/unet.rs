use std::fmt::{Debug, Display, Formatter};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;
use crate::cake::Forwarder;
use crate::models::llama3::{Cache, Config};
use crate::models::sd::ModelFile;
use crate::models::sd::util::pack_tensors;
use crate::StableDiffusionVersion;

pub struct UNet {
    unet_model: UNet2DConditionModel
}

impl Debug for UNet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for UNet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Forwarder for UNet {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        Err(anyhow!("load should never be called on UNet"))
    }

    async fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut Cache) -> anyhow::Result<Tensor> {
        todo!()
    }

    async fn forward_mut(&mut self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut Cache) -> anyhow::Result<Tensor> {
        todo!()
    }

    fn layer_name(&self) -> &str {
        todo!()
    }
}

impl UNet {
    pub fn load_model(name: Option<String>, use_flash_attn: bool, version: StableDiffusionVersion, use_f16: bool, device: &Device, dtype: DType, config: &StableDiffusionConfig) -> anyhow::Result<Box<Self>>
    where
        Self: Sized {

        let unet_weights = ModelFile::Unet.get(name, version, use_f16)?;
        let unet = config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;

        Ok(Box::new(Self{
            unet_model: unet,
        }))
    }

    pub async fn forward_unpacked(
        forwarder: &Box<dyn Forwarder>,
        latent_model_input: Tensor,
        text_embeddings: Tensor,
        timestep: usize,
        device: &Device,
        cache: & mut Cache
    ) -> anyhow::Result<Tensor> {

        // Pack the tensors to be sent into one
        let timestep_tensor = Tensor::from_slice(&*[timestep], 1, device)?;

        let tensors = Vec::from([
            latent_model_input,
            text_embeddings,
            timestep_tensor
        ]);

        let combined_tensor = pack_tensors(tensors, &device)?;
        Ok(forwarder.forward(&combined_tensor, 0, 0, cache).await?)
    }
}