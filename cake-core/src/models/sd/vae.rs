use std::fmt::{Debug, Display, Formatter};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, DiagonalGaussianDistribution};
use crate::cake::Forwarder;
use crate::models::llama3::{Cache, Config};
use crate::models::sd::ModelFile;
use crate::models::sd::util::pack_tensors;
use crate::StableDiffusionVersion;

pub struct VAE {
    vae_model: AutoEncoderKL
}

impl Debug for VAE {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for VAE {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Forwarder for VAE {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        Err(anyhow!("load should never be called on VAE"))
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

impl VAE {
    pub fn load_model(name: Option<String>, version: StableDiffusionVersion, use_f16: bool, device: &Device, dtype: DType, config: &StableDiffusionConfig) -> anyhow::Result<Box<Self>>
    where
        Self: Sized {
        let vae_weights = ModelFile::Vae.get(name, version, use_f16)?;
        let vae_model = config.build_vae(vae_weights, device, dtype)?;

        Ok(Box::new(Self{
            vae_model,
        }))
    }

    pub async fn encode(forwarder: &Box<dyn Forwarder>, image: Tensor, device: &Device, cache: &mut Cache) -> anyhow::Result<DiagonalGaussianDistribution> {
        let tensors = Vec::from([
            Tensor::from_slice(&*[1], 1, device)?,
            image
        ]);

        let combined_tensor = pack_tensors(tensors, device)?;

        let parameters = forwarder.forward(&combined_tensor, 0, 0, cache).await?;
        Ok(DiagonalGaussianDistribution::new(&parameters)?)
    }

    pub async fn decode(forwarder: &Box<dyn Forwarder>, latents: Tensor, device: &Device, cache: &mut Cache) -> anyhow::Result<Tensor> {
        let tensors = Vec::from([
            Tensor::from_slice(&*[0], 1, device)?,
            latents,
        ]);

        let combined_tensor = pack_tensors(tensors, device)?;

        let result = forwarder.forward(&combined_tensor, 0, 0, cache).await?;
        Ok(result)
    }
}