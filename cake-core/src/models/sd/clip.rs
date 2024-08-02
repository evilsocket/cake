use std::fmt::{Debug, Display, Formatter};
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use crate::cake::Forwarder;
use crate::models::llama3::{Cache, Config};
use crate::models::sd::sd::ModelFile;
use crate::StableDiffusionVersion;

pub struct Clip {
    clip_model: ClipTextTransformer
}

impl Debug for Clip {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for Clip {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Forwarder for Clip {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        Err(anyhow!("load should never be called on Clip"))
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

impl Clip {
    pub fn load_model(
        model_file: ModelFile,
        name: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
        device: &Device,
        dtype: DType,
        config: &stable_diffusion::clip::Config) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        let clip_weights = model_file.get(name, version, use_f16)?;
        let clip_model = stable_diffusion::build_clip_transformer(config, clip_weights, device, dtype)?;
        Ok(Box::new(Self {
            clip_model
        }))
    }
}
