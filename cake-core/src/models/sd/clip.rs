use std::fmt::{Debug, Display, Formatter};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::clip::ClipConfig;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use crate::cake::Forwarder;
use crate::models::llama3::{Cache, Config};

pub struct Clip {
    config: ClipConfig,
    clip_weights: Option<String>,

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
    fn load_text_model(name: String, vb: VarBuilder, cfg: &Config) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        Err(anyhow!("load_text_model should never be called on Clip"))
    }

    fn load_image_model(name: String, cfg: &StableDiffusionConfig) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        todo!()
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