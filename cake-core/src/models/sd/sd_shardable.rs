use std::fmt::{Debug, Display, Formatter};
use candle_core::Tensor;
use crate::cake::{Context, Forwarder};
use crate::models::llama3::{Cache};
use crate::models::sd::clip::Clip;
use crate::models::sd::unet::UNet;
use crate::models::sd::vae::VAE;

pub struct SDShardable {
    forwarder: dyn Forwarder,
    layer_name: String,
}

impl Debug for SDShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for SDShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Forwarder for SDShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        let mut model: dyn Forwarder;

        match name {
            String::from("vae") => {
                model = VAE::load(name.clone(), ctx)?;
            },
            String::from("clip") => {
                model = Clip::load(name.clone(), ctx)?;
            },
            String::from("clip2") => {
                model = Clip::load(name.clone(), ctx)?;
            },
            String::from("unet") => {
                model = UNet::load(name.clone(), ctx)?;
            },
        }

        Ok(Box::new(Self{
            forwarder: model,
            layer_name: name
        }))
    }

    async fn forward(& mut self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut Cache) -> anyhow::Result<Tensor> {
        self.forwarder.forward(x, index_pos, block_idx, cache)
    }

    async fn forward_mut(&mut self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut Cache) -> anyhow::Result<Tensor> {
        self.forwarder.forward_mut(x, index_pos, block_idx, cache)
    }

    async fn forward_batch(&mut self, _x: &Tensor, _batch: Vec<(String, usize, usize)>, _cache: &mut Cache) -> anyhow::Result<Tensor> {
        self.forwarder.forward_batch(_x, _batch, _cache)
    }

    fn layer_name(&self) -> &str {
        &*self.layer_name
    }

    fn ident(&self) -> &str {
        &*self.layer_name
    }
}
