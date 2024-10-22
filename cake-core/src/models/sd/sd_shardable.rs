use crate::cake::{Context, Forwarder};
use crate::models::sd::clip::Clip;
use crate::models::sd::unet::UNet;
use crate::models::sd::vae::VAE;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct SDShardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for SDShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for SDShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder>;

        match name.as_str() {
            "vae" => {
                model = VAE::load(name.clone(), ctx)?;
            }
            "clip" => {
                model = Clip::load(name.clone(), ctx)?;
            }
            "clip2" => {
                model = Clip::load(name.clone(), ctx)?;
            }
            "unet" => {
                model = UNet::load(name.clone(), ctx)?;
            }
            _ => {
                anyhow::bail!("Model name not recognized");
            }
        }

        Ok(Box::new(Self {
            forwarder: model,
            layer_name: name,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder.forward(x, index_pos, block_idx, ctx).await
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder
            .forward_mut(x, index_pos, block_idx, ctx)
            .await
    }

    async fn forward_batch(
        &mut self,
        _x: &Tensor,
        _batch: Vec<(String, usize, usize)>,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder.forward_batch(_x, _batch, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn ident(&self) -> &str {
        &self.layer_name
    }
}
