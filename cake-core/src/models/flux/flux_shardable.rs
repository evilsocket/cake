use crate::cake::{Context, Forwarder};
use crate::models::flux::clip::FluxClip;
use crate::models::flux::t5::FluxT5;
use crate::models::flux::transformer::FluxTransformer;
use crate::models::flux::vae::FluxVae;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct FluxShardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for FluxShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for FluxShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder> = match name.as_str() {
            "flux-transformer" => FluxTransformer::load(name.clone(), ctx)?,
            "flux-t5" => FluxT5::load(name.clone(), ctx)?,
            "flux-clip" => FluxClip::load(name.clone(), ctx)?,
            "flux-vae" => FluxVae::load(name.clone(), ctx)?,
            _ => anyhow::bail!("Flux component name not recognized: {}", name),
        };

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
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder.forward_batch(x, batch, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn ident(&self) -> &str {
        &self.layer_name
    }
}
