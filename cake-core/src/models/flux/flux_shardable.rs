//! FLUX component router for distributed workers.

use crate::cake::{Context, Forwarder};
use crate::models::flux::text_encoder::FluxTextEncoder;
use crate::models::flux::transformer::FluxTransformerForwarder;
use crate::models::flux::vae::FluxVAE;
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
            "flux_text_encoder" => FluxTextEncoder::load(name.clone(), ctx)?,
            "flux_transformer" => FluxTransformerForwarder::load(name.clone(), ctx)?,
            "flux_vae" => FluxVAE::load(name.clone(), ctx)?,
            _ => anyhow::bail!("Unknown FLUX component: {name}"),
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
