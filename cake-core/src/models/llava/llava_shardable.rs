//! LLaVA component router for distributed workers.
//!
//! Routes layer names to the appropriate Forwarder implementation:
//! - `"llava-vision"` → LlavaVision (CLIP vision tower)
//! - Standard layer names → Transformer (LLM decoder blocks)

use crate::cake::{Context, Forwarder};
use crate::models::common::Transformer;
use crate::models::llava::vision::LlavaVision;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct LlavaShardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for LlavaShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for LlavaShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder> = match name.as_str() {
            "llava-vision" => LlavaVision::load(name.clone(), ctx)?,
            _ => {
                // Standard transformer decoder block
                Transformer::load(name.clone(), ctx)?
            }
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
