use crate::cake::{Context, Forwarder};
use super::expert_forwarder::ExpertGroupForwarder;
use super::moe_block::MoeBlock;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

/// Dispatches layer names to the appropriate Mixtral component:
/// - `"model.layers.N"` → MoeBlock (attention + local experts)
/// - `"experts-group-N"` → ExpertGroupForwarder (remote expert serving)
#[derive(Debug)]
pub struct MixtralShardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for MixtralShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for MixtralShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder> = if name.starts_with("experts-group-") {
            ExpertGroupForwarder::load(name.clone(), ctx)?
        } else {
            // Standard MoE transformer block
            <MoeBlock as Forwarder>::load(name.clone(), ctx)?
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
