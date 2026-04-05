use anyhow::Result;
use crate::cake::{Context, Forwarder};
use super::t5_encoder::WanT5Encoder;
use super::transformer::WanTransformer;
use super::vae_forwarder::WanVae;

/// Component dispatcher for Wan2.2 model.
/// Routes layer names to the appropriate forwarder (transformer, VAE, T5).
/// Workers load the actual component; the dispatch is transparent.
pub struct WanShardable {
    inner: Box<dyn Forwarder>,
    name: String,
}

impl std::fmt::Debug for WanShardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WanShardable[{}]", self.name)
    }
}

impl std::fmt::Display for WanShardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", self.name)
    }
}

#[async_trait::async_trait]
impl Forwarder for WanShardable {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> where Self: Sized {
        let inner: Box<dyn Forwarder> = if name == "wan-t5" {
            WanT5Encoder::load(name.clone(), ctx)?
        } else if name.starts_with("wan-transformer") {
            WanTransformer::load(name.clone(), ctx)?
        } else if name == "wan-vae" {
            WanVae::load(name.clone(), ctx)?
        } else {
            anyhow::bail!("unknown Wan component: {}", name)
        };
        Ok(Box::new(Self { inner, name }))
    }

    async fn forward(&self, x: &candle_core::Tensor, i: usize, b: usize, ctx: &mut Context) -> Result<candle_core::Tensor> {
        self.inner.forward(x, i, b, ctx).await
    }

    async fn forward_mut(&mut self, x: &candle_core::Tensor, i: usize, b: usize, ctx: &mut Context) -> Result<candle_core::Tensor> {
        self.inner.forward_mut(x, i, b, ctx).await
    }

    fn layer_name(&self) -> &str { &self.name }
}
