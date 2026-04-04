use anyhow::Result;
use crate::cake::{Context, Forwarder};
use super::t5_encoder::WanT5Encoder;
use super::transformer::WanTransformer;
use super::vae_forwarder::WanVae;

/// Component dispatcher for Wan2.2 model.
/// Maps layer names to the appropriate forwarder type.
pub struct WanShardable;

impl WanShardable {
    pub fn load_component(name: &str, ctx: &Context) -> Result<Box<dyn Forwarder>> {
        if name == "wan-t5" {
            Ok(WanT5Encoder::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else if name.starts_with("wan-transformer") {
            Ok(WanTransformer::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else if name == "wan-vae" {
            Ok(WanVae::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else {
            anyhow::bail!("unknown Wan component: {}", name)
        }
    }
}

impl std::fmt::Debug for WanShardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WanShardable")
    }
}

impl std::fmt::Display for WanShardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WanShardable")
    }
}

#[async_trait::async_trait]
impl Forwarder for WanShardable {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> where Self: Sized {
        // WanShardable is a dispatcher, not a direct forwarder
        // Individual components are loaded via load_component
        Ok(Box::new(Self))
    }

    async fn forward(&self, _x: &candle_core::Tensor, _: usize, _: usize, _: &mut Context) -> Result<candle_core::Tensor> {
        anyhow::bail!("WanShardable::forward should not be called directly")
    }

    async fn forward_mut(&mut self, x: &candle_core::Tensor, i: usize, b: usize, ctx: &mut Context) -> Result<candle_core::Tensor> {
        self.forward(x, i, b, ctx).await
    }

    fn layer_name(&self) -> &str { "wan-shardable" }
}
