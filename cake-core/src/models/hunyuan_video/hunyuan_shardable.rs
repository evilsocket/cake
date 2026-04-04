use anyhow::Result;
use crate::cake::{Context, Forwarder};
use super::clip_encoder::HunyuanClipEncoder;
use super::t5_encoder::HunyuanTextEncoder;
use super::transformer::HunyuanTransformer;
use super::vae_forwarder::HunyuanVae;

/// Component dispatcher for HunyuanVideo model.
/// Maps layer names to the appropriate forwarder type.
pub struct HunyuanShardable;

impl HunyuanShardable {
    pub fn load_component(name: &str, ctx: &Context) -> Result<Box<dyn Forwarder>> {
        if name == "hunyuan-text" {
            Ok(HunyuanTextEncoder::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else if name == "hunyuan-clip" {
            Ok(HunyuanClipEncoder::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else if name.starts_with("hunyuan-transformer") {
            Ok(HunyuanTransformer::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else if name == "hunyuan-vae" {
            Ok(HunyuanVae::load(name.to_string(), ctx)? as Box<dyn Forwarder>)
        } else {
            anyhow::bail!("unknown HunyuanVideo component: {}", name)
        }
    }
}

impl std::fmt::Debug for HunyuanShardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanShardable")
    }
}

impl std::fmt::Display for HunyuanShardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanShardable")
    }
}

#[async_trait::async_trait]
impl Forwarder for HunyuanShardable {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> where Self: Sized {
        // Workers call Forwarder::load() with the component name.
        // We can't return a HunyuanShardable that wraps a component because the
        // Forwarder trait requires returning Box<Self>. Instead, workers should
        // use the component forwarders directly. For now, this is only called
        // during worker setup where load_component is the correct path.
        // This fallback exists for type-system compatibility.
        Ok(Box::new(Self))
    }

    async fn forward(&self, _x: &candle_core::Tensor, _: usize, _: usize, _: &mut Context) -> Result<candle_core::Tensor> {
        anyhow::bail!("HunyuanShardable::forward should not be called directly — use load_component() for workers")
    }

    async fn forward_mut(&mut self, x: &candle_core::Tensor, i: usize, b: usize, ctx: &mut Context) -> Result<candle_core::Tensor> {
        self.forward(x, i, b, ctx).await
    }

    fn layer_name(&self) -> &str { "hunyuan-shardable" }
}
