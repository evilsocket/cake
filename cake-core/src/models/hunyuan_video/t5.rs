use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};

/// HunyuanVideo T5-XXL text encoder Forwarder.
///
/// Layer name: `"hunyuan-t5"`
///
/// Reuses the same T5 architecture as LTX-Video and Flux.
#[derive(Debug)]
pub struct HunyuanT5 {
    name: String,
}

impl std::fmt::Display for HunyuanT5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl HunyuanT5 {
    pub fn load_model(_ctx: &Context) -> Result<Box<dyn Forwarder>> {
        log::warn!("HunyuanVideo T5 encoder: vendored model code not yet implemented");
        Ok(Box::new(Self {
            name: "hunyuan-t5".to_string(),
        }))
    }
}

#[async_trait]
impl Forwarder for HunyuanT5 {
    fn load(name: String, _ctx: &Context) -> Result<Box<Self>> {
        Ok(Box::new(Self { name }))
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        anyhow::bail!("HunyuanVideo T5 forward not yet implemented — vendored model code required")
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}
