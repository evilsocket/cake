use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};

/// HunyuanVideo CLIP-L text encoder Forwarder.
///
/// Layer name: `"hunyuan-clip"`
///
/// HunyuanVideo uses dual text encoders (T5-XXL + CLIP-L).
#[derive(Debug)]
pub struct HunyuanClip {
    name: String,
}

impl std::fmt::Display for HunyuanClip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl HunyuanClip {
    pub fn load_model(_ctx: &Context) -> Result<Box<dyn Forwarder>> {
        log::warn!("HunyuanVideo CLIP encoder: vendored model code not yet implemented");
        Ok(Box::new(Self {
            name: "hunyuan-clip".to_string(),
        }))
    }
}

#[async_trait]
impl Forwarder for HunyuanClip {
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
        anyhow::bail!(
            "HunyuanVideo CLIP forward not yet implemented — vendored model code required"
        )
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
