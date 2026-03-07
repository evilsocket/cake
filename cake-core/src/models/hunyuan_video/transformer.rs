use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};

/// HunyuanVideo DiT transformer Forwarder.
///
/// Layer name: `"hunyuan-transformer"`
///
/// This wraps the dual-stream DiT transformer. Once the vendored model code
/// is complete, this will load and run the full transformer weights.
#[derive(Debug)]
pub struct HunyuanTransformer {
    name: String,
}

impl std::fmt::Display for HunyuanTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl HunyuanTransformer {
    pub fn load_model(_ctx: &Context) -> Result<Box<dyn Forwarder>> {
        log::warn!("HunyuanVideo transformer: vendored model code not yet implemented");
        Ok(Box::new(Self {
            name: "hunyuan-transformer".to_string(),
        }))
    }
}

#[async_trait]
impl Forwarder for HunyuanTransformer {
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
        anyhow::bail!("HunyuanVideo transformer forward not yet implemented — vendored model code required")
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
