//! FLUX.2-klein MMDiT transformer Forwarder.

use crate::backends::ComputeBackend;
use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{pack_tensors, unpack_tensors};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use log::info;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use super::config::FluxModelFile;
use super::flux2_model::{Flux2Config, Flux2Transformer};

#[derive(Debug)]
pub struct FluxTransformerForwarder {
    pub model: Flux2Transformer,
}

impl Display for FluxTransformerForwarder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FluxTransformer (local)")
    }
}

#[async_trait]
impl Forwarder for FluxTransformerForwarder {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(&ctx.device, ctx.dtype, &ctx.args.model, ctx.backend.clone())
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        let img = &unpacked[0];
        let img_ids = &unpacked[1];
        let txt = &unpacked[2];
        let txt_ids = &unpacked[3];
        let timesteps = &unpacked[4];

        let result = self.model.forward(img, img_ids, txt, txt_ids, timesteps)?;
        Ok(result)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        "flux_transformer"
    }
}

impl FluxTransformerForwarder {
    /// Direct forward bypassing pack/unpack serialization.
    pub fn forward_direct(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
    ) -> anyhow::Result<Tensor> {
        Ok(self.model.forward(img, img_ids, txt, txt_ids, timesteps)?)
    }

    pub fn load_model(
        device: &Device,
        _dtype: DType,
        model_repo: &str,
        backend: Arc<dyn ComputeBackend>,
    ) -> anyhow::Result<Box<Self>> {
        let cfg = Flux2Config::klein_4b();

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .to_string_lossy()
            .to_string();

        let weights_path = FluxModelFile::Transformer.get(model_repo, &cache_dir)?;
        info!("loading FLUX transformer from {}", weights_path.display());

        // Load as native BF16 via mmap. The model handles dtype casting internally.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::BF16, device)?
        };

        let model = Flux2Transformer::load(vb, &cfg, backend)?;
        info!("FLUX transformer loaded ({} double + {} single blocks)", cfg.depth, cfg.depth_single);

        Ok(Box::new(Self { model }))
    }

    /// Forward with unpacked tensors (used by the pipeline directly).
    pub async fn forward_unpacked(
        forwarder: &mut Box<dyn Forwarder>,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let combined = pack_tensors(vec![img, img_ids, txt, txt_ids, timesteps], &ctx.device)?;
        forwarder.forward_mut(&combined, 0, 0, ctx).await
    }
}
