use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};
use crate::FluxVariant;
use async_trait::async_trait;
use candle_core::Tensor;
use candle_transformers::models::flux::{self, model::Flux as FluxModel};
use log::info;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct FluxTransformer {
    model: FluxModel,
}

impl Display for FluxTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "flux-transformer (local)")
    }
}

#[async_trait]
impl Forwarder for FluxTransformer {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(ctx)
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        let img = unpacked[0].to_dtype(ctx.dtype)?;
        let img_ids = unpacked[1].to_dtype(ctx.dtype)?;
        let txt = unpacked[2].to_dtype(ctx.dtype)?;
        let txt_ids = unpacked[3].to_dtype(ctx.dtype)?;
        let timesteps = unpacked[4].to_dtype(ctx.dtype)?;
        let y = unpacked[5].to_dtype(ctx.dtype)?;
        let guidance = if unpacked.len() > 6 {
            Some(unpacked[6].to_dtype(ctx.dtype)?)
        } else {
            None
        };

        info!("Flux transformer forwarding...");

        use flux::WithForward;
        Ok(self
            .model
            .forward(&img, &img_ids, &txt, &txt_ids, &timesteps, &y, guidance.as_ref())?)
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
        "flux-transformer"
    }
}

impl FluxTransformer {
    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let variant = ctx.args.flux_args.flux_variant;
        let cfg = match variant {
            FluxVariant::Dev => flux::model::Config::dev(),
            FluxVariant::Schnell => flux::model::Config::schnell(),
        };

        let weights_path = super::flux::FluxModelFile::Transformer.get(
            ctx.args.flux_args.flux_transformer.clone(),
            variant,
            &ctx.args.model,
        )?;

        info!("Loading Flux transformer from {:?}...", weights_path);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                ctx.dtype,
                &ctx.device,
            )?
        };
        let model = FluxModel::new(&cfg, vb)?;

        info!("Flux transformer loaded!");

        Ok(Box::new(Self { model }))
    }

    pub async fn forward_packed(
        forwarder: &mut Box<dyn Forwarder>,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Option<Tensor>,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let mut tensors = vec![img, img_ids, txt, txt_ids, timesteps, y];
        if let Some(g) = guidance {
            tensors.push(g);
        }
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }
}
