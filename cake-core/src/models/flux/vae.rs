use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};
use async_trait::async_trait;
use candle_core::Tensor;
use candle_transformers::models::flux::autoencoder::{self, AutoEncoder};
use log::info;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct FluxVae {
    model: AutoEncoder,
}

impl Display for FluxVae {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "flux-vae (local)")
    }
}

#[async_trait]
impl Forwarder for FluxVae {
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
        info!("Flux VAE forwarding...");

        let unpacked = unpack_tensors(x)?;

        // First tensor is direction: 1.0 = encode, 0.0 = decode
        let direction_vec: Vec<f32> = unpacked[0].to_vec1()?;
        let direction = *direction_vec.first().expect("Error retrieving direction");

        let input = unpacked[1].to_dtype(ctx.dtype)?;

        if direction == 1.0 {
            Ok(self.model.encode(&input)?)
        } else {
            Ok(self.model.decode(&input)?)
        }
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
        "flux-vae"
    }
}

impl FluxVae {
    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let variant = ctx.args.flux_args.flux_variant;

        let weights_path = super::flux::FluxModelFile::Vae.get(
            ctx.args.flux_args.flux_vae.clone(),
            variant,
            &ctx.args.model,
        )?;

        info!("Loading Flux VAE from {:?}...", weights_path);

        let cfg = autoencoder::Config::dev();

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                ctx.dtype,
                &ctx.device,
            )?
        };
        let model = AutoEncoder::new(&cfg, vb)?;

        info!("Flux VAE loaded!");

        Ok(Box::new(Self { model }))
    }

    #[allow(dead_code)]
    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        image: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = vec![
            Tensor::from_slice(&[1f32], 1, &ctx.device)?,
            image,
        ];
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }

    pub async fn decode(
        forwarder: &mut Box<dyn Forwarder>,
        latents: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = vec![
            Tensor::from_slice(&[0f32], 1, &ctx.device)?,
            latents,
        ];
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }
}
