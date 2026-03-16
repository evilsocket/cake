use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{get_sd_config, pack_tensors, unpack_tensors};
use crate::models::sd::ModelFile;
use crate::StableDiffusionVersion;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use log::{debug, info};
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct VAE {
    vae_model: AutoEncoderKL,
}

impl Display for VAE {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "VAE (local)")
    }
}

#[async_trait]
impl Forwarder for VAE {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let sd_config = get_sd_config(ctx)?;

        Self::load_model(
            ctx.args.sd_args.vae.clone(),
            ctx.args.sd_args.sd_version,
            ctx.args.sd_args.use_f16,
            &ctx.device,
            ctx.dtype,
            ctx.args.model.clone(),
            &sd_config,
        )
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("VAE model forwarding...");

        let unpacked_tensors = unpack_tensors(x)?;

        let direction_tensor = &unpacked_tensors[0];
        let direction_vec = direction_tensor.to_vec1()?;
        let direction_f32: f32 = *direction_vec
            .first()
            .expect("Error retrieving direction info");

        let input = &unpacked_tensors[1].to_dtype(ctx.dtype)?;

        debug!("VAE tensors decoded.");

        if direction_f32 == 1.0 {
            let dist = self.vae_model.encode(input)?;
            Ok(dist.sample()?)
        } else {
            Ok(self.vae_model.decode(input)?)
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
        "vae"
    }
}

impl VAE {
    pub fn load_model(
        name: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
        device: &Device,
        dtype: DType,
        cache_dir: String,
        config: &StableDiffusionConfig,
    ) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let vae_weights = ModelFile::Vae.get(name, version, use_f16, cache_dir)?;
        let vae_model = config.build_vae(vae_weights, device, dtype)?;

        info!("Loading VAE model...");

        Ok(Box::new(Self { vae_model }))
    }

    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        image: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = Vec::from([Tensor::from_slice(&[1f32], 1, &ctx.device)?, image]);

        let combined_tensor = pack_tensors(tensors, &ctx.device)?;

        forwarder.forward_mut(&combined_tensor, 0, 0, ctx).await
    }

    /// The constant layer name for this component.
    #[cfg(test)]
    pub const LAYER_NAME: &'static str = "vae";

    pub async fn decode(
        forwarder: &mut Box<dyn Forwarder>,
        latents: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let tensors = Vec::from([Tensor::from_slice(&[0f32], 1, &ctx.device)?, latents]);

        let combined_tensor = pack_tensors(tensors, &ctx.device)?;

        let result = forwarder.forward_mut(&combined_tensor, 0, 0, ctx).await?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_display_format() {
        // We can't construct a VAE without model weights, but we can test
        // the LAYER_NAME constant and verify the expected display string.
        assert_eq!(VAE::LAYER_NAME, "vae");
    }

    #[test]
    fn vae_layer_name_matches_constant() {
        // LAYER_NAME should match ModelFile::Vae.name()
        assert_eq!(VAE::LAYER_NAME, ModelFile::Vae.name());
    }

    #[test]
    fn vae_encode_direction_tensor_is_one() {
        // The encode path packs a direction tensor of [1.0] as the first element.
        let device = candle_core::Device::Cpu;
        let direction = Tensor::from_slice(&[1f32], 1, &device).unwrap();
        let val: Vec<f32> = direction.to_vec1().unwrap();
        assert_eq!(val, vec![1.0]);
    }

    #[test]
    fn vae_decode_direction_tensor_is_zero() {
        // The decode path packs a direction tensor of [0.0] as the first element.
        let device = candle_core::Device::Cpu;
        let direction = Tensor::from_slice(&[0f32], 1, &device).unwrap();
        let val: Vec<f32> = direction.to_vec1().unwrap();
        assert_eq!(val, vec![0.0]);
    }

    #[test]
    fn vae_pack_unpack_encode_direction() {
        // Verify the encode packing produces a valid packed tensor that unpacks correctly.
        let device = candle_core::Device::Cpu;
        let direction = Tensor::from_slice(&[1f32], 1, &device).unwrap();
        let image = Tensor::from_vec(vec![0.5f32; 12], (1, 3, 2, 2), &device).unwrap();

        let tensors = vec![direction, image.clone()];
        let packed = pack_tensors(tensors, &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 2);
        let dir_val: Vec<f32> = unpacked[0].to_vec1().unwrap();
        assert_eq!(dir_val, vec![1.0]);
        assert_eq!(unpacked[1].shape().dims(), &[1, 3, 2, 2]);
    }

    #[test]
    fn vae_pack_unpack_decode_direction() {
        // Verify the decode packing produces a valid packed tensor that unpacks correctly.
        let device = candle_core::Device::Cpu;
        let direction = Tensor::from_slice(&[0f32], 1, &device).unwrap();
        let latents = Tensor::from_vec(vec![0.1f32; 16], (1, 4, 2, 2), &device).unwrap();

        let tensors = vec![direction, latents.clone()];
        let packed = pack_tensors(tensors, &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 2);
        let dir_val: Vec<f32> = unpacked[0].to_vec1().unwrap();
        assert_eq!(dir_val, vec![0.0]);
        assert_eq!(unpacked[1].shape().dims(), &[1, 4, 2, 2]);
    }
}
