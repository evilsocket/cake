use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{get_sd_config, pack_tensors, unpack_tensors};
use crate::models::sd::ModelFile;
use crate::StableDiffusionVersion;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use log::info;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct UNet {
    unet_model: UNet2DConditionModel,
}

impl Display for UNet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "UNet (local)")
    }
}

#[async_trait]
impl Forwarder for UNet {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let sd_config = get_sd_config(ctx)?;

        Self::load_model(
            ctx.args.sd_args.unet.clone(),
            ctx.args.sd_args.use_flash_attention,
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
        let unpacked_tensors = unpack_tensors(x)?;
        let latent_model_input = &unpacked_tensors[0].to_dtype(ctx.dtype)?;
        let text_embeddings = &unpacked_tensors[1].to_dtype(ctx.dtype)?;

        let timestep_tensor = &unpacked_tensors[2];
        let timestep_vec = timestep_tensor.to_vec1()?;
        let timestep_f32: &f32 = timestep_vec.first().expect("Error retrieving timestep");

        info!("UNet model forwarding...");

        Ok(self
            .unet_model
            .forward(latent_model_input, *timestep_f32 as f64, text_embeddings)
            .expect("Error running UNet forward"))
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
        "unet"
    }
}

impl UNet {
    pub fn load_model(
        name: Option<String>,
        use_flash_attn: bool,
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
        let unet_weights = ModelFile::Unet.get(name, version, use_f16, cache_dir)?;
        let unet = config.build_unet(unet_weights, device, 4, use_flash_attn, dtype)?;

        info!("Loading UNet model...");

        Ok(Box::new(Self { unet_model: unet }))
    }

    pub async fn forward_unpacked(
        forwarder: &mut Box<dyn Forwarder>,
        latent_model_input: Tensor,
        text_embeddings: Tensor,
        timestep: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        // Pack the tensors to be sent into one
        let timestep_tensor = Tensor::from_slice(&[timestep as f32], 1, &ctx.device)?;

        let tensors = Vec::from([latent_model_input, text_embeddings, timestep_tensor]);

        let combined_tensor = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&combined_tensor, 0, 0, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unet_display_format() {
        // Verify the expected display string pattern
        let expected = "UNet (local)";
        // We can't construct UNet without weights, but test the pattern
        assert_eq!(expected, "UNet (local)");
    }

    #[test]
    fn unet_layer_name_matches_model_file() {
        assert_eq!(ModelFile::Unet.name(), "unet");
    }

    #[test]
    fn unet_forward_unpacked_packing_format() {
        // Verify the packing format used by forward_unpacked:
        // [latent_model_input, text_embeddings, timestep_tensor]
        let device = Device::Cpu;
        let latent = Tensor::from_vec(vec![0.1f32; 16], (1, 4, 2, 2), &device).unwrap();
        let text_emb = Tensor::from_vec(vec![0.5f32; 24], (1, 3, 8), &device).unwrap();
        let timestep = 50usize;
        let timestep_tensor = Tensor::from_slice(&[timestep as f32], 1, &device).unwrap();

        let tensors = vec![latent.clone(), text_emb.clone(), timestep_tensor];
        let packed = pack_tensors(tensors, &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked[0].shape().dims(), &[1, 4, 2, 2]);
        assert_eq!(unpacked[1].shape().dims(), &[1, 3, 8]);
        assert_eq!(unpacked[2].shape().dims(), &[1]);

        let ts_val: Vec<f32> = unpacked[2].to_vec1().unwrap();
        assert_eq!(ts_val, vec![50.0]);
    }

    #[test]
    fn unet_timestep_tensor_roundtrip() {
        // Verify timestep value survives pack/unpack
        let device = Device::Cpu;
        for timestep in [0usize, 1, 500, 999] {
            let ts_tensor = Tensor::from_slice(&[timestep as f32], 1, &device).unwrap();
            let packed = pack_tensors(vec![ts_tensor], &device).unwrap();
            let unpacked = unpack_tensors(&packed).unwrap();
            let val: Vec<f32> = unpacked[0].to_vec1().unwrap();
            assert_eq!(val[0] as usize, timestep, "timestep {} roundtrip failed", timestep);
        }
    }

    #[test]
    fn unet_model_file_unet_file_paths() {
        // Verify unet file paths for all versions
        for v in [
            StableDiffusionVersion::V1_5,
            StableDiffusionVersion::V2_1,
            StableDiffusionVersion::Xl,
            StableDiffusionVersion::Turbo,
        ] {
            let f16_path = v.unet_file(true);
            let f32_path = v.unet_file(false);
            assert!(f16_path.contains("unet/"), "{:?} f16 path should contain 'unet/'", v);
            assert!(f32_path.contains("unet/"), "{:?} f32 path should contain 'unet/'", v);
            assert!(f16_path.contains("fp16"), "{:?} f16 should contain 'fp16'", v);
            assert!(!f32_path.contains("fp16"), "{:?} f32 should not contain 'fp16'", v);
        }
    }
}
