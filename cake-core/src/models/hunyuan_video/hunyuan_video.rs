// HunyuanVideo (13B) text-to-video generation pipeline.
//
// Implements Generator + VideoGenerator for distributed inference.
//
// Architecture:
//   - Dual text encoder: LLaMA-based (4096-dim) + CLIP (768-dim pooled)
//   - Dual-stream DiT transformer (20 double + 40 single blocks)
//   - 3D causal VAE (16ch latent, 8x spatial, 4x temporal compression)
//   - Flow matching scheduler with shift=7.0

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use image::{ImageBuffer, Rgb};
use log::info;

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::pack_tensors;
use crate::models::{Generator, VideoGenerator};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;

use super::hunyuan_shardable::HunyuanShardable;
use super::transformer::HunyuanTransformer;
use super::vendored::scheduler::HunyuanFlowMatchScheduler;

pub struct HunyuanVideo {
    transformer: Box<dyn Forwarder>,
    text_encoder: Box<dyn Forwarder>,
    clip_encoder: Box<dyn Forwarder>,
    vae: Box<dyn Forwarder>,
    context: Context,
}

#[async_trait]
impl Generator for HunyuanVideo {
    type Shardable = HunyuanShardable;
    const MODEL_NAME: &'static str = "hunyuan-video";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        info!("Loading HunyuanVideo model...");

        // Load transformer
        let transformer: Box<dyn Forwarder> =
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer("hunyuan-transformer")
            {
                info!("node {node_name} will serve hunyuan-transformer");
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "hunyuan-transformer",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("hunyuan-transformer will be served locally");
                HunyuanTransformer::load("hunyuan-transformer".to_string(), ctx)?
            };

        // Load text encoder (LLaMA-based)
        let text_encoder: Box<dyn Forwarder> =
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer("hunyuan-text") {
                info!("node {node_name} will serve hunyuan-text");
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "hunyuan-text",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("hunyuan-text will be served locally");
                super::t5_encoder::HunyuanTextEncoder::load("hunyuan-text".to_string(), ctx)?
            };

        // Load CLIP encoder
        let clip_encoder: Box<dyn Forwarder> =
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer("hunyuan-clip") {
                info!("node {node_name} will serve hunyuan-clip");
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "hunyuan-clip",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("hunyuan-clip will be served locally");
                super::clip_encoder::HunyuanClipEncoder::load("hunyuan-clip".to_string(), ctx)?
            };

        // Load VAE
        let vae: Box<dyn Forwarder> =
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer("hunyuan-vae") {
                info!("node {node_name} will serve hunyuan-vae");
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "hunyuan-vae",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("hunyuan-vae will be served locally");
                super::vae_forwarder::HunyuanVae::load("hunyuan-vae".to_string(), ctx)?
            };

        info!("HunyuanVideo model loaded");

        Ok(Some(Box::new(Self {
            transformer,
            text_encoder,
            clip_encoder,
            vae,
            context: ctx.clone(),
        })))
    }
}

#[async_trait]
impl VideoGenerator for HunyuanVideo {
    async fn generate_video(&mut self, args: &ImageGenerationArgs) -> Result<VideoOutput> {
        let _prompt = &args.image_prompt;
        let seed = args.image_seed.unwrap_or(42);
        let num_frames = 45; // HunyuanVideo default
        let height = 720;    // 720p default
        let width = 1280;    // 720p default (16:9)
        let num_steps = 50;
        let guidance_scale = 6.0_f32;
        let shift = 7.0;

        let device = self.context.device.clone();
        let dtype = self.context.dtype;

        info!(
            "Generating HunyuanVideo: {}x{}, {} frames, {} steps, guidance={:.1}",
            width, height, num_frames, num_steps, guidance_scale,
        );

        // TODO: Text encoding (LLaMA-based + CLIP) - for now use placeholder zeros
        // The LLaMA encoder produces [B, L, 4096] hidden states
        // The CLIP encoder produces [B, 768] pooled embeddings
        let text_len = 256;
        let text_dim = 4096;
        info!("Text encoding not yet implemented - using placeholder embeddings");
        let context = Tensor::zeros((1, text_len, text_dim), dtype, &device)?;
        let uncond_context = Tensor::zeros((1, text_len, text_dim), dtype, &device)?;

        // Compute latent dimensions
        // HunyuanVideo VAE: 4x temporal compression, 8x spatial compression
        let lat_frames = (num_frames + 3) / 4; // ceil division by temporal_compression=4
        let lat_h = height / 8;
        let lat_w = width / 8;

        info!(
            "Latent shape: [{}, 16, {}, {}, {}]",
            1, lat_frames, lat_h, lat_w
        );

        // Generate initial noise
        device.set_seed(seed)?;
        let latents =
            Tensor::randn(0f32, 1.0, (1, 16, lat_frames, lat_h, lat_w), &device)?
                .to_dtype(dtype)?;

        // Flow matching schedule
        let scheduler = HunyuanFlowMatchScheduler::new(shift);
        let sigmas = scheduler.sigmas(num_steps);
        let timesteps = scheduler.timesteps(num_steps);

        info!(
            "Starting denoising ({} steps, shift={})...",
            num_steps, shift
        );

        let mut latents = latents;
        let do_cfg = guidance_scale > 1.0;

        for step in 0..num_steps {
            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t = timesteps[step] as f32;
            let dt = (sigma_next - sigma) as f64;

            let t_tensor = Tensor::from_slice(&[t], 1, &device)?.to_dtype(dtype)?;

            // Conditional pass
            let noise_pred = HunyuanTransformer::forward_packed(
                &mut self.transformer,
                latents.clone(),
                t_tensor.clone(),
                context.clone(),
                lat_frames,
                lat_h,
                lat_w,
                &mut self.context,
            )
            .await?;

            let noise_pred = if do_cfg {
                // Unconditional pass
                let noise_uncond = HunyuanTransformer::forward_packed(
                    &mut self.transformer,
                    latents.clone(),
                    t_tensor,
                    uncond_context.clone(),
                    lat_frames,
                    lat_h,
                    lat_w,
                    &mut self.context,
                )
                .await?;

                // CFG: pred = uncond + guidance_scale * (cond - uncond)
                // Cast intermediates to compute dtype to avoid F16 promotion issues
                let diff = (noise_pred - &noise_uncond)?.to_dtype(dtype)?;
                let scaled = diff.affine(guidance_scale as f64, 0.0)?.to_dtype(dtype)?;
                (noise_uncond.to_dtype(dtype)? + scaled)?
            } else {
                noise_pred
            };

            // Euler step: latents = latents + dt * velocity
            let noise_pred = noise_pred.to_dtype(dtype)?;
            let velocity = noise_pred.affine(dt, 0.0)?.to_dtype(dtype)?;
            latents = (latents + velocity)?;

            info!(
                "step {}/{}: sigma={:.4} -> {:.4}",
                step + 1,
                num_steps,
                sigma,
                sigma_next,
            );
        }

        // VAE decode
        info!("Decoding with VAE...");
        let direction = Tensor::from_slice(&[0f32], 1, &device)?;
        let packed = pack_tensors(vec![direction, latents], &device)?;
        let video = self
            .vae
            .forward_mut(&packed, 0, 0, &mut self.context)
            .await?;

        // Convert to frames
        info!("Converting to frames...");
        let video = ((video + 1.0)? * 127.5)?
            .clamp(0f32, 255f32)?
            .to_dtype(DType::U8)?
            .to_device(&Device::Cpu)?;

        let (_, _, num_out_frames, out_h, out_w) = video.dims5()?;
        let mut frames = Vec::with_capacity(num_out_frames);

        for f in 0..num_out_frames {
            let frame = video.narrow(2, f, 1)?.squeeze(2)?; // [1, 3, H, W]
            let frame = frame.squeeze(0)?; // [3, H, W]
            let frame = frame.permute((1, 2, 0))?.flatten_all()?; // [H*W*3]
            let pixels = frame.to_vec1::<u8>()?;
            if let Some(img) = ImageBuffer::from_raw(out_w as u32, out_h as u32, pixels) {
                frames.push(img);
            }
        }

        let fps = 24; // HunyuanVideo default
        info!(
            "Generated {} frames at {}x{}, {}fps",
            frames.len(),
            out_w,
            out_h,
            fps
        );

        Ok(VideoOutput::new(frames, fps, out_w as u32, out_h as u32))
    }
}
