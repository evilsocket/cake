use crate::cake::{Context, Forwarder};
use crate::models::ltx_video::ltx_video_shardable::LtxVideoShardable;
use crate::models::ltx_video::t5::LtxT5;
use crate::models::ltx_video::transformer::LtxTransformer;
use crate::models::ltx_video::vae_forwarder::LtxVae;
use crate::models::{Generator, VideoGenerator};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;
use anyhow::{Error as E, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use image::{ImageBuffer, Rgb};
use log::info;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::vendored::configs::get_config_by_version;
use super::vendored::scheduler::FlowMatchEulerDiscreteScheduler;
use super::vendored::t2v_pipeline::{self, LtxPipeline};

pub struct LtxVideo {
    t5_tokenizer: Tokenizer,
    t5_encoder: Box<dyn Forwarder>,
    transformer: Box<dyn Forwarder>,
    vae: Box<dyn Forwarder>,
    context: Context,
}

#[async_trait]
impl Generator for LtxVideo {
    type Shardable = LtxVideoShardable;
    const MODEL_NAME: &'static str = "ltx-video";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        let ltx_args = &context.args.ltx_args;
        let ltx_repo = ltx_args.ltx_repo();

        // Load T5 tokenizer
        info!("Loading T5 tokenizer...");
        let t5_tokenizer_path = if let Some(ref p) = ltx_args.ltx_t5_tokenizer {
            PathBuf::from(p)
        } else {
            // LTX ships spiece.model; use tokenizer.json from the repo or T5-XXL fallback
            resolve_hf_file(&ltx_repo, "tokenizer/tokenizer.json", &context.args.model)
                .or_else(|_| {
                    resolve_hf_file(
                        "google/t5-v1_1-xxl",
                        "tokenizer.json",
                        &context.args.model,
                    )
                })?
        };
        let t5_tokenizer = Tokenizer::from_file(&t5_tokenizer_path).map_err(E::msg)?;
        info!("T5 tokenizer loaded!");

        // T5 encoder
        info!("Loading T5 encoder...");
        let t5_encoder: Box<dyn Forwarder> =
            if let Some((node_name, node)) = context.topology.get_node_for_layer("ltx-t5") {
                info!("node {node_name} will serve ltx-t5");
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx-t5",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("T5 encoder will be served locally");
                LtxT5::load_model(context)?
            };
        info!("T5 encoder ready!");

        // VAE
        info!("Loading LTX VAE...");
        let vae: Box<dyn Forwarder> =
            if let Some((node_name, node)) = context.topology.get_node_for_layer("ltx-vae") {
                info!("node {node_name} will serve ltx-vae");
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx-vae",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("LTX VAE will be served locally");
                LtxVae::load_model(context)?
            };
        info!("LTX VAE ready!");

        // Transformer
        info!("Loading LTX transformer...");
        let transformer: Box<dyn Forwarder> = if let Some((node_name, node)) =
            context.topology.get_node_for_layer("ltx-transformer")
        {
            info!("node {node_name} will serve ltx-transformer");
            Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    "ltx-transformer",
                    context.args.cluster_key.as_deref(),
                )
                .await?,
            )
        } else {
            info!("LTX transformer will be served locally");
            LtxTransformer::load_model(context)?
        };
        info!("LTX transformer ready!");

        Ok(Some(Box::new(Self {
            t5_tokenizer,
            t5_encoder,
            transformer,
            vae,
            context: context.clone(),
        })))
    }
}

#[async_trait]
impl VideoGenerator for LtxVideo {
    async fn generate_video(
        &mut self,
        args: &ImageGenerationArgs,
    ) -> Result<VideoOutput> {
        let ImageGenerationArgs {
            image_prompt,
            image_seed,
            ..
        } = args;

        let ltx_args = &self.context.args.ltx_args;
        let version = &ltx_args.ltx_version;
        let full_config = get_config_by_version(version);
        let inference = &full_config.inference;

        let height = ltx_args.ltx_height;
        let width = ltx_args.ltx_width;
        let num_frames = ltx_args.ltx_num_frames;
        let num_steps = ltx_args.ltx_num_steps.unwrap_or(inference.num_inference_steps);
        let frame_rate = ltx_args.ltx_fps;
        let guidance_scale = inference.guidance_scale;

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }

        info!(
            "Generating LTX video: {}x{}, {} frames, {} steps, guidance={}, version={}",
            width, height, num_frames, num_steps, guidance_scale, version
        );

        // Transformer config for pack/unpack
        let tcfg = LtxTransformer::pipeline_config(version);
        let vae_spatial = full_config.vae.spatial_compression_ratio;
        let vae_temporal = full_config.vae.temporal_compression_ratio;
        let patch_size = tcfg.patch_size;
        let patch_size_t = tcfg.patch_size_t;

        // 1. Encode prompt with T5
        info!("Encoding prompt with T5...");
        let t5_tokens = self
            .t5_tokenizer
            .encode(image_prompt.as_str(), true)
            .map_err(E::msg)?;
        let t5_token_ids = t5_tokens.get_ids().to_vec();
        let t5_input =
            Tensor::new(t5_token_ids.as_slice(), &self.context.device)?.unsqueeze(0)?;
        let prompt_embeds = LtxT5::encode(&mut self.t5_encoder, t5_input.clone(), &mut self.context)
            .await?
            .to_dtype(self.context.dtype)?;
        info!("T5 encoding done: {:?}", prompt_embeds.shape());

        // Create attention mask (all 1s for actual tokens)
        let seq_len = prompt_embeds.dim(1)?;
        let prompt_mask = Tensor::ones((1, seq_len), DType::F32, &self.context.device)?
            .to_dtype(self.context.dtype)?;

        // 2. Prepare latents
        let latent_h = height / vae_spatial;
        let latent_w = width / vae_spatial;
        let latent_f = (num_frames - 1) / vae_temporal + 1;
        let num_channels = tcfg.in_channels; // 128

        let latents_5d = Tensor::randn(
            0f32,
            1f32,
            (1, num_channels, latent_f, latent_h, latent_w),
            &self.context.device,
        )?
        .to_dtype(self.context.dtype)?;

        // Pack latents: [B, C, F, H, W] -> [B, S, D]
        let mut latents =
            LtxPipeline::pack_latents(&latents_5d, patch_size, patch_size_t)?;

        // 3. Prepare RoPE video coordinates
        let video_coords = self.prepare_video_coords(
            latent_f,
            latent_h,
            latent_w,
            vae_temporal,
            vae_spatial,
            frame_rate,
        )?;

        // 4. Prepare scheduler
        let video_seq_len = latent_f * latent_h * latent_w;

        // Get timesteps from config or compute sigmas
        let timesteps: Vec<f32> = if let Some(ref ts) = inference.timesteps {
            ts.clone()
        } else {
            // Linspace from 1.0 to 1/num_steps
            let mut ts = Vec::with_capacity(num_steps);
            for i in 0..num_steps {
                ts.push(1.0 - (i as f32) / (num_steps as f32));
            }
            ts
        };

        // Compute mu for time shifting
        let sched_cfg = &full_config.scheduler;
        let base_seq = sched_cfg.base_image_seq_len.unwrap_or(256);
        let max_seq = sched_cfg.max_image_seq_len.unwrap_or(4096);
        let base_shift = sched_cfg.base_shift.unwrap_or(0.5);
        let max_shift = sched_cfg.max_shift.unwrap_or(1.15);
        let mu = t2v_pipeline::calculate_shift(
            video_seq_len,
            base_seq,
            max_seq,
            base_shift as f32,
            max_shift as f32,
        );

        // Initialize scheduler and set timesteps
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(full_config.scheduler.clone())?;
        let sigmas: Vec<f32> = timesteps.clone();
        scheduler.set_timesteps(
            None,
            &self.context.device,
            Some(&sigmas),
            Some(mu),
            None,
        )?;
        // Get timesteps as f32 vector
        let schedule: Vec<f32> = scheduler.timesteps.to_vec1()?;

        info!(
            "Denoising: {} steps, mu={:.4}, seq_len={}",
            schedule.len(),
            mu,
            video_seq_len
        );

        // 5. Denoising loop
        for (step, &t) in schedule.iter().enumerate() {
            let start_time = std::time::Instant::now();

            let b = latents.dim(0)?;
            let timestep_t =
                Tensor::full(t as f32, (b,), &self.context.device)?
                    .to_dtype(self.context.dtype)?;

            let noise_pred = LtxTransformer::forward_packed(
                &mut self.transformer,
                latents.to_dtype(self.context.dtype)?,
                prompt_embeds.clone(),
                timestep_t,
                prompt_mask.clone(),
                video_coords.clone(),
                latent_f,
                latent_h,
                latent_w,
                &mut self.context,
            )
            .await?
            .to_dtype(DType::F32)?;

            // Euler step
            let step_output = scheduler.step(&noise_pred, t, &latents, None)?;
            latents = step_output.prev_sample;

            let dt = start_time.elapsed().as_secs_f32();
            info!("step {}/{} done, {:.2}s", step + 1, schedule.len(), dt);
        }

        // 6. Unpack latents: [B, S, D] -> [B, C, F, H, W]
        let latents_5d = LtxPipeline::unpack_latents(
            &latents,
            latent_f,
            latent_h,
            latent_w,
            patch_size,
            patch_size_t,
        )?;

        // 7. Denormalize latents
        let vae_config = &full_config.vae;
        let latents_mean =
            Tensor::new(vae_config.latents_mean.as_slice(), &self.context.device)?
                .to_dtype(DType::F32)?;
        let latents_std =
            Tensor::new(vae_config.latents_std.as_slice(), &self.context.device)?
                .to_dtype(DType::F32)?;
        let latents_5d = LtxPipeline::denormalize_latents(
            &latents_5d.to_dtype(DType::F32)?,
            &latents_mean,
            &latents_std,
            vae_config.scaling_factor as f32,
        )?
        .to_dtype(self.context.dtype)?;

        // 8. Decode with VAE
        info!("Decoding with VAE...");
        let decode_timestep = inference
            .decode_timestep
            .as_ref()
            .and_then(|v| v.first().copied());
        let decode_noise_scale = inference
            .decode_noise_scale
            .as_ref()
            .and_then(|v| v.first().copied());

        // Optionally add noise for timestep-conditioned decoding
        let (latents_for_decode, vae_timestep) = if let Some(dt) = decode_timestep {
            let dns = decode_noise_scale.unwrap_or(dt);
            let noise = Tensor::randn(0f32, 1f32, latents_5d.dims(), &self.context.device)?
                .to_dtype(self.context.dtype)?;
            let scale =
                Tensor::full(dns, latents_5d.dims(), &self.context.device)?
                    .to_dtype(self.context.dtype)?;
            let one_minus = scale.affine(-1.0, 1.0)?;
            let noised = latents_5d.mul(&one_minus)?.add(&noise.mul(&scale)?)?;
            let ts = Tensor::full(dt, (1,), &self.context.device)?
                .to_dtype(self.context.dtype)?;
            (noised, Some(ts))
        } else {
            (latents_5d, None)
        };

        let decoded = LtxVae::decode(
            &mut self.vae,
            latents_for_decode,
            vae_timestep,
            &mut self.context,
        )
        .await?;

        // 9. Convert video frames to images
        let frames = self.video_tensor_to_images(&decoded)?;
        info!("Generated {} frames", frames.len());

        Ok(VideoOutput::new(
            frames,
            frame_rate,
            width as u32,
            height as u32,
        ))
    }
}

impl LtxVideo {
    /// Prepare 3D RoPE coordinates for the video latent grid.
    fn prepare_video_coords(
        &self,
        latent_f: usize,
        latent_h: usize,
        latent_w: usize,
        vae_temporal: usize,
        vae_spatial: usize,
        frame_rate: usize,
    ) -> Result<Tensor> {
        let device = &self.context.device;

        let grid_f = Tensor::arange(0u32, latent_f as u32, device)?.to_dtype(DType::F32)?;
        let grid_h = Tensor::arange(0u32, latent_h as u32, device)?.to_dtype(DType::F32)?;
        let grid_w = Tensor::arange(0u32, latent_w as u32, device)?.to_dtype(DType::F32)?;

        let f = grid_f
            .reshape((latent_f, 1, 1))?
            .broadcast_as((latent_f, latent_h, latent_w))?;
        let h = grid_h
            .reshape((1, latent_h, 1))?
            .broadcast_as((latent_f, latent_h, latent_w))?;
        let w = grid_w
            .reshape((1, 1, latent_w))?
            .broadcast_as((latent_f, latent_h, latent_w))?;

        // Stack [3, F, H, W] -> flatten -> [3, seq] -> transpose -> [seq, 3] -> [1, seq, 3]
        let coords = Tensor::stack(&[f.contiguous()?, h.contiguous()?, w.contiguous()?], 0)?
            .flatten_from(1)?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        // Apply causal fix and spatial scaling
        let vf = coords.i((.., .., 0))?;
        let vh = coords.i((.., .., 1))?;
        let vw = coords.i((.., .., 2))?;

        let ts_ratio = vae_temporal as f64;
        let sp_ratio = vae_spatial as f64;

        // CAUSAL FIX: (L * temporal_ratio + 1 - temporal_ratio).clamp(0) / frame_rate
        let vf = vf
            .affine(ts_ratio, 1.0 - ts_ratio)?
            .clamp(0.0f32, 1000.0f32)?
            .affine(1.0 / (frame_rate as f64), 0.0)?;

        // SPATIAL SCALE: L * spatial_ratio
        let vh = vh.affine(sp_ratio, 0.0)?;
        let vw = vw.affine(sp_ratio, 0.0)?;

        let video_coords =
            Tensor::stack(&[vf, vh, vw], candle_core::D::Minus1)?;

        Ok(video_coords)
    }

    /// Convert a decoded video tensor [B, C, T, H, W] to a list of RGB images (one per frame).
    fn video_tensor_to_images(
        &self,
        video: &Tensor,
    ) -> Result<Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>> {
        let mut result = Vec::new();

        // Video output is in [-1, 1] range, convert to [0, 255]
        let video = ((video.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
            .to_dtype(DType::U8)?
            .to_device(&Device::Cpu)?;

        let bsize = video.dim(0)?;
        for batch in 0..bsize {
            let batch_video = video.i(batch)?; // [C, T, H, W]
            let (channels, num_frames, height, width) = batch_video.dims4()?;
            if channels != 3 {
                anyhow::bail!("Expected 3 channels, got {}", channels);
            }

            for frame in 0..num_frames {
                let frame_tensor = batch_video.i((.., frame, .., ..))?; // [C, H, W]
                let frame_tensor = frame_tensor.permute((1, 2, 0))?.flatten_all()?;
                let pixels = frame_tensor.to_vec1::<u8>()?;

                let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(width as u32, height as u32, pixels)
                        .ok_or_else(|| anyhow::anyhow!("Error creating image buffer"))?;
                result.push(image);
            }
        }

        Ok(result)
    }
}

fn resolve_hf_file(repo: &str, file: &str, cache_dir: &str) -> Result<PathBuf> {
    let mut cache_path = PathBuf::from(cache_dir);
    cache_path.push("hub");
    let cache = Cache::new(cache_path);
    let api = ApiBuilder::from_cache(cache).build()?;
    let filename = api.model(repo.to_string()).get(file)?;
    Ok(filename)
}
