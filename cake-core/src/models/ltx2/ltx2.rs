use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use image::{ImageBuffer, Rgb};
use log::info;
use std::path::PathBuf;

use super::gemma::Ltx2Gemma;
use super::gemma_encoder::{gemma3_12b_config, Gemma3TextEncoder};
use super::ltx2_shardable::Ltx2Shardable;
use super::transformer::Ltx2Transformer;
use super::vae_forwarder::Ltx2Vae;
use super::vocoder::Ltx2Vocoder;
use super::vendored::config::{Ltx2SchedulerConfig, Ltx2TransformerConfig, Ltx2VaeConfig};
use super::vendored::pipeline::{
    build_video_positions, denormalize_latents, normalize_latents, pack_latents, unpack_latents,
};
use super::vendored::scheduler::{euler_step, Ltx2Scheduler};
use crate::cake::{Context, Forwarder};
use crate::models::{Generator, VideoGenerator};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;

/// LTX-2 model (19B audio+video generation).
///
/// Architecture:
/// - Asymmetric dual-stream DiT transformer (14B video + 5B audio)
/// - Gemma-3 12B text encoder (quantized to Q4)
/// - Video VAE decoder (native 4K support)
/// - Audio vocoder (synchronized with video)
///
/// Component topology:
/// ```yaml
/// gpu1:
///   host: "worker1:10128"
///   layers: ["ltx2-transformer"]  # ~19GB (FP8)
/// gpu2:
///   host: "worker2:10128"
///   layers: ["ltx2-gemma"]        # ~6GB (Q4)
/// # Master keeps ltx2-vae (~400MB) + ltx2-vocoder (~200MB)
/// ```
pub struct Ltx2 {
    gemma_encoder: Box<dyn Forwarder>,
    gemma_text_encoder: Option<Gemma3TextEncoder>,
    transformer: Box<dyn Forwarder>,
    vae: Box<dyn Forwarder>,
    #[allow(dead_code)]
    vocoder: Box<dyn Forwarder>,
    context: Context,
}

#[async_trait]
impl Generator for Ltx2 {
    type Shardable = Ltx2Shardable;
    const MODEL_NAME: &'static str = "ltx-2";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        info!("Loading LTX-2 components...");

        // Gemma-3 text encoder
        let gemma_encoder: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-gemma") {
                info!("ltx2-gemma will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-gemma",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                Ltx2Gemma::load_model(context)?
            };

        // Transformer
        let transformer: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-transformer") {
                info!("ltx2-transformer will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-transformer",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                Ltx2Transformer::load_model(&context)?
            };

        // VAE
        let vae: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-vae") {
                info!("ltx2-vae will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-vae",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                Ltx2Vae::load_model(context)?
            };

        // Vocoder
        let vocoder: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-vocoder") {
                info!("ltx2-vocoder will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-vocoder",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                Ltx2Vocoder::load_model(context)?
            };

        // Try to load Gemma-3 text encoder for direct text-to-video
        let gemma_text_encoder = match Self::try_load_gemma_encoder(context) {
            Ok(enc) => {
                info!("Gemma-3 text encoder loaded — text prompts are supported!");
                Some(enc)
            }
            Err(e) => {
                log::warn!(
                    "Gemma-3 text encoder not available: {}. \
                     Pre-computed embeddings must be provided.",
                    e
                );
                None
            }
        };

        info!("LTX-2 components loaded");

        Ok(Some(Box::new(Self {
            gemma_encoder,
            gemma_text_encoder,
            transformer,
            vae,
            vocoder,
            context: context.clone(),
        })))
    }
}

impl Ltx2 {
    /// Try to load the Gemma-3 12B model for text encoding.
    fn try_load_gemma_encoder(ctx: &Context) -> Result<Gemma3TextEncoder> {
        use hf_hub::api::sync::ApiBuilder;
        use hf_hub::Cache;

        let gemma_repo = "google/gemma-3-12b-pt";

        let mut cache_path = PathBuf::from(&ctx.args.model);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let model_api = api.model(gemma_repo.to_string());

        let tokenizer_path = model_api.get("tokenizer.json")?;

        // Parse config
        let config_path = model_api.get("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let gemma_config: candle_transformers::models::gemma3::Config =
            serde_json::from_str(&config_str).unwrap_or_else(|_| gemma3_12b_config());

        // Find safetensors files (handle sharded models)
        let model_paths = if let Ok(index_file) = model_api.get("model.safetensors.index.json") {
            let index_str = std::fs::read_to_string(&index_file)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
            let weight_map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index"))?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let mut paths = Vec::new();
            for shard in &shard_files {
                paths.push(model_api.get(shard)?);
            }
            paths
        } else {
            vec![model_api.get("model.safetensors")?]
        };

        Gemma3TextEncoder::load(
            &model_paths,
            &tokenizer_path,
            &gemma_config,
            ctx.dtype,
            &ctx.device,
        )
    }
}

#[async_trait]
impl VideoGenerator for Ltx2 {
    async fn generate_video(&mut self, args: &ImageGenerationArgs) -> Result<VideoOutput> {
        let ImageGenerationArgs {
            image_prompt: _,
            image_seed,
            ..
        } = args;

        let ltx_args = &self.context.args.ltx_args;

        let height = ltx_args.ltx_height;
        let width = ltx_args.ltx_width;
        let num_frames = ltx_args.ltx_num_frames;
        let num_steps = ltx_args.ltx_num_steps.unwrap_or(30);
        let frame_rate = ltx_args.ltx_fps;

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }

        let trans_config = Ltx2TransformerConfig::default();
        let vae_config = Ltx2VaeConfig::default();
        let sched_config = Ltx2SchedulerConfig::default();

        info!(
            "Generating LTX-2 video: {}x{}, {} frames, {} steps",
            width, height, num_frames, num_steps
        );

        // 1. Encode prompt with Gemma-3 → connector
        info!("Encoding prompt through text connector...");
        let prompt_text = if args.image_prompt.is_empty() {
            "a beautiful video"
        } else {
            &args.image_prompt
        };

        let (packed_embeds, text_mask) = if let Some(ref mut encoder) = self.gemma_text_encoder {
            // Use Gemma-3 encoder for real text encoding
            info!("Encoding text with Gemma-3: \"{}\"", prompt_text);
            encoder.encode(prompt_text)?
        } else {
            // Fallback: dummy packed embeddings (for testing without Gemma weights)
            log::warn!("Using dummy text embeddings (Gemma-3 not loaded)");
            let connector_seq_len = 1024usize;
            let packed_dim = trans_config.caption_channels * 49; // 3840 * 49 = 188160
            let dummy = Tensor::randn(
                0f32,
                1f32,
                (1, connector_seq_len, packed_dim),
                &self.context.device,
            )?
            .to_dtype(self.context.dtype)?;
            let mask = Tensor::ones(
                (1, connector_seq_len),
                DType::F32,
                &self.context.device,
            )?;
            (dummy, mask)
        };

        let prompt_embeds = Ltx2Gemma::encode(
            &mut self.gemma_encoder,
            packed_embeds,
            Some(text_mask),
            &mut self.context,
        )
        .await?
        .to_dtype(self.context.dtype)?;

        // The connector returns [B, seq_len, cross_attention_dim] with an attention mask.
        // The Gemma forwarder returns the embeddings; the mask is all-ones since
        // registers replace all padding. We use the full sequence.
        let ctx_seq_len = prompt_embeds.dim(1)?;
        let context_mask =
            Tensor::ones((1, ctx_seq_len), DType::F32, &self.context.device)?
                .to_dtype(self.context.dtype)?;

        info!("Text connector done: {:?}", prompt_embeds.shape());

        // 2. Prepare latents
        let latent_h = height / vae_config.spatial_compression_ratio;
        let latent_w = width / vae_config.spatial_compression_ratio;
        let latent_f = (num_frames - 1) / vae_config.temporal_compression_ratio + 1;
        let in_channels = trans_config.in_channels;

        let latents_5d = Tensor::randn(
            0f32,
            1f32,
            (1, in_channels, latent_f, latent_h, latent_w),
            &self.context.device,
        )?
        .to_dtype(self.context.dtype)?;

        // Normalize initial noise
        let latents_mean = Tensor::new(vae_config.latents_mean.as_slice(), &self.context.device)?;
        let latents_std = Tensor::new(vae_config.latents_std.as_slice(), &self.context.device)?;
        let latents_5d = normalize_latents(
            &latents_5d.to_dtype(DType::F32)?,
            &latents_mean,
            &latents_std,
            vae_config.scaling_factor,
        )?
        .to_dtype(self.context.dtype)?;

        // Pack latents: [B, C, F, H, W] -> [B, S, C] (patch_size=1)
        let mut latents = pack_latents(&latents_5d)?;

        // 3. Build video positions for RoPE
        let positions = build_video_positions(
            1, // batch_size
            latent_f,
            latent_h,
            latent_w,
            vae_config.temporal_compression_ratio,
            vae_config.spatial_compression_ratio,
            frame_rate,
            &self.context.device,
        )?;

        // 4. Prepare scheduler
        let num_tokens = latent_f * latent_h * latent_w;
        let scheduler = Ltx2Scheduler::new(sched_config);
        let sigmas = scheduler.execute(num_steps, num_tokens);

        info!(
            "Denoising: {} steps, {} tokens, sigma range {:.4}..{:.4}",
            num_steps,
            num_tokens,
            sigmas.first().unwrap_or(&0.0),
            sigmas.last().unwrap_or(&0.0),
        );

        // 5. Denoising loop
        for step in 0..num_steps {
            let start_time = std::time::Instant::now();

            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];

            let sigma_t = Tensor::full(sigma, (1,), &self.context.device)?
                .to_dtype(self.context.dtype)?;
            // Timestep = 1 - sigma (flow matching convention)
            let timestep_t = Tensor::full(1.0 - sigma, (1,), &self.context.device)?
                .to_dtype(self.context.dtype)?;

            // Scale input by sigma: noisy_input = sample * (1 - sigma) + noise * sigma
            // For velocity prediction, input is just the latents at current sigma level

            let velocity = Ltx2Transformer::forward_packed(
                &mut self.transformer,
                latents.to_dtype(self.context.dtype)?,
                sigma_t.clone(),
                timestep_t,
                positions.clone(),
                prompt_embeds.clone(),
                context_mask.clone(),
                &mut self.context,
            )
            .await?
            .to_dtype(DType::F32)?;

            // Euler step
            latents = euler_step(&latents.to_dtype(DType::F32)?, &velocity, sigma, sigma_next)?
                .to_dtype(self.context.dtype)?;

            let dt = start_time.elapsed().as_secs_f32();
            info!("step {}/{} done, sigma={:.4}, {:.2}s", step + 1, num_steps, sigma, dt);
        }

        // 6. Unpack latents: [B, S, C] -> [B, C, F, H, W]
        let latents_5d = unpack_latents(
            &latents.to_dtype(DType::F32)?,
            latent_f,
            latent_h,
            latent_w,
        )?;

        // 7. Denormalize latents
        let latents_5d = denormalize_latents(
            &latents_5d,
            &latents_mean,
            &latents_std,
            vae_config.scaling_factor,
        )?
        .to_dtype(self.context.dtype)?;

        // 8. Decode with VAE
        info!("Decoding with VAE...");
        let decoded = Ltx2Vae::decode(
            &mut self.vae,
            latents_5d,
            &mut self.context,
        )
        .await?;

        // 9. Convert video frames to images
        let frames = video_tensor_to_images(&decoded)?;
        info!("Generated {} frames", frames.len());

        Ok(VideoOutput::new(
            frames,
            frame_rate,
            width as u32,
            height as u32,
        ))
    }
}

/// Convert a decoded video tensor `[B, C, T, H, W]` to a list of RGB images.
///
/// Values are expected in `[-1, 1]` and are mapped to `[0, 255]` uint8.
fn video_tensor_to_images(video: &Tensor) -> Result<Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>> {
    let mut result = Vec::new();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_tensor_to_images_basic() {
        let device = Device::Cpu;
        // Create a simple [1, 3, 2, 4, 4] video tensor with values in [-1, 1]
        let video = Tensor::zeros((1, 3, 2, 4, 4), DType::F32, &device).unwrap();
        let frames = video_tensor_to_images(&video).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].width(), 4);
        assert_eq!(frames[0].height(), 4);
        // Zero maps to (0+1)*127.5 = 127
        assert_eq!(frames[0].get_pixel(0, 0)[0], 127);
    }

    #[test]
    fn test_video_tensor_to_images_clamping() {
        let device = Device::Cpu;
        // Values outside [-1, 1] should be clamped
        let video = Tensor::full(2.0f32, (1, 3, 1, 2, 2), &device).unwrap();
        let frames = video_tensor_to_images(&video).unwrap();
        assert_eq!(frames.len(), 1);
        // 2.0 clamped to 1.0, mapped to (1+1)*127.5 = 255
        assert_eq!(frames[0].get_pixel(0, 0)[0], 255);
    }

    #[test]
    fn test_video_tensor_to_images_multi_batch() {
        let device = Device::Cpu;
        let video = Tensor::zeros((2, 3, 3, 4, 4), DType::F32, &device).unwrap();
        let frames = video_tensor_to_images(&video).unwrap();
        // 2 batches * 3 frames = 6 total
        assert_eq!(frames.len(), 6);
    }
}
