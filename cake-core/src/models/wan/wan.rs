// Wan2.2-T2V-A14B video generation pipeline.
//
// Implements Generator + VideoGenerator for distributed inference.

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5;
use hf_hub::api::sync::ApiBuilder;
use image::{ImageBuffer, Rgb};
use log::info;
use tokenizers::Tokenizer;

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::pack_tensors;
use crate::models::{Generator, VideoGenerator};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;

use super::transformer::WanTransformer;
use super::wan_shardable::WanShardable;
use super::vendored::pipeline::{denormalize_latents, latent_spatial, num_latent_frames};
use super::vendored::scheduler::WanFlowMatchScheduler;

pub struct Wan {
    transformer: Box<dyn Forwarder>,
    vae: Box<dyn Forwarder>,
    context: Context,
}

#[async_trait]
impl Generator for Wan {
    type Shardable = WanShardable;
    const MODEL_NAME: &'static str = "wan";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        info!("Loading Wan2.2 video generation model...");

        // Load transformer
        let transformer: Box<dyn Forwarder> =
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer("wan-transformer") {
                info!("node {node_name} will serve wan-transformer");
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "wan-transformer",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("wan-transformer will be served locally");
                WanTransformer::load("wan-transformer".to_string(), ctx)?
            };

        // Load VAE
        let vae: Box<dyn Forwarder> =
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer("wan-vae") {
                info!("node {node_name} will serve wan-vae");
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "wan-vae",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("wan-vae will be served locally");
                super::vae_forwarder::WanVae::load("wan-vae".to_string(), ctx)?
            };

        info!("Wan2.2 model loaded");

        Ok(Some(Box::new(Self {
            transformer,
            vae,
            context: ctx.clone(),
        })))
    }
}

#[async_trait]
impl VideoGenerator for Wan {
    async fn generate_video(&mut self, args: &ImageGenerationArgs) -> Result<VideoOutput> {
        let prompt = &args.image_prompt;
        let seed = args.image_seed.unwrap_or(42);
        // Use LTX args for configurable video params, fall back to Wan defaults
        let num_frames = self.context.args.ltx_args.ltx_num_frames;
        let height = self.context.args.ltx_args.ltx_height;
        let width = self.context.args.ltx_args.ltx_width;
        let num_steps = self.context.args.ltx_args.ltx_num_steps.unwrap_or(40);
        let guidance_scale = args.guidance_scale.unwrap_or(4.0) as f32;
        // Wan diffusers default is shift=1.0 (linear schedule), not 5.0
        let shift = 1.0;

        let device = self.context.device.clone();
        let dtype = self.context.dtype;

        info!(
            "Generating Wan2.2 video: {}x{}, {} frames, {} steps, guidance={:.1}",
            width, height, num_frames, num_steps, guidance_scale,
        );

        // Text encoding: load UMT5-XXL on CPU, encode, then drop to free memory
        let text_len = 512;
        let (context, uncond_context) = encode_prompt_umt5(
            prompt, text_len, dtype, &device, &self.context.args.model,
        )?;

        // Compute latent dimensions
        let lat_frames = num_latent_frames(num_frames, 4);
        let (lat_h, lat_w) = latent_spatial(height, width, 8);

        info!("Latent shape: [{}, 16, {}, {}, {}]", 1, lat_frames, lat_h, lat_w);

        // Generate initial noise (load Python noise if available)
        let latents = if std::path::Path::new("/tmp/wan_python_noise.json").exists() {
            info!("Loading noise from /tmp/wan_python_noise.json");
            let data: serde_json::Value = serde_json::from_reader(
                std::fs::File::open("/tmp/wan_python_noise.json")?)?;
            let vals: Vec<f32> = data["data"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let shape: Vec<usize> = data["shape"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect();
            Tensor::from_vec(vals, shape.as_slice(), &device)?
        } else {
            device.set_seed(seed)?;
            Tensor::randn(0f32, 1.0, (1, 16, lat_frames, lat_h, lat_w), &device)?
        };

        // Flow matching schedule
        let scheduler = WanFlowMatchScheduler::new(shift);
        let sigmas = scheduler.sigmas(num_steps);
        let timesteps = scheduler.timesteps(num_steps);

        info!("Starting denoising ({} steps, shift={})...", num_steps, shift);

        let mut latents = latents;
        let do_cfg = guidance_scale > 1.0;

        for step in 0..num_steps {
            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];
            let t = timesteps[step] as f32;
            let dt = (sigma_next - sigma) as f64;

            let t_tensor = Tensor::from_slice(&[t], 1, &device)?.to_dtype(dtype)?;

            // Conditional pass (convert latents to model dtype for transformer)
            let noise_pred = WanTransformer::forward_packed(
                &mut self.transformer,
                latents.to_dtype(dtype)?,
                t_tensor.clone(),
                context.clone(),
                lat_frames, lat_h, lat_w,
                &mut self.context,
            )
            .await?;


            let noise_pred = if do_cfg {
                // Unconditional pass
                let noise_uncond = WanTransformer::forward_packed(
                    &mut self.transformer,
                    latents.to_dtype(dtype)?,
                    t_tensor,
                    uncond_context.clone(),
                    lat_frames, lat_h, lat_w,
                    &mut self.context,
                )
                .await?;

                // CFG in F32 for precision
                let pred_f32 = noise_pred.to_dtype(DType::F32)?;
                let uncond_f32 = noise_uncond.to_dtype(DType::F32)?;
                let diff = (&pred_f32 - &uncond_f32)?;
                (&uncond_f32 + diff.affine(guidance_scale as f64, 0.0)?)?
            } else {
                noise_pred
            };

            // Euler step in F32
            let pred_f32 = noise_pred.to_dtype(DType::F32)?;
            latents = (&latents + pred_f32.affine(dt, 0.0)?)?;



            info!(
                "step {}/{}: sigma={:.4} -> {:.4}",
                step + 1, num_steps, sigma, sigma_next,
            );
        }



        // VAE decode
        info!("Decoding with VAE...");
        let direction = Tensor::from_slice(&[0f32], 1, &device)?;
        let packed = pack_tensors(vec![direction, latents], &device)?;
        let video = self.vae.forward_mut(&packed, 0, 0, &mut self.context).await?;

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

        let fps = 16; // Wan default
        info!(
            "Generated {} frames at {}x{}, {}fps",
            frames.len(), out_w, out_h, fps
        );

        Ok(VideoOutput::new(frames, fps, out_w as u32, out_h as u32))
    }
}

/// Encode a text prompt using UMT5-XXL on CPU, pad/truncate to `max_len`,
/// and return (cond_embeds, uncond_embeds) on the target device.
fn encode_prompt_umt5(
    prompt: &str,
    max_len: usize,
    dtype: DType,
    target_device: &Device,
    model_path: &str,
) -> Result<(Tensor, Tensor)> {
    let cpu = Device::Cpu;

    // Resolve tokenizer from diffusers directory or HF
    let tokenizer_path = {
        let local = std::path::PathBuf::from(model_path).join("tokenizer/tokenizer.json");
        if local.exists() {
            local
        } else {
            let api = ApiBuilder::new().build()?;
            api.model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers".to_string())
                .get("tokenizer/tokenizer.json")?
        }
    };
    info!("Loading UMT5 tokenizer from {:?}...", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;

    // Tokenize prompt
    let encoding = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Truncate or pad to max_len
    token_ids.truncate(max_len);
    while token_ids.len() < max_len {
        token_ids.push(0); // pad_token_id = 0 for UMT5
    }
    let tokens = Tensor::new(token_ids.as_slice(), &cpu)?.unsqueeze(0)?;

    // Unconditional: empty string
    let uncond_encoding = tokenizer.encode("", true).map_err(anyhow::Error::msg)?;
    let mut uncond_ids: Vec<u32> = uncond_encoding.get_ids().to_vec();
    uncond_ids.truncate(max_len);
    while uncond_ids.len() < max_len {
        uncond_ids.push(0);
    }
    let uncond_tokens = Tensor::new(uncond_ids.as_slice(), &cpu)?.unsqueeze(0)?;

    // Load T5 config
    let config_path = {
        let local = std::path::PathBuf::from(model_path).join("text_encoder/config.json");
        if local.exists() {
            local
        } else {
            let api = ApiBuilder::new().build()?;
            api.model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers".to_string())
                .get("text_encoder/config.json")?
        }
    };
    info!("Loading UMT5 config from {:?}...", config_path);
    let mut config: t5::Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
    // UMT5 uses "umt5" model_type but T5 architecture — override if needed
    config.use_cache = false;

    // Load T5 weights on CPU
    let weight_dir = std::path::PathBuf::from(model_path).join("text_encoder");
    let weight_files: Vec<std::path::PathBuf> = if weight_dir.exists() {
        let mut files: Vec<_> = std::fs::read_dir(&weight_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect();
        files.sort();
        files
    } else {
        info!("Downloading UMT5-XXL weights from HuggingFace...");
        let api = ApiBuilder::new().build()?;
        let model_api = api.model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers".to_string());
        let mut files = Vec::new();
        for i in 1..=5 {
            let f = model_api.get(&format!("text_encoder/model-{i:05}-of-00005.safetensors"))?;
            files.push(f);
        }
        files
    };

    info!("Loading UMT5-XXL encoder on CPU ({} shards)...", weight_files.len());
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &cpu)?
    };
    let mut model = t5::T5EncoderModel::load(vb, &config)?;
    info!("UMT5-XXL loaded! Encoding prompt...");

    // Encode
    let cond_embeds = model.forward(&tokens)?;
    let uncond_embeds = model.forward(&uncond_tokens)?;
    info!("Text encoding done: cond={:?}, uncond={:?}", cond_embeds.shape(), uncond_embeds.shape());

    // Transfer to target device and dtype, drop model to free RAM
    let cond = cond_embeds.to_dtype(dtype)?.to_device(target_device)?;
    let uncond = uncond_embeds.to_dtype(dtype)?.to_device(target_device)?;
    drop(model);

    Ok((cond, uncond))
}
