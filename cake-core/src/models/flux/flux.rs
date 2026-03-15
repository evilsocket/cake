//! FLUX.2-klein image generation pipeline.
//!
//! Implements `Generator` + `ImageGenerator` for the FLUX.2-klein-4B model.
//! Components (text encoder, transformer, VAE) can each run locally or
//! on remote workers via the Forwarder abstraction.

use crate::cake::{Context, Forwarder};
use crate::models::flux::config::FluxModelFile;
use crate::models::flux::flux_shardable::FluxShardable;
use crate::models::flux::text_encoder::FluxTextEncoder;
use crate::models::flux::transformer::FluxTransformerForwarder;
use crate::models::flux::vae::FluxVAE;
use crate::models::sd::util::unpack_tensors;
use crate::models::{Generator, ImageGenerator};
use crate::ImageGenerationArgs;
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::flux::sampling;
use image::{ImageBuffer, Rgb};
use log::info;
use tokenizers::Tokenizer;

pub struct FluxGen {
    tokenizer: Tokenizer,
    // Components are loaded on-demand to fit in VRAM.
    // Text encoder is run once then dropped before the transformer loads.
    context: Context,
    height: usize,
    width: usize,
}

#[async_trait]
impl Generator for FluxGen {
    type Shardable = FluxShardable;
    const MODEL_NAME: &'static str = "flux";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        let model_repo = &context.args.model;

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .to_string_lossy()
            .to_string();

        // Load tokenizer (lightweight, always in memory)
        info!("Loading FLUX tokenizer...");
        let tokenizer_path = FluxModelFile::Tokenizer.get(model_repo, &cache_dir)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
        info!("FLUX tokenizer loaded");

        // Pre-download all component files so generate_image doesn't block on network
        info!("Pre-downloading FLUX model files...");
        let _ = FluxModelFile::TextEncoder.get_all(model_repo, &cache_dir)?;
        let _ = FluxModelFile::Transformer.get(model_repo, &cache_dir)?;
        let _ = FluxModelFile::Vae.get(model_repo, &cache_dir)?;
        info!("All FLUX model files ready");

        let height = context.args.flux_args.height;
        let width = context.args.flux_args.width;

        Ok(Some(Box::new(Self {
            tokenizer,
            context: context.clone(),
            height,
            width,
        })))
    }
}

#[async_trait]
impl ImageGenerator for FluxGen {
    async fn generate_image<F>(
        &mut self,
        args: &ImageGenerationArgs,
        mut callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static,
    {
        let ImageGenerationArgs {
            image_prompt,
            image_seed,
            ..
        } = args;

        let num_steps = self.context.args.flux_args.num_steps;

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }

        let dev = self.context.device.clone();

        // 1. Tokenize with chat template (required for FLUX.2-klein Qwen3 encoder)
        info!("Tokenizing prompt: \"{}\"", image_prompt);
        let chat_text = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            image_prompt
        );
        let tokens = self
            .tokenizer
            .encode(chat_text.as_str(), false)
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let token_ids = tokens.get_ids().to_vec();
        info!("Tokenized to {} tokens", token_ids.len());
        let token_tensor =
            Tensor::new(token_ids.as_slice(), &dev)?.unsqueeze(0)?;

        // 2. Text encode — run on CPU to avoid GPU VRAM contention
        info!("Loading and running text encoder (CPU)...");
        let txt = {
            let cpu_ctx = Context {
                device: Device::Cpu,
                dtype: DType::F32,
                ..self.context.clone()
            };
            let mut text_encoder = FluxTextEncoder::load(
                FluxModelFile::TextEncoder.name().to_string(),
                &cpu_ctx,
            )?;
            let mut cpu_ctx = cpu_ctx;
            let enc_output = text_encoder
                .forward_mut(&token_tensor.to_device(&Device::Cpu)?, 0, 0, &mut cpu_ctx)
                .await?;
            let enc_tensors = unpack_tensors(&enc_output)?;
            enc_tensors[0].clone()
        };
        info!("Text encoding done (shape={:?})", txt.shape());

        // 3. Load transformer (now fits in VRAM)
        info!("Loading transformer...");
        let mut transformer: Box<dyn Forwarder> = FluxTransformerForwarder::load(
            FluxModelFile::Transformer.name().to_string(),
            &self.context,
        )?;

        // Move text embeddings back to GPU
        let txt = txt.to_device(&dev)?;

        // 4. Generate noise latents (32-channel for FLUX.2-klein)
        info!("Generating noise latents ({}x{})...", self.height, self.width);
        let h_lat2 = self.height.div_ceil(16) * 2;
        let w_lat2 = self.width.div_ceil(16) * 2;
        let img = Tensor::randn(0f32, 1., (1, 32, h_lat2, w_lat2), &dev)?;
        let h_half = h_lat2 / 2;
        let w_half = w_lat2 / 2;

        let img_packed = img
            .reshape((1, 32, h_half, 2, w_half, 2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((1, h_half * w_half, 32 * 4))?; // (1, seq, 128)

        // Build 4-axis image IDs: [T=0, H=h_coord, W=w_coord, L=0]
        let dtype = img_packed.dtype();
        let img_ids = Tensor::stack(
            &[
                Tensor::full(0u32, (h_half, w_half), &dev)?,   // T axis
                Tensor::arange(0u32, h_half as u32, &dev)?     // H axis
                    .reshape((h_half, 1))?
                    .broadcast_as((h_half, w_half))?,
                Tensor::arange(0u32, w_half as u32, &dev)?     // W axis
                    .reshape((1, w_half))?
                    .broadcast_as((h_half, w_half))?,
                Tensor::full(0u32, (h_half, w_half), &dev)?,   // L axis
            ],
            2,
        )?
        .to_dtype(dtype)?
        .reshape((1, h_half * w_half, 4))?;

        // Text IDs: [T=0, H=0, W=0, L=seq_pos]
        let txt_seq_len = txt.dim(1)?;
        let txt_l = Tensor::arange(0u32, txt_seq_len as u32, &dev)?
            .to_dtype(dtype)?
            .unsqueeze(0)?; // (1, seq)
        let txt_zeros = Tensor::zeros((1, txt_seq_len), dtype, &dev)?;
        let txt_ids = Tensor::stack(
            &[&txt_zeros, &txt_zeros, &txt_zeros, &txt_l],
            2, // stack along last dim
        )?; // (1, seq, 4)

        // 5. Schedule (flow-matching Euler steps with dynamic shifting)
        let image_seq_len = h_half * w_half;
        // Dynamic shift: mu = base_shift + (max_shift - base_shift) * (seq_len - 256) / (4096 - 256)
        let base_shift = 0.5_f64;
        let max_shift = 1.15_f64;
        let timesteps = sampling::get_schedule(num_steps, Some((image_seq_len, base_shift, max_shift)));

        // Debug: print tensor norms
        let txt_norm: f32 = txt.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        let img_norm: f32 = img_packed.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        info!("txt_norm={txt_norm:.4}, img_norm={img_norm:.4}, txt_shape={:?}, img_shape={:?}",
              txt.shape(), img_packed.shape());
        info!("Starting denoising ({num_steps} steps)...");

        // 6. Denoise loop (flow-matching Euler ODE)
        let mut img = img_packed;

        for (step, window) in timesteps.windows(2).enumerate() {
            let (t_curr, t_prev) = (window[0], window[1]);
            let t_vec = Tensor::full(t_curr as f32, 1, &dev)?;

            let pred = FluxTransformerForwarder::forward_unpacked(
                &mut transformer,
                img.clone(),
                img_ids.clone(),
                txt.clone(),
                txt_ids.clone(),
                t_vec,
                &mut self.context,
            )
            .await?;

            let pred_norm: f32 = pred.to_dtype(DType::F32)?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            // Euler step: img = img + pred * (t_prev - t_curr)
            let pred = pred.to_dtype(img.dtype())?;
            img = (img + pred * (t_prev - t_curr))?;
            let img_norm_after: f32 = img.to_dtype(DType::F32)?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

            info!("step {}/{num_steps}: t={t_curr:.3}→{t_prev:.3}, pred_norm={pred_norm:.4}, img_norm={img_norm_after:.4}", step + 1);
        }

        // Drop transformer to free VRAM for VAE
        drop(transformer);

        // 7. BN denormalization on patchified latents: (b, seq, 128)
        info!("Loading VAE and decoding latents...");
        let vae_loaded = FluxVAE::load_model(
            &self.context.device,
            self.context.dtype,
            &self.context.args.model,
        )?;
        // Access the BN stats from the loaded VAE for denormalization
        let bn_mean = &vae_loaded.bn_running_mean;
        let bn_var = &vae_loaded.bn_running_var;
        let bn_eps = 0.0001_f64;
        let bn_std = bn_var
            .to_dtype(DType::F32)?
            .broadcast_add(&Tensor::new(&[bn_eps as f32], bn_var.device())?)?
            .sqrt()?;
        // img: (b, seq, 128), bn_std/bn_mean: (128,) → broadcast as (1, 1, 128)
        let img = img.to_dtype(DType::F32)?;
        let img = img
            .broadcast_mul(&bn_std.unsqueeze(0)?.unsqueeze(0)?)?
            .broadcast_add(&bn_mean.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?)?;

        // 8. Unpatchify: (b, h*w, 128) → (b, 32, h*2, w*2)
        let img = img
            .reshape((1, h_half, w_half, 32, 2, 2))?    // (b, h, w, c, ph, pw)
            .permute((0, 3, 1, 4, 2, 5))?               // (b, c, h, ph, w, pw)
            .reshape((1, 32, h_half * 2, w_half * 2))?;  // (b, 32, h_lat, w_lat)

        // 9. VAE decode
        let mut vae: Box<dyn Forwarder> = vae_loaded;
        let decoded = FluxVAE::decode(&mut vae, img, &mut self.context).await?;

        // 9. Convert to RGB images
        let images = ((decoded / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;

        let image_tensor = images.i(0)?;
        let (channel, height, width) = image_tensor.dims3()?;
        if channel != 3 {
            anyhow::bail!("expected 3 channels, got {channel}");
        }
        let image_tensor = image_tensor.permute((1, 2, 0))?.flatten_all()?;
        let pixels = image_tensor.to_vec1::<u8>()?;

        let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, pixels)
                .ok_or_else(|| anyhow!("failed to create image buffer"))?;

        info!("Image generated ({}x{})", width, height);
        callback(vec![image]);

        Ok(())
    }
}
