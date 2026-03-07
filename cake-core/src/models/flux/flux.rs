use crate::cake::{Context, Forwarder};
use crate::models::flux::clip::FluxClip;
use crate::models::flux::flux_shardable::FluxShardable;
use crate::models::flux::t5::FluxT5;
use crate::models::flux::transformer::FluxTransformer;
use crate::models::flux::vae::FluxVae;
use crate::models::{Generator, ImageGenerator};
use crate::{FluxVariant, ImageGenerationArgs};
use anyhow::{Error as E, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::flux::sampling;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use image::{ImageBuffer, Rgb};
use log::{debug, info};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const FLUX_DEV_REPO: &str = "black-forest-labs/FLUX.1-dev";
const FLUX_SCHNELL_REPO: &str = "black-forest-labs/FLUX.1-schnell";

/// Identifies a Flux model file for HuggingFace resolution.
#[derive(Debug, Clone, Copy)]
pub enum FluxModelFile {
    Transformer,
    Vae,
    ClipWeights,
    ClipTokenizer,
    T5Config,
    T5Tokenizer,
}

impl FluxModelFile {
    fn repo_and_path(&self, variant: FluxVariant) -> (&'static str, &'static str) {
        let flux_repo = match variant {
            FluxVariant::Dev => FLUX_DEV_REPO,
            FluxVariant::Schnell => FLUX_SCHNELL_REPO,
        };
        match self {
            Self::Transformer => (
                flux_repo,
                match variant {
                    FluxVariant::Dev => "flux1-dev.safetensors",
                    FluxVariant::Schnell => "flux1-schnell.safetensors",
                },
            ),
            Self::Vae => (flux_repo, "ae.safetensors"),
            Self::ClipWeights => (flux_repo, "text_encoder/model.safetensors"),
            Self::ClipTokenizer => (flux_repo, "tokenizer/tokenizer.json"),
            Self::T5Config => (flux_repo, "text_encoder_2/config.json"),
            Self::T5Tokenizer => (flux_repo, "tokenizer_2/tokenizer.json"),
        }
    }

    pub fn get(
        &self,
        override_path: Option<String>,
        variant: FluxVariant,
        cache_dir: &str,
    ) -> Result<PathBuf> {
        if let Some(path) = override_path {
            return Ok(PathBuf::from(path));
        }
        let (repo, file) = self.repo_and_path(variant);
        let mut cache_path = PathBuf::from(cache_dir);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let filename = api.model(repo.to_string()).get(file)?;
        Ok(filename)
    }
}

/// Get T5 weight file paths (handles sharded weights).
pub fn get_t5_weight_files(
    override_path: Option<String>,
    variant: FluxVariant,
    cache_dir: &str,
) -> Result<Vec<PathBuf>> {
    if let Some(path) = override_path {
        // If user specifies a path, use it directly (single file or comma-separated)
        return Ok(path.split(',').map(|p| PathBuf::from(p.trim())).collect());
    }

    let flux_repo = match variant {
        FluxVariant::Dev => FLUX_DEV_REPO,
        FluxVariant::Schnell => FLUX_SCHNELL_REPO,
    };

    let mut cache_path = PathBuf::from(cache_dir);
    cache_path.push("hub");
    let cache = Cache::new(cache_path);
    let api = ApiBuilder::from_cache(cache).build()?;
    let model_api = api.model(flux_repo.to_string());

    // Try single file first
    if let Ok(path) = model_api.get("text_encoder_2/model.safetensors") {
        return Ok(vec![path]);
    }

    // Fall back to 2-shard format
    let shard1 = model_api.get("text_encoder_2/model-00001-of-00002.safetensors")?;
    let shard2 = model_api.get("text_encoder_2/model-00002-of-00002.safetensors")?;
    Ok(vec![shard1, shard2])
}

pub struct Flux {
    t5_tokenizer: Tokenizer,
    clip_tokenizer: Tokenizer,
    t5_encoder: Box<dyn Forwarder>,
    clip_encoder: Box<dyn Forwarder>,
    transformer: Box<dyn Forwarder>,
    vae: Box<dyn Forwarder>,
    variant: FluxVariant,
    context: Context,
}

#[async_trait]
impl Generator for Flux {
    type Shardable = FluxShardable;
    const MODEL_NAME: &'static str = "flux";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        let flux_args = &context.args.flux_args;
        let variant = flux_args.flux_variant;

        // Load T5 tokenizer
        info!("Loading T5 tokenizer...");
        let t5_tokenizer_path = FluxModelFile::T5Tokenizer.get(
            flux_args.flux_t5_tokenizer.clone(),
            variant,
            &context.args.model,
        )?;
        let t5_tokenizer = Tokenizer::from_file(&t5_tokenizer_path).map_err(E::msg)?;
        info!("T5 tokenizer loaded!");

        // Load CLIP tokenizer
        info!("Loading CLIP tokenizer...");
        let clip_tokenizer_path = FluxModelFile::ClipTokenizer.get(
            flux_args.flux_clip_tokenizer.clone(),
            variant,
            &context.args.model,
        )?;
        let clip_tokenizer = Tokenizer::from_file(&clip_tokenizer_path).map_err(E::msg)?;
        info!("CLIP tokenizer loaded!");

        // T5 encoder
        info!("Loading T5 encoder...");
        let t5_encoder: Box<dyn Forwarder> =
            if let Some((node_name, node)) = context.topology.get_node_for_layer("flux-t5") {
                info!("node {node_name} will serve flux-t5");
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "flux-t5",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("T5 encoder will be served locally");
                FluxT5::load_model(context)?
            };
        info!("T5 encoder ready!");

        // CLIP encoder
        info!("Loading CLIP encoder...");
        let clip_encoder: Box<dyn Forwarder> =
            if let Some((node_name, node)) = context.topology.get_node_for_layer("flux-clip") {
                info!("node {node_name} will serve flux-clip");
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "flux-clip",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("CLIP encoder will be served locally");
                FluxClip::load_model(context)?
            };
        info!("CLIP encoder ready!");

        // VAE
        info!("Loading Flux VAE...");
        let vae: Box<dyn Forwarder> =
            if let Some((node_name, node)) = context.topology.get_node_for_layer("flux-vae") {
                info!("node {node_name} will serve flux-vae");
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "flux-vae",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                info!("Flux VAE will be served locally");
                FluxVae::load_model(context)?
            };
        info!("Flux VAE ready!");

        // Transformer
        info!("Loading Flux transformer...");
        let transformer: Box<dyn Forwarder> = if let Some((node_name, node)) =
            context.topology.get_node_for_layer("flux-transformer")
        {
            info!("node {node_name} will serve flux-transformer");
            Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    "flux-transformer",
                    context.args.cluster_key.as_deref(),
                )
                .await?,
            )
        } else {
            info!("Flux transformer will be served locally");
            FluxTransformer::load_model(context)?
        };
        info!("Flux transformer ready!");

        Ok(Some(Box::new(Self {
            t5_tokenizer,
            clip_tokenizer,
            t5_encoder,
            clip_encoder,
            transformer,
            vae,
            variant,
            context: context.clone(),
        })))
    }
}

#[async_trait]
impl ImageGenerator for Flux {
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
            num_samples,
            image_seed,
            ..
        } = args;

        let flux_args = &self.context.args.flux_args;
        let height = flux_args.flux_height;
        let width = flux_args.flux_width;
        let guidance_scale = flux_args.flux_guidance_scale;
        let num_steps = flux_args.flux_num_steps.unwrap_or(match self.variant {
            FluxVariant::Dev => 50,
            FluxVariant::Schnell => 4,
        });

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }

        info!(
            "Generating Flux image: {}x{}, {} steps, guidance={}, variant={:?}",
            width, height, num_steps, guidance_scale, self.variant
        );

        // 1. Encode prompt with T5
        info!("Encoding prompt with T5...");
        let t5_tokens = self
            .t5_tokenizer
            .encode(image_prompt.as_str(), true)
            .map_err(E::msg)?;
        let t5_token_ids = t5_tokens.get_ids().to_vec();
        let t5_input =
            Tensor::new(t5_token_ids.as_slice(), &self.context.device)?.unsqueeze(0)?;
        let txt = FluxT5::encode(&mut self.t5_encoder, t5_input, &mut self.context)
            .await?
            .to_dtype(self.context.dtype)?;
        info!("T5 encoding done: {:?}", txt.shape());

        // 2. Encode prompt with CLIP
        info!("Encoding prompt with CLIP...");
        let clip_tokens = self
            .clip_tokenizer
            .encode(image_prompt.as_str(), true)
            .map_err(E::msg)?;
        let mut clip_token_ids = clip_tokens.get_ids().to_vec();
        // Pad CLIP tokens to max_position_embeddings (77)
        let clip_pad_id = *self
            .clip_tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .unwrap_or(&49407);
        while clip_token_ids.len() < 77 {
            clip_token_ids.push(clip_pad_id);
        }
        clip_token_ids.truncate(77);
        let clip_input =
            Tensor::new(clip_token_ids.as_slice(), &self.context.device)?.unsqueeze(0)?;
        let vec = FluxClip::encode(&mut self.clip_encoder, clip_input, &mut self.context)
            .await?
            .to_dtype(self.context.dtype)?;
        info!("CLIP encoding done: {:?}", vec.shape());

        for sample_idx in 0..(*num_samples) {
            info!("Generating sample {}/{}...", sample_idx + 1, num_samples);

            // 3. Generate initial noise
            let img = sampling::get_noise(1, height, width, &self.context.device)?
                .to_dtype(self.context.dtype)?;

            // 4. Create state (sets up img_ids, txt_ids, etc.)
            let state = sampling::State::new(&txt, &vec, &img)?;

            // 5. Get timestep schedule
            let img_seq_len = state.img.dim(1)?;
            let timesteps = sampling::get_schedule(
                num_steps,
                match self.variant {
                    FluxVariant::Dev => Some((img_seq_len, 0.5, 1.15)),
                    FluxVariant::Schnell => None,
                },
            );

            debug!("Timesteps: {:?}", timesteps);

            // 6. Denoising loop (rectified flow Euler integration)
            let mut img = state.img.clone();
            let guidance_tensor = if self.variant == FluxVariant::Dev {
                Some(
                    Tensor::full(guidance_scale as f32, img.dims()[0], &self.context.device)?
                        .to_dtype(self.context.dtype)?,
                )
            } else {
                None
            };

            for (step, (&t_curr, &t_prev)) in
                timesteps.iter().zip(timesteps[1..].iter()).enumerate()
            {
                let start_time = std::time::Instant::now();

                let t_vec = Tensor::full(t_curr as f32, img.dims()[0], &self.context.device)?
                    .to_dtype(self.context.dtype)?;

                let pred = FluxTransformer::forward_packed(
                    &mut self.transformer,
                    img.clone(),
                    state.img_ids.clone(),
                    state.txt.clone(),
                    state.txt_ids.clone(),
                    t_vec,
                    state.vec.clone(),
                    guidance_tensor.clone(),
                    &mut self.context,
                )
                .await?;

                img = (&img + &pred * (t_prev - t_curr))?;

                let dt = start_time.elapsed().as_secs_f32();
                info!("step {}/{} done, {:.2}s", step + 1, num_steps, dt);
            }

            // 7. Unpack from patches back to spatial
            let img = sampling::unpack(&img, height, width)?;

            // 8. Decode with VAE
            info!("Decoding with VAE...");
            let decoded = FluxVae::decode(&mut self.vae, img, &mut self.context).await?;

            // 9. Convert to image
            let images = self.tensor_to_images(&decoded)?;
            callback(images);
        }

        Ok(())
    }
}

impl Flux {
    fn tensor_to_images(
        &self,
        images: &Tensor,
    ) -> Result<Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>> {
        let mut result = Vec::new();

        // Flux VAE output is in [-1, 1] range, convert to [0, 255]
        let images = ((images.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
            .to_dtype(DType::U8)?
            .to_device(&Device::Cpu)?;

        let bsize = images.dim(0)?;
        for batch in 0..bsize {
            let image_tensor = images.i(batch)?;
            let (channel, height, width) = image_tensor.dims3()?;
            if channel != 3 {
                anyhow::bail!("Expected 3 channels, got {}", channel);
            }
            let image_tensor = image_tensor.permute((1, 2, 0))?.flatten_all()?;
            let pixels = image_tensor.to_vec1::<u8>()?;

            let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width as u32, height as u32, pixels)
                    .ok_or_else(|| anyhow::anyhow!("Error creating image buffer"))?;
            result.push(image);
        }

        Ok(result)
    }
}
