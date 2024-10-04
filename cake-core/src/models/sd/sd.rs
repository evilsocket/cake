use crate::cake::{Context, Forwarder};
use crate::models::sd::clip::Clip;
use crate::models::sd::safe_scheduler::SafeScheduler;
use crate::models::sd::sd_shardable::SDShardable;
use crate::models::sd::unet::UNet;
use crate::models::sd::vae::VAE;
use crate::models::{Generator, ImageGenerator};
use crate::{ImageGenerationArgs, SDArgs, StableDiffusionVersion};
use anyhow::{Error as E, Result};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use image::{ImageBuffer, Rgb};
use log::{debug, info};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl ModelFile {
    pub fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
        cache_dir: String,
    ) -> Result<std::path::PathBuf> {
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };
                let mut cache_path = std::path::PathBuf::from(cache_dir.as_str());
                cache_path.push("hub");

                debug!("Model cache dir: {:?}", cache_path);

                let cache = Cache::new(cache_path);
                let api = ApiBuilder::from_cache(cache).build()?;

                let filename = api.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match *self {
            ModelFile::Tokenizer => "tokenizer",
            ModelFile::Tokenizer2 => "tokenizer_2",
            ModelFile::Clip => "clip",
            ModelFile::Clip2 => "clip2",
            ModelFile::Unet => "unet",
            ModelFile::Vae => "vae",
        }
    }
}

pub struct SD {
    tokenizer: Tokenizer,
    pad_id: u32,
    tokenizer_2: Option<Tokenizer>,
    pad_id_2: Option<u32>,
    text_model: Box<dyn Forwarder>,
    text_model_2: Option<Box<dyn Forwarder>>,
    vae: Box<dyn Forwarder>,
    unet: Box<dyn Forwarder>,
    sd_version: StableDiffusionVersion,
    sd_config: StableDiffusionConfig,
    context: Context,
}

#[async_trait]
impl Generator for SD {
    type Shardable = SDShardable;
    const MODEL_NAME: &'static str = "stable-diffusion";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        let SDArgs {
            tokenizer,
            tokenizer_2,
            sd_version,
            use_f16,
            width,
            height,
            sliced_attention_size,
            clip,
            clip2,
            vae,
            unet,
            use_flash_attention,
            ..
        } = &context.args.sd_args;

        let sd_config = match *sd_version {
            StableDiffusionVersion::V1_5 => {
                StableDiffusionConfig::v1_5(*sliced_attention_size, *height, *width)
            }
            StableDiffusionVersion::V2_1 => {
                StableDiffusionConfig::v2_1(*sliced_attention_size, *height, *width)
            }
            StableDiffusionVersion::Xl => {
                StableDiffusionConfig::sdxl(*sliced_attention_size, *height, *width)
            }
            StableDiffusionVersion::Turbo => {
                StableDiffusionConfig::sdxl_turbo(*sliced_attention_size, *height, *width)
            }
        };

        // Tokenizer
        info!("Loading the Tokenizer...");

        let tokenizer_file = ModelFile::Tokenizer;
        let tokenizer = tokenizer_file.get(
            tokenizer.clone(),
            *sd_version,
            *use_f16,
            context.args.model.clone(),
        )?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

        let pad_id = match &sd_config.clip.pad_with {
            Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
        };

        info!("Tokenizer loaded!");

        // Tokenizer 2

        let mut tokenizer_2_option: Option<Tokenizer> = None;
        let mut pad_id_2: Option<u32> = None;

        if let StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo = sd_version {
            info!("Loading the Tokenizer 2...");

            let tokenizer_2_file = ModelFile::Tokenizer2;
            let tokenizer_2 = tokenizer_2_file.get(
                tokenizer_2.clone(),
                *sd_version,
                *use_f16,
                context.args.model.clone(),
            )?;
            let tokenizer_2 = Tokenizer::from_file(tokenizer_2).map_err(E::msg)?;

            if let Some(clip2) = &sd_config.clip2 {
                pad_id_2 = match &clip2.pad_with {
                    Some(padding) => {
                        Some(*tokenizer_2.get_vocab(true).get(padding.as_str()).unwrap())
                    }
                    None => Some(*tokenizer_2.get_vocab(true).get("<|endoftext|>").unwrap()),
                };
            }

            tokenizer_2_option = Some(tokenizer_2);

            info!("Tokenizer 2 loaded!");
        }

        // Clip
        info!("Loading the Clip text model.");

        let text_model: Box<dyn Forwarder>;

        if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Clip.name())
        {
            info!("node {node_name} will serve Clip");
            text_model = Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    ModelFile::Clip.name(),
                )
                .await?,
            );
        } else {
            info!("Clip will be served locally");
            text_model = Clip::load_model(
                ModelFile::Clip,
                clip.clone(),
                *sd_version,
                *use_f16,
                &context.device,
                context.dtype,
                context.args.model.clone(),
                &sd_config.clip,
            )?;
        }

        info!("Clip text model loaded!");

        // Clip 2

        let mut text_model_2: Option<Box<dyn Forwarder>> = None;
        if let StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo = sd_version {
            info!("Loading the Clip 2 text model.");

            if let Some((node_name, node)) =
                context.topology.get_node_for_layer(ModelFile::Clip2.name())
            {
                info!("node {node_name} will serve clip2");
                text_model_2 = Some(Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        ModelFile::Clip2.name(),
                    )
                    .await?,
                ));
            } else {
                info!("Clip 2 will be served locally");
                text_model_2 = Some(Clip::load_model(
                    ModelFile::Clip2,
                    clip2.clone(),
                    *sd_version,
                    *use_f16,
                    &context.device,
                    context.dtype,
                    context.args.model.clone(),
                    sd_config.clip2.as_ref().unwrap(),
                )?);
            }

            info!("Clip 2 text model loaded!");
        }

        // VAE
        info!("Loading the VAE...");

        let vae_model: Box<dyn Forwarder>;

        if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Vae.name())
        {
            info!("node {node_name} will serve VAE");
            vae_model = Box::new(
                crate::cake::Client::new(context.device.clone(), &node.host, ModelFile::Vae.name())
                    .await?,
            );
        } else {
            info!("VAE will be served locally");
            vae_model = VAE::load_model(
                vae.clone(),
                *sd_version,
                *use_f16,
                &context.device,
                context.dtype,
                context.args.model.clone(),
                &sd_config,
            )?;
        }

        info!("VAE loaded!");

        // Unet
        info!("Loading the UNet.");

        let unet_model: Box<dyn Forwarder>;
        if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Unet.name())
        {
            info!("node {node_name} will serve UNet");
            unet_model = Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    ModelFile::Unet.name(),
                )
                .await?,
            );
        } else {
            info!("UNet will be served locally");
            unet_model = UNet::load_model(
                unet.clone(),
                *use_flash_attention,
                *sd_version,
                *use_f16,
                &context.device,
                context.dtype,
                context.args.model.clone(),
                &sd_config,
            )?;
        }

        info!("UNet loaded!");

        Ok(Some(Box::new(Self {
            tokenizer,
            sd_version: *sd_version,
            sd_config,
            pad_id,
            text_model,
            tokenizer_2: tokenizer_2_option,
            pad_id_2,
            text_model_2,
            vae: vae_model,
            unet: unet_model,
            context: context.clone(),
        })))
    }
}

#[async_trait]
impl ImageGenerator for SD {
    async fn generate_image<F>(
        &mut self,
        args: &ImageGenerationArgs,
        mut callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static,
    {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        let ImageGenerationArgs {
            image_prompt,
            uncond_prompt,
            n_steps,
            num_samples,
            bsize,
            tracing,
            guidance_scale,
            img2img,
            img2img_strength,
            image_seed,
            intermediary_images,
            ..
        } = args;

        let sd_version = self.sd_version;

        if !(0. ..=1.).contains(img2img_strength) {
            anyhow::bail!("img2img-strength should be between 0 and 1, got {img2img_strength}")
        }

        let _guard = if *tracing {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };

        let guidance_scale = match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => &match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };
        let n_steps = match n_steps {
            Some(n_steps) => n_steps,
            None => &match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 30,
                StableDiffusionVersion::Turbo => 1,
            },
        };

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }
        let use_guide_scale = guidance_scale > &1.0;

        let mut text_embeddings: Vec<Tensor> = Vec::new();

        let text_embeddings_1 = self
            .text_embeddings(&image_prompt, &uncond_prompt, use_guide_scale, true)
            .await?;

        text_embeddings.push(text_embeddings_1);

        if let StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo = sd_version {
            let text_embeddings_2 = self
                .text_embeddings(&image_prompt, &uncond_prompt, use_guide_scale, false)
                .await?;

            text_embeddings.push(text_embeddings_2);
        }

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat((*bsize, 1, 1))?;
        debug!("{text_embeddings:?}");

        let init_latent_dist_sample = match &img2img {
            None => None,
            Some(image) => {
                let image = image_preprocess(image)?.to_device(&self.context.device)?;
                Some(VAE::encode(&mut self.vae, image, &mut self.context).await?)
            }
        };

        let t_start = if img2img.is_some() {
            *n_steps - (*n_steps as f64 * img2img_strength) as usize
        } else {
            0
        };

        let vae_scale = match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };

        let safe_scheduler = SafeScheduler {
            scheduler: self.sd_config.build_scheduler(*n_steps)?,
        };

        for idx in 0..(*num_samples) {
            let timesteps = safe_scheduler.scheduler.timesteps();
            let latents = match &init_latent_dist_sample {
                Some(init_latent_dist) => {
                    let latents =
                        (init_latent_dist * vae_scale)?.to_device(&self.context.device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        safe_scheduler
                            .scheduler
                            .add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }

                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        (
                            *bsize,
                            4,
                            self.sd_config.height / 8,
                            self.sd_config.width / 8,
                        ),
                        &&self.context.device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * safe_scheduler.scheduler.init_noise_sigma())?
                }
            };

            let mut latents = latents.to_dtype(self.context.dtype)?;

            debug!("Starting sampling...");

            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let start_time = std::time::Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                let latent_model_input = safe_scheduler
                    .scheduler
                    .scale_model_input(latent_model_input, timestep)?;

                debug!("UNet forwarding...");

                let noise_pred = UNet::forward_unpacked(
                    &mut self.unet,
                    latent_model_input,
                    text_embeddings.clone(),
                    timestep,
                    &mut self.context,
                )
                .await?;

                debug!("UNet forwarding completed!");

                let noise_pred = if use_guide_scale {
                    debug!("Applying guidance scale...");

                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond
                        + ((noise_pred_text - noise_pred_uncond)? * *guidance_scale)?)?
                } else {
                    noise_pred
                };

                debug!("Scheduler stepping...");

                latents = safe_scheduler
                    .scheduler
                    .step(&noise_pred, timestep, &latents)?;

                let dt = start_time.elapsed().as_secs_f32();
                info!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

                if *intermediary_images != 0 && timestep_index % *intermediary_images == 0 {
                    let intermediary_batched_images =
                        self.split_images(&latents, vae_scale, *bsize).await?;
                    callback(intermediary_batched_images);
                }
            }

            debug!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );

            let batched_images = self.split_images(&latents, vae_scale, *bsize).await?;

            callback(batched_images);
        }

        Ok(())
    }
}

impl SD {
    async fn split_images(
        &mut self,
        latents: &Tensor,
        vae_scale: f64,
        bsize: usize,
    ) -> Result<Vec<ImageBuffer<image::Rgb<u8>, Vec<u8>>>> {
        let mut images_vec = Vec::new();

        let scaled = (latents / vae_scale)?;
        let images = VAE::decode(&mut self.vae, scaled, &mut self.context).await?;
        let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
        for batch in 0..bsize {
            let image_tensor = images.i(batch)?;
            let (channel, height, width) = image_tensor.dims3()?;
            if channel != 3 {
                anyhow::bail!("save_image expects an input of shape (3, height, width)")
            }
            let image_tensor = image_tensor.permute((1, 2, 0))?.flatten_all()?;
            let pixels = image_tensor.to_vec1::<u8>()?;

            let image: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                match ImageBuffer::from_raw(width as u32, height as u32, pixels) {
                    Some(image) => image,
                    None => anyhow::bail!("Error splitting images"),
                };
            images_vec.push(image)
        }
        Ok(images_vec)
    }

    async fn text_embeddings(
        &mut self,
        prompt: &str,
        uncond_prompt: &str,
        use_guide_scale: bool,
        first: bool,
    ) -> Result<Tensor> {
        let tokenizer;
        let text_model;
        let pad_id;
        let max_token_embeddings;

        if first {
            tokenizer = &self.tokenizer;
            text_model = &mut self.text_model;
            pad_id = self.pad_id;
            max_token_embeddings = self.sd_config.clip.max_position_embeddings;
        } else {
            tokenizer = self.tokenizer_2.as_ref().unwrap();
            text_model = self.text_model_2.as_mut().unwrap();
            pad_id = self.pad_id_2.unwrap();
            max_token_embeddings = self
                .sd_config
                .clip2
                .as_ref()
                .unwrap()
                .max_position_embeddings;
        }

        info!("Running with prompt \"{prompt}\".");

        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        if tokens.len() > max_token_embeddings {
            anyhow::bail!(
                "the prompt is too long, {} > max-tokens ({})",
                tokens.len(),
                max_token_embeddings
            )
        }

        while tokens.len() < max_token_embeddings {
            tokens.push(pad_id)
        }

        let tokens = Tensor::new(tokens.as_slice(), &&self.context.device)?.unsqueeze(0)?;

        let text_embeddings = text_model
            .forward_mut(&tokens, 0, 0, &mut self.context)
            .await?;

        let text_embeddings = if use_guide_scale {
            let mut uncond_tokens = tokenizer
                .encode(uncond_prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            if uncond_tokens.len() > max_token_embeddings {
                anyhow::bail!(
                    "the negative prompt is too long, {} > max-tokens ({})",
                    uncond_tokens.len(),
                    max_token_embeddings
                )
            }
            while uncond_tokens.len() < max_token_embeddings {
                uncond_tokens.push(pad_id)
            }

            let uncond_tokens =
                Tensor::new(uncond_tokens.as_slice(), &&self.context.device)?.unsqueeze(0)?;

            info!("Clip forwarding...");
            let uncond_embeddings = text_model
                .forward_mut(&uncond_tokens, 0, 0, &mut self.context)
                .await?;

            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(self.context.dtype)?
        } else {
            text_embeddings.to_dtype(self.context.dtype)?
        };

        Ok(text_embeddings)
    }
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}
