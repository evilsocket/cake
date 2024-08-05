use clap::Parser;
use tokenizers::Tokenizer;
use crate::cake::{Context, Forwarder};
use anyhow::{Error as E, Result};
use candle_core::{D, Device, DType, IndexOp, Tensor};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use candle_transformers::models::stable_diffusion::vae::DiagonalGaussianDistribution;
use image::{DynamicImage, ImageBuffer};
use tracing_subscriber::fmt::time;
use crate::models::{Generator, ImageGenerator};
use crate::{Args, SDArgs, StableDiffusionVersion};
use crate::models::llama3::Cache;
use crate::models::sd::clip::Clip;
use crate::models::sd::sd_shardable::SDShardable;
use crate::models::sd::unet::UNet;
use crate::models::sd::util::{get_device, pack_tensors};
use crate::models::sd::vae::VAE;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ImageGenerationArgs {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    #[arg(long, default_value = "")]
    uncond_prompt: String,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    /// The number of steps to run the diffusion for.
    #[arg(long)]
    n_steps: Option<usize>,

    /// The number of samples to generate iteratively.
    #[arg(long, default_value_t = 1)]
    num_samples: usize,

    /// The numbers of samples to generate simultaneously.
    #[arg[long, default_value_t = 1]]
    bsize: usize,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    #[arg(long)]
    guidance_scale: Option<f64>,

    #[arg(long, value_name = "FILE")]
    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    #[arg(long, default_value_t = 0.8)]
    img2img_strength: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

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
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
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
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
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
    dtype: DType,
    sd_version: StableDiffusionVersion,
    sd_config: StableDiffusionConfig,
    device: Device,
    cache: Cache
}

impl Generator for SD {
    type Shardable = SDShardable;
    const MODEL_NAME: &'static str = "stable-diffusion";

    async fn load(context: Context) -> Result<Option<Box<Self>>> {

        let Args {
            cpu,
            ..
        } = context.args;

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
        } = context.args.sd_args;

        let dtype = if use_f16 { DType::F16 } else { DType::F32 };
        let device = get_device(cpu)?;
        let cache = context.cache.clone();

        let sd_config = match sd_version {
            StableDiffusionVersion::V1_5 => {
                StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::V2_1 => {
                StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Xl => {
                StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Turbo => StableDiffusionConfig::sdxl_turbo(
                sliced_attention_size,
                height,
                width,
            ),
        };

        // Tokenizer
        println!("Loading the Tokenizer.");

        let tokenizer_file = ModelFile::Tokenizer;
        let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;

        let pad_id = match &sd_config.clip.pad_with {
            Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
            None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
        };

        // Tokenizer 2
        println!("Loading the Tokenizer 2.");

        let mut tokenizer_2_option: Option<Tokenizer> = None;
        let mut pad_id_2: Option<u32> = None;

        if let Some(StableDiffusionVersion::Xl) | Some(StableDiffusionVersion::Turbo) = sd_version {
            let tokenizer_2_file = ModelFile::Tokenizer2;
            let tokenizer_2 = tokenizer_2_file.get(tokenizer_2, sd_version, use_f16)?;
            let tokenizer_2 = Tokenizer::from_file(tokenizer_2).map_err(E::msg)?;

            if let Some(clip2) = &sd_config.clip2 {
                pad_id_2 = match &clip2.pad_with {
                    Some(padding) => Some(*tokenizer_2.get_vocab(true).get(padding.as_str()).unwrap()),
                    None => Some(*tokenizer_2.get_vocab(true).get("<|endoftext|>").unwrap()),
                };
            }
        }

        // Clip
        println!("Loading the Clip text model.");

        let text_model: dyn Forwarder;

        if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Clip.name()) {
            log::debug!("node {node_name} will serve Clip");
            text_model = Box::new(
                crate::cake::Client::new(context.device.clone(), &node.host, ModelFile::Clip.name())
                    .await?,
            );
        } else {
            log::debug!("Clip will be served locally");
            text_model = Clip::load_model(
                ModelFile::Clip,
                clip,
                sd_version,
                use_f16,
                &device,
                dtype,
                &sd_config.clip
            );
        }

        // Clip 2
        println!("Loading the Clip 2 text model.");

        let mut text_model_2: Option<dyn Forwarder> = None;
        if let Some(StableDiffusionVersion::Xl) | Some(StableDiffusionVersion::Turbo) = sd_version {
            if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Clip2.name()) {
                log::debug!("node {node_name} will serve clip2");
                text_model_2 = Some(Box::new(
                    crate::cake::Client::new(context.device.clone(), &node.host, ModelFile::Clip2.name())
                        .await?,
                ));
            } else {
                log::debug!("Clip 2 will be served locally");
                text_model_2 = Some(Clip::load_model(
                    ModelFile::Clip2,
                    clip2,
                    sd_version,
                    use_f16,
                    &device,
                    dtype,
                    sd_config.clip2.as_ref().unwrap()
                ));
            }
        }

        // VAE
        println!("Loading the VAE.");

        let vae_model: dyn Forwarder;

        if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Vae.name()) {
            log::debug!("node {node_name} will serve VAE");
            vae_model = Box::new(
                crate::cake::Client::new(context.device.clone(), &node.host, ModelFile::Vae.name())
                    .await?,
            );
        } else {
            log::debug!("VAE will be served locally");
            vae_model = VAE::load_model(
                vae,
                sd_version,
                use_f16,
                &device,
                dtype,
                &sd_config,
            )
        }

        // Unet
        println!("Loading the UNet.");

        let unet_model: dyn Forwarder;
        if let Some((node_name, node)) = context.topology.get_node_for_layer(ModelFile::Unet.name()) {
            log::debug!("node {node_name} will serve UNet");
            unet_model = Box::new(
                crate::cake::Client::new(context.device.clone(), &node.host, ModelFile::Unet.name())
                    .await?,
            );
        } else {
            log::debug!("UNet will be served locally");
            unet_model = UNet::load_model(
                unet,
                use_flash_attention,
                sd_version,
                use_f16,
                &device,
                dtype,
                &sd_config,
            )
        }

        Ok(Some(Box::new(Self {
            tokenizer,
            dtype,
            sd_version,
            sd_config,
            pad_id,
            text_model,
            tokenizer_2: tokenizer_2_option,
            pad_id_2,
            text_model_2,
            device,
            cache,
            vae: vae_model,
            unet: unet_model,
        })))
    }
}

impl ImageGenerator for SD {
    async fn generate_image(&mut self, args: ImageGenerationArgs) -> Result<Vec<ImageBuffer<image::Rgb<u8>, Vec<u8>>>> {
        use tracing_chrome::ChromeLayerBuilder;
        use tracing_subscriber::prelude::*;

        let ImageGenerationArgs {
            prompt,
            uncond_prompt,
            n_steps,
            num_samples,
            bsize,
            tracing,
            guidance_scale,
            img2img,
            img2img_strength,
            seed,
            ..
        } = args;

        let sd_version = self.sd_version;

        if !(0. ..=1.).contains(&img2img_strength) {
            anyhow::bail!("img2img-strength should be between 0 and 1, got {img2img_strength}")
        }

        let _guard = if tracing {
            let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
            tracing_subscriber::registry().with(chrome_layer).init();
            Some(guard)
        } else {
            None
        };

        let guidance_scale = match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };
        let n_steps = match n_steps {
            Some(n_steps) => n_steps,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 30,
                StableDiffusionVersion::Turbo => 1,
            },
        };

        let scheduler = self.sd_config.build_scheduler(n_steps)?;

        if let Some(seed) = seed {
            self.device.set_seed(seed)?;
        }
        let use_guide_scale = guidance_scale > 1.0;

        let which = match sd_version {
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
            _ => vec![true],
        };

        let text_embeddings = which
            .iter()
            .map(|first| {
                self.text_embeddings(&prompt, &uncond_prompt, use_guide_scale, *first)
                    .expect("Error generating text embeddings");
            })
            .collect::<Result<Vec<_>>>()?;

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;
        println!("{text_embeddings:?}");

        let init_latent_dist_sample = match &img2img {
            None => None,
            Some(image) => {
                let image = image_preprocess(image)?.to_device(&self.device)?;
                Some(VAE::encode(&self.vae, image, &self.device, &mut self.cache).await?)
            }
        };

        let t_start = if img2img.is_some() {
            n_steps - (n_steps as f64 * img2img_strength) as usize
        } else {
            0
        };

        let vae_scale = match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };

        let mut final_images = Vec::new();

        for idx in 0..num_samples {
            let timesteps = scheduler.timesteps();
            let latents = match &init_latent_dist_sample {

                Some(init_latent_dist) => {
                    let latents = (init_latent_dist * vae_scale)?.to_device(&self.device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        scheduler.add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }

                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        (bsize, 4, self.sd_config.height / 8, self.sd_config.width / 8),
                        &self.device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };

            let mut latents = latents.to_dtype(self.dtype)?;

            println!("starting sampling");

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

                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;

                println!("UNet forwarding.");

                let noise_pred = UNet::forward_unpacked(
                    &self.unet,
                    latent_model_input,
                    text_embeddings.clone(),
                    timestep,
                    &self.device,
                    &mut self.cache,
                ).await?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                let dt = start_time.elapsed().as_secs_f32();
                println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );

            let mut batched_images = self.split_images(
                &latents,
                vae_scale,
                bsize,
            ).await?;

            final_images.append(&mut batched_images);
        }

        Ok(final_images)
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
        let images = VAE::decode(&self.vae, scaled, &self.device, &mut self.cache).await?;
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
                    None => anyhow::bail!("error saving image {p:?}"),
            };
            images_vec.push(image)
        }
        Ok(images_vec)
    }

    fn text_embeddings(
        & mut self,
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
            text_model = &self.text_model;
            pad_id = self.pad_id;
            max_token_embeddings = self.sd_config.clip.max_position_embeddings;
        } else {
            tokenizer = self.tokenizer_2.as_ref().unwrap();
            text_model = self.text_model_2.as_ref().unwrap();
            pad_id = self.pad_id_2.unwrap();
            max_token_embeddings = self.sd_config.clip2.as_ref().unwrap().max_position_embeddings;
        }

        println!("Running with prompt \"{prompt}\".");

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

        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        let cache = &mut self.cache;

        let text_embeddings = text_model.forward(&tokens, 0, 0, cache)?;

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

            let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            println!("Clip forwarding.");
            let uncond_embeddings = text_model.forward(&uncond_tokens, 0, 0, cache)?;

            Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(self.dtype)?
        } else {
            text_embeddings.to_dtype(self.dtype)?
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

