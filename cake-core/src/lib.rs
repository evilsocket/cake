//! This is the core library where all Cake logic is implemented.
#[macro_use]
extern crate anyhow;

use cake::Mode;

use clap::{Parser, ValueEnum};
use serde::Deserialize;

pub mod cake;
pub mod models;
pub mod utils;

#[derive(Copy, Clone, Parser, Default, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum)]
pub enum ModelType {
    #[default]
    TextModel,
    ImageModel,
}

#[derive(Clone, Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// GPU device index.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Mode.
    #[arg(long, default_value_t, value_enum)]
    pub mode: Mode,
    /// Worker name.
    #[arg(long)]
    pub name: Option<String>,
    /// Binding address and port for workers.
    #[arg(long, default_value = "127.0.0.1:10128")]
    pub address: String,
    /// Enable OpenAI compatible chat completion API.
    #[arg(long)]
    pub api: Option<String>,
    /// Llama3 model data path.
    #[arg(long, default_value = "./cake-data/Meta-Llama-3-8B/")]
    pub model: String,
    /// Topology file.
    #[arg(long, default_value = "./cake-data/topology.yml")]
    pub topology: String,
    /// The initial prompt.
    #[arg(long, default_value = "The sky is blue because ")]
    pub prompt: String,
    /// The system prompt.
    #[arg(long, default_value = "You are a helpful AI assistant.")]
    pub system_prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 100)]
    pub sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Only sample among the top K samples.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    pub repeat_last_n: usize,
    /// Use different dtype than f16
    #[arg(long)]
    pub dtype: Option<String>,

    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    pub cpu: bool,

    #[arg(long, default_value = "text-model")]
    pub model_type: ModelType,

    #[clap(flatten)]
    pub sd_args: SDArgs,

    #[clap(flatten)]
    pub sd_img_gen_args: ImageGenerationArgs,
}

#[derive(Clone, Parser, Default, Debug)]
pub struct SDArgs {
    #[arg(long = "sd-tokenizer")]
    pub tokenizer: Option<String>,

    #[arg(long = "sd-tokenizer-2")]
    pub tokenizer_2: Option<String>,

    #[arg(long = "sd-version", value_enum, default_value = "v1-5")]
    sd_version: StableDiffusionVersion,

    #[arg(long = "sd-use-f16", default_value_t = true)]
    use_f16: bool,

    #[arg(long = "sd-width")]
    width: Option<usize>,

    #[arg(long = "sd-height")]
    height: Option<usize>,

    #[arg(long = "sd-sliced-attention-size")]
    sliced_attention_size: Option<usize>,

    #[arg(long = "sd-clip")]
    clip: Option<String>,

    #[arg(long = "sd-clip2")]
    clip2: Option<String>,

    #[arg(long = "sd-vae")]
    vae: Option<String>,

    #[arg(long = "sd-unet")]
    unet: Option<String>,

    #[arg(long = "sd-use-flash-attention", default_value_t = false)]
    use_flash_attention: bool,
}

fn default_prompt() -> String {
    "A very realistic photo of a rusty robot walking on a sandy beach".to_string()
}

fn empty_str() -> String {
    "".to_string()
}

fn usize_one() -> usize {
    1
}

fn default_img2img_strength() -> f64 {
    0.8
}

#[derive(Clone, Parser, Default, Debug, Deserialize)]
pub struct ImageGenerationArgs {
    /// The prompt to be used for image generation.
    #[arg(
        long = "sd-image-prompt",
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    #[serde(rename(deserialize = "sd-image-prompt"), default = "default_prompt")]
    image_prompt: String,

    #[arg(long = "sd-uncond-prompt", default_value = "")]
    #[serde(rename(deserialize = "sd-uncond-prompt"), default = "empty_str")]
    uncond_prompt: String,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long = "sd-tracing", default_value_t = false)]
    #[serde(rename(deserialize = "sd-tracing"), default)]
    tracing: bool,

    /// The number of steps to run the diffusion for.
    #[arg(long = "sd-n-steps")]
    #[serde(rename(deserialize = "sd-n-steps"))]
    n_steps: Option<usize>,

    /// The number of samples to generate iteratively.
    #[arg(long = "sd-num-samples", default_value_t = 1)]
    #[serde(rename(deserialize = "sd-num-samples"), default = "usize_one")]
    num_samples: usize,

    /// The numbers of samples to generate simultaneously.
    #[arg(long = "sd-bsize", default_value_t = 1)]
    #[serde(rename(deserialize = "sd-bsize"), default = "usize_one")]
    bsize: usize,

    /// Generate intermediary images every n steps.
    #[arg(long = "sd-intermediary-images", default_value_t = 0, action)]
    #[serde(rename(deserialize = "sd-intermediary-images"), default)]
    intermediary_images: usize,

    #[arg(long = "sd-guidance-scale")]
    #[serde(rename(deserialize = "sd-guidance-scale"))]
    guidance_scale: Option<f64>,

    #[arg(long = "sd-img2img", value_name = "FILE")]
    #[serde(rename(deserialize = "sd-img2img"))]
    img2img: Option<String>,

    /// The strength, indicates how much to transform the initial image. The
    /// value must be between 0 and 1, a value of 1 discards the initial image
    /// information.
    #[arg(long = "sd-img2img-strength", default_value_t = 0.8)]
    #[serde(
        rename(deserialize = "sd-img2img-strength"),
        default = "default_img2img_strength"
    )]
    img2img_strength: f64,

    /// The seed to use when generating random samples.
    #[arg(long = "sd-seed")]
    #[serde(rename(deserialize = "sd-seed"))]
    image_seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, Default)]
pub enum StableDiffusionVersion {
    #[default]
    V1_5,
    V2_1,
    Xl,
    Turbo,
}

impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}
