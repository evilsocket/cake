//! This is the core library where all Cake logic is implemented.
#[macro_use]
extern crate anyhow;

use cake::Mode;

use clap::{Parser, ValueEnum};
use serde::Deserialize;

pub mod cake;
pub mod models;
pub mod utils;
pub mod video;

#[derive(Copy, Clone, Parser, Default, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum)]
pub enum ModelType {
    #[default]
    TextModel,
    ImageModel,
}

/// Supported image model architectures.
#[derive(Copy, Clone, Parser, Default, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum)]
pub enum ImageModelArch {
    /// Auto-detect (defaults to Stable Diffusion)
    #[default]
    Auto,
    /// Stable Diffusion (v1.5, v2.1, XL, Turbo)
    SD,
    /// FLUX.2-klein (flow-matching transformer)
    Flux,
    /// Lightricks LTX-Video (0.9.x series)
    LtxVideo,
    /// Lightricks LTX-2 (19B audio+video, Gemma-3 text encoder)
    Ltx2,
}

/// Supported text model architectures.
#[derive(Copy, Clone, Parser, Default, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum)]
pub enum TextModelArch {
    /// Auto-detect from config.json
    #[default]
    Auto,
    /// LLaMA family (also covers SmolLM2, DeepSeek-R1 distilled)
    Llama,
    /// Qwen2/Qwen2.5 family
    Qwen2,
    /// Qwen3.5 hybrid linear/full attention
    Qwen3_5,
    /// Qwen3 dense (GQA + QK-norm)
    Qwen3,
    /// Qwen3 MoE sparse mixture-of-experts
    Qwen3Moe,
    /// Qwen3.5 MoE sparse mixture-of-experts (35B-A3B)
    Qwen3_5Moe,
    /// Phi-4-mini / Phi-4 (pre-fused QKV + gate_up)
    Phi4,
    /// Mistral family (standard GQA + optional sliding window)
    Mistral,
    /// Gemma 3 (interleaved local/global attention)
    Gemma3,
    /// Falcon3 (standard GQA, Apache 2.0)
    Falcon3,
    /// OLMo 2 (post-norm, QK-norm)
    OLMo2,
    /// EXAONE 4.0 (3:1 local/global hybrid, QK-norm)
    EXAONE4,
}

#[derive(Clone, Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// GPU device index.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Mode (set by subcommand, not directly by user).
    #[arg(skip)]
    pub mode: Mode,
    /// Worker name.
    #[arg(long)]
    pub name: Option<String>,
    /// Binding address and port for workers.
    #[arg(long, default_value = "0.0.0.0:10128")]
    pub address: String,
    /// Enable OpenAI compatible chat completion API.
    #[arg(long)]
    pub api: Option<String>,
    /// Path to model directory, or HuggingFace repo ID (e.g., Qwen/Qwen2.5-Coder-1.5B-Instruct).
    #[arg(long, default_value = "./cake-data/Meta-Llama-3-8B/")]
    pub model: String,
    /// Topology file.
    #[arg(long)]
    pub topology: Option<String>,
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
    #[arg(short = 'n', long, default_value_t = 2048)]
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

    /// Cluster key for zero-config mDNS discovery and PSK authentication.
    /// When set on both master and workers, enables automatic discovery,
    /// layer assignment, and model data push without topology files.
    #[arg(long, env = "CAKE_CLUSTER_KEY")]
    pub cluster_key: Option<String>,

    /// How long to wait for worker discovery (seconds). 0 = skip discovery.
    #[arg(long, default_value_t = 10)]
    pub discovery_timeout: u64,

    /// Stop discovery as soon as this many workers have been found (0 = wait full timeout).
    #[arg(long, default_value_t = 0)]
    pub min_workers: usize,

    /// Optional basic auth for the web UI (format: "user:pass").
    #[arg(long)]
    pub ui_auth: Option<String>,

    /// Topology built during zero-config setup (not a CLI arg).
    #[arg(skip)]
    pub topology_override: Option<cake::Topology>,

    /// Draft model for speculative decoding (path or HuggingFace repo).
    /// Must share the same tokenizer as the main model.
    /// Example: --draft-model Qwen/Qwen2.5-0.5B-Instruct
    #[arg(long)]
    pub draft_model: Option<String>,

    /// Number of speculative tokens to draft before verification (default: 4).
    #[arg(long, default_value_t = 4)]
    pub spec_tokens: usize,

    /// Run on CPU rather than on GPU.
    #[arg(long, default_value_t = false)]
    pub cpu: bool,

    #[arg(long, default_value = "text-model")]
    pub model_type: ModelType,

    /// Text model architecture (auto-detected from config.json if omitted).
    #[arg(long, default_value = "auto")]
    pub text_model_arch: TextModelArch,

    /// Image model architecture (defaults to auto/stable-diffusion).
    #[arg(long, default_value = "auto")]
    pub image_model_arch: ImageModelArch,

    #[clap(flatten)]
    pub sd_args: SDArgs,

    #[clap(flatten)]
    pub sd_img_gen_args: ImageGenerationArgs,

    #[clap(flatten)]
    pub flux_args: FluxArgs,

    #[clap(flatten)]
    pub ltx_args: LtxVideoArgs,
}

#[derive(Clone, clap::Args, Default, Debug)]
pub struct FluxArgs {
    /// Image height for FLUX generation.
    #[arg(long = "flux-height", default_value_t = 1024, id = "flux_height")]
    pub height: usize,

    /// Image width for FLUX generation.
    #[arg(long = "flux-width", default_value_t = 1024, id = "flux_width")]
    pub width: usize,

    /// Number of denoising steps (4 for distilled FLUX.2-klein).
    #[arg(long = "flux-steps", default_value_t = 4, id = "flux_steps")]
    pub num_steps: usize,

    /// Guidance scale for classifier-free guidance.
    #[arg(long = "flux-guidance", default_value_t = 3.5, id = "flux_guidance")]
    pub guidance_scale: f64,
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


#[derive(Clone, Parser, Default, Debug)]
pub struct LtxVideoArgs {
    /// LTX-Video model version (e.g., "0.9.8-13b-distilled").
    #[arg(long = "ltx-version", default_value = "0.9.8-13b-distilled")]
    pub ltx_version: String,

    /// Override HuggingFace repo for LTX-Video weights.
    #[arg(long = "ltx-model")]
    pub ltx_model: Option<String>,

    /// Override path to LTX transformer weights (safetensors).
    #[arg(long = "ltx-transformer")]
    pub ltx_transformer: Option<String>,

    /// Override path to T5-XXL encoder weights (safetensors, comma-separated for sharded).
    #[arg(long = "ltx-t5")]
    pub ltx_t5: Option<String>,

    /// Override path to T5 config.json.
    #[arg(long = "ltx-t5-config")]
    pub ltx_t5_config: Option<String>,

    /// Override path to T5 tokenizer (tokenizer.json).
    #[arg(long = "ltx-t5-tokenizer")]
    pub ltx_t5_tokenizer: Option<String>,

    /// Override path to LTX VAE weights (safetensors).
    #[arg(long = "ltx-vae")]
    pub ltx_vae: Option<String>,

    /// Number of video frames to generate.
    #[arg(long = "ltx-num-frames", default_value_t = 41)]
    pub ltx_num_frames: usize,

    /// Video frame rate.
    #[arg(long = "ltx-fps", default_value_t = 24)]
    pub ltx_fps: usize,

    /// Output video height.
    #[arg(long = "ltx-height", default_value_t = 512)]
    pub ltx_height: usize,

    /// Output video width.
    #[arg(long = "ltx-width", default_value_t = 704)]
    pub ltx_width: usize,

    /// Number of sampling steps (default from model config).
    #[arg(long = "ltx-num-steps")]
    pub ltx_num_steps: Option<usize>,

    /// STG (Spatio-Temporal Guidance) scale. 0 to disable. Default: 1.0.
    #[arg(long = "ltx-stg-scale")]
    pub ltx_stg_scale: Option<f32>,

    /// STG block index to perturb. Default: 28 (LTX-2.3).
    #[arg(long = "ltx-stg-block")]
    pub ltx_stg_block: Option<usize>,

    /// Guidance rescale factor. Prevents oversaturation. Default: 0.7.
    #[arg(long = "ltx-rescale")]
    pub ltx_rescale: Option<f32>,

    /// Path to a GGUF file for quantized Gemma-3 (runs on GPU instead of CPU).
    /// Example: --ltx-gemma-gguf /path/to/gemma-3-12b-pt-Q4_K_M.gguf
    #[arg(long = "ltx-gemma-gguf")]
    pub ltx_gemma_gguf: Option<String>,
}

impl LtxVideoArgs {
    /// Get the HuggingFace repo ID for the LTX-Video model.
    pub fn ltx_repo(&self) -> String {
        if let Some(ref repo) = self.ltx_model {
            return repo.clone();
        }
        match self.ltx_version.as_str() {
            // LTX-2.3 (22B, improved training + gated attention)
            "2.3" | "2.3-dev" | "2.3-22b-dev" => "Lightricks/LTX-2.3".to_string(),
            "2.3-distilled" | "2.3-22b-distilled" => "Lightricks/LTX-2.3".to_string(),

            // LTX-2 (19B, audio+video, Gemma-3 text encoder)
            "2-19b-dev" | "2.0" | "2" => "Lightricks/LTX-2".to_string(),
            "2-19b-distilled" => "Lightricks/LTX-2".to_string(),

            // LTX-Video 0.9.8
            "0.9.8-13b-distilled" | "0.9.8-13b" => {
                "Lightricks/LTX-Video-0.9.8-13b-distilled".to_string()
            }
            "0.9.8-13b-dev" => "Lightricks/LTX-Video-0.9.8-13b-dev".to_string(),
            "0.9.8-2b-distilled" | "0.9.8-distilled" => {
                "Lightricks/LTX-Video-0.9.8-distilled".to_string()
            }

            // LTX-Video 0.9.6
            "0.9.6-distilled" | "0.9.6-2b-distilled" => {
                "Lightricks/LTX-Video-0.9.6-distilled".to_string()
            }
            "0.9.6-dev" | "0.9.6-2b-dev" => "Lightricks/LTX-Video-0.9.6-dev".to_string(),

            _ => "Lightricks/LTX-Video".to_string(),
        }
    }

    /// Whether this is an LTX-2.3 model (gated attention, 8 connector blocks).
    pub fn is_ltx23(&self) -> bool {
        let repo = self.ltx_repo();
        repo.contains("LTX-2.3") || self.ltx_version.starts_with("2.3")
    }
}
