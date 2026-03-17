//! FLUX.2-klein configuration and model file resolution.

use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use std::path::PathBuf;

/// FLUX model component files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluxModelFile {
    Tokenizer,
    TextEncoder,
    Transformer,
    Vae,
}

impl FluxModelFile {
    /// Resolve the file path, downloading from HuggingFace if needed.
    /// Returns the list of files to download for this component.
    fn files(&self) -> Vec<&'static str> {
        match self {
            Self::Tokenizer => vec!["tokenizer/tokenizer.json"],
            Self::TextEncoder => vec![
                "text_encoder/model.safetensors.index.json",
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors",
                "text_encoder/config.json",
            ],
            Self::Transformer => vec!["transformer/diffusion_pytorch_model.safetensors"],
            Self::Vae => vec!["vae/diffusion_pytorch_model.safetensors"],
        }
    }

    /// Download and return the primary file path for this component.
    pub fn get(&self, repo_id: &str, cache_dir: &str) -> Result<PathBuf> {
        let mut cache_path = PathBuf::from(cache_dir);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let repo = api.model(repo_id.to_string());

        let files = self.files();
        let mut primary_path = None;
        for file in &files {
            log::info!("downloading {} ...", file);
            let path = repo
                .get(file)
                .map_err(|e| anyhow::anyhow!("failed to download {}: {}", file, e))?;
            if primary_path.is_none() {
                primary_path = Some(path);
            }
        }
        Ok(primary_path.unwrap())
    }

    /// Get all downloaded file paths for this component.
    pub fn get_all(&self, repo_id: &str, cache_dir: &str) -> Result<Vec<PathBuf>> {
        let mut cache_path = PathBuf::from(cache_dir);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let repo = api.model(repo_id.to_string());

        let mut paths = Vec::new();
        for file in self.files() {
            log::info!("downloading {} ...", file);
            let path = repo
                .get(file)
                .map_err(|e| anyhow::anyhow!("failed to download {}: {}", file, e))?;
            paths.push(path);
        }
        Ok(paths)
    }

    /// Component name for topology layer assignment.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tokenizer => "flux_tokenizer",
            Self::TextEncoder => "flux_text_encoder",
            Self::Transformer => "flux_transformer",
            Self::Vae => "flux_vae",
        }
    }
}

/// FLUX.2-klein transformer configuration.
///
/// Values derived from `transformer/config.json` of `black-forest-labs/FLUX.2-klein-4B`.
pub fn flux2_klein_transformer_config() -> candle_transformers::models::flux::model::Config {
    candle_transformers::models::flux::model::Config {
        in_channels: 128,
        vec_in_dim: 768,        // TODO: verify from model weights
        context_in_dim: 7680,   // joint_attention_dim from config.json
        hidden_size: 3072,      // num_attention_heads * attention_head_dim = 24 * 128
        mlp_ratio: 3.0,
        num_heads: 24,
        depth: 5,               // num_layers (double blocks)
        depth_single_blocks: 20, // num_single_layers
        axes_dim: vec![32, 32, 32, 32], // axes_dims_rope (4 axes)
        theta: 2000,            // rope_theta
        qkv_bias: true,
        guidance_embed: false,  // guidance_embeds from config.json
    }
}

/// FLUX.2-klein VAE configuration.
///
/// Values from `vae/config.json` — uses KL autoencoder with 32 latent channels.
pub fn flux2_klein_vae_config() -> candle_transformers::models::flux::autoencoder::Config {
    candle_transformers::models::flux::autoencoder::Config {
        resolution: 256,
        in_channels: 3,
        ch: 128,
        out_ch: 3,
        ch_mult: vec![1, 2, 4, 4],
        num_res_blocks: 2,
        z_channels: 32,        // latent_channels from config.json (vs 16 for FLUX.1)
        scale_factor: 0.3611,
        shift_factor: 0.1159,
    }
}

// ── FLUX.1-dev configuration ────────────────────────────────────────────────────

/// FLUX.1-dev model component file resolution.
///
/// Uses a single bundled checkpoint from Comfy-Org/flux1-dev (17.2GB)
/// that contains transformer + CLIP-L + T5-XXL + VAE.
/// Tokenizers are downloaded separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flux1ModelFile {
    /// Bundled checkpoint: transformer + CLIP + T5 + VAE
    Checkpoint,
    /// CLIP tokenizer from openai/clip-vit-large-patch14
    ClipTokenizer,
    /// T5 tokenizer from google/t5-v1_1-xxl
    T5Tokenizer,
}

impl Flux1ModelFile {
    fn repo_and_file(&self) -> (&'static str, &'static str) {
        match self {
            Self::Checkpoint => ("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
            Self::ClipTokenizer => ("openai/clip-vit-large-patch14", "tokenizer.json"),
            Self::T5Tokenizer => ("google/t5-v1_1-xxl", "spiece.model"),
        }
    }

    /// Download and return the file path.
    pub fn get(&self, cache_dir: &str) -> Result<PathBuf> {
        let (repo_id, filename) = self.repo_and_file();
        let mut cache_path = PathBuf::from(cache_dir);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let repo = api.model(repo_id.to_string());
        log::info!("downloading {}/{} ...", repo_id, filename);
        let path = repo
            .get(filename)
            .map_err(|e| anyhow::anyhow!("failed to download {}/{}: {}", repo_id, filename, e))?;
        Ok(path)
    }
}

/// Tensor name prefixes in the bundled Comfy-Org checkpoint.
pub mod flux1_prefixes {
    pub const TRANSFORMER: &str = "model.diffusion_model";
    pub const CLIP: &str = "text_encoders.clip_l.transformer";
    pub const T5: &str = "text_encoders.t5xxl.transformer";
    pub const VAE: &str = "vae";
}
