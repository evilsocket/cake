//! FLUX image generation models.
//!
//! Supports both FLUX.2-klein-4B and FLUX.1-dev:
//! - FLUX.2-klein: Qwen3 text encoder, custom VAE (32 latent channels)
//! - FLUX.1-dev: CLIP-L + T5-XXL encoders, standard VAE (16 latent channels), FP8 weights

pub mod config;
#[allow(clippy::module_inception)]
mod flux;
pub mod flux2_model;
pub mod flux2_vae;
mod flux_shardable;
pub mod text_encoder;
pub mod transformer;
mod vae;

// FLUX.1-dev
mod clip_encoder;
mod flux1;
pub mod flux1_model;
mod t5_encoder;

pub use flux::FluxGen;
pub use flux1::Flux1Gen;
pub use flux_shardable::FluxShardable;
