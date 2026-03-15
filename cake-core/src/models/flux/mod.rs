//! FLUX.2-klein image generation model.
//!
//! Implements a flow-matching transformer (MMDiT) pipeline for text-to-image generation.
//! Uses Qwen3 as text encoder, a dual-stream + single-stream transformer for denoising,
//! and a KL autoencoder VAE for latent ↔ pixel conversion.

mod config;
mod flux;
mod flux2_model;
mod flux_shardable;
mod text_encoder;
mod transformer;
mod vae;

pub use flux::FluxGen;
pub use flux_shardable::FluxShardable;
