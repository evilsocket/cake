//! LTX-2 model implementation (19B audio+video generation).
//!
//! Component-based topology (same pattern as LTX-Video / HunyuanVideo):
//! - `ltx2-transformer` — Asymmetric dual-stream DiT (14B video + 5B audio)
//! - `ltx2-gemma` — Gemma-3 12B text encoder
//! - `ltx2-vae` — Video VAE decoder
//! - `ltx2-vocoder` — Audio vocoder

pub mod vendored;

mod ltx2;
mod ltx2_shardable;
mod gemma;
pub(crate) mod gemma_encoder;
mod transformer;
mod vae_forwarder;
mod vocoder;

pub use ltx2::*;
