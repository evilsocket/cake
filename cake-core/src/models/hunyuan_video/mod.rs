//! HunyuanVideo model implementation.
//!
//! Follows the same component-based topology pattern as LTX-Video:
//! - `hunyuan-transformer` — Dual-stream DiT transformer
//! - `hunyuan-t5` — T5-XXL text encoder
//! - `hunyuan-clip` — CLIP-L text encoder
//! - `hunyuan-vae` — 3D VAE decoder
pub mod vendored;

mod clip;
mod hunyuan_video;
mod hunyuan_video_shardable;
mod t5;
mod transformer;
mod vae_forwarder;

pub use hunyuan_video::*;
