//! VibeVoice-Realtime-0.5B text-to-speech model.
//!
//! Architecture: Qwen2.5-0.5B LLM backbone + 4-layer diffusion prediction head
//! + σ-VAE acoustic decoder. Generates streaming audio at 24kHz.
//!
//! Components:
//! - `tts_language_model`: Qwen2.5-0.5B (24 layers, 896 hidden, reuses Qwen2 code)
//! - `prediction_head`: 4-layer DiT with AdaLN + DDPM v-prediction
//! - `acoustic_tokenizer.decoder`: σ-VAE Conv1d decoder (~340M params)
//! - `acoustic_connector`: MLP mapping VAE latents to LLM space
//! - `tts_eos_classifier`: Binary MLP for end-of-speech detection

pub mod config;
pub mod prediction_head;
pub mod acoustic_connector;
pub mod eos_classifier;
pub mod ddpm;
pub mod vae_decoder;
#[allow(clippy::module_inception)]
pub mod vibevoice;

pub use vibevoice::{VibeVoiceTTS, save_wav};
