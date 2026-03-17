//! VibeVoice text-to-speech models.
//!
//! Supports two model variants:
//! - **VibeVoice-Realtime-0.5B** (streaming): Split LM (4 base + 20 TTS layers, 896 hidden)
//! - **VibeVoice-1.5B** (non-streaming): Single 28-layer Qwen2 LM (1536 hidden) + semantic tokenizer
//!
//! Shared components: prediction head, DPM-Solver++, σ-VAE decoder, acoustic connector.

// Shared components
pub mod config;
pub mod config_1_5b;
pub mod prediction_head;
pub mod acoustic_connector;
pub mod eos_classifier;
pub mod ddpm;
pub mod vae_decoder;
pub mod vae_encoder;

// 0.5B streaming model
pub mod voice_prompt;
#[allow(clippy::module_inception)]
pub mod vibevoice;

// 1.5B non-streaming model
pub mod vibevoice_1_5b;

pub use vibevoice::{VibeVoiceTTS, save_wav};
pub use vibevoice_1_5b::VibeVoice1_5B;
pub use voice_prompt::VoicePrompt;
