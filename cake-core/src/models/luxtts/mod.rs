//! LuxTTS text-to-speech model — lightweight voice cloning TTS using Zipformer + flow matching.
//!
//! Architecture:
//! - **Text encoder**: 4 Zipformer layers at dim=192 (always on master)
//! - **FM decoder**: 16 Zipformer layers at dim=512 (shardable across workers)
//! - **Vocos vocoder**: ISTFT-based neural vocoder (always on master)
//! - **Flow matching**: 4-step Euler ODE solver
//!
//! Supports distributed inference: FM decoder layers are the shardable units.

pub mod activations;
pub mod bias_norm;
pub mod block;
pub mod bypass_module;
pub mod config;
pub mod convolution_module;
pub mod euler_solver;
pub mod feedforward;
pub mod mel;
pub mod model;
pub mod nonlin_attention;
pub mod rel_pos_attention;
pub mod text_encoder;
pub mod tokenizer;
pub mod vocos;
pub mod zipformer_layer;

pub use block::ZipformerBlock;
pub use config::LuxTTSConfig;
pub use model::LuxTTS;
pub use vocos::save_wav;
