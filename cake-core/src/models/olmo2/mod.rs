//! OLMo 2 model implementation (OLMo2ForCausalLM).
mod block;
pub mod config;
mod model;

pub use block::*;
pub use config::*;
pub use model::*;

crate::impl_model_for_text!(OLMo2);
