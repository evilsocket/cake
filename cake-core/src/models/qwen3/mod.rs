//! Qwen3 dense model implementation (Qwen3ForCausalLM).
pub mod config;
mod model;

pub use config::*;
pub use model::*;

crate::impl_model_for_text!(Qwen3);
