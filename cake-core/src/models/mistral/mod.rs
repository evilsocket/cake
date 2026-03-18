//! Mistral model implementation (MistralForCausalLM).
pub mod config;
mod model;

pub use config::*;
pub use model::*;

crate::impl_model_for_text!(Mistral);
