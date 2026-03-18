//! Gemma 3 model implementation (Gemma3ForCausalLM).
mod block;
pub mod config;
mod history;
mod model;

pub use block::*;
pub use config::*;
pub use model::*;

crate::impl_model_for_text!(Gemma3);
