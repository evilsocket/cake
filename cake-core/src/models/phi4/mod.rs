//! Phi-4-mini and Phi-4 model implementations (Phi3ForCausalLM / Phi4ForCausalLM).
pub mod config;
mod history;
mod model;

pub use config::*;
pub use model::*;

crate::impl_model_for_text!(Phi4);
