//! EXAONE 4.0 model implementation (ExaoneForCausalLM).
mod block;
pub mod config;
mod model;

pub use block::*;
pub use config::*;
pub use model::*;

crate::impl_model_for_text!(EXAONE4);
