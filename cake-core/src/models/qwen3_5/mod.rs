//! Qwen3.5 hybrid linear/full attention model implementation.
pub mod block;
pub mod config;
pub mod full_attention;
pub mod linear_attention;
mod model;

pub use block::*;
pub use config::*;
pub use model::*;

crate::impl_model_for_text!(Qwen3_5);
