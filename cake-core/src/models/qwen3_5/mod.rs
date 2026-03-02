//! Qwen3.5 hybrid linear/full attention model implementation.
mod block;
mod config;
mod full_attention;
mod linear_attention;
mod model;

pub use config::*;
pub use model::*;
