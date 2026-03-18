pub mod block;
pub mod config;
pub mod moe;
mod model;

pub use config::*;
pub use model::*;

crate::impl_model_for_text!(Qwen3_5Moe);
