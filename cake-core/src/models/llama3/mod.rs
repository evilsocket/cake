//! LLaMA3 model implementation.
pub mod config;
mod history;
mod llama;

pub use config::*;
pub use history::*;
pub use llama::*;

crate::impl_model_for_text!(LLama);
