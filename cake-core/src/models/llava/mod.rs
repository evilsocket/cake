//! LLaVA (Large Language and Vision Assistant) model implementation.
//!
//! Combines a CLIP vision tower + MM projector + LLM (Llama) for multimodal
//! inference. The vision tower and LLM layers can be distributed across workers.
mod config;
mod llava;
mod llava_shardable;
mod vision;

pub use config::*;
pub use llava::*;
