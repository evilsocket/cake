//! Mixtral Mixture of Experts model implementation.
//!
//! Supports distributed expert-parallel inference where groups of experts
//! can be served by different workers.
mod config;
mod expert_forwarder;
mod mixtral;
mod mixtral_shardable;
mod moe_block;

pub use config::*;
pub use mixtral::*;
