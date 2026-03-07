//! Vendored HunyuanVideo model components.
//!
//! These will be ported from the HuggingFace diffusers reference implementation
//! (Apache 2.0) or community Rust ports when available.
//!
//! For now, this module provides the type definitions and configuration structures
//! needed for the Cake integration layer.

#[allow(dead_code, unused_imports, clippy::too_many_arguments)]
pub mod config;
#[allow(dead_code, unused_imports, clippy::too_many_arguments)]
pub mod scheduler;

pub use config::*;
pub use scheduler::*;
