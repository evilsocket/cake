//! Fused kernel implementations for inference-critical operations.
//!
//! Organized by operation category. Each module contains CustomOp struct
//! definitions with CPU, CUDA, and Metal dispatch.

#[cfg(feature = "cuda")]
mod cuda_kernels;
#[cfg(feature = "metal")]
mod metal_shaders;

mod f8_dequant;
mod activations;
mod normalization;
mod elementwise;
mod convolution;

pub use f8_dequant::*;
pub use activations::*;
pub use normalization::*;
pub use elementwise::*;
pub use convolution::*;

// Re-export kernel constants for use in CustomOp cuda_fwd/metal_fwd methods
#[cfg(feature = "cuda")]
pub(crate) use cuda_kernels::FUSED_OPS_PTX;
#[cfg(feature = "metal")]
pub(crate) use metal_shaders::FUSED_OPS_MSL;
