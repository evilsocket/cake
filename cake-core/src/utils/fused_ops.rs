//! Re-exports from [`crate::backends::fused_ops`].
//!
//! The fused kernel implementations have moved into the `backends` module where they
//! are used as the internal engine for all `ComputeBackend` trait implementations.
//! This module re-exports the public API for backward compatibility.

pub use crate::backends::fused_ops::*;
