//! This module contains Cake protocol specific objects and constants.

/// Cake protocol header magic value.
pub(crate) const PROTO_MAGIC: u32 = 0x104F4C7;

/// Cake protocol message max size (1 GB).
/// Increased from 512 MB to support high-resolution video tensor transport
/// (e.g., 768×1024 @ 97 frames produces ~873 MB F32 VAE output).
pub(crate) const MESSAGE_MAX_SIZE: u32 = 1024 * 1024 * 1024;

mod message;

pub use message::*;
