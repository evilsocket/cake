//! This module contains Cake protocol specific objects and constants.

/// Cake protocol header magic value.
pub(crate) const PROTO_MAGIC: u32 = 0x104F4C7;

/// Pre-shifted magic for fast header construction (magic in upper 32 bits of u64).
pub(crate) const PROTO_MAGIC_U64_HIGH: u64 = (PROTO_MAGIC as u64) << 32;

/// Cake protocol message max size.
pub(crate) const MESSAGE_MAX_SIZE: u32 = 512 * 1024 * 1024;

mod message;

pub use message::*;
