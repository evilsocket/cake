//! This module contains Cake protocol specific objects and constants.

/// Cake protocol header magic value.
const PROTO_MAGIC: u32 = 0x104F4C7;

/// Cake protocol message max size.
const MESSAGE_MAX_SIZE: u32 = 512 * 1024 * 1024;

mod message;

pub use message::*;
