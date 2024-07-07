const PROTO_MAGIC: u32 = 0x104F4C7;

// does more than this even make sense?
const MESSAGE_MAX_SIZE: u32 = 512 * 1024 * 1024;

mod message;

pub use message::*;
