//! Vendored from https://github.com/FerrisMind/candle-video (Apache 2.0, by FerrisMind)
//! with minimal modifications (import path adaptation, nightly feature removal).

#[allow(dead_code, unused_imports, clippy::too_many_arguments, clippy::type_complexity)]
pub mod configs;
#[allow(dead_code, unused_imports, clippy::too_many_arguments)]
pub mod ltx_transformer;
#[allow(dead_code, unused_imports, clippy::too_many_arguments)]
pub mod scheduler;
#[allow(dead_code, unused_imports, clippy::too_many_arguments)]
pub mod t2v_pipeline;
#[allow(dead_code, unused_imports, clippy::too_many_arguments, clippy::type_complexity)]
pub mod vae;

pub use configs::*;
pub use ltx_transformer::*;
pub use scheduler::FlowMatchEulerDiscreteScheduler;
pub use scheduler::FlowMatchEulerDiscreteSchedulerConfig;
pub use vae::AutoencoderKLLtxVideo;
pub use vae::AutoencoderKLLtxVideoConfig;
