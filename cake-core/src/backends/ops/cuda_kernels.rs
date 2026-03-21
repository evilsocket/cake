//! CUDA PTX kernel source (compiled from src/cuda/fused_ops.cu by build.rs).
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/fused_ops_ptx.rs"));
}
pub(crate) const FUSED_OPS_PTX: &str = ptx::FUSED_OPS;
