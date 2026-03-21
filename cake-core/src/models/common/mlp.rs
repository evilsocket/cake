use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::backends::ComputeBackend;

/// Multi-perceptron implementation with fused gate+up projection.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
    use_gelu: bool,
    backend: Arc<dyn ComputeBackend>,
}

impl MLP {
    /// Execute MLP(x).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.gate_up_proj.forward(x)?;
        let gate = fused.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = fused.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let x = if self.use_gelu {
            (gate.gelu()? * up)?
        } else {
            self.backend.silu_mul(&gate.contiguous()?, &up.contiguous()?)?
        };
        self.down_proj.forward(&x)
    }

    /// Load this block from the VarBuilder given the specific configuration.
    pub fn load(vb: VarBuilder, cfg: &super::Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        let gate_up_w = if cfg.fused_gate_up_proj {
            // Phi-3/4 style: weights already fused as 'gate_up_proj'
            vb.pp("gate_up_proj").get((2 * i_size, h_size), "weight")?
        } else {
            // Standard: fuse gate_proj and up_proj into a single matmul
            let gate_w = vb.pp("gate_proj").get((i_size, h_size), "weight")?;
            let up_w = vb.pp("up_proj").get((i_size, h_size), "weight")?;
            Tensor::cat(&[&gate_w, &up_w], 0)?
        };
        let gate_up_proj = Linear::new(gate_up_w, None);

        let down_w = vb.pp("down_proj").get((h_size, i_size), "weight")?;
        let down_proj = Linear::new(down_w, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: i_size,
            use_gelu: cfg.use_gelu_mlp,
            backend,
        })
    }
}
