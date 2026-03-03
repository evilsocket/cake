use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

/// Multi-perceptron implementation with fused gate+up projection.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl MLP {
    /// Execute MLP(x).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.gate_up_proj.forward(x)?;
        let gate = fused.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = fused.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let x = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&x)
    }

    /// Load this block from the VarBuilder given the specific configuration.
    pub fn load(vb: VarBuilder, cfg: &super::Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        // Fuse gate_proj and up_proj into a single matmul
        let gate_w = vb.pp("gate_proj").get((i_size, h_size), "weight")?;
        let up_w = vb.pp("up_proj").get((i_size, h_size), "weight")?;
        let fused_w = Tensor::cat(&[&gate_w, &up_w], 0)?;
        let gate_up_proj = Linear::new(fused_w, None);

        let down_w = vb.pp("down_proj").get((h_size, i_size), "weight")?;
        let down_proj = Linear::new(down_w, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: i_size,
        })
    }
}
