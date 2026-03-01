use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias as linear, Linear, Module, VarBuilder};

/// Multi-perceptron implementation.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    /// Execute MLP(x).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x)
    }

    /// Load this block from the VarBuilder given the specific configuration.
    pub fn load(vb: VarBuilder, cfg: &super::Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_proj = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let up_proj = linear(h_size, i_size, vb.pp("up_proj"))?;
        let down_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}
