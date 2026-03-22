use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

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
    /// Linear forward: x @ w^T + bias via backend matmul.
    /// Handles broadcasting for batched inputs (3D x against 2D weight).
    fn linear_forward(&self, x: &Tensor, linear: &Linear) -> Result<Tensor> {
        let w = linear.weight().t()?;
        // For batched input (batch, seq, hidden) × (hidden, out), reshape to 2D,
        // matmul, then reshape back — avoids candle's 3D×2D shape mismatch.
        let x_dims = x.dims();
        let out = if x_dims.len() > 2 {
            let leading: usize = x_dims[..x_dims.len() - 1].iter().product();
            let inner = *x_dims.last().unwrap();
            let x_2d = x.reshape((leading, inner))?;
            let out_2d = self.backend.matmul(&x_2d, &w)?;
            let out_dim = out_2d.dim(1)?;
            let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
            out_shape.push(out_dim);
            out_2d.reshape(out_shape.as_slice())?
        } else {
            self.backend.matmul(x, &w)?
        };
        match linear.bias() {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }

    /// Execute MLP(x).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.linear_forward(x, &self.gate_up_proj)?;
        let gate = fused.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = fused.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let x = if self.use_gelu {
            (gate.gelu()? * up)?
        } else {
            self.backend.silu_mul(&gate.contiguous()?, &up.contiguous()?)?
        };
        self.linear_forward(&x, &self.down_proj)
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
        let gate_up_w = backend.preprocess_linear_weight(&gate_up_w)?;
        let gate_up_proj = Linear::new(gate_up_w, None);

        let down_w = vb.pp("down_proj").get((h_size, i_size), "weight")?;
        let down_w = backend.preprocess_linear_weight(&down_w)?;
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
