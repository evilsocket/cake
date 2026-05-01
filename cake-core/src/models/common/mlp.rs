use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;
use crate::utils::quantized_linear::LinearWeight;

/// Attempt to load a quantized linear weight from a VarBuilder.
///
/// Returns `Some(LinearWeight::Quantized)` if the prefix has `.scales` and
/// the `.weight` tensor is U32 (packed 4-bit from MetalMlxBackend).
/// Returns `None` if the tensor is not quantized (caller falls back to dense).
pub(crate) fn try_load_quantized(vb: &VarBuilder) -> Result<Option<LinearWeight>> {
    if !vb.contains_tensor("scales") {
        return Ok(None);
    }
    // Try loading the weight — if it's U32, it's packed 4-bit.
    if let Ok(weight) = vb.get_unchecked("weight") {
        if weight.dtype() == candle_core::DType::U32 {
            let scales = vb.get_unchecked("scales")?;
            let biases = vb.get_unchecked("biases")?;
            let packed_cols = weight.dim(1)?;
            let num_quant_groups = scales.dim(1)?;
            let group_size = (packed_cols * 8) / num_quant_groups;
            return Ok(Some(LinearWeight::quantized(
                weight, scales, biases, group_size,
            )));
        }
    }
    Ok(None)
}

/// Multi-perceptron implementation with fused gate+up projection.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct MLP {
    gate_up_proj_weight: LinearWeight,
    down_proj_weight: LinearWeight,
    intermediate_size: usize,
    use_gelu: bool,
    backend: Arc<dyn ComputeBackend>,
}

impl MLP {
    /// Execute MLP(x).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.gate_up_proj_weight.forward(x, None, &*self.backend)?;
        let gate = fused.narrow(D::Minus1, 0, self.intermediate_size)?;
        let up = fused.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let x = if self.use_gelu {
            (self.backend.gelu(&gate)? * up)?
        } else {
            self.backend
                .silu_mul(&gate.contiguous()?, &up.contiguous()?)?
        };
        self.down_proj_weight.forward(&x, None, &*self.backend)
    }

    /// Load this block from the VarBuilder given the specific configuration.
    pub fn load(
        vb: VarBuilder,
        cfg: &super::Config,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        let gate_up_proj_weight = if cfg.fused_gate_up_proj {
            // Phi-3/4 style: weights already fused as 'gate_up_proj'
            let vb_proj = vb.pp("gate_up_proj");
            if let Some(qw) = try_load_quantized(&vb_proj)? {
                qw
            } else {
                LinearWeight::Dense(backend.preprocess_linear_weight(
                    &vb_proj.get((2 * i_size, h_size), "weight")?,
                )?)
            }
        } else {
            // Standard: fuse gate_proj and up_proj into a single matmul
            let gate_q = try_load_quantized(&vb.pp("gate_proj"))?;
            let up_q = try_load_quantized(&vb.pp("up_proj"))?;
            match (gate_q, up_q) {
                (Some(g), Some(u)) => {
                    // Both quantized: fuse packed weights along dim 0 (out_features)
                    fuse_quantized(&[g, u])?
                }
                _ => {
                    // Dense: fuse raw weights then preprocess
                    let gate_w = vb.pp("gate_proj").get((i_size, h_size), "weight")?;
                    let up_w = vb.pp("up_proj").get((i_size, h_size), "weight")?;
                    LinearWeight::Dense(
                        backend.preprocess_linear_weights(&[&gate_w, &up_w])?,
                    )
                }
            }
        };

        let down_proj_weight = {
            let vb_down = vb.pp("down_proj");
            if let Some(qw) = try_load_quantized(&vb_down)? {
                qw
            } else {
                let w = vb_down.get((h_size, i_size), "weight")?;
                LinearWeight::Dense(backend.preprocess_linear_weight(&w)?)
            }
        };

        Ok(Self {
            gate_up_proj_weight,
            down_proj_weight,
            intermediate_size: i_size,
            use_gelu: cfg.use_gelu_mlp,
            backend,
        })
    }
}

/// Fuse multiple quantized linear weights by concatenating along dim 0.
pub(crate) fn fuse_quantized_pub(weights: &[LinearWeight]) -> Result<LinearWeight> {
    fuse_quantized(weights)
}

fn fuse_quantized(weights: &[LinearWeight]) -> Result<LinearWeight> {
    use crate::utils::quantized_linear::QuantizedWeight;

    let qws: Vec<&QuantizedWeight> = weights
        .iter()
        .map(|w| match w {
            LinearWeight::Quantized(qw) => Ok(qw),
            LinearWeight::Dense(_) => candle_core::bail!("fuse_quantized: expected Quantized"),
        })
        .collect::<Result<_>>()?;

    let packed_refs: Vec<&Tensor> = qws.iter().map(|qw| &qw.packed).collect();
    let scales_refs: Vec<&Tensor> = qws.iter().map(|qw| &qw.scales).collect();
    let biases_refs: Vec<&Tensor> = qws.iter().map(|qw| &qw.biases).collect();

    Ok(LinearWeight::quantized(
        Tensor::cat(&packed_refs, 0)?,
        Tensor::cat(&scales_refs, 0)?,
        Tensor::cat(&biases_refs, 0)?,
        qws[0].group_size,
    ))
}
