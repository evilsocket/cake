//! Expert weight provider abstraction for Mixture-of-Experts models.
//!
//! [`ExpertProvider`] decouples expert weight access from storage strategy.
//! Two implementations:
//! - [`ResidentProvider`]: all experts in RAM (current default behavior)
//! - `DiskProvider`: streams expert weights from disk on demand (Flash-MoE)

use std::sync::Arc;

use candle_core::{Result, Tensor};

/// Weights for a single expert's SwiGLU FFN.
#[derive(Debug, Clone)]
pub struct ExpertWeights {
    /// Gate projection: (intermediate_size, hidden_size)
    pub gate_proj: Tensor,
    /// Up projection: (intermediate_size, hidden_size)
    pub up_proj: Tensor,
    /// Down projection: (hidden_size, intermediate_size)
    pub down_proj: Tensor,
}

/// Trait for providing expert weight tensors on demand.
///
/// MoE `forward()` methods call `get_expert(idx)` to obtain the three weight
/// matrices (gate, up, down) for a selected expert. The provider determines
/// whether these come from RAM, disk, or network.
pub trait ExpertProvider: Send + Sync + std::fmt::Debug {
    /// Get weight tensors for the expert at `idx`.
    fn get_expert(&self, idx: usize) -> Result<ExpertWeights>;
    /// Number of experts available.
    fn num_experts(&self) -> usize;
    /// Downcast support for provider-specific fast paths.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// All experts resident in RAM as stacked 3D tensors.
///
/// This wraps the existing weight layout where all experts are loaded at init
/// time and stacked into `(num_experts, intermediate, hidden)` tensors.
/// Used by Qwen3 MoE (128 experts, stacked).
#[derive(Debug)]
pub struct StackedResidentProvider {
    /// (num_experts, intermediate_size, hidden_size)
    gate_proj: Tensor,
    /// (num_experts, intermediate_size, hidden_size)
    up_proj: Tensor,
    /// (num_experts, hidden_size, intermediate_size)
    down_proj: Tensor,
    /// Fused gate+up: (num_experts, 2*intermediate_size, hidden_size)
    gate_up_proj: Tensor,
    num_experts: usize,
}

impl StackedResidentProvider {
    pub fn new(gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor, num_experts: usize) -> Self {
        let gate_up_proj = Tensor::cat(&[&gate_proj, &up_proj], 1)
            .expect("failed to fuse gate+up stacked tensors");
        Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj,
            num_experts,
        }
    }

    /// Access the full stacked tensors (for batched index_select in fast path).
    pub fn gate_proj(&self) -> &Tensor {
        &self.gate_proj
    }
    pub fn up_proj(&self) -> &Tensor {
        &self.up_proj
    }
    pub fn down_proj(&self) -> &Tensor {
        &self.down_proj
    }
    /// Fused gate+up: (num_experts, 2*intermediate_size, hidden_size)
    pub fn gate_up_proj(&self) -> &Tensor {
        &self.gate_up_proj
    }
}

impl ExpertProvider for StackedResidentProvider {
    fn get_expert(&self, idx: usize) -> Result<ExpertWeights> {
        Ok(ExpertWeights {
            gate_proj: self.gate_proj.get(idx)?,
            up_proj: self.up_proj.get(idx)?,
            down_proj: self.down_proj.get(idx)?,
        })
    }

    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// All experts resident in RAM as individual weight tensors.
///
/// This wraps the existing layout where each expert has separate
/// gate/up/down weight tensors. Used by Qwen3.5 MoE (256 experts, individual).
#[derive(Debug)]
pub struct IndividualResidentProvider {
    /// Per-expert gate projection weights.
    gate_weights: Vec<Tensor>,
    /// Per-expert up projection weights.
    up_weights: Vec<Tensor>,
    /// Per-expert down projection weights.
    down_weights: Vec<Tensor>,
}

impl IndividualResidentProvider {
    pub fn new(
        gate_weights: Vec<Tensor>,
        up_weights: Vec<Tensor>,
        down_weights: Vec<Tensor>,
    ) -> Self {
        assert_eq!(gate_weights.len(), up_weights.len());
        assert_eq!(gate_weights.len(), down_weights.len());
        Self {
            gate_weights,
            up_weights,
            down_weights,
        }
    }
}

impl ExpertProvider for IndividualResidentProvider {
    fn get_expert(&self, idx: usize) -> Result<ExpertWeights> {
        Ok(ExpertWeights {
            gate_proj: self.gate_weights[idx].clone(),
            up_proj: self.up_weights[idx].clone(),
            down_proj: self.down_weights[idx].clone(),
        })
    }

    fn num_experts(&self) -> usize {
        self.gate_weights.len()
    }
}

/// Type alias for shared expert provider.
pub type SharedExpertProvider = Arc<dyn ExpertProvider>;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_expert_weights_struct() {
        let g = Tensor::zeros((64, 32), DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((64, 32), DType::F32, &Device::Cpu).unwrap();
        let d = Tensor::zeros((32, 64), DType::F32, &Device::Cpu).unwrap();
        let ew = ExpertWeights {
            gate_proj: g,
            up_proj: u,
            down_proj: d,
        };
        assert_eq!(ew.gate_proj.dims(), &[64, 32]);
        assert_eq!(ew.down_proj.dims(), &[32, 64]);
    }

    #[test]
    fn test_stacked_resident_provider() {
        let n = 4;
        let g = Tensor::zeros((n, 64, 32), DType::F32, &Device::Cpu).unwrap();
        let u = Tensor::zeros((n, 64, 32), DType::F32, &Device::Cpu).unwrap();
        let d = Tensor::zeros((n, 32, 64), DType::F32, &Device::Cpu).unwrap();
        let provider = StackedResidentProvider::new(g, u, d, n);
        assert_eq!(provider.num_experts(), 4);

        let ew = provider.get_expert(2).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[64, 32]);
        assert_eq!(ew.up_proj.dims(), &[64, 32]);
        assert_eq!(ew.down_proj.dims(), &[32, 64]);
    }

    #[test]
    fn test_individual_resident_provider() {
        let g: Vec<Tensor> = (0..3)
            .map(|_| Tensor::zeros((64, 32), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let u: Vec<Tensor> = (0..3)
            .map(|_| Tensor::zeros((64, 32), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let d: Vec<Tensor> = (0..3)
            .map(|_| Tensor::zeros((32, 64), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let provider = IndividualResidentProvider::new(g, u, d);
        assert_eq!(provider.num_experts(), 3);

        let ew = provider.get_expert(1).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[64, 32]);
    }
}
