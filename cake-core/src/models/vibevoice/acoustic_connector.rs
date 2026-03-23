//! Acoustic connector: maps VAE latent space to LLM hidden space.
//!
//! Simple MLP: fc1(vae_dim → hidden) + RmsNorm + fc2(hidden → hidden).

use std::sync::Arc;

use candle_core::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

#[derive(Debug, Clone)]
pub struct AcousticConnector {
    fc1_weight: Tensor,
    fc1_bias: Option<Tensor>,
    norm_weight: Tensor,
    norm_eps: f32,
    fc2_weight: Tensor,
    fc2_bias: Option<Tensor>,
    backend: Arc<dyn ComputeBackend>,
}

impl AcousticConnector {
    pub fn load(vb: VarBuilder, vae_dim: usize, hidden: usize, eps: f64, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let fc1_weight = vb.pp("fc1").get((hidden, vae_dim), "weight")?;
        let fc1_bias = vb.pp("fc1").get(hidden, "bias").ok();
        let norm_weight = vb.pp("norm").get(hidden, "weight")?;
        let norm_eps = eps as f32;
        let fc2_weight = vb.pp("fc2").get((hidden, hidden), "weight")?;
        let fc2_bias = vb.pp("fc2").get(hidden, "bias").ok();
        Ok(Self { fc1_weight, fc1_bias, norm_weight, norm_eps, fc2_weight, fc2_bias, backend })
    }

    /// Map acoustic VAE latent (batch, vae_dim) → LLM hidden (batch, hidden).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.backend.linear_forward(x, &self.fc1_weight, self.fc1_bias.as_ref())?;
        let h = self.backend.rms_norm(&h, &self.norm_weight, self.norm_eps)?;
        self.backend.linear_forward(&h, &self.fc2_weight, self.fc2_bias.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
        use rand::{Rng, SeedableRng};
        let numel: usize = shape.iter().product();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_acoustic_connector_shape() {
        let vae_dim = 64;
        let hidden = 128;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("fc1.weight".into(), make_tensor(&[hidden, vae_dim], 1));
        map.insert("fc1.bias".into(), make_tensor(&[hidden], 2));
        map.insert("norm.weight".into(), Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap());
        map.insert("fc2.weight".into(), make_tensor(&[hidden, hidden], 3));
        map.insert("fc2.bias".into(), make_tensor(&[hidden], 4));

        let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        let backend = crate::backends::create_backend(&Device::Cpu);
        let conn = AcousticConnector::load(vb, vae_dim, hidden, 1e-5, backend).unwrap();

        let x = make_tensor(&[2, vae_dim], 10);
        let y = conn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, hidden]);
    }
}
