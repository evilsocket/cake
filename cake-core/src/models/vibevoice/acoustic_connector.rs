//! Acoustic connector: maps VAE latent space to LLM hidden space.
//!
//! Simple MLP: fc1(vae_dim → hidden) + RmsNorm + fc2(hidden → hidden).

use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};

#[derive(Debug, Clone)]
pub struct AcousticConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl AcousticConnector {
    pub fn load(vb: VarBuilder, vae_dim: usize, hidden: usize, eps: f64) -> Result<Self> {
        let fc1 = candle_nn::linear(vae_dim, hidden, vb.pp("fc1"))?;
        let norm = candle_nn::rms_norm(hidden, eps, vb.pp("norm"))?;
        let fc2 = candle_nn::linear(hidden, hidden, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }

    /// Map acoustic VAE latent (batch, vae_dim) → LLM hidden (batch, hidden).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = self.norm.forward(&h)?;
        self.fc2.forward(&h)
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
        let conn = AcousticConnector::load(vb, vae_dim, hidden, 1e-5).unwrap();

        let x = make_tensor(&[2, vae_dim], 10);
        let y = conn.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, hidden]);
    }
}
