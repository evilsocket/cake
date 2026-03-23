//! End-of-speech binary classifier.
//!
//! 2-layer MLP that detects when the model should stop generating speech frames.
//! Input: LLM hidden state → sigmoid → probability of EOS.

use std::sync::Arc;

use candle_core::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

#[derive(Debug, Clone)]
pub struct EosClassifier {
    fc1_weight: Tensor,
    fc1_bias: Option<Tensor>,
    fc2_weight: Tensor,
    fc2_bias: Option<Tensor>,
    backend: Arc<dyn ComputeBackend>,
}

impl EosClassifier {
    pub fn load(vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        // Weight shapes inferred from safetensors: fc1 + fc2
        let fc1_weight = vb.pp("fc1").get_unchecked("weight")?;
        let fc1_bias = Some(vb.pp("fc1").get_unchecked("bias")?);
        let fc2_weight = vb.pp("fc2").get_unchecked("weight")?;
        let fc2_bias = Some(vb.pp("fc2").get_unchecked("bias")?);
        Ok(Self { fc1_weight, fc1_bias, fc2_weight, fc2_bias, backend })
    }

    /// Predict EOS probability.
    /// Input: (batch, hidden_size) LLM hidden state.
    /// Output: (batch, 1) probability (after sigmoid).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.backend.linear_forward(x, &self.fc1_weight, self.fc1_bias.as_ref())?;
        let h = self.backend.silu(&h)?;
        let h = self.backend.linear_forward(&h, &self.fc2_weight, self.fc2_bias.as_ref())?;
        self.backend.sigmoid(&h)
    }

    /// Check if generation should stop (probability > threshold).
    pub fn should_stop(&self, x: &Tensor, threshold: f32) -> Result<bool> {
        let prob = self.forward(x)?;
        let val: f32 = prob
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?[0];
        Ok(val > threshold)
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
    fn test_eos_classifier_shape() {
        let hidden = 64;
        let mid = 32;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("fc1.weight".into(), make_tensor(&[mid, hidden], 1));
        map.insert("fc1.bias".into(), make_tensor(&[mid], 2));
        map.insert("fc2.weight".into(), make_tensor(&[1, mid], 3));
        map.insert("fc2.bias".into(), make_tensor(&[1], 4));

        let backend = crate::backends::create_backend(&Device::Cpu);
        let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        let cls = EosClassifier::load(vb, backend).unwrap();

        let x = make_tensor(&[1, hidden], 10);
        let prob = cls.forward(&x).unwrap();
        assert_eq!(prob.dims(), &[1, 1]);

        // Sigmoid output should be in [0, 1]
        let val: f32 = prob.flatten_all().unwrap().to_vec1().unwrap()[0];
        assert!((0.0..=1.0).contains(&val));
    }
}
