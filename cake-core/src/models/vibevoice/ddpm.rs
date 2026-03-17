//! DDPM noise scheduler with v-prediction support.
//!
//! Implements the reverse diffusion process matching diffusers' DDPMScheduler
//! with squaredcos_cap_v2 beta schedule and v-prediction parameterization.

use candle_core::Result;
use candle_core::Tensor;

/// DDPM scheduler state.
#[derive(Debug, Clone)]
pub struct DdpmScheduler {
    /// Cumulative product of alphas (indexed by training timestep).
    alphas_cumprod: Vec<f64>,
    /// Inference timesteps (descending, e.g. [900, 800, ..., 100, 0]).
    timesteps: Vec<usize>,
}

impl DdpmScheduler {
    /// Create a DDPM scheduler matching diffusers' squaredcos_cap_v2.
    pub fn new_cosine(num_train_steps: usize, num_inference_steps: usize) -> Self {
        let s = 0.008_f64;
        let max_beta = 0.999_f64;

        // Compute alpha_bar function
        let alpha_bar_fn = |t: f64| -> f64 {
            ((t / num_train_steps as f64 + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2)
                .cos()
                .powi(2)
        };

        // Compute betas from consecutive alpha_bar ratios (squaredcos_cap_v2)
        let mut betas = Vec::with_capacity(num_train_steps);
        for i in 0..num_train_steps {
            let beta = (1.0 - alpha_bar_fn((i + 1) as f64) / alpha_bar_fn(i as f64)).min(max_beta);
            betas.push(beta);
        }

        // Compute cumulative product of alphas
        let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
        let mut cumprod = 1.0_f64;
        for b in &betas {
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        // Diffusers-compatible timestep spacing: [900, 800, ..., 100, 0]
        let step_ratio = num_train_steps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio)
            .collect();

        Self {
            alphas_cumprod,
            timesteps,
        }
    }

    /// Get the inference timesteps.
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get alpha_cumprod for a given timestep.
    fn alpha_cumprod(&self, t: usize) -> f64 {
        if t < self.alphas_cumprod.len() {
            self.alphas_cumprod[t]
        } else {
            *self.alphas_cumprod.last().unwrap_or(&0.0)
        }
    }

    /// Single DDPM reverse step with v-prediction.
    ///
    /// v-prediction: v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x0
    /// Recover: x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
    ///          eps = sqrt(1-alpha_bar) * x_t + sqrt(alpha_bar) * v
    pub fn step(
        &self,
        v_pred: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let alpha_bar_t = self.alpha_cumprod(timestep);
        let sqrt_alpha = alpha_bar_t.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar_t).sqrt();

        // Predict x_0 and epsilon
        let x0_pred = ((sample * sqrt_alpha)? - (v_pred * sqrt_one_minus_alpha)?)?;
        let eps_pred = ((v_pred * sqrt_alpha)? + (sample * sqrt_one_minus_alpha)?)?;

        // Determine previous timestep
        let t_idx = self.timesteps.iter().position(|&t| t == timestep);
        let prev_t = match t_idx {
            Some(idx) if idx + 1 < self.timesteps.len() => Some(self.timesteps[idx + 1]),
            _ => None,
        };

        match prev_t {
            None | Some(0) => Ok(x0_pred), // Last step: return prediction
            Some(pt) => {
                let alpha_bar_prev = self.alpha_cumprod(pt);
                let sqrt_alpha_prev = alpha_bar_prev.sqrt();
                let sqrt_one_minus_prev = (1.0 - alpha_bar_prev).sqrt();
                // Deterministic DDPM: x_{t-1} = sqrt(a_{t-1}) * x0 + sqrt(1-a_{t-1}) * eps
                let prev = ((x0_pred * sqrt_alpha_prev)? + (eps_pred * sqrt_one_minus_prev)?)?;
                Ok(prev)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cosine_schedule_properties() {
        let sched = DdpmScheduler::new_cosine(1000, 20);
        // alpha_cumprod[0] should be close to 1
        assert!((sched.alpha_cumprod(0) - 0.999959).abs() < 0.001);
        // alpha_cumprod[999] should be close to 0
        assert!(sched.alpha_cumprod(999) < 0.001);
        // Monotonically decreasing
        for i in 1..999 {
            assert!(sched.alpha_cumprod(i) <= sched.alpha_cumprod(i - 1) + 1e-10);
        }
    }

    #[test]
    fn test_timesteps_count() {
        let sched = DdpmScheduler::new_cosine(1000, 10);
        assert_eq!(sched.timesteps().len(), 10);
        assert_eq!(sched.timesteps()[0], 900);
        assert_eq!(*sched.timesteps().last().unwrap(), 0);
    }

    #[test]
    fn test_ddpm_step_shape() {
        let sched = DdpmScheduler::new_cosine(1000, 20);
        let v_pred = Tensor::randn(0f32, 1., (2, 64), &Device::Cpu).unwrap();
        let sample = Tensor::randn(0f32, 1., (2, 64), &Device::Cpu).unwrap();
        let result = sched.step(&v_pred, sched.timesteps()[0], &sample).unwrap();
        assert_eq!(result.dims(), &[2, 64]);
    }

    #[test]
    fn test_ddpm_full_loop_converges() {
        let sched = DdpmScheduler::new_cosine(1000, 20);
        let mut sample = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();
        let zero = Tensor::zeros((1, 64), DType::F32, &Device::Cpu).unwrap();
        for &t in sched.timesteps() {
            sample = sched.step(&zero, t, &sample).unwrap();
        }
        let vals: Vec<f32> = sample.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_alpha_cumprod_matches_diffusers() {
        let sched = DdpmScheduler::new_cosine(1000, 10);
        // These values come from diffusers DDPMScheduler with squaredcos_cap_v2
        assert!((sched.alpha_cumprod(0) - 0.999959).abs() < 0.0001);
        assert!((sched.alpha_cumprod(500) - 0.492285).abs() < 0.001);
        assert!((sched.alpha_cumprod(900) - 0.023616).abs() < 0.001);
    }
}
