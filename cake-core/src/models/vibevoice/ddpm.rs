//! DDPM noise scheduler with v-prediction support.
//!
//! Implements the reverse diffusion process for the VibeVoice prediction head.
//! Uses cosine beta schedule and v-prediction parameterization.

use candle_core::{Result, Tensor};

/// DDPM scheduler state.
#[derive(Debug, Clone)]
pub struct DdpmScheduler {
    /// Cumulative product of alphas: ᾱ_t
    alphas_cumprod: Vec<f64>,
    /// Inference timesteps (descending from T to 0).
    timesteps: Vec<usize>,
    /// Total training timesteps.
    num_train_steps: usize,
}

impl DdpmScheduler {
    /// Create a new DDPM scheduler with cosine beta schedule.
    pub fn new_cosine(num_train_steps: usize, num_inference_steps: usize) -> Self {
        // Cosine schedule: ᾱ_t = cos²((t/T + s) / (1 + s) * π/2)
        let s = 0.008;
        let alphas_cumprod: Vec<f64> = (0..=num_train_steps)
            .map(|t| {
                ((t as f64 / num_train_steps as f64 + s) / (1.0 + s)
                    * std::f64::consts::FRAC_PI_2)
                    .cos()
                    .powi(2)
            })
            .collect();

        // Normalize, clip to [0.0001, 0.9999] (matching diffusers)
        let alpha_0 = alphas_cumprod[0];
        let alphas_cumprod: Vec<f64> = alphas_cumprod
            .iter()
            .map(|a| (a / alpha_0).clamp(0.0001, 0.9999))
            .collect();

        // Diffusers-compatible timestep spacing: [900, 800, ..., 100, 0]
        let step_ratio = num_train_steps / num_inference_steps;
        let timesteps: Vec<usize> = (0..num_inference_steps)
            .rev()
            .map(|i| i * step_ratio)
            .collect();

        Self {
            alphas_cumprod,
            timesteps,
            num_train_steps,
        }
    }

    /// Get the inference timesteps.
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Get ᾱ_t for a given timestep.
    fn alpha_cumprod(&self, t: usize) -> f64 {
        self.alphas_cumprod[t.min(self.num_train_steps)]
    }

    /// Single DDPM reverse step with v-prediction.
    ///
    /// Given current noisy sample x_t and model prediction v_pred at timestep t,
    /// compute x_{t-1} (or x_0 for the last step).
    ///
    /// v-prediction: v = α_t * ε - σ_t * x_0
    /// → x_0 = α_t * x_t - σ_t * v
    /// → ε = σ_t * x_t + α_t * v
    pub fn step(
        &self,
        v_pred: &Tensor,
        timestep: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let alpha_bar_t = self.alpha_cumprod(timestep);
        let alpha_t = alpha_bar_t.sqrt();
        let sigma_t = (1.0 - alpha_bar_t).sqrt();

        // Predict x_0 from v-prediction: x_0 = α_t * x_t - σ_t * v
        let x0_pred = ((sample * alpha_t)? - (v_pred * sigma_t)?)?;

        // Determine previous timestep
        let t_idx = self.timesteps.iter().position(|&t| t == timestep);
        let prev_t = match t_idx {
            Some(idx) if idx + 1 < self.timesteps.len() => self.timesteps[idx + 1],
            _ => 0, // Last step → return x_0
        };

        if prev_t == 0 {
            return Ok(x0_pred);
        }

        let alpha_bar_prev = self.alpha_cumprod(prev_t);
        let alpha_prev = alpha_bar_prev.sqrt();
        let sigma_prev = (1.0 - alpha_bar_prev).sqrt();

        // Predict noise: ε = σ_t * x_t + α_t * v
        let noise_pred = ((sample * sigma_t)? + (v_pred * alpha_t)?)?;

        // DDPM posterior: x_{t-1} = α_{t-1} * x_0 + σ_{t-1} * ε
        // (deterministic, no added noise during inference)
        let prev = ((x0_pred * alpha_prev)? + (noise_pred * sigma_prev)?)?;
        Ok(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cosine_schedule_properties() {
        let sched = DdpmScheduler::new_cosine(1000, 20);

        // ᾱ_0 should be ~1
        assert!((sched.alpha_cumprod(0) - 1.0).abs() < 0.01);
        // ᾱ_T should be ~0
        assert!(sched.alpha_cumprod(999) < 0.01);
        // Monotonically decreasing
        for i in 1..1000 {
            assert!(sched.alpha_cumprod(i) <= sched.alpha_cumprod(i - 1));
        }
    }

    #[test]
    fn test_timesteps_count() {
        let sched = DdpmScheduler::new_cosine(1000, 20);
        assert_eq!(sched.timesteps().len(), 20);
        // Should be descending
        for w in sched.timesteps().windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn test_ddpm_step_shape() {
        let sched = DdpmScheduler::new_cosine(1000, 20);
        let v_pred = Tensor::randn(0f32, 1., (2, 64), &Device::Cpu).unwrap();
        let sample = Tensor::randn(0f32, 1., (2, 64), &Device::Cpu).unwrap();

        let t = sched.timesteps()[0]; // First (largest) timestep
        let result = sched.step(&v_pred, t, &sample).unwrap();
        assert_eq!(result.dims(), &[2, 64]);
    }

    #[test]
    fn test_ddpm_full_loop_converges() {
        let sched = DdpmScheduler::new_cosine(1000, 20);
        let mut sample = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();

        // Zero v-prediction should converge to a fixed point
        let zero_pred = Tensor::zeros((1, 64), DType::F32, &Device::Cpu).unwrap();
        for &t in sched.timesteps() {
            sample = sched.step(&zero_pred, t, &sample).unwrap();
        }
        // Output should be finite
        let vals: Vec<f32> = sample.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}
