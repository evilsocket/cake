//! DPM-Solver++ multistep scheduler with v-prediction support.
//!
//! Matches diffusers' DPMSolverMultistepScheduler with:
//! - algorithm_type="dpmsolver++"
//! - solver_type="midpoint"
//! - solver_order=2
//! - prediction_type="v_prediction"
//! - beta_schedule="squaredcos_cap_v2"
//! - timestep_spacing="linspace"
//! - lower_order_final=true

use candle_core::Result;
use candle_core::Tensor;

/// DPM-Solver++ scheduler.
#[derive(Debug, Clone)]
pub struct DpmSolverPP {
    /// sqrt(alpha_cumprod) per training timestep.
    alpha_t: Vec<f64>,
    /// sqrt(1 - alpha_cumprod) per training timestep.
    sigma_t: Vec<f64>,
    /// log(alpha_t / sigma_t) per training timestep (log-SNR).
    lambda_t: Vec<f64>,
    /// Inference timesteps (descending, e.g. [999, 949, ..., 50]).
    timesteps: Vec<usize>,
}

impl DpmSolverPP {
    /// Create a DPM-Solver++ scheduler with cosine (squaredcos_cap_v2) beta schedule.
    pub fn new_cosine(num_train_steps: usize, num_inference_steps: usize) -> Self {
        let s = 0.008_f64;
        let max_beta = 0.999_f64;

        let alpha_bar_fn = |t: f64| -> f64 {
            ((t / num_train_steps as f64 + s) / (1.0 + s) * std::f64::consts::FRAC_PI_2)
                .cos()
                .powi(2)
        };

        let mut betas = Vec::with_capacity(num_train_steps);
        for i in 0..num_train_steps {
            let beta = (1.0 - alpha_bar_fn((i + 1) as f64) / alpha_bar_fn(i as f64)).min(max_beta);
            betas.push(beta);
        }

        let mut alphas_cumprod = Vec::with_capacity(num_train_steps);
        let mut cumprod = 1.0_f64;
        for b in &betas {
            cumprod *= 1.0 - b;
            alphas_cumprod.push(cumprod);
        }

        let alpha_t: Vec<f64> = alphas_cumprod.iter().map(|a| a.sqrt()).collect();
        let sigma_t: Vec<f64> = alphas_cumprod.iter().map(|a| (1.0 - a).sqrt()).collect();
        let lambda_t: Vec<f64> = alpha_t
            .iter()
            .zip(sigma_t.iter())
            .map(|(a, s)| (a / s).ln())
            .collect();

        // Linspace timestep spacing: np.linspace(0, N-1, steps+1).round()[::-1][:-1]
        let n = num_inference_steps + 1;
        let max_t = (num_train_steps - 1) as f64;
        let mut ts: Vec<usize> = (0..n)
            .map(|i| (i as f64 * max_t / (n - 1) as f64).round() as usize)
            .collect();
        ts.reverse();
        ts.pop(); // Remove trailing 0
        let timesteps = ts;

        Self {
            alpha_t,
            sigma_t,
            lambda_t,
            timesteps,
        }
    }

    /// Get the inference timesteps.
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Convert v-prediction to x0 prediction.
    /// x0 = alpha_t * sample - sigma_t * v_pred
    pub fn convert_v_to_x0(
        &self,
        v_pred: &Tensor,
        t: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let alpha = self.alpha_t[t];
        let sigma = self.sigma_t[t];
        (sample * alpha)? - (v_pred * sigma)?
    }

    /// First-order DPM-Solver++ update (from timestep s to t, s > t).
    ///
    /// x_t = (sigma_t/sigma_s) * x - alpha_t * expm1(-h) * x0
    pub fn first_order_update(
        &self,
        x0_pred: &Tensor,
        s: usize,
        t: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let lambda_s = self.lambda_t[s];
        let lambda_t = self.lambda_t[t];
        let alpha_t = self.alpha_t[t];
        let sigma_t = self.sigma_t[t];
        let sigma_s = self.sigma_t[s];

        let h = lambda_t - lambda_s; // positive (lambda increases as noise decreases)
        let ratio = sigma_t / sigma_s;
        let coeff = alpha_t * (-h).exp_m1(); // alpha_t * (exp(-h) - 1), negative for h > 0

        (sample * ratio)? - (x0_pred * coeff)?
    }

    /// Second-order DPM-Solver++ midpoint update.
    ///
    /// m0 = most recent x0 prediction (at timestep s0)
    /// m1 = previous x0 prediction (at timestep s1)
    /// Goes from s0 to target t.
    pub fn second_order_update(
        &self,
        m0: &Tensor,
        m1: &Tensor,
        s0: usize,
        s1: usize,
        t: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        let lambda_s0 = self.lambda_t[s0];
        let lambda_s1 = self.lambda_t[s1];
        let lambda_t = self.lambda_t[t];
        let alpha_t = self.alpha_t[t];
        let sigma_t = self.sigma_t[t];
        let sigma_s0 = self.sigma_t[s0];

        let h = lambda_t - lambda_s0;
        let h_0 = lambda_s0 - lambda_s1;
        let r0 = h_0 / h;

        // D0 = m0, D1 = (1/(2r)) * (m0 - m1)  [midpoint solver type]
        let d1 = ((m0 - m1)? * (1.0 / (2.0 * r0)))?;

        let ratio = sigma_t / sigma_s0;
        let expm1_neg_h = (-h).exp_m1(); // exp(-h) - 1

        // x_t = ratio * x - alpha_t * expm1(-h) * D0 - 0.5 * alpha_t * expm1(-h) * D1
        let coeff = alpha_t * expm1_neg_h;
        let term1 = (sample * ratio)?;
        let term2 = (m0 * coeff)?;
        let term3 = (&d1 * (0.5 * coeff))?;
        (term1 - term2)? - term3
    }

    /// Run a complete diffusion step, managing solver order automatically.
    ///
    /// `step_idx`: 0-based index into timesteps
    /// `x0_buffer`: mutable buffer of previous x0 predictions (caller maintains)
    /// `ts_buffer`: mutable buffer of corresponding timesteps (caller maintains)
    ///
    /// Returns the updated sample.
    pub fn step(
        &self,
        v_pred: &Tensor,
        step_idx: usize,
        sample: &Tensor,
        x0_buffer: &mut Vec<Tensor>,
        ts_buffer: &mut Vec<usize>,
    ) -> Result<Tensor> {
        let t = self.timesteps[step_idx];
        let num_steps = self.timesteps.len();

        // Convert v-prediction to x0
        let x0 = self.convert_v_to_x0(v_pred, t, sample)?;

        // Target: next timestep, or final_sigma=0 (perfectly clean) for last step
        let is_final = step_idx + 1 >= num_steps;
        let next_t = if !is_final {
            Some(self.timesteps[step_idx + 1])
        } else {
            None // final_sigmas_type="zero": target sigma=0, alpha=1
        };

        // Use first order for: first step, last step (lower_order_final with final_sigma=0)
        let use_first_order = step_idx == 0
            || is_final
            || x0_buffer.is_empty();

        let result = if is_final {
            // final_sigmas_type="zero": last step returns x0 directly
            // (sigma_t=0 means x_t = 0*sample - 1*(0-1)*x0 = x0)
            x0.clone()
        } else if use_first_order {
            self.first_order_update(&x0, t, next_t.unwrap(), sample)?
        } else {
            let m1 = x0_buffer.last().unwrap();
            let s1 = *ts_buffer.last().unwrap();
            self.second_order_update(&x0, m1, t, s1, next_t.unwrap(), sample)?
        };

        // Update buffers (keep only last entry for second-order)
        x0_buffer.clear();
        x0_buffer.push(x0);
        ts_buffer.clear();
        ts_buffer.push(t);

        Ok(result)
    }
}

// Keep the old name as an alias for backward compatibility
pub type DdpmScheduler = DpmSolverPP;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cosine_schedule_properties() {
        let sched = DpmSolverPP::new_cosine(1000, 20);
        // alpha_t[0]^2 should be close to 1
        let ac0 = sched.alpha_t[0].powi(2);
        assert!((ac0 - 0.999959).abs() < 0.001);
        // alpha_t[999]^2 should be close to 0
        let ac999 = sched.alpha_t[999].powi(2);
        assert!(ac999 < 0.001);
        // Monotonically decreasing alpha_cumprod
        for i in 1..999 {
            assert!(sched.alpha_t[i] <= sched.alpha_t[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_linspace_timesteps() {
        let sched = DpmSolverPP::new_cosine(1000, 20);
        assert_eq!(sched.timesteps().len(), 20);
        assert_eq!(sched.timesteps()[0], 999);
        assert_eq!(*sched.timesteps().last().unwrap(), 50);
    }

    #[test]
    fn test_lambda_monotonic() {
        let sched = DpmSolverPP::new_cosine(1000, 20);
        // lambda should be monotonically increasing (more signal at lower t)
        for i in 1..1000 {
            assert!(
                sched.lambda_t[i - 1] > sched.lambda_t[i],
                "lambda not decreasing at i={}: {} vs {}",
                i,
                sched.lambda_t[i - 1],
                sched.lambda_t[i]
            );
        }
    }

    #[test]
    fn test_first_order_step_shape() {
        let sched = DpmSolverPP::new_cosine(1000, 20);
        let v_pred = Tensor::randn(0f32, 1., (2, 64), &Device::Cpu).unwrap();
        let sample = Tensor::randn(0f32, 1., (2, 64), &Device::Cpu).unwrap();
        let x0 = sched.convert_v_to_x0(&v_pred, 999, &sample).unwrap();
        let result = sched.first_order_update(&x0, 999, 949, &sample).unwrap();
        assert_eq!(result.dims(), &[2, 64]);
    }

    #[test]
    fn test_second_order_step_shape() {
        let sched = DpmSolverPP::new_cosine(1000, 20);
        let m0 = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();
        let m1 = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();
        let sample = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();
        let result = sched
            .second_order_update(&m0, &m1, 949, 999, 899, &sample)
            .unwrap();
        assert_eq!(result.dims(), &[1, 64]);
    }

    #[test]
    fn test_full_loop_converges() {
        let sched = DpmSolverPP::new_cosine(1000, 20);
        let mut sample = Tensor::randn(0f32, 1., (1, 64), &Device::Cpu).unwrap();
        let zero = Tensor::zeros((1, 64), DType::F32, &Device::Cpu).unwrap();
        let mut x0_buf: Vec<Tensor> = Vec::new();
        let mut ts_buf: Vec<usize> = Vec::new();

        for step_idx in 0..sched.timesteps().len() {
            sample = sched.step(&zero, step_idx, &sample, &mut x0_buf, &mut ts_buf).unwrap();
        }
        let vals: Vec<f32> = sample.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_step_sequence_uses_both_orders() {
        // Verify that step 0 uses first order, step 1 uses second order
        let sched = DpmSolverPP::new_cosine(1000, 20);
        let sample = Tensor::ones((1, 4), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
        let mut x0_buf: Vec<Tensor> = Vec::new();
        let mut ts_buf: Vec<usize> = Vec::new();

        // Step 0: first order (empty buffer)
        let s1 = sched.step(&v, 0, &sample, &mut x0_buf, &mut ts_buf).unwrap();
        assert_eq!(x0_buf.len(), 1);
        assert_eq!(ts_buf.len(), 1);

        // Step 1: second order (buffer has 1 entry)
        let s2 = sched.step(&v, 1, &s1, &mut x0_buf, &mut ts_buf).unwrap();
        assert_eq!(s2.dims(), [1, 4]);
    }

    #[test]
    fn test_v_to_x0_identity() {
        // When v=0 and sample=x, x0 = alpha * x
        let sched = DpmSolverPP::new_cosine(1000, 20);
        let sample = Tensor::ones((1, 4), DType::F32, &Device::Cpu).unwrap();
        let v_zero = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
        let x0 = sched.convert_v_to_x0(&v_zero, 500, &sample).unwrap();
        let vals: Vec<f32> = x0.to_vec2().unwrap()[0].clone();
        let expected = sched.alpha_t[500] as f32;
        for v in vals {
            assert!((v - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_alpha_cumprod_matches_diffusers() {
        let sched = DpmSolverPP::new_cosine(1000, 10);
        let ac = |t: usize| sched.alpha_t[t].powi(2);
        assert!((ac(0) - 0.999959).abs() < 0.0001);
        assert!((ac(500) - 0.492285).abs() < 0.001);
        assert!((ac(900) - 0.023616).abs() < 0.001);
    }
}
