//! LTX-2 scheduler: token-count-dependent sigma shifting.
//!
//! Generates sigma schedules with `flux_time_shift` and optional stretch-to-terminal.

use candle_core::{Result, Tensor};

use super::config::Ltx2SchedulerConfig;

/// flux_time_shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
fn flux_time_shift(mu: f32, sigma_power: f32, t: f32) -> f32 {
    let emu = mu.exp();
    if t <= 0.0 || t >= 1.0 {
        return t;
    }
    let base = (1.0 / t - 1.0).powf(sigma_power);
    emu / (emu + base)
}

/// LTX-2 scheduler.
pub struct Ltx2Scheduler {
    config: Ltx2SchedulerConfig,
}

impl Ltx2Scheduler {
    pub fn new(config: Ltx2SchedulerConfig) -> Self {
        Self { config }
    }

    /// Compute sigma schedule for a given number of tokens and steps.
    ///
    /// Returns `(steps + 1)` sigma values from ~1.0 down to 0.0.
    /// Matches Python diffusers FlowMatchEulerDiscreteScheduler:
    /// 1. Generate N sigmas (no zero) via linspace
    /// 2. Apply flux_time_shift
    /// 3. Apply stretch_to_terminal (on N sigmas, without trailing zero)
    /// 4. Append 0.0 at the end
    pub fn execute(&self, steps: usize, num_tokens: usize) -> Vec<f32> {
        let shift = self.compute_shift(num_tokens);

        // Generate N sigmas from 1.0 down to 1/steps (no zero)
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| 1.0 - (i as f32 / steps as f32))
            .collect();

        // Apply flux_time_shift
        for s in sigmas.iter_mut() {
            *s = flux_time_shift(shift, self.config.power, *s);
        }

        // Optional stretch to terminal (before appending zero)
        if let Some(terminal) = self.config.stretch_terminal {
            stretch_to_terminal(&mut sigmas, terminal);
        }

        // Append terminal zero
        sigmas.push(0.0);

        sigmas
    }

    fn compute_shift(&self, num_tokens: usize) -> f32 {
        // Dynamic shift: log-linear interpolation between base_shift and max_shift
        // based on token count (matches diffusers FlowMatchEulerDiscreteScheduler).
        // base_image_seq_len=1024, max_image_seq_len=4096 from scheduler config.
        let base_seq = 1024.0f32;
        let max_seq = 4096.0f32;

        let m = (self.config.max_shift - self.config.base_shift)
            / (max_seq - base_seq);
        let b = self.config.base_shift - m * base_seq;
        let mu = (num_tokens as f32) * m + b;
        mu
    }
}

fn stretch_to_terminal(sigmas: &mut [f32], terminal: f32) {
    if sigmas.is_empty() {
        return;
    }
    let last = *sigmas.last().unwrap();
    let one_minus_last = 1.0 - last;
    let denom = 1.0 - terminal;
    if denom.abs() < 1e-12 {
        return;
    }
    let scale = one_minus_last / denom;
    for s in sigmas.iter_mut() {
        let one_minus = 1.0 - *s;
        *s = 1.0 - (one_minus / scale);
    }
}

/// Euler diffusion step: sample + velocity * dt.
///
/// `sample`: current latent, `[B, T, D]`
/// `velocity`: model prediction (velocity)
/// `sigma`: current sigma (scalar)
/// `sigma_next`: next sigma (scalar)
pub fn euler_step(
    sample: &Tensor,
    velocity: &Tensor,
    sigma: f32,
    sigma_next: f32,
) -> Result<Tensor> {
    let dt = sigma_next - sigma;
    let scaled = velocity.affine(dt as f64, 0.0)?;
    sample.broadcast_add(&scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_flux_time_shift_boundaries() {
        // t=0 and t=1 are identity
        assert_eq!(flux_time_shift(1.0, 1.0, 0.0), 0.0);
        assert_eq!(flux_time_shift(1.0, 1.0, 1.0), 1.0);
    }

    #[test]
    fn test_flux_time_shift_midpoint() {
        // At t=0.5 with mu=0 (exp(0)=1), sigma=1: 1 / (1 + (1/0.5 - 1)^1) = 1/2 = 0.5
        let v = flux_time_shift(0.0, 1.0, 0.5);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_flux_time_shift_positive_mu() {
        // Positive mu shifts schedule toward 1 (more denoising at start)
        let v_low = flux_time_shift(0.5, 1.0, 0.5);
        let v_high = flux_time_shift(2.0, 1.0, 0.5);
        assert!(v_high > v_low);
    }

    #[test]
    fn test_scheduler_produces_correct_length() {
        let config = Ltx2SchedulerConfig::default();
        let scheduler = Ltx2Scheduler::new(config);
        let sigmas = scheduler.execute(20, 1024);
        assert_eq!(sigmas.len(), 21); // steps + 1
    }

    #[test]
    fn test_scheduler_monotonically_decreasing() {
        let config = Ltx2SchedulerConfig::default();
        let scheduler = Ltx2Scheduler::new(config);
        let sigmas = scheduler.execute(30, 2048);
        for i in 1..sigmas.len() {
            assert!(
                sigmas[i] <= sigmas[i - 1],
                "Sigma at step {} ({}) > step {} ({})",
                i,
                sigmas[i],
                i - 1,
                sigmas[i - 1]
            );
        }
        // All sigmas should be non-negative
        for (i, s) in sigmas.iter().enumerate() {
            assert!(*s >= 0.0, "Sigma at step {} ({}) is negative", i, s);
        }
        // Last sigma should be 0.0
        assert_eq!(*sigmas.last().unwrap(), 0.0);
    }

    #[test]
    fn test_scheduler_starts_near_one() {
        let config = Ltx2SchedulerConfig::default();
        let scheduler = Ltx2Scheduler::new(config);
        let sigmas = scheduler.execute(20, 1024);
        // First sigma should be close to 1 (shifted)
        assert!(sigmas[0] > 0.8);
    }

    #[test]
    fn test_scheduler_more_tokens_more_shift() {
        let config = Ltx2SchedulerConfig::default();
        let scheduler = Ltx2Scheduler::new(config);
        let sigmas_small = scheduler.execute(20, 256);
        let sigmas_large = scheduler.execute(20, 4096);
        // More tokens = more shift = higher initial sigma
        assert!(sigmas_large[1] > sigmas_small[1]);
    }

    #[test]
    fn test_euler_step() {
        let device = Device::Cpu;
        let sample = Tensor::ones((1, 4, 3), candle_core::DType::F32, &device).unwrap();
        let velocity = Tensor::full(2.0f32, (1, 4, 3), &device).unwrap();
        // dt = sigma_next - sigma = 0.8 - 1.0 = -0.2
        let result = euler_step(&sample, &velocity, 1.0, 0.8).unwrap();
        let val: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // sample + velocity * dt = 1.0 + 2.0 * (-0.2) = 0.6
        for v in &val {
            assert!((*v - 0.6).abs() < 1e-6, "Expected 0.6, got {}", v);
        }
    }
}
