/// Flow matching Euler discrete scheduler for HunyuanVideo.
///
/// Similar to Wan's scheduler: linear sigma schedule with configurable shift.
/// HunyuanVideo uses shift=7.0 by default.

#[derive(Debug, Clone)]
pub struct HunyuanFlowMatchScheduler {
    pub shift: f64,
    pub num_train_timesteps: usize,
}

impl HunyuanFlowMatchScheduler {
    pub fn new(shift: f64) -> Self {
        Self {
            shift,
            num_train_timesteps: 1000,
        }
    }

    /// Compute the sigma schedule for the given number of inference steps.
    /// Returns `num_steps + 1` sigma values from sigma_max to 0.
    pub fn sigmas(&self, num_steps: usize) -> Vec<f64> {
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        for i in 0..=num_steps {
            let t = 1.0 - (i as f64 / num_steps as f64);
            // Apply shift: sigma = shift * t / (1 + (shift - 1) * t)
            let sigma = if self.shift != 1.0 {
                self.shift * t / (1.0 + (self.shift - 1.0) * t)
            } else {
                t
            };
            sigmas.push(sigma);
        }
        sigmas
    }

    /// Compute timesteps from sigmas (sigma * num_train_timesteps).
    /// Returns `num_steps` timestep values (excludes the final sigma=0).
    pub fn timesteps(&self, num_steps: usize) -> Vec<f64> {
        self.sigmas(num_steps)
            .iter()
            .take(num_steps)
            .map(|s| s * self.num_train_timesteps as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmas_length() {
        let sched = HunyuanFlowMatchScheduler::new(7.0);
        let sigmas = sched.sigmas(50);
        assert_eq!(sigmas.len(), 51); // num_steps + 1
    }

    #[test]
    fn test_sigmas_endpoints() {
        let sched = HunyuanFlowMatchScheduler::new(7.0);
        let sigmas = sched.sigmas(50);
        assert!((sigmas[0] - 1.0).abs() < 1e-12, "first sigma should be 1.0");
        assert!(sigmas[50].abs() < 1e-12, "last sigma should be 0.0");
    }

    #[test]
    fn test_sigmas_monotonically_decreasing() {
        let sched = HunyuanFlowMatchScheduler::new(7.0);
        let sigmas = sched.sigmas(50);
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] >= sigmas[i + 1],
                "sigmas should be monotonically decreasing: sigmas[{}]={} < sigmas[{}]={}",
                i, sigmas[i], i + 1, sigmas[i + 1]
            );
        }
    }

    #[test]
    fn test_sigmas_shift_1_is_linear() {
        let sched = HunyuanFlowMatchScheduler::new(1.0);
        let sigmas = sched.sigmas(10);
        // shift=1.0 should produce linear schedule: 1.0, 0.9, 0.8, ..., 0.0
        for (i, s) in sigmas.iter().enumerate() {
            let expected = 1.0 - i as f64 / 10.0;
            assert!(
                (s - expected).abs() < 1e-12,
                "shift=1 should be linear: sigmas[{}]={} expected {}",
                i, s, expected
            );
        }
    }

    #[test]
    fn test_sigmas_shift_7_front_loaded() {
        // Higher shift front-loads the sigma schedule (more denoising early)
        let sched = HunyuanFlowMatchScheduler::new(7.0);
        let sigmas = sched.sigmas(10);
        // At midpoint (i=5, t=0.5): sigma = 7*0.5 / (1 + 6*0.5) = 3.5 / 4.0 = 0.875
        let mid = sigmas[5];
        assert!(
            (mid - 0.875).abs() < 1e-12,
            "midpoint sigma with shift=7 should be 0.875, got {}",
            mid
        );
    }

    #[test]
    fn test_timesteps_length() {
        let sched = HunyuanFlowMatchScheduler::new(7.0);
        let timesteps = sched.timesteps(50);
        assert_eq!(timesteps.len(), 50); // excludes final sigma=0
    }

    #[test]
    fn test_timesteps_range() {
        let sched = HunyuanFlowMatchScheduler::new(7.0);
        let timesteps = sched.timesteps(50);
        // First timestep should be sigma[0] * 1000 = 1000
        assert!((timesteps[0] - 1000.0).abs() < 1e-9);
        // All timesteps should be in (0, 1000]
        for (i, t) in timesteps.iter().enumerate() {
            assert!(
                *t > 0.0 && *t <= 1000.0,
                "timestep[{}]={} out of range (0, 1000]",
                i, t
            );
        }
    }
}
