/// Flow matching Euler discrete scheduler for Wan2.2.
///
/// Simpler than LTX-2's scheduler: fixed shift, linear sigma schedule.

#[derive(Debug, Clone)]
pub struct WanFlowMatchScheduler {
    pub shift: f64,
    pub num_train_timesteps: usize,
}

impl WanFlowMatchScheduler {
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
            let sigma = self.shift * t / (1.0 + (self.shift - 1.0) * t);
            sigmas.push(sigma);
        }
        sigmas
    }

    /// Compute timesteps from sigmas (sigma * num_train_timesteps).
    pub fn timesteps(&self, num_steps: usize) -> Vec<f64> {
        self.sigmas(num_steps)
            .iter()
            .map(|s| s * self.num_train_timesteps as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wan_sigmas_length() {
        let sched = WanFlowMatchScheduler::new(5.0);
        let sigmas = sched.sigmas(40);
        assert_eq!(sigmas.len(), 41);
    }

    #[test]
    fn test_wan_sigmas_endpoints() {
        let sched = WanFlowMatchScheduler::new(5.0);
        let sigmas = sched.sigmas(40);
        assert!((sigmas[0] - 1.0).abs() < 1e-12, "first sigma should be 1.0, got {}", sigmas[0]);
        assert!(sigmas[40].abs() < 1e-12, "last sigma should be 0.0, got {}", sigmas[40]);
    }

    #[test]
    fn test_wan_sigmas_monotonically_decreasing() {
        let sched = WanFlowMatchScheduler::new(5.0);
        let sigmas = sched.sigmas(40);
        for i in 0..sigmas.len() - 1 {
            assert!(sigmas[i] >= sigmas[i + 1],
                "sigmas should decrease: [{i}]={} > [{}]={}", sigmas[i], i + 1, sigmas[i + 1]);
        }
    }

    #[test]
    fn test_wan_shift_5_midpoint() {
        // At midpoint (t=0.5): sigma = 5*0.5 / (1 + 4*0.5) = 2.5 / 3.0 = 5/6
        let sched = WanFlowMatchScheduler::new(5.0);
        let sigmas = sched.sigmas(10);
        let mid = sigmas[5];
        let expected = 5.0 * 0.5 / (1.0 + 4.0 * 0.5);
        assert!((mid - expected).abs() < 1e-12, "midpoint: got {mid}, expected {expected}");
    }

    #[test]
    fn test_wan_timesteps_length() {
        let sched = WanFlowMatchScheduler::new(5.0);
        let timesteps = sched.timesteps(40);
        // Wan timesteps() includes the final sigma=0 (maps to all sigmas, not num_steps)
        assert_eq!(timesteps.len(), 41);
    }

    #[test]
    fn test_wan_timesteps_first_is_1000() {
        let sched = WanFlowMatchScheduler::new(5.0);
        let timesteps = sched.timesteps(40);
        assert!((timesteps[0] - 1000.0).abs() < 1e-9, "first timestep should be 1000");
    }
}
