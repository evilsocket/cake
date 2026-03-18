//! Euler ODE solver for flow matching.
//!
//! Flow matching update:
//!   x_1_pred = x + (1 - t) * v
//!   x_0_pred = x - t * v
//!   x_next = (1 - t_next) * x_0_pred + t_next * x_1_pred
//!   (last step: x = x_1_pred)

use anyhow::Result;
use candle_core::Tensor;

/// Euler solver for flow matching.
#[derive(Debug, Clone)]
pub struct EulerSolver {
    pub num_steps: usize,
    pub t_shift: f32,
}

impl EulerSolver {
    pub fn new(num_steps: usize, t_shift: f32) -> Self {
        Self { num_steps, t_shift }
    }

    /// Generate the time schedule: linspace(0, 1, num_steps+1) with t_shift applied.
    pub fn time_schedule(&self) -> Vec<f32> {
        let mut times = Vec::with_capacity(self.num_steps + 1);
        for i in 0..=self.num_steps {
            let t = i as f32 / self.num_steps as f32;
            let t_shifted = if (self.t_shift - 1.0).abs() > 1e-6 {
                self.t_shift * t / (1.0 + (self.t_shift - 1.0) * t)
            } else {
                t
            };
            times.push(t_shifted);
        }
        times
    }

    /// Perform one flow matching Euler step.
    /// Returns updated x for the next timestep.
    pub fn step(
        x: &Tensor,
        v: &Tensor,
        t_cur: f32,
        t_next: f32,
        is_last: bool,
    ) -> Result<Tensor> {
        // x_1_pred = x + (1 - t_cur) * v
        let x_1_pred = (x + (v * (1.0 - t_cur) as f64)?)?;

        if is_last {
            // Last step: return x_1_pred directly
            return Ok(x_1_pred);
        }

        // x_0_pred = x - t_cur * v
        let x_0_pred = (x - (v * t_cur as f64)?)?;

        // x_next = (1 - t_next) * x_0_pred + t_next * x_1_pred
        let result = ((&x_0_pred * (1.0 - t_next) as f64)? + (&x_1_pred * t_next as f64)?)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_time_schedule_no_shift() {
        let solver = EulerSolver::new(4, 1.0);
        let times = solver.time_schedule();
        assert_eq!(times.len(), 5);
        assert!((times[0]).abs() < 1e-6);
        assert!((times[4] - 1.0).abs() < 1e-6);
        for i in 0..4 {
            assert!((times[i + 1] - times[i] - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_time_schedule_with_shift() {
        let solver = EulerSolver::new(4, 2.0);
        let times = solver.time_schedule();
        assert_eq!(times.len(), 5);
        assert!((times[0]).abs() < 1e-6);
        assert!((times[4] - 1.0).abs() < 1e-6);
        assert!(times[1] > 0.25); // shifted forward
    }

    #[test]
    fn test_euler_step_last() {
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let v = Tensor::new(&[0.5f32, 1.0, 1.5], &Device::Cpu).unwrap();
        // Last step: x_1_pred = x + (1 - t) * v, with t=0.75
        let x_new = EulerSolver::step(&x, &v, 0.75, 1.0, true)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // x_1_pred = [1+0.25*0.5, 2+0.25*1.0, 3+0.25*1.5] = [1.125, 2.25, 3.375]
        assert!((x_new[0] - 1.125).abs() < 1e-6);
        assert!((x_new[1] - 2.25).abs() < 1e-6);
        assert!((x_new[2] - 3.375).abs() < 1e-6);
    }

    #[test]
    fn test_euler_step_intermediate() {
        let x = Tensor::new(&[0.0f32], &Device::Cpu).unwrap();
        let v = Tensor::new(&[4.0f32], &Device::Cpu).unwrap();
        // t_cur=0.25, t_next=0.5, not last
        // x_1_pred = 0 + 0.75 * 4 = 3.0
        // x_0_pred = 0 - 0.25 * 4 = -1.0
        // x_next = 0.5 * (-1.0) + 0.5 * 3.0 = 1.0
        let x_new = EulerSolver::step(&x, &v, 0.25, 0.5, false)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((x_new[0] - 1.0).abs() < 1e-6);
    }
}
