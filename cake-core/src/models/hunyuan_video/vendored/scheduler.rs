use anyhow::Result;
use candle_core::{Device, Tensor};

/// Flow matching Euler discrete scheduler for HunyuanVideo.
///
/// Similar to LTX-Video's FlowMatchEulerDiscreteScheduler but with
/// HunyuanVideo-specific defaults and shift parameters.
pub struct HunyuanScheduler {
    pub num_inference_steps: usize,
    pub shift: f64,
    timesteps: Vec<f64>,
    sigmas: Vec<f64>,
}

impl HunyuanScheduler {
    pub fn new(num_inference_steps: usize) -> Self {
        let shift = 7.0; // HunyuanVideo default shift

        let mut timesteps = Vec::with_capacity(num_inference_steps + 1);
        let mut sigmas = Vec::with_capacity(num_inference_steps + 1);

        for i in 0..=num_inference_steps {
            let t = 1.0 - (i as f64 / num_inference_steps as f64);
            let sigma = t;
            timesteps.push(t * 1000.0);
            sigmas.push(sigma);
        }

        Self {
            num_inference_steps,
            shift,
            timesteps,
            sigmas,
        }
    }

    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// Perform one Euler step.
    pub fn step(
        &self,
        model_output: &Tensor,
        sample: &Tensor,
        sigma: f64,
        sigma_next: f64,
    ) -> Result<Tensor> {
        let dt = sigma_next - sigma;
        Ok((sample + model_output * dt)?)
    }

    /// Create initial noise latents.
    pub fn create_noise(
        batch_size: usize,
        channels: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        device: &Device,
    ) -> Result<Tensor> {
        Ok(Tensor::randn(
            0f32,
            1f32,
            (batch_size, channels, num_frames, height, width),
            device,
        )?)
    }
}
