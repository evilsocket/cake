//! FlowMatchEulerDiscreteScheduler (Euler, discrete) ported from the attached Python implementation.
//!
//! Note: This is a standalone scheduler implementation (no Diffusers ConfigMixin/SchedulerMixin layer).
//! It keeps the same math and branching as the source file.

use super::t2v_pipeline::{Scheduler, SchedulerConfig, TimestepsSpec};
use candle_core::{DType, Device, Result, Tensor, bail};
use statrs::distribution::{Beta, ContinuousCDF};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TimeShiftType {
    Exponential,
    Linear,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlowMatchEulerDiscreteSchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f32,
    pub use_dynamic_shifting: bool,

    pub base_shift: Option<f32>,
    pub max_shift: Option<f32>,
    pub base_image_seq_len: Option<usize>,
    pub max_image_seq_len: Option<usize>,

    pub invert_sigmas: bool,
    pub shift_terminal: Option<f32>,

    pub use_karras_sigmas: bool,
    pub use_exponential_sigmas: bool,
    pub use_beta_sigmas: bool,

    pub time_shift_type: TimeShiftType,
    pub stochastic_sampling: bool,
}

impl Default for FlowMatchEulerDiscreteSchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            // Official Lightricks config from LTX-Video 0.9.5
            use_dynamic_shifting: false,
            base_shift: Some(0.5),
            max_shift: Some(1.15),
            base_image_seq_len: Some(256),
            max_image_seq_len: Some(4096),
            invert_sigmas: false,
            shift_terminal: None,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
            time_shift_type: TimeShiftType::Exponential,
            stochastic_sampling: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteSchedulerOutput {
    pub prev_sample: Tensor,
}

#[derive(Debug)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub config: FlowMatchEulerDiscreteSchedulerConfig,

    // Stored as tensors for convenient device/dtype conversion.
    pub timesteps: Tensor, // shape [n] (not appended)
    sigmas: Tensor,    // shape [n+1] (terminal appended)
    timesteps_cpu: Vec<f32>,
    sigmas_cpu: Vec<f32>, // includes terminal appended

    sigma_min: f32,
    sigma_max: f32,

    step_index: Option<usize>,
    begin_index: Option<usize>,
    num_inference_steps: Option<usize>,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(config: FlowMatchEulerDiscreteSchedulerConfig) -> Result<Self> {
        if config.use_beta_sigmas as u32
            + config.use_exponential_sigmas as u32
            + config.use_karras_sigmas as u32
            > 1
        {
            bail!(
                "Only one of use_beta_sigmas/use_exponential_sigmas/use_karras_sigmas can be enabled."
            );
        }

        // Equivalent to:
        // timesteps = np.linspace(1, N, N, dtype=float32)[::-1]
        // sigmas = timesteps / N
        let n = config.num_train_timesteps;
        let mut ts: Vec<f32> = (1..=n).map(|v| v as f32).collect();
        ts.reverse();

        let mut sigmas: Vec<f32> = ts.iter().map(|t| t / n as f32).collect();

        // If not dynamic shifting: apply fixed shift at init (as in Python).
        if !config.use_dynamic_shifting {
            sigmas = sigmas
                .into_iter()
                .map(|s| {
                    let shift = config.shift;
                    shift * s / (1.0 + (shift - 1.0) * s)
                })
                .collect();
            ts = sigmas.iter().map(|s| s * n as f32).collect();
        } else {
            // Python keeps unshifted schedule here and does shifting in set_timesteps(mu=...)
            ts = sigmas.iter().map(|s| s * n as f32).collect();
        }

        // Store on CPU by default.
        let device = Device::Cpu;
        let timesteps_t = Tensor::from_vec(ts.clone(), (ts.len(),), &device)?;
        let sigmas_t = Tensor::from_vec(sigmas.clone(), (sigmas.len(),), &device)?;

        let sigma_min = *sigmas.last().unwrap_or(&0.0);
        let sigma_max = *sigmas.first().unwrap_or(&1.0);

        // Note: during init, Python does NOT append terminal sigma; this is done in set_timesteps.
        // But we keep a consistent internal representation: append terminal in sigmas/sigmas_cpu.
        let mut sigmas_cpu = sigmas.clone();
        sigmas_cpu.push(0.0);
        let sigmas_with_terminal =
            Tensor::cat(&[sigmas_t, Tensor::zeros((1,), DType::F32, &device)?], 0)?;

        Ok(Self {
            config,
            timesteps: timesteps_t,
            sigmas: sigmas_with_terminal,
            timesteps_cpu: ts,
            sigmas_cpu,
            sigma_min,
            sigma_max,
            step_index: None,
            begin_index: None,
            num_inference_steps: None,
        })
    }

    pub fn shift(&self) -> f32 {
        self.config.shift
    }

    pub fn step_index(&self) -> Option<usize> {
        self.step_index
    }

    pub fn begin_index(&self) -> Option<usize> {
        self.begin_index
    }

    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = Some(begin_index);
    }

    pub fn set_shift(&mut self, shift: f32) {
        self.config.shift = shift;
    }

    fn sigma_to_t(&self, sigma: f32) -> f32 {
        sigma * self.config.num_train_timesteps as f32
    }

    fn time_shift_scalar(&self, mu: f32, sigma: f32, t: f32) -> f32 {
        match self.config.time_shift_type {
            TimeShiftType::Exponential => {
                // exp(mu) / (exp(mu) + (1/t - 1)^sigma)
                let emu = mu.exp();
                let base = (1.0 / t - 1.0).powf(sigma);
                emu / (emu + base)
            }
            TimeShiftType::Linear => {
                // mu / (mu + (1/t - 1)^sigma)
                let base = (1.0 / t - 1.0).powf(sigma);
                mu / (mu + base)
            }
        }
    }

    fn stretch_shift_to_terminal_vec(&self, t: &mut [f32]) -> Result<()> {
        let shift_terminal = match self.config.shift_terminal {
            Some(v) => v,
            None => return Ok(()),
        };
        if t.is_empty() {
            return Ok(());
        }
        let one_minus_last = 1.0 - t[t.len() - 1];
        let denom = 1.0 - shift_terminal;
        if denom.abs() < 1e-12 {
            bail!("shift_terminal too close to 1.0, would divide by zero.");
        }
        let scale_factor = one_minus_last / denom;
        for v in t.iter_mut() {
            let one_minus_z = 1.0 - *v;
            *v = 1.0 - (one_minus_z / scale_factor);
        }
        Ok(())
    }

    fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
        if steps == 0 {
            return vec![];
        }
        if steps == 1 {
            return vec![start];
        }
        let denom = (steps - 1) as f32;
        (0..steps)
            .map(|i| start + (end - start) * (i as f32) / denom)
            .collect()
    }

    fn convert_to_karras(&self, in_sigmas: &[f32], num_inference_steps: usize) -> Vec<f32> {
        let sigma_min = in_sigmas.last().copied().unwrap_or(self.sigma_min);
        let sigma_max = in_sigmas.first().copied().unwrap_or(self.sigma_max);

        let rho: f32 = 7.0;
        let ramp = Self::linspace(0.0, 1.0, num_inference_steps);

        let min_inv_rho = sigma_min.powf(1.0 / rho);
        let max_inv_rho = sigma_max.powf(1.0 / rho);

        ramp.into_iter()
            .map(|r| (max_inv_rho + r * (min_inv_rho - max_inv_rho)).powf(rho))
            .collect()
    }

    fn convert_to_exponential(&self, in_sigmas: &[f32], num_inference_steps: usize) -> Vec<f32> {
        let sigma_min = in_sigmas.last().copied().unwrap_or(self.sigma_min);
        let sigma_max = in_sigmas.first().copied().unwrap_or(self.sigma_max);

        let start = sigma_max.ln();
        let end = sigma_min.ln();
        let logs = Self::linspace(start, end, num_inference_steps);
        logs.into_iter().map(|v| v.exp()).collect()
    }

    fn convert_to_beta(
        &self,
        in_sigmas: &[f32],
        num_inference_steps: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Vec<f32>> {
        let sigma_min = in_sigmas.last().copied().unwrap_or(self.sigma_min);
        let sigma_max = in_sigmas.first().copied().unwrap_or(self.sigma_max);

        // ppf for timesteps in: 1 - linspace(0, 1, steps)
        let ts = Self::linspace(0.0, 1.0, num_inference_steps)
            .into_iter()
            .map(|v| 1.0 - v as f64)
            .collect::<Vec<_>>();

        let dist = Beta::new(alpha, beta).map_err(|e| candle_core::Error::msg(format!("{e:?}")))?;

        let mut out = Vec::with_capacity(num_inference_steps);
        for t in ts {
            let ppf = dist.inverse_cdf(t); // matches scipy.stats.beta.ppf
            let s = sigma_min as f64 + ppf * ((sigma_max - sigma_min) as f64);
            out.push(s as f32);
        }
        Ok(out)
    }

    pub fn set_timesteps(
        &mut self,
        num_inference_steps: Option<usize>,
        device: &Device,
        sigmas: Option<&[f32]>,
        mu: Option<f32>,
        timesteps: Option<&[f32]>,
    ) -> Result<()> {
        if self.config.use_dynamic_shifting && mu.is_none() {
            bail!("mu must be provided when use_dynamic_shifting = true.");
        }

        if sigmas
            .zip(timesteps)
            .is_some_and(|(s, t)| s.len() != t.len())
        {
            bail!("sigmas and timesteps must have the same length.");
        }

        let mut num_inference_steps = num_inference_steps;
        if let Some(n) = num_inference_steps {
            if sigmas.is_some_and(|s| s.len() != n) {
                bail!("sigmas length must match num_inference_steps.");
            }
            if timesteps.is_some_and(|t| t.len() != n) {
                bail!("timesteps length must match num_inference_steps.");
            }
        } else {
            // Infer from provided sigmas/timesteps.
            if let Some(s) = sigmas {
                num_inference_steps = Some(s.len());
            } else if let Some(t) = timesteps {
                num_inference_steps = Some(t.len());
            } else {
                bail!(
                    "num_inference_steps must be provided if neither sigmas nor timesteps are provided."
                );
            }
        }
        let num_inference_steps = num_inference_steps.unwrap();
        self.num_inference_steps = Some(num_inference_steps);

        // 1) Prepare default timesteps/sigmas arrays (Vec<f32>).
        let is_timesteps_provided = timesteps.is_some();
        let mut ts_vec: Option<Vec<f32>> = timesteps.map(|t| t.to_vec());

        let mut sigmas_vec: Vec<f32> = if let Some(s) = sigmas {
            s.to_vec()
        } else {
            // if timesteps is None => construct timesteps linearly in t-space
            let timesteps_vec = match ts_vec.take() {
                Some(v) => v,
                None => {
                    let start = self.sigma_to_t(self.sigma_max);
                    let end = self.sigma_to_t(self.sigma_min);
                    Self::linspace(start, end, num_inference_steps)
                }
            };
            let s = timesteps_vec
                .iter()
                .map(|t| *t / self.config.num_train_timesteps as f32)
                .collect::<Vec<_>>();
            ts_vec = Some(timesteps_vec);
            s
        };

        // 2) Perform shifting (dynamic or fixed)
        if let Some(mu) = mu {
            // Use exponential time shift (SD3 style)
            sigmas_vec = sigmas_vec
                .into_iter()
                .map(|t| self.time_shift_scalar(mu, 1.0, t))
                .collect();
        } else if self.config.use_dynamic_shifting {
            bail!("mu must be provided when use_dynamic_shifting = true.");
        } else {
            // Use standard linear/rational shift
            let shift = self.config.shift;
            sigmas_vec = sigmas_vec
                .into_iter()
                .map(|s| shift * s / (1.0 + (shift - 1.0) * s))
                .collect();
        }

        // 3) Optional stretch to terminal
        if self.config.shift_terminal.is_some() {
            self.stretch_shift_to_terminal_vec(&mut sigmas_vec)?;
        }

        // 4) Optional conversion to karras/exponential/beta
        if self.config.use_karras_sigmas {
            sigmas_vec = self.convert_to_karras(&sigmas_vec, num_inference_steps);
        } else if self.config.use_exponential_sigmas {
            sigmas_vec = self.convert_to_exponential(&sigmas_vec, num_inference_steps);
        } else if self.config.use_beta_sigmas {
            sigmas_vec = self.convert_to_beta(&sigmas_vec, num_inference_steps, 0.6, 0.6)?;
        }

        // 5) timesteps tensor
        let mut timesteps_vec: Vec<f32> = if is_timesteps_provided {
            ts_vec.unwrap_or_else(|| {
                sigmas_vec
                    .iter()
                    .map(|s| s * self.config.num_train_timesteps as f32)
                    .collect()
            })
        } else {
            sigmas_vec
                .iter()
                .map(|s| s * self.config.num_train_timesteps as f32)
                .collect()
        };

        // 6) Optional invert sigmas + append terminal sigma
        if self.config.invert_sigmas {
            for v in sigmas_vec.iter_mut() {
                *v = 1.0 - *v;
            }
            timesteps_vec = sigmas_vec
                .iter()
                .map(|s| s * self.config.num_train_timesteps as f32)
                .collect();
            sigmas_vec.push(1.0);
        } else {
            sigmas_vec.push(0.0);
        }

        self.sigmas_cpu = sigmas_vec.clone();
        self.timesteps_cpu = timesteps_vec.clone();

        self.sigmas = Tensor::from_vec(sigmas_vec, (self.sigmas_cpu.len(),), device)?;
        self.timesteps = Tensor::from_vec(timesteps_vec, (self.timesteps_cpu.len(),), device)?;

        // Reset indices like in Python.
        self.step_index = None;
        self.begin_index = None;

        Ok(())
    }

    pub fn index_for_timestep(
        &self,
        timestep: f32,
        schedule_timesteps: Option<&[f32]>,
    ) -> Result<usize> {
        let st = schedule_timesteps.unwrap_or(&self.timesteps_cpu);
        let mut indices = Vec::new();
        for (i, &v) in st.iter().enumerate() {
            if (v - timestep).abs() < 1e-6 {
                indices.push(i);
            }
        }
        if indices.is_empty() {
            bail!("timestep not found in schedule_timesteps.");
        }
        let pos = if indices.len() > 1 { 1 } else { 0 };
        Ok(indices[pos])
    }

    fn init_step_index(&mut self, timestep: f32) -> Result<()> {
        if self.begin_index.is_none() {
            self.step_index = Some(self.index_for_timestep(timestep, None)?);
        } else {
            self.step_index = self.begin_index;
        }
        Ok(())
    }

    /// Forward process in flow-matching: sample <- sigma * noise + (1 - sigma) * sample
    pub fn scale_noise(
        &self,
        sample: &Tensor,
        timestep: &Tensor,
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = sample.device();

        // timestep is expected to be 1D (batch). For scalar, allow rank 0.
        let ts: Vec<f32> = match timestep.rank() {
            0 => vec![timestep.to_scalar::<f32>()?],
            1 => timestep.to_vec1::<f32>()?,
            r => bail!("timestep must be rank 0 or 1, got rank={r}"),
        };

        // Resolve indices the same way as Python (begin_index/step_index rules).
        let mut step_indices = Vec::with_capacity(ts.len());
        if self.begin_index.is_none() {
            for &t in ts.iter() {
                step_indices.push(self.index_for_timestep(t, Some(&self.timesteps_cpu))?);
            }
        } else if let Some(si) = self.step_index {
            step_indices.extend(std::iter::repeat(si).take(ts.len()));
        } else {
            let bi = self.begin_index.unwrap_or(0);
            step_indices.extend(std::iter::repeat(bi).take(ts.len()));
        }

        // Gather sigmas and reshape/broadcast to sample rank.
        let gathered = step_indices
            .into_iter()
            .map(|idx| self.sigmas_cpu[idx])
            .collect::<Vec<f32>>();

        let mut sigma =
            Tensor::from_vec(gathered, (ts.len(),), device)?.to_dtype(sample.dtype())?;
        while sigma.rank() < sample.rank() {
            sigma = sigma.unsqueeze(sigma.rank())?;
        }

        let noise = match noise {
            Some(n) => n.clone(),
            None => Tensor::randn(0f32, 1f32, sample.shape(), device)?.to_dtype(sample.dtype())?,
        };

        let one_minus_sigma = sigma.affine(-1.0, 1.0)?;
        let a = sigma.broadcast_mul(&noise)?;
        let b = one_minus_sigma.broadcast_mul(sample)?;
        a.broadcast_add(&b)
    }

    /// One Euler step.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: f32,
        sample: &Tensor,
        per_token_timesteps: Option<&Tensor>,
    ) -> Result<FlowMatchEulerDiscreteSchedulerOutput> {
        if self.step_index.is_none() {
            self.init_step_index(timestep)?;
        }

        // Upcast to f32 (Python does: sample = sample.to(torch.float32)).
        let mut sample_f = sample.to_dtype(DType::F32)?;

        let device = sample_f.device();

        let (current_sigma, next_sigma, dt) = if let Some(per_token_ts) = per_token_timesteps {
            // per_token_sigmas = per_token_timesteps / num_train_timesteps
            let per_token_sigmas =
                per_token_ts.affine(1.0 / self.config.num_train_timesteps as f64, 0.0)?;

            // sigmas = self.sigmas[:, None, None]
            let sigmas_t = self
                .sigmas
                .to_device(device)?
                .to_dtype(per_token_sigmas.dtype())?
                .unsqueeze(1)?
                .unsqueeze(2)?;

            // lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            let threshold = per_token_sigmas.unsqueeze(0)?.affine(1.0, -1e-6)?;
            let lower_mask = sigmas_t.broadcast_lt(&threshold)?; // bool-like (u8) tensor
            let lower_mask_f = lower_mask.to_dtype(per_token_sigmas.dtype())?;

            // lower_sigmas = lower_mask * sigmas
            let lower_sigmas = lower_mask_f.broadcast_mul(&sigmas_t)?;

            // lower_sigmas, _ = lower_sigmas.max(dim=0)
            let lower_sigmas = lower_sigmas.max(0)?; // reduce over sigma dimension -> shape like per_token_sigmas

            // current_sigma = per_token_sigmas[..., None]
            // next_sigma = lower_sigmas[..., None]
            let current_sigma = per_token_sigmas.unsqueeze(per_token_sigmas.rank())?;
            let next_sigma = lower_sigmas.unsqueeze(lower_sigmas.rank())?;

            // dt = current_sigma - next_sigma
            let dt = current_sigma.broadcast_sub(&next_sigma)?;
            (current_sigma, next_sigma, dt)
        } else {
            let idx = self.step_index.expect("step_index must be initialized");
            let sigma = self.sigmas_cpu[idx];
            let sigma_next = self.sigmas_cpu[idx + 1];

            // In Python (non per-token): dt = sigma_next - sigma
            let dt = sigma_next - sigma;

            let current_sigma = Tensor::new(sigma, device)?.to_dtype(DType::F32)?;
            let next_sigma = Tensor::new(sigma_next, device)?.to_dtype(DType::F32)?;
            let dt = Tensor::new(dt, device)?.to_dtype(DType::F32)?;
            (current_sigma, next_sigma, dt)
        };

        let prev_sample = if self.config.stochastic_sampling {
            // x0 = sample - current_sigma * model_output
            let cs = current_sigma
                .broadcast_as(sample_f.shape())?
                .to_dtype(DType::F32)?;
            let x0 =
                sample_f.broadcast_sub(&cs.broadcast_mul(&model_output.to_dtype(DType::F32)?)?)?;

            // noise = randn_like(sample)
            let noise = Tensor::randn(0f32, 1f32, sample_f.shape(), device)?;

            // prev_sample = (1 - next_sigma) * x0 + next_sigma * noise
            let ns = next_sigma
                .broadcast_as(sample_f.shape())?
                .to_dtype(DType::F32)?;
            let one_minus_ns = ns.affine(-1.0, 1.0)?;
            let a = one_minus_ns.broadcast_mul(&x0)?;
            let b = ns.broadcast_mul(&noise)?;
            a.broadcast_add(&b)?
        } else {
            // prev_sample = sample + dt * model_output
            let dt = dt.broadcast_as(sample_f.shape())?.to_dtype(DType::F32)?;
            let scaled = model_output.to_dtype(DType::F32)?.broadcast_mul(&dt)?;
            sample_f = sample_f.broadcast_add(&scaled)?;
            sample_f
        };

        // Increment step index.
        if let Some(si) = self.step_index.as_mut() {
            *si += 1;
        }

        // PRECISION FIX: Keep result in F32 to prevent error accumulation over multiple steps.
        // The pipeline will convert to model dtype only when needed for transformer forward.
        // Previously: prev_sample.to_dtype(model_output.dtype())? for non per-token case
        // Now: Always return F32 to maintain precision throughout denoising loop.

        Ok(FlowMatchEulerDiscreteSchedulerOutput { prev_sample })
    }

    pub fn timesteps(&self) -> &Tensor {
        &self.timesteps
    }

    pub fn sigmas(&self) -> &Tensor {
        &self.sigmas
    }

    pub fn len(&self) -> usize {
        self.config.num_train_timesteps
    }

    pub fn is_empty(&self) -> bool {
        self.config.num_train_timesteps == 0
    }
}

impl Scheduler for FlowMatchEulerDiscreteScheduler {
    fn config(&self) -> &SchedulerConfig {
        // We need to return a reference to SchedulerConfig.
        // Since FlowMatchEulerDiscreteSchedulerConfig doesn't match exactly,
        // and trait returns reference, we either need to store SchedulerConfig
        // or change trait to return Cow or Clone.
        // For now, let's assume we can't change the trait (it returns &).
        // Hack: return a static default or store it.
        // The LtxPipeline uses this config mainly for `calculate_shift`.
        // Let's rely on LtxPipeline using its own defaults if we don't change this,
        // OR add a field to struct.
        // Simplest: use a lazy_static or constant if possible, or just unimplemented if not strictly used dynamic.
        // Converting:
        // base_image_seq_len: 256
        // max_image_seq_len: 4096
        // base_shift: 0.5
        // max_shift: 1.15

        // BETTER: allow implementing struct to own the config.
        // But for now, I'll store a `SchedulerConfig` inside `FlowMatchEulerDiscreteScheduler`?
        // No, that changes the struct definition.

        // Let's implement it by adding a phantom static or leaking? No.
        // Let's just create a static instance for now as LTX uses fixed params.
        static DEFAULT_CONFIG: std::sync::OnceLock<SchedulerConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(SchedulerConfig::default)
    }

    fn order(&self) -> usize {
        1
    }

    fn set_timesteps(&mut self, spec: TimestepsSpec, device: &Device, mu: f32) -> Result<Vec<i64>> {
        let (num, ts, sig) = match spec {
            TimestepsSpec::Steps(n) => (Some(n), None, None),
            TimestepsSpec::Timesteps(t) => (
                None,
                Some(t.iter().map(|&x| x as f32).collect::<Vec<f32>>()),
                None,
            ),
            TimestepsSpec::Sigmas(s) => (None, None, Some(s)),
        };

        self.set_timesteps(num, device, sig.as_deref(), Some(mu), ts.as_deref())?;
        let t = self.timesteps.to_vec1::<f32>()?;
        Ok(t.into_iter().map(|x| x as i64).collect())
    }

    fn step(&mut self, noise_pred: &Tensor, timestep: i64, latents: &Tensor) -> Result<Tensor> {
        // We cast timestep to f32 as underlying scheduler expects f32 (usually) or i64?
        // Scheduler::step takes timestep: f32.
        let ts = timestep as f32;
        let out = self.step(noise_pred, ts, latents, None)?;
        Ok(out.prev_sample)
    }
}
