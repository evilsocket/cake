//! VibeVoice diffusion prediction head.
//!
//! A 4-layer DiT-style diffusion head with AdaLN modulation and SwiGLU FFN.
//! Takes noisy acoustic latents (64-dim) + LLM hidden states (896-dim) +
//! timestep → predicts noise (v-prediction).

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

/// Sinusoidal timestep embedding → MLP.
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    mlp_0_weight: Tensor,
    mlp_2_weight: Tensor,
    backend: Arc<dyn ComputeBackend>,
}

impl TimestepEmbedder {
    pub fn load(vb: VarBuilder, hidden_size: usize, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let mlp_0_weight = vb.pp("mlp").pp("0").get((hidden_size, 256), "weight")?;
        let mlp_2_weight = vb.pp("mlp").pp("2").get((hidden_size, hidden_size), "weight")?;
        Ok(Self { mlp_0_weight, mlp_2_weight, backend })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // Sinusoidal embedding: t → (batch, 256)
        let half_dim = 128;
        let emb = {
            // Compute frequency vector on CPU — avoids arange + to_dtype + mul + exp tensor ops
            let decay = -f64::ln(10000.0) / half_dim as f64;
            let freq_data: Vec<f32> = (0..half_dim).map(|j| (j as f64 * decay).exp() as f32).collect();
            let freq = Tensor::new(freq_data.as_slice(), t.device())?.unsqueeze(0)?;
            let t_f32 = t.to_dtype(DType::F32)?;
            let args = t_f32.unsqueeze(1)?.broadcast_mul(&freq)?;
            Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(t.dtype())?
        };
        // MLP: 256 → hidden → hidden with SiLU
        let h = self.backend.linear_forward(&emb, &self.mlp_0_weight, None)?;
        let h = self.backend.silu(&h)?;
        self.backend.linear_forward(&h, &self.mlp_2_weight, None)
    }
}

/// SwiGLU feed-forward network.
#[derive(Debug, Clone)]
struct FeedForward {
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    backend: Arc<dyn ComputeBackend>,
}

impl FeedForward {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let gate_proj_weight = vb.pp("gate_proj").get((intermediate, hidden), "weight")?;
        let up_proj_weight = vb.pp("up_proj").get((intermediate, hidden), "weight")?;
        let down_proj_weight = vb.pp("down_proj").get((hidden, intermediate), "weight")?;
        Ok(Self { gate_proj_weight, up_proj_weight, down_proj_weight, backend })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.backend.linear_forward(x, &self.gate_proj_weight, None)?;
        let up = self.backend.linear_forward(x, &self.up_proj_weight, None)?;
        // Fused silu(gate) * up — 1 kernel instead of 2
        let gated = self.backend.silu_mul(&gate, &up)?;
        self.backend.linear_forward(&gated, &self.down_proj_weight, None)
    }
}

/// Single diffusion block with AdaLN modulation.
#[derive(Debug, Clone)]
struct DiffusionBlock {
    norm_weight: Tensor,
    eps: f32,
    ffn: FeedForward,
    ada_ln_weight: Tensor,
    backend: Arc<dyn ComputeBackend>,
}

impl DiffusionBlock {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize, eps: f64, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let norm_weight = vb.pp("norm").get(hidden, "weight")?;
        let ffn = FeedForward::load(vb.pp("ffn"), hidden, intermediate, backend.clone())?;
        let ada_ln_weight = vb.pp("adaLN_modulation").pp("1").get((3 * hidden, hidden), "weight")?;
        Ok(Self {
            norm_weight,
            eps: eps as f32,
            ffn,
            ada_ln_weight,
            backend,
        })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = self.backend.silu(cond)
            .and_then(|c| self.backend.linear_forward(&c, &self.ada_ln_weight, None))?;
        let chunks = modulation.chunk(3, D::Minus1)?;
        let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);

        let h = self.backend.adaln_modulate(x, &self.norm_weight, scale, shift, self.eps)?;
        let h = self.ffn.forward(&h)?;
        x + h.broadcast_mul(gate)?
    }

    /// Forward with pre-computed silu(cond).
    fn forward_with_silu_cond(&self, x: &Tensor, silu_cond: &Tensor) -> Result<Tensor> {
        let modulation = self.backend.linear_forward(silu_cond, &self.ada_ln_weight, None)?;
        let chunks = modulation.chunk(3, D::Minus1)?;
        let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);

        let h = self.backend.adaln_modulate(x, &self.norm_weight, scale, shift, self.eps)?;
        let h = self.ffn.forward(&h)?;
        x + h.broadcast_mul(gate)?
    }
}

/// Final output layer with RMSNorm (no affine) + AdaLN + linear projection.
#[derive(Debug, Clone)]
struct FinalLayer {
    norm_weight: Tensor,
    eps: f32,
    ada_ln_weight: Tensor,
    linear_weight: Tensor,
    backend: Arc<dyn ComputeBackend>,
}

impl FinalLayer {
    fn load(vb: VarBuilder, hidden: usize, latent: usize, eps: f64, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        // norm_final is RMSNorm with elementwise_affine=False — use weight=ones
        let norm_weight = Tensor::ones(hidden, candle_core::DType::F32, vb.device())?
            .to_dtype(vb.dtype())?;
        let ada_ln_weight = vb.pp("adaLN_modulation").pp("1").get((2 * hidden, hidden), "weight")?;
        let linear_weight = vb.pp("linear").get((latent, hidden), "weight")?;
        Ok(Self {
            norm_weight,
            eps: eps as f32,
            ada_ln_weight,
            linear_weight,
            backend,
        })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = self.backend.silu(cond)
            .and_then(|c| self.backend.linear_forward(&c, &self.ada_ln_weight, None))?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let h = self.backend.adaln_modulate(x, &self.norm_weight, scale, shift, self.eps)?;
        self.backend.linear_forward(&h, &self.linear_weight, None)
    }

    fn forward_with_silu_cond(&self, x: &Tensor, silu_cond: &Tensor) -> Result<Tensor> {
        let modulation = self.backend.linear_forward(silu_cond, &self.ada_ln_weight, None)?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let h = self.backend.adaln_modulate(x, &self.norm_weight, scale, shift, self.eps)?;
        self.backend.linear_forward(&h, &self.linear_weight, None)
    }
}

/// Complete prediction head for diffusion.
#[derive(Debug, Clone)]
pub struct PredictionHead {
    t_embedder: TimestepEmbedder,
    noisy_images_proj_weight: Tensor,
    cond_proj_weight: Tensor,
    layers: Vec<DiffusionBlock>,
    final_layer: FinalLayer,
    /// Pre-computed timestep embeddings (one per inference step).
    pre_t_embeddings: Vec<Tensor>,
    backend: Arc<dyn ComputeBackend>,
}

impl PredictionHead {
    pub fn load(
        vb: VarBuilder,
        cfg: &super::config::DiffusionHeadConfig,
        timesteps: &[usize],
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let latent = cfg.latent_size;
        let intermediate = (h as f64 * cfg.head_ffn_ratio) as usize;

        let t_embedder = TimestepEmbedder::load(vb.pp("t_embedder"), h, backend.clone())?;
        let noisy_images_proj_weight = vb.pp("noisy_images_proj").get((h, latent), "weight")?;
        let cond_proj_weight = vb.pp("cond_proj").get((h, h), "weight")?;

        let mut layers = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            layers.push(DiffusionBlock::load(
                vb.pp("layers").pp(i),
                h,
                intermediate,
                cfg.rms_norm_eps,
                backend.clone(),
            )?);
        }

        let final_layer = FinalLayer::load(vb.pp("final_layer"), h, latent, cfg.rms_norm_eps, backend.clone())?;

        // Pre-compute timestep embeddings for all inference steps (avoids ~12 kernels/step)
        let pre_t_embeddings: Vec<Tensor> = timesteps
            .iter()
            .map(|&t| {
                // The doubled timestep (batch=2 for CFG) — same value repeated
                let t_tensor = Tensor::new(&[t as f32, t as f32], vb.device())
                    .unwrap()
                    .to_dtype(vb.dtype())
                    .unwrap();
                t_embedder.forward(&t_tensor).unwrap()
            })
            .collect();

        Ok(Self {
            t_embedder,
            noisy_images_proj_weight,
            cond_proj_weight,
            layers,
            final_layer,
            pre_t_embeddings,
            backend,
        })
    }

    /// Single denoising prediction step.
    /// x: (batch, latent_dim) noisy acoustic latents
    /// t: (batch,) timestep scalars
    /// condition: (batch, hidden_size) LLM hidden states
    pub fn forward(&self, x: &Tensor, t: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let mut h = self.backend.linear_forward(x, &self.noisy_images_proj_weight, None)?;
        let t_emb = self.t_embedder.forward(t)?;
        let c_proj = self.backend.linear_forward(condition, &self.cond_proj_weight, None)?;
        let c = (c_proj + t_emb)?;

        for layer in &self.layers {
            h = layer.forward(&h, &c)?;
        }

        self.final_layer.forward(&h, &c)
    }

    /// Project condition once per frame (constant across all diffusion steps).
    /// Returns the projected condition to be reused.
    pub fn project_condition(&self, condition: &Tensor) -> Result<Tensor> {
        self.backend.linear_forward(condition, &self.cond_proj_weight, None)
    }

    /// Optimized forward using pre-computed timestep embedding and projected condition.
    /// Caches silu(cond) across blocks (saves 4 silu kernels per step).
    pub fn forward_fast(
        &self,
        x: &Tensor,
        step_idx: usize,
        cond_proj: &Tensor,
    ) -> Result<Tensor> {
        let mut h = self.backend.linear_forward(x, &self.noisy_images_proj_weight, None)?;

        // Use pre-computed timestep embedding (saves ~12 kernel launches)
        let t_emb = &self.pre_t_embeddings[step_idx];
        let c = (cond_proj + t_emb)?;

        // Compute silu(c) once, reuse across all blocks (saves 4 silu kernels)
        let silu_c = self.backend.silu(&c)?;

        for layer in &self.layers {
            h = layer.forward_with_silu_cond(&h, &silu_c)?;
        }

        self.final_layer.forward_with_silu_cond(&h, &silu_c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
        use rand::{Rng, SeedableRng};
        let numel: usize = shape.iter().product();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn make_prediction_head_vb() -> VarBuilder<'static> {
        let h = 32; // small for testing
        let latent = 8;
        let intermediate = 96; // h * 3

        let mut map: HashMap<String, Tensor> = HashMap::new();

        // TimestepEmbedder
        map.insert("t_embedder.mlp.0.weight".into(), make_tensor(&[h, 256], 1));
        map.insert("t_embedder.mlp.2.weight".into(), make_tensor(&[h, h], 2));

        // Projections
        map.insert("noisy_images_proj.weight".into(), make_tensor(&[h, latent], 3));
        map.insert("cond_proj.weight".into(), make_tensor(&[h, h], 4));

        // 2 layers (small for test)
        for i in 0..2 {
            let p = format!("layers.{i}");
            map.insert(format!("{p}.norm.weight"), make_tensor(&[h], 10 + i as u64));
            map.insert(format!("{p}.adaLN_modulation.1.weight"), make_tensor(&[3 * h, h], 20 + i as u64));
            map.insert(format!("{p}.ffn.gate_proj.weight"), make_tensor(&[intermediate, h], 30 + i as u64));
            map.insert(format!("{p}.ffn.up_proj.weight"), make_tensor(&[intermediate, h], 40 + i as u64));
            map.insert(format!("{p}.ffn.down_proj.weight"), make_tensor(&[h, intermediate], 50 + i as u64));
        }

        // Final layer
        map.insert("final_layer.adaLN_modulation.1.weight".into(), make_tensor(&[2 * h, h], 60));
        map.insert("final_layer.linear.weight".into(), make_tensor(&[latent, h], 61));

        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn test_prediction_head_forward_shape() {
        let cfg = super::super::config::DiffusionHeadConfig {
            ddpm_num_inference_steps: 20,
            ddpm_num_steps: 1000,
            head_layers: 2,
            hidden_size: 32,
            latent_size: 8,
            head_ffn_ratio: 3.0,
            prediction_type: "v_prediction".into(),
            rms_norm_eps: 1e-5,
            ddpm_beta_schedule: "cosine".into(),
        };

        let vb = make_prediction_head_vb();
        let sched = super::super::ddpm::DpmSolverPP::new_cosine(1000, 20);
        let backend = crate::backends::create_backend(&Device::Cpu);
        let head = PredictionHead::load(vb, &cfg, sched.timesteps(), backend).unwrap();

        let x = make_tensor(&[2, 8], 100);       // (batch=2, latent=8)
        let t = Tensor::new(&[0.5f32, 0.3], &Device::Cpu).unwrap(); // (batch=2,)
        let cond = make_tensor(&[2, 32], 101);    // (batch=2, hidden=32)

        let out = head.forward(&x, &t, &cond).unwrap();
        assert_eq!(out.dims(), &[2, 8]); // (batch, latent)
    }

    #[test]
    fn test_timestep_embedder_shape() {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("mlp.0.weight".into(), make_tensor(&[32, 256], 1));
        map.insert("mlp.2.weight".into(), make_tensor(&[32, 32], 2));
        let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);

        let backend = crate::backends::create_backend(&Device::Cpu);
        let emb = TimestepEmbedder::load(vb, 32, backend).unwrap();
        let t = Tensor::new(&[0.5f32, 0.8], &Device::Cpu).unwrap();
        let out = emb.forward(&t).unwrap();
        assert_eq!(out.dims(), &[2, 32]);
    }
}
