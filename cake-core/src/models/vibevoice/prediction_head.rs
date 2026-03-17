//! VibeVoice diffusion prediction head.
//!
//! A 4-layer DiT-style diffusion head with AdaLN modulation and SwiGLU FFN.
//! Takes noisy acoustic latents (64-dim) + LLM hidden states (896-dim) +
//! timestep → predicts noise (v-prediction).

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias as linear, Linear, RmsNorm, VarBuilder};

/// Sinusoidal timestep embedding → MLP.
#[derive(Debug, Clone)]
pub struct TimestepEmbedder {
    mlp_0: Linear,
    mlp_2: Linear,
}

impl TimestepEmbedder {
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let mlp_0 = linear(256, hidden_size, vb.pp("mlp").pp("0"))?;
        let mlp_2 = linear(hidden_size, hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self { mlp_0, mlp_2 })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // Sinusoidal embedding: t → (batch, 256)
        let half_dim = 128;
        let emb = {
            let freq = Tensor::arange(0u32, half_dim as u32, t.device())?
                .to_dtype(DType::F32)?;
            let freq = (freq * (-f64::ln(10000.0) / half_dim as f64))?.exp()?;
            let t_f32 = t.to_dtype(DType::F32)?;
            let args = t_f32.unsqueeze(1)?.broadcast_mul(&freq.unsqueeze(0)?)?;
            Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(t.dtype())?
        };
        // MLP: 256 → hidden → hidden with SiLU
        let h = candle_nn::ops::silu(&self.mlp_0.forward(&emb)?)?;
        self.mlp_2.forward(&h)
    }
}

/// SwiGLU feed-forward network.
#[derive(Debug, Clone)]
struct FeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        let gate_proj = linear(hidden, intermediate, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden, intermediate, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate, hidden, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Single diffusion block with AdaLN modulation.
#[derive(Debug, Clone)]
struct DiffusionBlock {
    norm: RmsNorm,
    ffn: FeedForward,
    ada_ln: Linear,
}

impl DiffusionBlock {
    fn load(vb: VarBuilder, hidden: usize, intermediate: usize, eps: f64) -> Result<Self> {
        let norm = candle_nn::rms_norm(hidden, eps, vb.pp("norm"))?;
        let ffn = FeedForward::load(vb.pp("ffn"), hidden, intermediate)?;
        // adaLN produces 3 * hidden = (scale, shift, gate) for modulation
        let ada_ln = linear(hidden, 3 * hidden, vb.pp("adaLN_modulation").pp("1"))?;
        Ok(Self { norm, ffn, ada_ln })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = candle_nn::ops::silu(cond)
            .and_then(|c| self.ada_ln.forward(&c))?;
        let chunks = modulation.chunk(3, D::Minus1)?;
        let (scale, shift, gate) = (&chunks[0], &chunks[1], &chunks[2]);

        // AdaLN: norm(x) * (1 + scale) + shift
        let h = self.norm.forward(x)?;
        let h = h.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?;
        let h = self.ffn.forward(&h)?;
        // Gate
        x + h.broadcast_mul(gate)?
    }
}

/// Final output layer with AdaLN and linear projection.
#[derive(Debug, Clone)]
struct FinalLayer {
    ada_ln: Linear,
    linear: Linear,
}

impl FinalLayer {
    fn load(vb: VarBuilder, hidden: usize, latent: usize) -> Result<Self> {
        let ada_ln = linear(hidden, 2 * hidden, vb.pp("adaLN_modulation").pp("1"))?;
        let linear_proj = linear(hidden, latent, vb.pp("linear"))?;
        Ok(Self { ada_ln, linear: linear_proj })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let modulation = candle_nn::ops::silu(cond)
            .and_then(|c| self.ada_ln.forward(&c))?;
        let chunks = modulation.chunk(2, D::Minus1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]);
        let h = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?;
        self.linear.forward(&h)
    }
}

/// Complete prediction head for diffusion.
#[derive(Debug, Clone)]
pub struct PredictionHead {
    t_embedder: TimestepEmbedder,
    noisy_images_proj: Linear,
    cond_proj: Linear,
    layers: Vec<DiffusionBlock>,
    final_layer: FinalLayer,
}

impl PredictionHead {
    pub fn load(vb: VarBuilder, cfg: &super::config::DiffusionHeadConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let latent = cfg.latent_size;
        let intermediate = (h as f64 * cfg.head_ffn_ratio) as usize;

        let t_embedder = TimestepEmbedder::load(vb.pp("t_embedder"), h)?;
        let noisy_images_proj = linear(latent, h, vb.pp("noisy_images_proj"))?;
        let cond_proj = linear(h, h, vb.pp("cond_proj"))?;

        let mut layers = Vec::with_capacity(cfg.head_layers);
        for i in 0..cfg.head_layers {
            layers.push(DiffusionBlock::load(
                vb.pp("layers").pp(i),
                h,
                intermediate,
                cfg.rms_norm_eps,
            )?);
        }

        let final_layer = FinalLayer::load(vb.pp("final_layer"), h, latent)?;

        Ok(Self {
            t_embedder,
            noisy_images_proj,
            cond_proj,
            layers,
            final_layer,
        })
    }

    /// Single denoising prediction step.
    /// x: (batch, latent_dim) noisy acoustic latents
    /// t: (batch,) timestep scalars
    /// condition: (batch, hidden_size) LLM hidden states
    pub fn forward(&self, x: &Tensor, t: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let t_emb = self.t_embedder.forward(t)?;
        let x_proj = self.noisy_images_proj.forward(x)?;
        let c_proj = self.cond_proj.forward(condition)?;

        // Fuse: h = x_proj + c_proj, cond = c_proj + t_emb
        let mut h = (x_proj + &c_proj)?;
        let cond = (c_proj + t_emb)?;

        for layer in &self.layers {
            h = layer.forward(&h, &cond)?;
        }

        self.final_layer.forward(&h, &cond)
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
        let head = PredictionHead::load(vb, &cfg).unwrap();

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

        let emb = TimestepEmbedder::load(vb, 32).unwrap();
        let t = Tensor::new(&[0.5f32, 0.8], &Device::Cpu).unwrap();
        let out = emb.forward(&t).unwrap();
        assert_eq!(out.dims(), &[2, 32]);
    }
}
