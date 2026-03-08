//! Adaptive Layer Norm (AdaLN) for LTX-2.
//!
//! Timestep → sinusoidal embedding → SiLU → Linear → per-block modulation params.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Sinusoidal timestep embedding (PixArt-Alpha style).
#[derive(Debug)]
struct Timesteps {
    dim: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
}

impl Timesteps {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            flip_sin_to_cos: true,
            downscale_freq_shift: 0.0,
        }
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let half_dim = self.dim / 2;

        // exp(-log(10000) * i / half_dim) for i in 0..half_dim
        let exponent: Vec<f32> = (0..half_dim)
            .map(|i| {
                let freq = -(10000.0f64.ln()) * (i as f64)
                    / ((half_dim as f64) - self.downscale_freq_shift);
                freq.exp() as f32
            })
            .collect();

        let freqs = Tensor::new(exponent, device)?; // [half_dim]
        let t = t.to_dtype(DType::F32)?;

        // t: [B] or [B, T], freqs: [half_dim]
        // Outer product: [B, half_dim]
        let args = if t.rank() == 1 {
            t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?
        } else {
            // [B, T] -> [B, T, half_dim]
            t.unsqueeze(t.rank())?.broadcast_mul(
                &freqs
                    .reshape(std::iter::repeat(1).take(t.rank()).chain([half_dim]).collect::<Vec<_>>())?,
            )?
        };

        let (cos, sin) = if self.flip_sin_to_cos {
            (args.cos()?, args.sin()?)
        } else {
            (args.sin()?, args.cos()?)
        };

        Tensor::cat(&[cos, sin], args.rank() - 1)
    }
}

/// Two-layer MLP for timestep projection.
#[derive(Debug)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = candle_nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = candle_nn::ops::silu(&x)?;
        self.linear_2.forward(&x)
    }
}

/// PixArt-Alpha combined timestep + size embeddings.
#[derive(Debug)]
struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    timestep: Timesteps,
    time_proj: TimestepEmbedding,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep = Timesteps::new(256);
        let time_proj = TimestepEmbedding::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self {
            timestep,
            time_proj,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_emb = self.timestep.forward(t)?;
        // Timesteps produces F32 (sinusoidal); convert to weight dtype before Linear
        let weight_dtype = self.time_proj.linear_1.weight().dtype();
        let t_emb = t_emb.to_dtype(weight_dtype)?;
        self.time_proj.forward(&t_emb)
    }
}

/// AdaLayerNormSingle: timestep → embedding → SiLU → Linear → per-block params.
///
/// Returns `(modulation_params, embedded_timestep)`.
/// `modulation_params` shape: `[B, embedding_coefficient * dim]`.
#[derive(Debug)]
pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: Linear,
}

impl AdaLayerNormSingle {
    pub fn new(
        embedding_dim: usize,
        embedding_coefficient: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(embedding_dim, vb.pp("emb"))?;
        let linear = candle_nn::linear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            vb.pp("linear"),
        )?;
        Ok(Self { emb, linear })
    }

    /// Returns `(modulation_params, raw_embedded_timestep)`.
    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded = self.emb.forward(timestep)?;
        let params = candle_nn::ops::silu(&embedded)?;
        let params = self.linear.forward(&params)?;
        Ok((params, embedded))
    }
}

/// Caption/text projection: Linear → GELU → Linear.
#[derive(Debug)]
pub struct TextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl TextProjection {
    pub fn new(
        caption_channels: usize,
        inner_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_1 = candle_nn::linear(caption_channels, inner_dim, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear(inner_dim, inner_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl Module for TextProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = x.gelu()?;
        self.linear_2.forward(&x)
    }
}
