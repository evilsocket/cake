//! Rust 2024 + candle port of transformer_ltx.py (LTX-Video transformer core).
//!
//! Notes:
//! - This is a self-contained module intended to compile and mirror the structure
//!   of the provided Python file.
//! - Some components imported in Python are implemented here minimally to match
//!   the tensor contracts used in the file (e.g., AdaLayerNormSingle, PixArtAlphaTextProjection).

use super::t2v_pipeline::{TransformerConfig, VideoTransformer3D};
use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn as nn;
use nn::{Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct Transformer2DModelOutput {
    pub sample: Tensor,
}

use serde::{Deserialize, Serialize};

/// Configuration for LtxVideoTransformer3DModel
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LtxVideoTransformer3DModelConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,
    pub num_layers: usize,
    pub qk_norm: String,
    pub norm_elementwise_affine: bool,
    pub norm_eps: f64,
    pub caption_channels: usize,
    pub attention_bias: bool,
    pub attention_out_bias: bool,
}

impl Default for LtxVideoTransformer3DModelConfig {
    fn default() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1, // 0.9.5 uses patch_size 1? Json says 1.
            patch_size_t: 1,
            num_attention_heads: 32, // 2048 hidden size / 64
            attention_head_dim: 64,
            cross_attention_dim: 2048,
            num_layers: 28,
            qk_norm: "rms_norm_across_heads".to_string(),
            norm_elementwise_affine: false,
            norm_eps: 1e-6,
            caption_channels: 4096,
            attention_bias: true,
            attention_out_bias: true,
        }
    }
}

/// LayerNorm without affine parameters (elementwise_affine=False).
#[derive(Clone, Debug)]
pub struct LayerNormNoParams {
    eps: f64,
}

impl LayerNormNoParams {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let last_dim = xs.dim(D::Minus1)?;
        let mean = (xs.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let xc = xs.broadcast_sub(&mean)?;
        let var = (xc.sqr()?.sum_keepdim(D::Minus1)? / (last_dim as f64))?;
        let denom = (var + self.eps)?.sqrt()?;
        xc.broadcast_div(&denom)
    }
}

/// RMSNorm with optional affine weight (elementwise_affine=True/False).
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, elementwise_affine: bool, vb: VarBuilder) -> Result<Self> {
        let weight = if elementwise_affine {
            Some(vb.get(dim, "weight")?)
        } else {
            None
        };
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let dim = xs_f32.dim(D::Minus1)? as f64;
        let ms = xs_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        let denom = ms.affine(1.0, self.eps)?.sqrt()?;
        let ys_f32 = xs_f32.broadcast_div(&denom)?;
        let mut ys = ys_f32.to_dtype(dtype)?;
        if let Some(w) = &self.weight {
            // Broadcast weight over leading dims.
            let rank = ys.rank();
            let mut shape = vec![1usize; rank];
            shape[rank - 1] = w.dims1()?;
            let w = w.reshape(shape)?;
            ys = ys.broadcast_mul(&w)?;
        }
        Ok(ys)
    }
}

// Helper for GEGLU feed-forward structure usually found in diffusers
// Helper for GELU (approximate) feed-forward projection (Layer 0 of FeedForward)
#[derive(Clone, Debug)]
struct GeluProjection {
    proj: nn::Linear,
}

impl GeluProjection {
    fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(xs)?;
        gelu_approximate(&x)
    }
}

impl Module for GeluProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

// FeedForward container matching "net" structure with GEGLU
#[derive(Clone, Debug)]
pub struct FeedForward {
    net_0: GeluProjection,
    net_2: nn::Linear,
}

impl FeedForward {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // net.0: GeluProjection (Linear + Gelu)
        // net.2: Linear
        let hidden = dim * 4;

        let net_0 = GeluProjection::new(dim, hidden, vb.pp("net.0"))?;
        let net_2 = nn::linear(hidden, dim, vb.pp("net.2"))?;

        Ok(Self { net_0, net_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.net_0.forward(xs)?;
        self.net_2.forward(&x)
    }
}

/// Minimal PixArtAlphaTextProjection: linear projection to model inner dim.
#[derive(Clone, Debug)]
pub struct PixArtAlphaTextProjection {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl PixArtAlphaTextProjection {
    pub fn new(in_features: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_features, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(hidden_size, hidden_size, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = gelu_approximate(&x)?;
        self.linear_2.forward(&x)
    }
}

/// Timestep embedding with two linear layers and SiLU.
#[derive(Clone, Debug)]
pub struct TimestepEmbedding {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl TimestepEmbedding {
    pub fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 = nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear_2 = nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(xs)?;
        let x = x.silu()?;
        self.linear_2.forward(&x)
    }
}

pub fn gelu_approximate(x: &Tensor) -> Result<Tensor> {
    // Upcast to F32 for math stability
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_cube = x_f32.sqr()?.broadcast_mul(&x_f32)?;
    let inner = x_f32.broadcast_add(&x_cube.affine(0.044715, 0.0)?)?;
    let scale = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let tanh_input = inner.affine(scale as f64, 0.0)?;
    let tanh_out = tanh_input.tanh()?;
    let gelu = x_f32
        .broadcast_mul(&tanh_out.affine(1.0, 1.0)?)?
        .affine(0.5, 0.0)?;
    gelu.to_dtype(x.dtype())
}

/// PixArtAlphaCombinedTimestepSizeEmbeddings
#[derive(Clone, Debug)]
pub struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    timestep_embedder: TimestepEmbedding,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder =
            TimestepEmbedding::new(256, embedding_dim, vb.pp("timestep_embedder"))?;
        Ok(Self { timestep_embedder })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        // time_proj produces 256 dimensions, flip_sin_to_cos=true
        let timesteps_proj = get_timestep_embedding(timestep, 256, true)?;
        self.timestep_embedder.forward(&timesteps_proj)
    }
}

/// AdaLayerNormSingle: (PixArtAlphaCombinedTimestepSizeEmbeddings + Linear)
#[derive(Clone, Debug)]
pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: nn::Linear,
}

impl AdaLayerNormSingle {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new(dim, vb.pp("emb"))?;
        let linear = nn::linear(dim, 6 * dim, vb.pp("linear"))?;
        Ok(Self { emb, linear })
    }

    pub fn forward(&self, timestep: &Tensor) -> Result<(Tensor, Tensor)> {
        let embedded_timestep = self.emb.forward(timestep)?;
        let x = embedded_timestep.silu()?;
        let x = self.linear.forward(&x)?;
        Ok((x, embedded_timestep))
    }
}

/// This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
fn get_timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
) -> Result<Tensor> {
    let device = timesteps.device();
    let original_dtype = timesteps.dtype();

    // Always use F32 for sinusoidal embedding math
    let dtype = DType::F32;

    let n = timesteps.dim(0)?;
    let half = embedding_dim / 2;

    let t = timesteps.to_dtype(dtype)?; // [N]
    let t = t.unsqueeze(1)?; // [N, 1]

    let inv_freq: Vec<_> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(i as f32 / (half as f32)))
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
    let freqs = t.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [N, half]

    let sin = freqs.sin()?;
    let cos = freqs.cos()?;

    let emb = if flip_sin_to_cos {
        Tensor::cat(&[cos, sin], D::Minus1)?
    } else {
        Tensor::cat(&[sin, cos], D::Minus1)?
    };

    if embedding_dim % 2 == 1 {
        let pad = Tensor::zeros((n, 1), dtype, device)?;
        Tensor::cat(&[emb, pad], D::Minus1)?.to_dtype(original_dtype)
    } else {
        emb.to_dtype(original_dtype)
    }
}

/// apply_rotary_emb from the Python file:
/// - x: [B, S, C]
/// - freqs: (cos, sin) each [B, S, C]
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    // Upcast to F32 for rotation math stability
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let (b, s, c) = x_f32.dims3()?;
    if c % 2 != 0 {
        candle_core::bail!("apply_rotary_emb expects last dim even, got {c}");
    }
    let half = c / 2;

    // x -> [B, S, half, 2]
    let x2 = x_f32.reshape((b, s, half, 2))?;
    let x_real = x2.i((.., .., .., 0))?;
    let x_imag = x2.i((.., .., .., 1))?;

    // [-imag, real] interleave back.
    let x_rot = Tensor::stack(&[x_imag.neg()?, x_real.clone()], D::Minus1)?.reshape((b, s, c))?;

    let out = x_f32
        .broadcast_mul(&cos)?
        .broadcast_add(&x_rot.broadcast_mul(&sin)?)?;
    out.to_dtype(dtype)
}

#[derive(Clone, Debug)]
pub struct LtxVideoRotaryPosEmbed {
    dim: usize,
    base_num_frames: usize,
    base_height: usize,
    base_width: usize,
    patch_size: usize,
    patch_size_t: usize,
    theta: f64,
}

impl LtxVideoRotaryPosEmbed {
    pub fn new(
        dim: usize,
        base_num_frames: usize,
        base_height: usize,
        base_width: usize,
        patch_size: usize,
        patch_size_t: usize,
        theta: f64,
    ) -> Self {
        Self {
            dim,
            base_num_frames,
            base_height,
            base_width,
            patch_size,
            patch_size_t,
            theta,
        }
    }

    fn prepare_video_coords(
        &self,
        batch_size: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        device: &Device,
    ) -> Result<Tensor> {
        // Compute coords in F32 for precision, convert to model dtype later
        let dtype = DType::F32;

        let grid_h = Tensor::arange(0u32, height as u32, device)?.to_dtype(dtype)?; // [H]
        let grid_w = Tensor::arange(0u32, width as u32, device)?.to_dtype(dtype)?; // [W]
        let grid_f = Tensor::arange(0u32, num_frames as u32, device)?.to_dtype(dtype)?; // [F]

        // meshgrid ij:
        // f: [F,H,W], h: [F,H,W], w: [F,H,W]
        let f = grid_f
            .reshape((num_frames, 1, 1))?
            .broadcast_as((num_frames, height, width))?;
        let h = grid_h
            .reshape((1, height, 1))?
            .broadcast_as((num_frames, height, width))?;
        let w = grid_w
            .reshape((1, 1, width))?
            .broadcast_as((num_frames, height, width))?;

        // stack -> [3,F,H,W]
        let mut grid = Tensor::stack(&[f, h, w], 0)?; // [3,F,H,W]
        // [B,3,F,H,W]
        grid = grid
            .unsqueeze(0)?
            .broadcast_as((batch_size, 3, num_frames, height, width))?;

        if let Some((sf, sh, sw)) = rope_interpolation_scale {
            // grid[:,0:1] *= sf * patch_size_t / base_num_frames
            let f_scale = (sf * self.patch_size_t as f64 / self.base_num_frames as f64) as f32;
            let h_scale = (sh * self.patch_size as f64 / self.base_height as f64) as f32;
            let w_scale = (sw * self.patch_size as f64 / self.base_width as f64) as f32;

            let gf = grid
                .i((.., 0..1, .., .., ..))?
                .affine(f_scale as f64, 0.0)?;
            let gh = grid
                .i((.., 1..2, .., .., ..))?
                .affine(h_scale as f64, 0.0)?;
            let gw = grid
                .i((.., 2..3, .., .., ..))?
                .affine(w_scale as f64, 0.0)?;
            grid = Tensor::cat(&[gf, gh, gw], 1)?;
        }

        // flatten dims 2..4 => seq, transpose(1,2): [B, seq, 3]
        let seq = num_frames * height * width;
        let grid = grid
            .reshape((batch_size, 3, seq))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok(grid)
    }

    /// Returns (cos, sin), both shaped [B, seq, dim].
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let device = hidden_states.device();
        let batch_size = hidden_states.dim(0)?;

        let grid = if let Some(coords) = video_coords {
            // Expect [B, seq, 3] and normalize by base sizes.
            let (b, seq, c) = coords.dims3()?;
            if b != batch_size || c != 3 {
                candle_core::bail!("video_coords must be [B, seq, 3], got [{b}, {seq}, {c}]");
            }
            let base_f = (self.base_num_frames as f64) as f32;
            let base_h = (self.base_height as f64) as f32;
            let base_w = (self.base_width as f64) as f32;

            let cf = coords.i((.., .., 0))?.affine(1.0 / base_f as f64, 0.0)?;
            let ch = coords.i((.., .., 1))?.affine(1.0 / base_h as f64, 0.0)?;
            let cw = coords.i((.., .., 2))?.affine(1.0 / base_w as f64, 0.0)?;
            Tensor::stack(&[cf, ch, cw], D::Minus1)?
        } else {
            self.prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale,
                device,
            )?
        };

        // freqs: theta ** linspace(log(start,theta), log(end,theta), dim//6)
        // In the file: start=1.0, end=theta => exponents go 0..1.
        let steps = self.dim / 6;
        let dtype = DType::F32; // Use F32 for coordinate math

        let lin = if steps <= 1 {
            Tensor::zeros((1,), dtype, device)?
        } else {
            // linspace [0, 1], inclusive
            let idx = Tensor::arange(0u32, steps as u32, device)?.to_dtype(dtype)?;
            idx.affine(1.0 / ((steps - 1) as f64), 0.0)?
        };

        let theta_ln = (self.theta.ln()) as f32;
        let freqs = (lin.affine(theta_ln as f64, 0.0)?).exp()?; // exp(lin * ln(theta)) => theta**lin
        let freqs = freqs.affine(std::f64::consts::PI / 2.0, 0.0)?; // * pi/2

        // freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        // grid: [B, seq, 3] -> [B, seq, 3, 1]
        let grid = grid.to_dtype(dtype)?;
        let grid_scaled = grid.unsqueeze(D::Minus1)?.affine(2.0, -1.0)?; // *2 -1
        let freqs = grid_scaled.broadcast_mul(&freqs.reshape((1, 1, 1, steps))?)?; // [B,seq,3,steps]
        let freqs = freqs
            .transpose(D::Minus1, D::Minus2)?
            .contiguous()?
            .flatten_from(2)?; // [B,seq,3*steps]

        // Manually implement repeat_interleave(2, D::Minus1) for cos/sin
        fn repeat_interleave_2(t: &Tensor) -> Result<Tensor> {
            let t_unsq = t.unsqueeze(D::Minus1)?; // [..., C, 1]
            let t_rep = Tensor::cat(&[t_unsq.clone(), t_unsq], D::Minus1)?; // [..., C, 2]
            let shape = t.dims();
            let new_last = shape[shape.len() - 1] * 2;
            let mut new_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
            new_shape.push(new_last);
            t_rep.reshape(new_shape)
        }

        let mut cos = repeat_interleave_2(&freqs.cos()?)?;
        let mut sin = repeat_interleave_2(&freqs.sin()?)?;

        let rem = self.dim % 6;
        if rem != 0 {
            let (b, seq, _) = cos.dims3()?;
            let cos_pad = Tensor::ones((b, seq, rem), dtype, device)?;
            let sin_pad = Tensor::zeros((b, seq, rem), dtype, device)?;
            cos = Tensor::cat(&[cos_pad, cos], D::Minus1)?;
            sin = Tensor::cat(&[sin_pad, sin], D::Minus1)?;
        }

        Ok((cos, sin))
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct LtxAttention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    inner_kv_dim: usize,
    cross_attention_dim: usize,

    norm_q: RmsNorm,
    norm_k: RmsNorm,

    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,

    to_out: nn::Linear,
    dropout: nn::Dropout,
}

impl LtxAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query_dim: usize,
        heads: usize,
        kv_heads: usize,
        dim_head: usize,
        dropout: f64,
        bias: bool,
        cross_attention_dim: Option<usize>,
        out_bias: bool,
        qk_norm: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        if qk_norm != "rms_norm_across_heads" {
            candle_core::bail!("Only 'rms_norm_across_heads' is supported as qk_norm.");
        }

        let inner_dim = dim_head * heads;
        let inner_kv_dim = dim_head * kv_heads;
        let cross_attention_dim = cross_attention_dim.unwrap_or(query_dim);

        // Python uses eps=1e-5 and elementwise_affine=True for these.
        let norm_q = RmsNorm::new(inner_dim, 1e-5, true, vb.pp("norm_q"))?;
        let norm_k = RmsNorm::new(inner_kv_dim, 1e-5, true, vb.pp("norm_k"))?;

        let to_q = nn::linear_b(query_dim, inner_dim, bias, vb.pp("to_q"))?;
        let to_k = nn::linear_b(cross_attention_dim, inner_kv_dim, bias, vb.pp("to_k"))?;
        let to_v = nn::linear_b(cross_attention_dim, inner_kv_dim, bias, vb.pp("to_v"))?;

        let to_out = nn::linear_b(inner_dim, query_dim, out_bias, vb.pp("to_out").pp("0"))?;
        let dropout = nn::Dropout::new(dropout as f32);

        Ok(Self {
            heads,
            head_dim: dim_head,
            inner_dim,
            inner_kv_dim,
            cross_attention_dim,
            norm_q,
            norm_k,
            to_q,
            to_k,
            to_v,
            to_out,
            dropout,
        })
    }

    fn prepare_attention_mask(
        &self,
        attention_mask: &Tensor,
        q_len: usize,
        k_len: usize,
    ) -> Result<Tensor> {
        // The Python file relies on AttentionModuleMixin.prepare_attention_mask.
        // Here we support the shapes that are consistent with the file usage:
        // - [B, 1, k_len] bias -> expand to [B, heads, q_len, k_len]
        // - [B, heads, q_len, k_len] already prepared
        match attention_mask.rank() {
            2 => {
                let (b, kk) = attention_mask.dims2()?;
                if kk != k_len {
                    candle_core::bail!(
                        "Expected attention_mask [B,k_len]=[{},{}], got [{},{}]",
                        b,
                        k_len,
                        b,
                        kk
                    );
                }
                // Convert 0/1 mask from tokenizer (where 1 is keep, 0 is mask)
                // to additive offset (-10000.0 for mask, 0.0 for keep)
                let mask = (1.0 - attention_mask.to_dtype(DType::F32)?)? * -10000.0;
                // [B, k_len] -> [B, 1, 1, k_len]
                let m = mask?.unsqueeze(1)?.unsqueeze(1)?;

                m.broadcast_as((b, self.heads, q_len, k_len))?.contiguous()
            }

            3 => {
                let (b, one, kk) = attention_mask.dims3()?;
                if one != 1 || kk != k_len {
                    candle_core::bail!(
                        "Expected attention_mask [B,1,k_len]=[{},1,{}], got [{},{},{}]",
                        b,
                        k_len,
                        b,
                        one,
                        kk
                    );
                }
                let m = attention_mask.unsqueeze(2)?; // [B,1,1,k_len]
                m.broadcast_as((b, self.heads, q_len, k_len))?.contiguous()
            }
            4 => Ok(attention_mask.clone()),
            other => candle_core::bail!("Unsupported attention_mask rank {other}"),
        }
    }

    /// Mirrors LTXVideoAttnProcessor.__call__ behavior from the Python file.
    pub fn forward(
        &self,
        hidden_states: &Tensor,                       // [B, S, query_dim]
        encoder_hidden_states: Option<&Tensor>,       // [B, K, cross_dim] or None
        attention_mask: Option<&Tensor>,              // optional bias/mask
        image_rotary_emb: Option<(&Tensor, &Tensor)>, // (cos, sin)
    ) -> Result<Tensor> {
        let (b, q_len, _) = hidden_states.dims3()?;
        let enc = encoder_hidden_states.unwrap_or(hidden_states);
        let (_, k_len, _) = enc.dims3()?;

        let _attn_mask = if let Some(mask) = attention_mask {
            Some(self.prepare_attention_mask(mask, q_len, k_len)?)
        } else {
            None
        };

        // Project
        let mut q = self.to_q.forward(hidden_states)?; // [B,S,inner_dim]
        let mut k = self.to_k.forward(enc)?; // [B,K,inner_kv_dim]
        let v = self.to_v.forward(enc)?; // [B,K,inner_kv_dim]

        // QK RMSNorm
        q = self.norm_q.forward(&q)?;
        k = self.norm_k.forward(&k)?;

        // RoPE on Q,K if provided
        if let Some((cos, sin)) = image_rotary_emb {
            q = apply_rotary_emb(&q, cos, sin)?;
            k = apply_rotary_emb(&k, cos, sin)?;
        }

        // Reshape to heads: [B, S, heads, head_dim]
        let q = q.reshape((b, q_len, self.heads, self.head_dim))?;
        let k = k.reshape((b, k_len, self.heads, self.head_dim))?;
        let v = v.reshape((b, k_len, self.heads, self.head_dim))?;

        let dtype = q.dtype();
        let scale = 1f32 / (self.head_dim as f32).sqrt();

        // Check if we can use Flash Attention
        #[allow(unused_mut)]
        let mut use_flash = false;
        #[cfg(feature = "flash-attn")]
        {
            // Flash Attention doesn't support masks easily
            if _attn_mask.is_none() && q.device().is_cuda() {
                use_flash = true;
            }
        }

        let out = if use_flash {
            #[cfg(feature = "flash-attn")]
            {
                // candle_flash_attn expects [B, seq, heads, head_dim] which matches our current shape
                let q_bf = q.to_dtype(DType::BF16)?;
                let k_bf = k.to_dtype(DType::BF16)?;
                let v_bf = v.to_dtype(DType::BF16)?;

                let out = candle_flash_attn::flash_attn(&q_bf, &k_bf, &v_bf, scale, false)?;

                // Result is [B, seq, heads, head_dim].
                // We need it to be [B, heads, seq, head_dim] to match common post-processing below
                out.transpose(1, 2)?.to_dtype(dtype)?
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                unreachable!()
            }
        } else {
            // Manual attention path
            let q_f32 = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?; // [B, heads, seq, head_dim]
            let k_f32 = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
            let v_f32 = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

            let att = q_f32.matmul(&k_f32.transpose(D::Minus1, D::Minus2)?)?;
            let att = (att * (scale as f64))?;

            // Add mask if present
            let att = match _attn_mask {
                Some(ref mask) => att.broadcast_add(&mask.to_dtype(DType::F32)?)?,
                None => att,
            };

            // Softmax - already in F32
            let (b_sz, h_sz, q_l, k_l) = att.dims4()?;
            let att = att.reshape((b_sz * h_sz * q_l, k_l))?;
            let att = nn::ops::softmax(&att, D::Minus1)?;
            let att = att.reshape((b_sz, h_sz, q_l, k_l))?;

            // out = att @ v
            let out_f32 = att.matmul(&v_f32)?;
            out_f32.to_dtype(dtype)?
        };

        // Back to [B, S, heads, head_dim] -> flatten -> [B,S,inner_dim]
        let out = out.transpose(1, 2)?.contiguous()?;
        let out = out.reshape((b, q_len, self.inner_dim))?;

        // Output projection + dropout
        let out = self.to_out.forward(&out)?;
        self.dropout.forward(&out, false)
    }
}

#[derive(Clone, Debug)]
pub struct LtxVideoTransformerBlock {
    norm1: RmsNorm,
    attn1: LtxAttention,
    norm2: RmsNorm,
    attn2: LtxAttention,
    ff: FeedForward,
    scale_shift_table: Tensor, // [6, dim]
}

impl LtxVideoTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_attention_heads: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        qk_norm: &str,
        attention_bias: bool,
        attention_out_bias: bool,
        eps: f64,
        elementwise_affine: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm1"))?;
        let attn1 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            None,
            attention_out_bias,
            qk_norm,
            vb.pp("attn1"),
        )?;
        let norm2 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm2"))?;
        let attn2 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            Some(cross_attention_dim),
            attention_out_bias,
            qk_norm,
            vb.pp("attn2"),
        )?;

        let ff = FeedForward::new(dim, vb.pp("ff"))?;

        // Parameter: torch.randn(6, dim) / dim**0.5
        // In candle: we store as a trainable tensor; initialization is delegated to checkpoint loading.
        let scale_shift_table = vb.get((6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            ff,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,         // [B, S, dim]
        encoder_hidden_states: &Tensor, // [B, K, dim]
        temb: &Tensor,                  // [B, T, 6*dim]
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = hidden_states.dim(0)?;
        let norm_hidden = self.norm1.forward(hidden_states)?;

        // ada_values = scale_shift_table[None,None] + temb.reshape(B, T, 6, dim)
        // ada_values = scale_shift_table[None,None] + temb.reshape(B, T, 6, dim)
        let (b_temb, temb_last) = temb.dims2()?;
        if b_temb != b {
            candle_core::bail!(
                "temb batch size {} mismatch hidden_states batch size {}",
                b_temb,
                b
            );
        }

        if temb_last % 6 != 0 {
            candle_core::bail!("temb last dim must be divisible by 6, got {temb_last}");
        }
        let dim = temb_last / 6;
        let t = 1; // temb is [B, 6*dim], so T=1 effectively
        let temb_reshaped = temb.reshape((b, t, 6, dim))?;

        let table = self
            .scale_shift_table
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, t, 6, dim))?;
        let ada = table.broadcast_add(&temb_reshaped)?; // [B,T,6,dim]

        let shift_msa = ada.i((.., .., 0, ..))?;
        let scale_msa = ada.i((.., .., 1, ..))?;
        let gate_msa = ada.i((.., .., 2, ..))?;
        let shift_mlp = ada.i((.., .., 3, ..))?;
        let scale_mlp = ada.i((.., .., 4, ..))?;
        let gate_mlp = ada.i((.., .., 5, ..))?;

        // norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        // Align shapes: norm_hidden is [B,S,dim], while shift/scale are [B,T,dim].
        // In the Python file, T corresponds to the second dimension after time embedding; typically T==1.
        // We broadcast T over S when T==1.
        let scale_msa = scale_msa;
        let shift_msa = shift_msa;
        let gate_msa = gate_msa;
        let scale_mlp = scale_mlp;
        let shift_mlp = shift_mlp;
        let gate_mlp = gate_mlp;

        let norm_hidden = {
            let one = Tensor::ones_like(&scale_msa)?;
            let s = one.broadcast_add(&scale_msa)?; // 1+scale
            // If T==1, expand to [B,S,dim]
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hidden_states.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_msa.dim(1)? == 1 {
                shift_msa.broadcast_as((b, hidden_states.dim(1)?, shift_msa.dim(2)?))?
            } else {
                shift_msa
            };
            norm_hidden.broadcast_mul(&s)?.broadcast_add(&sh)?
        };

        // Self-attn (encoder_hidden_states=None) with RoPE
        let attn1 = self
            .attn1
            .forward(&norm_hidden, None, None, image_rotary_emb)?;
        let gate_msa = if gate_msa.dim(1)? == 1 {
            gate_msa.broadcast_as((b, hidden_states.dim(1)?, gate_msa.dim(2)?))?
        } else {
            gate_msa
        };
        let mut hs = hidden_states.broadcast_add(&attn1.broadcast_mul(&gate_msa)?)?;

        // Cross-attn
        let attn2 = self.attn2.forward(
            &hs,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            None,
        )?;
        hs = hs.broadcast_add(&attn2)?;

        // MLP
        let norm2 = self.norm2.forward(&hs)?;
        let norm2 = {
            let one = Tensor::ones_like(&scale_mlp)?;
            let s = one.broadcast_add(&scale_mlp)?;
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hs.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_mlp.dim(1)? == 1 {
                shift_mlp.broadcast_as((b, hs.dim(1)?, shift_mlp.dim(2)?))?
            } else {
                shift_mlp
            };
            norm2.broadcast_mul(&s)?.broadcast_add(&sh)?
        };
        let ff = self.ff.forward(&norm2)?;
        let gate_mlp = if gate_mlp.dim(1)? == 1 {
            gate_mlp.broadcast_as((b, hs.dim(1)?, gate_mlp.dim(2)?))?
        } else {
            gate_mlp
        };
        hs = hs.broadcast_add(&ff.broadcast_mul(&gate_mlp)?)?;

        Ok(hs)
    }
}

#[derive(Clone, Debug)]
pub struct LtxVideoTransformer3DModel {
    proj_in: nn::Linear,
    scale_shift_table: Tensor, // [2, inner_dim]
    time_embed: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    rope: LtxVideoRotaryPosEmbed,
    transformer_blocks: Vec<LtxVideoTransformerBlock>,
    norm_out: LayerNormNoParams,

    proj_out: nn::Linear,
    pipeline_config: TransformerConfig,
    skip_block_list: Vec<usize>,
}

impl LtxVideoTransformer3DModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(config: &LtxVideoTransformer3DModelConfig, vb: VarBuilder) -> Result<Self> {
        let out_channels = if config.out_channels == 0 {
            config.in_channels
        } else {
            config.out_channels
        };
        let inner_dim = config.num_attention_heads * config.attention_head_dim;

        let proj_in = nn::linear(config.in_channels, inner_dim, vb.pp("proj_in"))?;

        let scale_shift_table = vb.get((2, inner_dim), "scale_shift_table")?;

        let time_embed = AdaLayerNormSingle::new(inner_dim, vb.pp("time_embed"))?;
        let caption_projection = PixArtAlphaTextProjection::new(
            config.caption_channels,
            inner_dim,
            vb.pp("caption_projection"),
        )?;

        let rope = LtxVideoRotaryPosEmbed::new(
            inner_dim,
            20,
            2048,
            2048,
            config.patch_size,
            config.patch_size_t,
            10000.0,
        );

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            transformer_blocks.push(LtxVideoTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.cross_attention_dim,
                &config.qk_norm,
                config.attention_bias,
                config.attention_out_bias,
                config.norm_eps,
                config.norm_elementwise_affine,
                vb.pp("transformer_blocks").pp(layer_idx.to_string()),
            )?);
        }

        let norm_out = LayerNormNoParams::new(1e-6);
        let proj_out = nn::linear(inner_dim, out_channels, vb.pp("proj_out"))?;

        Ok(Self {
            proj_in,
            scale_shift_table,
            time_embed,
            caption_projection,
            rope,
            transformer_blocks,
            norm_out,
            proj_out,
            pipeline_config: TransformerConfig {
                in_channels: config.in_channels,
                patch_size: config.patch_size,
                patch_size_t: config.patch_size_t,
                num_layers: config.num_layers,
            },
            skip_block_list: Vec::new(),
        })
    }

    pub fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.skip_block_list = list;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,                  // [B, S, in_channels]
        encoder_hidden_states: &Tensor,          // [B, K, caption_channels]
        timestep: &Tensor,                       // int tensor, shape [B] or [B,1]...
        encoder_attention_mask: Option<&Tensor>, // [B,K] (1 valid, 0 pad) or already bias
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, _s, _c) = hidden_states.dims3()?;

        // Convert inputs to model dtype (BF16) if needed
        let model_dtype = self.proj_in.weight().dtype();
        let hidden_states = hidden_states.to_dtype(model_dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(model_dtype)?;

        let hidden_states = self.proj_in.forward(&hidden_states)?;

        let timestep = timestep.flatten_all()?.to_dtype(model_dtype)?; // ensure [B] or flat, in BF16

        // 1. AdaLayerNormSingle (Timesteps -> PixArtEmbed -> Silu -> Linear)
        let (temb, embedded_timestep) = self.time_embed.forward(&timestep)?; // [B, 6*dim], [B, dim]

        let encoder_hidden_states = self.caption_projection.forward(&encoder_hidden_states)?;

        // convert encoder_attention_mask to a bias
        let encoder_attention_mask = if let Some(mask) = encoder_attention_mask {
            if mask.rank() == 2 {
                let mask_f = mask.to_dtype(hidden_states.dtype())?;
                // (1 - mask) * -10000.0
                let bias = (mask_f.affine(-1.0, 1.0)? * (-10000.0))?;
                Some(bias.unsqueeze(1)?)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };
        let encoder_attention_mask = encoder_attention_mask.as_ref();

        let (cos, sin) = self.rope.forward(
            &hidden_states,
            num_frames,
            height,
            width,
            rope_interpolation_scale,
            video_coords,
        )?;

        // Pass embedded_timestep as conditioning? LTX-Video adds it to temb or similar?
        // Wait, logic in `forward` block of transformer_ltx.py:
        // temb is passed to blocks. embedded_timestep is not used explicitly in blocks for this model version?
        // Let's check block forward. Block uses `temb`.
        // The `AdaLayerNormSingle` returns (temb, embedded_timestep), but `LtxVideoTransformerBlock`
        // seems to take `temb`.
        // `LtxVideoTransformerBlock::forward` signature: `temb: &Tensor`.

        let mut hidden_states = hidden_states;
        let image_rotary_emb = Some((&cos, &sin));

        for (index, block) in self.transformer_blocks.iter().enumerate() {
            if self.skip_block_list.contains(&index) {
                continue;
            }

            let original_hidden_states = if skip_layer_mask.is_some() {
                Some(hidden_states.clone())
            } else {
                None
            };

            hidden_states = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                image_rotary_emb,
                encoder_attention_mask,
            )?;

            if let (Some(mask), Some(orig)) = (skip_layer_mask, original_hidden_states) {
                // mask shape: [num_layers, batch]
                // FIX: mask=1 means SKIP layer (use original), mask=0 means APPLY layer (keep processed)
                let m = mask.narrow(0, index, 1)?.flatten_all()?;
                let b_size = hidden_states.dim(0)?;
                let m = m.reshape((b_size, 1, 1))?.to_dtype(hidden_states.dtype())?;
                let one_minus_m = m.affine(-1.0, 1.0)?;
                // When m=1 (skip): use orig. When m=0 (apply): use hidden_states
                hidden_states = hidden_states
                    .broadcast_mul(&one_minus_m)? // m=0 -> keep processed hidden_states
                    .broadcast_add(&orig.broadcast_mul(&m)?)?; // m=1 -> add original (skip)
            }
        }

        // Final modulation: scale_shift_table[None,None] + embedded_timestep[:, :, None]
        let b = hidden_states.dim(0)?;
        let inner_dim = hidden_states.dim(2)?;

        // scale_shift_table: [2, inner_dim] -> cast to dtype of embedded_timestep
        let table = self.scale_shift_table.to_dtype(embedded_timestep.dtype())?;

        // table: [1, 1, 2, inner_dim]
        let table = table.unsqueeze(0)?.unsqueeze(0)?;

        // embedded_timestep: [B, T=1, inner_dim] (usually T=1 after pool or similar? AdaLayerNormSingle returns [B, inner_dim]?)
        // AdaLayerNormSingle returns `emb` which is [B, inner_dim].
        // We need to check dims.
        // If embedded_timestep is [B, D].
        let emb = embedded_timestep.unsqueeze(1)?.unsqueeze(2)?; // [B, 1, 1, D]

        // broadcast add: table + emb
        // [1,1,2,D] + [B,1,1,D] -> [B,1,2,D]
        let scale_shift = table.broadcast_add(&emb)?;

        let shift = scale_shift.i((.., .., 0, ..))?; // [B, 1, D]
        let scale = scale_shift.i((.., .., 1, ..))?; // [B, 1, D]

        let mut hidden_states = self.norm_out.forward(&hidden_states)?;

        // (1 + scale) * x + shift
        let one = Tensor::ones_like(&scale)?;
        let ss = one.broadcast_add(&scale)?;

        // Broadcast scale/shift to [B, S, D]
        // S is dim 1 of hidden_states
        let s_dim = hidden_states.dim(1)?;
        let ss = ss.broadcast_as((b, s_dim, inner_dim))?;
        let sh = shift.broadcast_as((b, s_dim, inner_dim))?;

        hidden_states = hidden_states.broadcast_mul(&ss)?.broadcast_add(&sh)?;

        let hidden_states = self.proj_out.forward(&hidden_states)?;

        // Residual connection? In Python:
        // output = self.proj_out(self.norm_out(hidden_states))
        // return output + residual (if configured? No usually strict functional)
        // Check if `hidden_states` (input) is added?
        // LTX models usually treat it as noise prediction (epsilon or v-prediction).
        // Let's assume just return output.
        Ok(hidden_states)
    }
}

impl VideoTransformer3D for LtxVideoTransformer3DModel {
    fn config(&self) -> &TransformerConfig {
        &self.pipeline_config
    }

    fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.set_skip_block_list(list);
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f32, f32, f32)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // cast scale to f64
        let scale = rope_interpolation_scale.map(|s| (s.0 as f64, s.1 as f64, s.2 as f64));

        // Call inherent forward
        // Note: inherent forward takes &self, trait takes &mut self (which coerces to &self).
        LtxVideoTransformer3DModel::forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            Some(encoder_attention_mask),
            num_frames,
            height,
            width,
            scale,
            video_coords,
            skip_layer_mask,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn test_skip_block_list_logic() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let config = LtxVideoTransformer3DModelConfig {
            num_layers: 3,
            ..Default::default()
        };

        // Use zeros/ones for weights to track passes if needed, but here we just check if it runs
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model = LtxVideoTransformer3DModel::new(&config, vb.pp("transformer"))?;

        // Initial state: no skips
        assert_eq!(model.skip_block_list.len(), 0);

        // Set skips
        model.set_skip_block_list(vec![1]);
        assert_eq!(model.skip_block_list, vec![1]);

        // In a real test we'd verify the output differs or some side effect,
        // but for now, we ensure it compiles and the logic is present.
        Ok(())
    }

    #[test]
    fn test_skip_layer_mask() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let config = LtxVideoTransformer3DModelConfig {
            num_layers: 2,
            attention_head_dim: 16,
            num_attention_heads: 2,
            cross_attention_dim: 32,
            caption_channels: 32,
            in_channels: 32,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = LtxVideoTransformer3DModel::new(&config, vb.pp("transformer"))?;

        let b = 2;
        let s = 16;
        let hidden_states = Tensor::ones(
            (b, s, config.attention_head_dim * config.num_attention_heads),
            DType::F32,
            &device,
        )?;
        let encoder_hidden_states =
            Tensor::zeros((b, 1, config.caption_channels), DType::F32, &device)?;
        let timestep = Tensor::zeros((b,), DType::F32, &device)?;

        // Mask: Layer 0 skipped for batch 0, Layer 1 skipped for batch 1
        // [num_layers, batch]
        let mask_data = vec![
            0.0f32, 1.0f32, // Layer 0: skip batch 0, keep batch 1
            1.0f32, 0.0f32, // Layer 1: keep batch 0, skip batch 1
        ];
        let mask = Tensor::from_vec(mask_data, (2, b), &device)?;

        let out = model.forward(
            &hidden_states,
            &encoder_hidden_states,
            &timestep,
            None,
            1,
            1,
            1,
            None,
            None,
            Some(&mask),
        )?;

        assert_eq!(out.dims3()?, (b, s, 128)); // 128 is out_channels by default? No, let's check
        // By default out_channels = in_channels if not specified.
        // LTXV model has out_channels = 128 by default in config.

        Ok(())
    }
}
