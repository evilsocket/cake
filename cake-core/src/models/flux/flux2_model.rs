//! FLUX.2-klein MMDiT transformer implementation.
//!
//! Architecture: dual-stream + single-stream transformer with shared modulation,
//! matching the `Flux2Transformer2DModel` from diffusers. Weight names follow
//! the diffusers convention (not the BFL FLUX.1 convention).

use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

use crate::backends::ComputeBackend;

// ── Helpers ────────────────────────────────────────────────────────────────────

pub fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
    const TIME_FACTOR: f64 = 1000.;
    const MAX_PERIOD: f64 = 10000.;
    if dim % 2 == 1 {
        candle_core::bail!("{dim} is odd")
    }
    let dev = t.device();
    let half = dim / 2;
    let t = (t * TIME_FACTOR)?;
    let arange = Tensor::arange(0, half as u32, dev)?.to_dtype(DType::F32)?;
    let freqs = (arange * (-MAX_PERIOD.ln() / half as f64))?.exp()?;
    let args = t
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(dtype)
}

/// Positional embedding matching diffusers Flux2PosEmbed exactly.
/// Returns (cos, sin) each of shape [S, head_dim] with repeat_interleave.
#[derive(Debug, Clone)]
pub struct Flux2PosEmbed {
    axes_dim: Vec<usize>,
    /// Pre-computed inverse frequencies per axis (cached to avoid reallocation)
    inv_freqs: Vec<Vec<f64>>,
}

impl Flux2PosEmbed {
    fn new(theta: usize, axes_dim: Vec<usize>) -> Self {
        let theta_f = theta as f64;
        let inv_freqs: Vec<Vec<f64>> = axes_dim.iter().map(|&dim| {
            (0..dim).step_by(2).map(|j| 1.0 / theta_f.powf(j as f64 / dim as f64)).collect()
        }).collect();
        let _ = theta; // used for inv_freqs computation above
        Self { axes_dim, inv_freqs }
    }

    /// Public constructor for testing.
    pub fn new_pub(theta: usize, axes_dim: Vec<usize>) -> Self {
        Self::new(theta, axes_dim)
    }

    /// Compute (cos, sin) PE for given position IDs [S, num_axes].
    pub fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut all_cos = Vec::with_capacity(self.axes_dim.len());
        let mut all_sin = Vec::with_capacity(self.axes_dim.len());
        let pos = ids.to_dtype(DType::F64)?;
        let seq_len = ids.dim(0)?;

        for (i, &dim) in self.axes_dim.iter().enumerate() {
            let half = dim / 2;
            let p = pos.get_on_dim(D::Minus1, i)?;

            let inv_freq = Tensor::from_slice(&self.inv_freqs[i], (1, half), ids.device())?;

            let freqs = p.unsqueeze(1)?.broadcast_mul(&inv_freq)?;
            let cos = freqs.cos()?.to_dtype(DType::F32)?;
            let sin = freqs.sin()?.to_dtype(DType::F32)?;

            // repeat_interleave(2): [S, half] → [S, dim]
            let cos = cos.unsqueeze(2)?.broadcast_as((seq_len, half, 2))?.reshape((seq_len, dim))?;
            let sin = sin.unsqueeze(2)?.broadcast_as((seq_len, half, 2))?.reshape((seq_len, dim))?;

            all_cos.push(cos);
            all_sin.push(sin);
        }

        let cos = Tensor::cat(&all_cos, D::Minus1)?;
        let sin = Tensor::cat(&all_sin, D::Minus1)?;
        Ok((cos, sin))
    }
}

/// Apply rotary embeddings matching diffusers apply_rotary_emb exactly.
/// x: [B, H, S, D], cos/sin: [S, D]
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let dtype = x.dtype();

    // cos/sin: [S, D] → [1, 1, S, D] for broadcasting
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;

    let x = x.to_dtype(DType::F32)?;

    // Split into interleaved pairs: [B, H, S, D] → [B, H, S, D/2, 2]
    let x_pairs = x.reshape((b, h, s, d / 2, 2))?;
    let x_real = x_pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let x_imag = x_pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

    // Compute real and imaginary parts of rotation separately:
    // out_real = x_real * cos_half - x_imag * sin_half
    // out_imag = x_imag * cos_half + x_real * sin_half
    // Then interleave back to [B, H, S, D]
    let cos_half = cos.reshape((1, 1, s, d / 2, 2))?.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let sin_half = sin.reshape((1, 1, s, d / 2, 2))?.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;

    let out_real = (&x_real * &cos_half)? - (&x_imag * &sin_half)?;
    let out_imag = (&x_imag * &cos_half)? + (&x_real * &sin_half)?;

    let out = Tensor::stack(&[&out_real?, &out_imag?], D::Minus1)?.reshape((b, h, s, d))?;
    out.to_dtype(dtype)
}

/// Linear projection that ensures inputs match weight dtype.
/// This is needed because modulate() returns in the input dtype which might be F32
/// from layer_norm, but the weight is BF16.
fn linear_matched(backend: &dyn ComputeBackend, weight: &Tensor, bias: Option<&Tensor>, x: &Tensor) -> Result<Tensor> {
    let w_dtype = weight.dtype();
    backend.linear_forward(&x.to_dtype(w_dtype)?, weight, bias)
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
    let w_dtype = q.dtype();
    let q = apply_rope(q, pe_cos, pe_sin)?.contiguous()?;
    let k = apply_rope(k, pe_cos, pe_sin)?.contiguous()?;
    let dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();

    // Flash Attention on CUDA — O(n) memory, faster for long sequences
    #[cfg(feature = "flash-attn")]
    if q.rank() == 4 && matches!(q.device(), candle_core::Device::Cuda(_)) {
        // flash_attn needs F16 in (batch, seq, heads, head_dim) layout
        let q16 = q.to_dtype(DType::F16)?.transpose(1, 2)?.contiguous()?;
        let k16 = k.to_dtype(DType::F16)?.transpose(1, 2)?.contiguous()?;
        let v16 = v.to_dtype(DType::F16)?.transpose(1, 2)?.contiguous()?;
        let attn = candle_flash_attn::flash_attn(&q16, &k16, &v16, scale as f32, false)?
            .to_dtype(w_dtype)?;
        // (batch, seq, heads, head_dim) → (batch, heads, seq, head_dim) → flatten
        return attn.transpose(1, 2)?.flatten_from(2);
    }

    // Metal: try F32 SDPA (non-causal, no mask — image gen doesn't need causal masking)
    #[cfg(feature = "metal")]
    if matches!(q.device(), candle_core::Device::Metal(_)) {
        let q32 = q.to_dtype(DType::F32)?;
        let k32 = k.to_dtype(DType::F32)?;
        let v32 = v.to_dtype(DType::F32)?;
        if let Ok(attn) = backend.sdpa(&q32, &k32, &v32, None, false, scale as f32) {
            return attn.to_dtype(w_dtype)?.transpose(1, 2)?.flatten_from(2);
        }
        // Fallback to manual if SDPA fails (threadgroup memory exceeded)
    }

    // CPU fallback: manual SDPA in F32
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?.to_dtype(DType::F32)?;
    let k = k.flatten_to(batch_dims.len() - 1)?.to_dtype(DType::F32)?;
    let v = v.flatten_to(batch_dims.len() - 1)?.to_dtype(DType::F32)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale)?;
    let last_dim = attn_weights.rank() - 1;
    let attn_scores = backend.softmax(&attn_weights, last_dim)?
        .matmul(&v)?
        .to_dtype(w_dtype)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    let x = attn_scores.reshape(batch_dims)?;
    x.transpose(1, 2)?.flatten_from(2)
}

/// RmsNorm that works with F16 inputs by computing in F32 internally.
#[derive(Debug, Clone)]
struct QkNorm {
    weight: Tensor,
    eps: f64,
}

impl QkNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        x_normed.broadcast_mul(&w)?.to_dtype(in_dtype)
    }
}

// ── MLP ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GatedMLP {
    linear_in_weight: Tensor,  // fused gate + up: (2*mlp_hidden, hidden)
    linear_out_weight: Tensor, // down: (hidden, mlp_hidden)
    mlp_hidden: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl GatedMLP {
    fn load(vb: VarBuilder, hidden: usize, mlp_hidden: usize, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let linear_in_weight = vb.pp("linear_in").get((2 * mlp_hidden, hidden), "weight")?;
        let linear_out_weight = vb.pp("linear_out").get((hidden, mlp_hidden), "weight")?;
        Ok(Self { linear_in_weight, linear_out_weight, mlp_hidden, backend })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = linear_matched(&*self.backend, &self.linear_in_weight, None, x)?;

        let gate = fused.narrow(D::Minus1, 0, self.mlp_hidden)?;
        let up = fused.narrow(D::Minus1, self.mlp_hidden, self.mlp_hidden)?;
        let x = self.backend.silu_mul(&gate, &up)?;
        linear_matched(&*self.backend, &self.linear_out_weight, None, &x)
    }
}

// ── Double Stream Block ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    // Image attention
    to_q_weight: Tensor,
    to_k_weight: Tensor,
    to_v_weight: Tensor,
    to_out_weight: Tensor,
    norm_q: QkNorm,
    norm_k: QkNorm,
    // Text attention
    add_q_proj_weight: Tensor,
    add_k_proj_weight: Tensor,
    add_v_proj_weight: Tensor,
    to_add_out_weight: Tensor,
    norm_added_q: QkNorm,
    norm_added_k: QkNorm,
    // MLPs
    ff: GatedMLP,
    ff_context: GatedMLP,
    num_heads: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl DoubleStreamBlock {
    fn load(vb: VarBuilder, cfg: &Flux2Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_hidden = (h as f64 * cfg.mlp_ratio) as usize;
        let attn = vb.pp("attn");

        Ok(Self {
            to_q_weight: attn.pp("to_q").get((h, h), "weight")?,
            to_k_weight: attn.pp("to_k").get((h, h), "weight")?,
            to_v_weight: attn.pp("to_v").get((h, h), "weight")?,
            to_out_weight: attn.pp("to_out").pp("0").get((h, h), "weight")?,
            norm_q: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_q"))?,
            norm_k: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_k"))?,
            add_q_proj_weight: attn.pp("add_q_proj").get((h, h), "weight")?,
            add_k_proj_weight: attn.pp("add_k_proj").get((h, h), "weight")?,
            add_v_proj_weight: attn.pp("add_v_proj").get((h, h), "weight")?,
            to_add_out_weight: attn.pp("to_add_out").get((h, h), "weight")?,
            norm_added_q: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_added_q"))?,
            norm_added_k: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_added_k"))?,
            ff: GatedMLP::load(vb.pp("ff"), h, mlp_hidden, backend.clone())?,
            ff_context: GatedMLP::load(vb.pp("ff_context"), h, mlp_hidden, backend.clone())?,
            num_heads: cfg.num_heads,
            backend,
        })
    }

    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &[Tensor],
        txt_mod: &[Tensor],
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let head_dim = img.dim(D::Minus1)? / self.num_heads;

        // Modulate + norm (computes in F32 internally, returns in input dtype)
        let img_norm = modulate(img, &img_mod[0], &img_mod[1])?;
        let txt_norm = modulate(txt, &txt_mod[0], &txt_mod[1])?;

        // Q/K/V projections (linear_matched ensures dtype compatibility)
        let img_q = linear_matched(&*self.backend, &self.to_q_weight, None, &img_norm)?;
        let img_k = linear_matched(&*self.backend, &self.to_k_weight, None, &img_norm)?;
        let img_v = linear_matched(&*self.backend, &self.to_v_weight, None, &img_norm)?;
        let txt_q = linear_matched(&*self.backend, &self.add_q_proj_weight, None, &txt_norm)?;
        let txt_k = linear_matched(&*self.backend, &self.add_k_proj_weight, None, &txt_norm)?;
        let txt_v = linear_matched(&*self.backend, &self.add_v_proj_weight, None, &txt_norm)?;

        // Reshape + QK-norm
        let (b, img_seq, _) = img_q.dims3()?;
        let (_, txt_seq, _) = txt_q.dims3()?;

        let reshape_norm = |x: Tensor, norm: &QkNorm, seq: usize| -> Result<Tensor> {
            let x = x.reshape((b, seq, self.num_heads, head_dim))?;
            norm.forward(&x)?.transpose(1, 2)
        };

        let img_q = reshape_norm(img_q, &self.norm_q, img_seq)?;
        let img_k = reshape_norm(img_k, &self.norm_k, img_seq)?;
        let img_v = img_v.reshape((b, img_seq, self.num_heads, head_dim))?.transpose(1, 2)?;
        let txt_q = reshape_norm(txt_q, &self.norm_added_q, txt_seq)?;
        let txt_k = reshape_norm(txt_k, &self.norm_added_k, txt_seq)?;
        let txt_v = txt_v.reshape((b, txt_seq, self.num_heads, head_dim))?.transpose(1, 2)?;

        // Joint attention
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;
        let attn_out = attention(&q, &k, &v, pe_cos, pe_sin, &*self.backend)?;

        // Split + output projections + gated residual
        let txt_attn = attn_out.narrow(1, 0, txt_seq)?;
        let img_attn = attn_out.narrow(1, txt_seq, img_seq)?;

        let img = (img + img_mod[2].unsqueeze(1)?.broadcast_mul(&linear_matched(&*self.backend, &self.to_out_weight, None, &img_attn)?)?)?;
        let txt = (txt + txt_mod[2].unsqueeze(1)?.broadcast_mul(&linear_matched(&*self.backend, &self.to_add_out_weight, None, &txt_attn)?)?)?;

        // MLP with modulation
        let img_ff = self.ff.forward(&modulate(&img, &img_mod[3], &img_mod[4])?)?;
        let txt_ff = self.ff_context.forward(&modulate(&txt, &txt_mod[3], &txt_mod[4])?)?;

        let img = (img + img_mod[5].unsqueeze(1)?.broadcast_mul(&img_ff)?)?;
        let txt = (txt + txt_mod[5].unsqueeze(1)?.broadcast_mul(&txt_ff)?)?;

        Ok((img, txt))
    }
}

// ── Single Stream Block ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    to_qkv_mlp_proj_weight: Tensor,
    to_out_weight: Tensor,
    norm_q: QkNorm,
    norm_k: QkNorm,
    num_heads: usize,
    hidden_size: usize,
    mlp_hidden: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl SingleStreamBlock {
    fn load(vb: VarBuilder, cfg: &Flux2Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_hidden = (h as f64 * cfg.mlp_ratio) as usize;
        let attn = vb.pp("attn");
        // Fused QKV + MLP gate+up: 3*h + 2*mlp_hidden
        let qkv_mlp_dim = 3 * h + 2 * mlp_hidden;
        let out_dim = h + mlp_hidden; // attention_out + mlp_out

        Ok(Self {
            to_qkv_mlp_proj_weight: attn.pp("to_qkv_mlp_proj").get((qkv_mlp_dim, h), "weight")?,
            to_out_weight: attn.pp("to_out").get((h, out_dim), "weight")?,
            norm_q: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_q"))?,
            norm_k: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_k"))?,
            num_heads: cfg.num_heads,
            hidden_size: h,
            mlp_hidden,
            backend,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mod_tensors: &[Tensor],
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let head_dim = self.hidden_size / self.num_heads;

        let x_norm = modulate(x, &mod_tensors[0], &mod_tensors[1])?;
        let fused = linear_matched(&*self.backend, &self.to_qkv_mlp_proj_weight, None, &x_norm)?;

        let h = self.hidden_size;
        let q = fused.narrow(D::Minus1, 0, h)?;
        let k = fused.narrow(D::Minus1, h, h)?;
        let v = fused.narrow(D::Minus1, 2 * h, h)?;
        let mlp_fused = fused.narrow(D::Minus1, 3 * h, 2 * self.mlp_hidden)?;

        let q = q.reshape((b, seq, self.num_heads, head_dim))?;
        let q = self.norm_q.forward(&q)?.transpose(1, 2)?;
        let k = k.reshape((b, seq, self.num_heads, head_dim))?;
        let k = self.norm_k.forward(&k)?.transpose(1, 2)?;
        let v = v.reshape((b, seq, self.num_heads, head_dim))?.transpose(1, 2)?;

        let attn_out = attention(&q, &k, &v, pe_cos, pe_sin, &*self.backend)?;

        let gate = mlp_fused.narrow(D::Minus1, 0, self.mlp_hidden)?;
        let up = mlp_fused.narrow(D::Minus1, self.mlp_hidden, self.mlp_hidden)?;
        let mlp_out = self.backend.silu_mul(&gate, &up)?;

        let combined = Tensor::cat(&[&attn_out, &mlp_out], D::Minus1)?;
        let out = linear_matched(&*self.backend, &self.to_out_weight, None, &combined)?;

        x + mod_tensors[2].unsqueeze(1)?.broadcast_mul(&out)?
    }
}

// ── Modulation helper ──────────────────────────────────────────────────────────

/// Adaptive LayerNorm: norm(x) * (1 + scale) + shift
/// LayerNorm computes in F32, then casts back. Scale/shift arithmetic in input dtype (BF16).
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_norm = layer_norm_forward(x)?.to_dtype(dtype)?;
    let scale = scale.unsqueeze(1)?;
    let shift = shift.unsqueeze(1)?;
    x_norm.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)
}

/// Manual layer norm (no learned params, eps=1e-6).
/// Always computes and returns in F32.
fn layer_norm_forward(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(var + 1e-6)?.sqrt()?)
}

// ── MLP Embedder ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    linear_1_weight: Tensor,
    linear_2_weight: Tensor,
    backend: Arc<dyn ComputeBackend>,
}

impl MlpEmbedder {
    pub fn load(vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let linear_1_weight = vb.pp("linear_1").get((3072, 256), "weight")?;
        let linear_2_weight = vb.pp("linear_2").get((3072, 3072), "weight")?;
        Ok(Self { linear_1_weight, linear_2_weight, backend })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.backend.linear_forward(x, &self.linear_1_weight, None)?;
        let h = self.backend.silu(&h)?;
        self.backend.linear_forward(&h, &self.linear_2_weight, None)
    }
}

// ── Main Model ─────────────────────────────────────────────────────────────────

/// FLUX.2-klein transformer configuration.
#[derive(Debug, Clone)]
pub struct Flux2Config {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: f64,
    pub depth: usize,           // double blocks
    pub depth_single: usize,    // single blocks
    pub in_channels: usize,     // latent input channels
    pub context_in_dim: usize,  // text encoder output dim
    pub axes_dim: Vec<usize>,   // RoPE axes dimensions
    pub theta: usize,           // RoPE theta
}

impl Flux2Config {
    pub fn klein_4b() -> Self {
        Self {
            hidden_size: 3072,
            num_heads: 24,
            head_dim: 128,
            mlp_ratio: 3.0,
            depth: 5,
            depth_single: 20,
            in_channels: 128,
            context_in_dim: 7680,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
        }
    }
}

/// FLUX.2-klein MMDiT transformer.
#[derive(Debug, Clone)]
pub struct Flux2Transformer {
    pub x_embedder_weight: Tensor,
    pub context_embedder_weight: Tensor,
    pub time_embedder: MlpEmbedder,
    pub pe_embedder: Flux2PosEmbed,
    pub double_mod_img_weight: Tensor,
    pub double_mod_txt_weight: Tensor,
    pub single_mod_weight: Tensor,
    pub double_blocks: Vec<DoubleStreamBlock>,
    pub single_blocks: Vec<SingleStreamBlock>,
    pub norm_out_weight: Tensor,
    pub proj_out_weight: Tensor,
    pub cfg: Flux2Config,
    backend: Arc<dyn ComputeBackend>,
}

impl Flux2Transformer {
    pub fn load(vb: VarBuilder, cfg: &Flux2Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h = cfg.hidden_size;

        let x_embedder_weight = vb.pp("x_embedder").get((h, cfg.in_channels), "weight")?;
        let context_embedder_weight = vb.pp("context_embedder").get((h, cfg.context_in_dim), "weight")?;
        let time_embedder = MlpEmbedder::load(vb.pp("time_guidance_embed").pp("timestep_embedder"), backend.clone())?;

        let pe_embedder = Flux2PosEmbed::new(cfg.theta, cfg.axes_dim.clone());

        // Shared modulation layers
        let double_mod_img_weight = vb.pp("double_stream_modulation_img").pp("linear").get((6 * h, h), "weight")?;
        let double_mod_txt_weight = vb.pp("double_stream_modulation_txt").pp("linear").get((6 * h, h), "weight")?;
        let single_mod_weight = vb.pp("single_stream_modulation").pp("linear").get((3 * h, h), "weight")?;

        // Double blocks
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("transformer_blocks");
        for i in 0..cfg.depth {
            double_blocks.push(DoubleStreamBlock::load(vb_d.pp(i), cfg, backend.clone())?);
        }

        // Single blocks
        let mut single_blocks = Vec::with_capacity(cfg.depth_single);
        let vb_s = vb.pp("single_transformer_blocks");
        for i in 0..cfg.depth_single {
            single_blocks.push(SingleStreamBlock::load(vb_s.pp(i), cfg, backend.clone())?);
        }

        // Final layer
        let norm_out_weight = vb.pp("norm_out").pp("linear").get((2 * h, h), "weight")?;
        let proj_out_weight = vb.pp("proj_out").get((cfg.in_channels, h), "weight")?;

        Ok(Self {
            x_embedder_weight,
            context_embedder_weight,
            time_embedder,
            pe_embedder,
            double_mod_img_weight,
            double_mod_txt_weight,
            single_mod_weight,
            double_blocks,
            single_blocks,
            norm_out_weight,
            proj_out_weight,
            cfg: cfg.clone(),
            backend,
        })
    }

    pub fn forward(
        &self,
        img: &Tensor,      // (b, img_seq, in_channels)
        img_ids: &Tensor,   // (b, img_seq, 4)
        txt: &Tensor,       // (b, txt_seq, context_in_dim)
        txt_ids: &Tensor,   // (b, txt_seq, 4)
        timesteps: &Tensor,  // (b,)
    ) -> Result<Tensor> {
        // Determine weight dtype from first weight tensor
        let w_dtype = self.x_embedder_weight.dtype();

        // Cast inputs to match weight dtype for matmul compatibility
        let img = img.to_dtype(w_dtype)?;
        let txt = txt.to_dtype(w_dtype)?;

        // Embed inputs (F32 for precision)
        let mut img = linear_matched(&*self.backend, &self.x_embedder_weight, None, &img)?;
        let mut txt = linear_matched(&*self.backend, &self.context_embedder_weight, None, &txt)?;

        // Timestep conditioning (compute in F32, cast back)
        let vec = timestep_embedding(&timesteps.to_dtype(DType::F32)?, 256, DType::F32)?
            .to_dtype(w_dtype)?;
        let vec = self.time_embedder.forward(&vec)?;

        // Positional embeddings
        let img_ids_2d = if img_ids.dims().len() == 3 { img_ids.squeeze(0)? } else { img_ids.clone() };
        let txt_ids_2d = if txt_ids.dims().len() == 3 { txt_ids.squeeze(0)? } else { txt_ids.clone() };
        let (img_cos, img_sin) = self.pe_embedder.forward(&img_ids_2d)?;
        let (txt_cos, txt_sin) = self.pe_embedder.forward(&txt_ids_2d)?;
        let pe_cos = Tensor::cat(&[&txt_cos, &img_cos], 0)?.to_dtype(w_dtype)?;
        let pe_sin = Tensor::cat(&[&txt_sin, &img_sin], 0)?.to_dtype(w_dtype)?;

        // Compute shared modulations (F32 for precision)
        let vec_silu = self.backend.silu(&vec)?;
        let img_mod_all = linear_matched(&*self.backend, &self.double_mod_img_weight, None, &vec_silu)?;
        let txt_mod_all = linear_matched(&*self.backend, &self.double_mod_txt_weight, None, &vec_silu)?;
        let single_mod_all = linear_matched(&*self.backend, &self.single_mod_weight, None, &vec_silu)?;

        // Split modulations into chunks
        let img_mods = img_mod_all.chunk(6, D::Minus1)?;
        let txt_mods = txt_mod_all.chunk(6, D::Minus1)?;
        let single_mods = single_mod_all.chunk(3, D::Minus1)?;

        // Double stream blocks
        for block in &self.double_blocks {
            let (new_img, new_txt) = block.forward(&img, &txt, &img_mods, &txt_mods, &pe_cos, &pe_sin)?;
            img = new_img;
            txt = new_txt;
        }

        // Single stream blocks
        let txt_seq = txt.dim(1)?;
        let mut merged = Tensor::cat(&[&txt, &img], 1)?;
        for block in &self.single_blocks {
            merged = block.forward(&merged, &single_mods, &pe_cos, &pe_sin)?;
        }

        // Extract image portion
        let img = merged.narrow(1, txt_seq, merged.dim(1)? - txt_seq)?;

        // Final layer: adaptive norm + projection (F32 for precision)
        let final_mod = linear_matched(&*self.backend, &self.norm_out_weight, None, &self.backend.silu(&vec)?)?;
        let final_chunks = final_mod.chunk(2, D::Minus1)?;
        let img = modulate(&img, &final_chunks[0], &final_chunks[1])?;
        linear_matched(&*self.backend, &self.proj_out_weight, None, &img)
    }
}
