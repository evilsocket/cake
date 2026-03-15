//! FLUX.2-klein MMDiT transformer implementation.
//!
//! Architecture: dual-stream + single-stream transformer with shared modulation,
//! matching the `Flux2Transformer2DModel` from diffusers. Weight names follow
//! the diffusers convention (not the BFL FLUX.1 convention).

use candle_core::{DType, Result, Tensor, D, Module};
use candle_nn::{Linear, VarBuilder};

// ── Helpers ────────────────────────────────────────────────────────────────────

pub(crate) fn timestep_embedding(t: &Tensor, dim: usize, dtype: DType) -> Result<Tensor> {
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
struct Flux2PosEmbed {
    theta: usize,
    axes_dim: Vec<usize>,
}

impl Flux2PosEmbed {
    fn new(theta: usize, axes_dim: Vec<usize>) -> Self {
        Self { theta, axes_dim }
    }

    /// Compute (cos, sin) PE for given position IDs [S, num_axes].
    fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut all_cos = Vec::new();
        let mut all_sin = Vec::new();
        let pos = ids.to_dtype(DType::F64)?;

        for i in 0..self.axes_dim.len() {
            let dim = self.axes_dim[i];
            let half = dim / 2;
            let p = pos.get_on_dim(D::Minus1, i)?; // [S]
            let theta = self.theta as f64;

            let inv_freq: Vec<f64> = (0..dim)
                .step_by(2)
                .map(|j| 1.0 / theta.powf(j as f64 / dim as f64))
                .collect();
            let inv_freq = Tensor::from_vec(inv_freq, (1, half), ids.device())?
                .to_dtype(DType::F64)?;

            let freqs = p.unsqueeze(1)?.broadcast_mul(&inv_freq)?; // [S, half]
            let cos = freqs.cos()?.to_dtype(DType::F32)?;
            let sin = freqs.sin()?.to_dtype(DType::F32)?;

            // repeat_interleave(2): each frequency applies to a pair of elements
            // [S, half] → [S, dim] by repeating each value twice
            let cos = cos.unsqueeze(2)?.broadcast_as((ids.dim(0)?, half, 2))?.reshape((ids.dim(0)?, dim))?;
            let sin = sin.unsqueeze(2)?.broadcast_as((ids.dim(0)?, half, 2))?.reshape((ids.dim(0)?, dim))?;

            all_cos.push(cos);
            all_sin.push(sin);
        }

        let cos = Tensor::cat(&all_cos, D::Minus1)?; // [S, head_dim]
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

    // Split into interleaved pairs: reshape to (..., D/2, 2), unbind last dim
    let x_pairs = x.reshape((b, h, s, d / 2, 2))?;
    let x_real = x_pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?; // [B, H, S, D/2]
    let x_imag = x_pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

    // x_rotated = [-x_imag, x_real] interleaved
    let neg_x_imag = x_imag.neg()?;
    let x_rotated = Tensor::stack(&[&neg_x_imag, &x_real], D::Minus1)?
        .reshape((b, h, s, d))?;

    // out = x * cos + x_rotated * sin
    let out = (x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?)?;
    out.to_dtype(dtype)
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe_cos: &Tensor, pe_sin: &Tensor) -> Result<Tensor> {
    let w_dtype = q.dtype();
    let q = apply_rope(q, pe_cos, pe_sin)?.contiguous()?;
    let k = apply_rope(k, pe_cos, pe_sin)?.contiguous()?;
    let dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    // Compute attention in F32 for softmax precision
    let q = q.flatten_to(batch_dims.len() - 1)?.to_dtype(DType::F32)?;
    let k = k.flatten_to(batch_dims.len() - 1)?.to_dtype(DType::F32)?;
    let v = v.flatten_to(batch_dims.len() - 1)?.to_dtype(DType::F32)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?
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
pub(crate) struct GatedMLP {
    linear_in: Linear,  // fused gate + up: (hidden, 2*mlp_hidden)
    linear_out: Linear, // down: (mlp_hidden, hidden)
    mlp_hidden: usize,
}

impl GatedMLP {
    fn load(vb: VarBuilder, hidden: usize, mlp_hidden: usize) -> Result<Self> {
        let linear_in = candle_nn::linear_no_bias(hidden, 2 * mlp_hidden, vb.pp("linear_in"))?;
        let linear_out = candle_nn::linear_no_bias(mlp_hidden, hidden, vb.pp("linear_out"))?;
        Ok(Self { linear_in, linear_out, mlp_hidden })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fused = self.linear_in.forward(x)?;

        let gate = fused.narrow(D::Minus1, 0, self.mlp_hidden)?;
        let up = fused.narrow(D::Minus1, self.mlp_hidden, self.mlp_hidden)?;
        let x = (gate.silu()? * up)?;
        self.linear_out.forward(&x)
    }
}

// ── Double Stream Block ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct DoubleStreamBlock {
    // Image attention
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: QkNorm,
    norm_k: QkNorm,
    // Text attention
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    to_add_out: Linear,
    norm_added_q: QkNorm,
    norm_added_k: QkNorm,
    // MLPs
    ff: GatedMLP,
    ff_context: GatedMLP,
    num_heads: usize,
}

impl DoubleStreamBlock {
    fn load(vb: VarBuilder, cfg: &Flux2Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_hidden = (h as f64 * cfg.mlp_ratio) as usize;
        let attn = vb.pp("attn");

        Ok(Self {
            to_q: candle_nn::linear_no_bias(h, h, attn.pp("to_q"))?,
            to_k: candle_nn::linear_no_bias(h, h, attn.pp("to_k"))?,
            to_v: candle_nn::linear_no_bias(h, h, attn.pp("to_v"))?,
            to_out: candle_nn::linear_no_bias(h, h, attn.pp("to_out").pp("0"))?,
            norm_q: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_q"))?,
            norm_k: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_k"))?,
            add_q_proj: candle_nn::linear_no_bias(h, h, attn.pp("add_q_proj"))?,
            add_k_proj: candle_nn::linear_no_bias(h, h, attn.pp("add_k_proj"))?,
            add_v_proj: candle_nn::linear_no_bias(h, h, attn.pp("add_v_proj"))?,
            to_add_out: candle_nn::linear_no_bias(h, h, attn.pp("to_add_out"))?,
            norm_added_q: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_added_q"))?,
            norm_added_k: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_added_k"))?,
            ff: GatedMLP::load(vb.pp("ff"), h, mlp_hidden)?,
            ff_context: GatedMLP::load(vb.pp("ff_context"), h, mlp_hidden)?,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_mod: &[Tensor],
        txt_mod: &[Tensor],
        pe_cos: &Tensor,
        pe_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let head_dim = img.dim(D::Minus1)? / self.num_heads;

        // Modulate + norm (adaptive LayerNorm)
        let img_norm = modulate(img, &img_mod[0], &img_mod[1])?;
        let txt_norm = modulate(txt, &txt_mod[0], &txt_mod[1])?;

        // Image Q/K/V
        let img_q = self.to_q.forward(&img_norm)?;
        let img_k = self.to_k.forward(&img_norm)?;
        let img_v = self.to_v.forward(&img_norm)?;

        // Text Q/K/V
        let txt_q = self.add_q_proj.forward(&txt_norm)?;
        let txt_k = self.add_k_proj.forward(&txt_norm)?;
        let txt_v = self.add_v_proj.forward(&txt_norm)?;


        // Reshape to (b, heads, seq, head_dim) and apply QK-norm
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

        // Joint attention: concat text + image
        let q = Tensor::cat(&[&txt_q, &img_q], 2)?; // (b, heads, txt+img, head_dim)
        let k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 2)?;

        let attn_out = attention(&q, &k, &v, pe_cos, pe_sin)?; // (b, txt+img, hidden)

        // Split attention output
        let txt_attn = attn_out.narrow(1, 0, txt_seq)?;
        let img_attn = attn_out.narrow(1, txt_seq, img_seq)?;

        // Apply output projections + gated residual
        let img = (img + img_mod[2].unsqueeze(1)?.broadcast_mul(&self.to_out.forward(&img_attn)?)?)?;
        let txt = (txt + txt_mod[2].unsqueeze(1)?.broadcast_mul(&self.to_add_out.forward(&txt_attn)?)?)?;

        // MLP with modulation
        let img_mod_mlp = modulate(&img, &img_mod[3], &img_mod[4])?;
        let img_ff = self.ff.forward(&img_mod_mlp)?;
        let txt_mod_mlp = modulate(&txt, &txt_mod[3], &txt_mod[4])?;
        let txt_ff = self.ff_context.forward(&txt_mod_mlp)?;

        let img = (img + img_mod[5].unsqueeze(1)?.broadcast_mul(&img_ff)?)?;
        let txt = (txt + txt_mod[5].unsqueeze(1)?.broadcast_mul(&txt_ff)?)?;

        Ok((img, txt))
    }
}

// ── Single Stream Block ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SingleStreamBlock {
    to_qkv_mlp_proj: Linear,
    to_out: Linear,
    norm_q: QkNorm,
    norm_k: QkNorm,
    num_heads: usize,
    hidden_size: usize,
    mlp_hidden: usize,
}

impl SingleStreamBlock {
    fn load(vb: VarBuilder, cfg: &Flux2Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let mlp_hidden = (h as f64 * cfg.mlp_ratio) as usize;
        let attn = vb.pp("attn");
        // Fused QKV + MLP gate+up: 3*h + 2*mlp_hidden
        let qkv_mlp_dim = 3 * h + 2 * mlp_hidden;
        let out_dim = h + mlp_hidden; // attention_out + mlp_out

        Ok(Self {
            to_qkv_mlp_proj: candle_nn::linear_no_bias(h, qkv_mlp_dim, attn.pp("to_qkv_mlp_proj"))?,
            to_out: candle_nn::linear_no_bias(out_dim, h, attn.pp("to_out"))?,
            norm_q: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_q"))?,
            norm_k: QkNorm::load(cfg.head_dim, 1e-6, attn.pp("norm_k"))?,
            num_heads: cfg.num_heads,
            hidden_size: h,
            mlp_hidden,
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
        let fused = self.to_qkv_mlp_proj.forward(&x_norm)?;

        // Split: Q, K, V (each hidden_size), then MLP gate+up (2*mlp_hidden)
        let h = self.hidden_size;
        let q = fused.narrow(D::Minus1, 0, h)?;
        let k = fused.narrow(D::Minus1, h, h)?;
        let v = fused.narrow(D::Minus1, 2 * h, h)?;
        let mlp_fused = fused.narrow(D::Minus1, 3 * h, 2 * self.mlp_hidden)?;

        // QK-norm + reshape
        let q = q.reshape((b, seq, self.num_heads, head_dim))?;
        let q = self.norm_q.forward(&q)?.transpose(1, 2)?;
        let k = k.reshape((b, seq, self.num_heads, head_dim))?;
        let k = self.norm_k.forward(&k)?.transpose(1, 2)?;
        let v = v.reshape((b, seq, self.num_heads, head_dim))?.transpose(1, 2)?;

        // Attention
        let attn_out = attention(&q, &k, &v, pe_cos, pe_sin)?; // (b, seq, hidden)

        // MLP: gated SiLU
        let gate = mlp_fused.narrow(D::Minus1, 0, self.mlp_hidden)?;
        let up = mlp_fused.narrow(D::Minus1, self.mlp_hidden, self.mlp_hidden)?;
        let mlp_out = (gate.silu()? * up)?;

        // Concat attention + MLP, project out
        let combined = Tensor::cat(&[&attn_out, &mlp_out], D::Minus1)?;
        let out = self.to_out.forward(&combined)?;

        // Gated residual
        Ok((x + mod_tensors[2].unsqueeze(1)?.broadcast_mul(&out)?)?)
    }
}

// ── Modulation helper ──────────────────────────────────────────────────────────

/// Adaptive LayerNorm: norm(x) * (1 + scale) + shift
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    // Compute norm in F32, then scale/shift in input dtype
    let x_norm = layer_norm_forward(x)?.to_dtype(dtype)?;
    let scale = scale.unsqueeze(1)?;
    let shift = shift.unsqueeze(1)?;
    let ones = Tensor::ones_like(&scale)?;
    x_norm.broadcast_mul(&(scale + ones)?)?.broadcast_add(&shift)
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
pub(crate) struct MlpEmbedder {
    linear_1: Linear,
    linear_2: Linear,
}

impl MlpEmbedder {
    pub(crate) fn load(vb: VarBuilder) -> Result<Self> {
        let linear_1 = candle_nn::linear_no_bias(256, 3072, vb.pp("linear_1"))?;
        let linear_2 = candle_nn::linear_no_bias(3072, 3072, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.linear_1)?.silu()?.apply(&self.linear_2)
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
    x_embedder: Linear,
    context_embedder: Linear,
    time_embedder: MlpEmbedder,
    pe_embedder: Flux2PosEmbed,
    // Shared modulation
    double_mod_img: Linear,
    double_mod_txt: Linear,
    single_mod: Linear,
    // Blocks
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    // Final layer
    norm_out: Linear,
    proj_out: Linear,
    cfg: Flux2Config,
}

impl Flux2Transformer {
    pub fn load(vb: VarBuilder, cfg: &Flux2Config) -> Result<Self> {
        let h = cfg.hidden_size;

        let x_embedder = candle_nn::linear_no_bias(cfg.in_channels, h, vb.pp("x_embedder"))?;
        let context_embedder = candle_nn::linear_no_bias(cfg.context_in_dim, h, vb.pp("context_embedder"))?;
        let time_embedder = MlpEmbedder::load(vb.pp("time_guidance_embed").pp("timestep_embedder"))?;

        let pe_embedder = Flux2PosEmbed::new(cfg.theta, cfg.axes_dim.clone());

        // Shared modulation layers
        let double_mod_img = candle_nn::linear_no_bias(h, 6 * h, vb.pp("double_stream_modulation_img").pp("linear"))?;
        let double_mod_txt = candle_nn::linear_no_bias(h, 6 * h, vb.pp("double_stream_modulation_txt").pp("linear"))?;
        let single_mod = candle_nn::linear_no_bias(h, 3 * h, vb.pp("single_stream_modulation").pp("linear"))?;

        // Double blocks
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("transformer_blocks");
        for i in 0..cfg.depth {
            double_blocks.push(DoubleStreamBlock::load(vb_d.pp(i), cfg)?);
        }

        // Single blocks
        let mut single_blocks = Vec::with_capacity(cfg.depth_single);
        let vb_s = vb.pp("single_transformer_blocks");
        for i in 0..cfg.depth_single {
            single_blocks.push(SingleStreamBlock::load(vb_s.pp(i), cfg)?);
        }

        // Final layer
        let norm_out = candle_nn::linear_no_bias(h, 2 * h, vb.pp("norm_out").pp("linear"))?;
        let proj_out = candle_nn::linear_no_bias(h, cfg.in_channels, vb.pp("proj_out"))?;

        Ok(Self {
            x_embedder,
            context_embedder,
            time_embedder,
            pe_embedder,
            double_mod_img,
            double_mod_txt,
            single_mod,
            double_blocks,
            single_blocks,
            norm_out,
            proj_out,
            cfg: cfg.clone(),
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
        let w_dtype = self.x_embedder.weight().dtype();

        // Cast inputs to match weight dtype for matmul compatibility
        let img = img.to_dtype(w_dtype)?;
        let txt = txt.to_dtype(w_dtype)?;

        // Embed inputs
        let mut img = self.x_embedder.forward(&img)?;
        let mut txt = self.context_embedder.forward(&txt)?;

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

        // Compute shared modulations
        let vec_silu = vec.silu()?;
        let img_mod_all = Module::forward(&self.double_mod_img, &vec_silu)?;
        let txt_mod_all = Module::forward(&self.double_mod_txt, &vec_silu)?;
        let single_mod_all = Module::forward(&self.single_mod, &vec_silu)?;

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

        // Final layer: adaptive norm + projection
        let final_mod = Module::forward(&self.norm_out, &vec.silu()?)?;
        let final_chunks = final_mod.chunk(2, D::Minus1)?;
        let img = modulate(&img, &final_chunks[0], &final_chunks[1])?;
        self.proj_out.forward(&img)
    }
}
