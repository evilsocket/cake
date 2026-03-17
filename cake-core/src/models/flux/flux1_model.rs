//! FLUX.1-dev MMDiT transformer with FP8 weight storage.
//!
//! Forked from candle_transformers::models::flux::model::Flux to support FP8 weights
//! that stay in F8E4M3 on GPU (~12GB) and are dequantized to BF16 per-layer during forward.
//! Weight names follow the BFL convention (double_blocks, single_blocks, etc.).

use candle_core::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, RmsNorm, VarBuilder};

// ── Config ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Config {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
}

impl Config {
    pub fn dev() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
        }
    }
}

// ── FP8-aware Linear ───────────────────────────────────────────────────────────

/// Linear layer that stores weights in their native dtype (possibly F8E4M3)
/// and dequantizes to F16 on first forward call, caching the result.
#[derive(Debug)]
pub struct Fp8Linear {
    /// Original F8E4M3 weight (dropped after first dequant to save VRAM).
    f8_weight: Option<Tensor>,
    /// Cached F16 weight (populated on first forward call).
    weight: std::sync::RwLock<Option<Tensor>>,
    bias: Option<Tensor>,
}

impl Clone for Fp8Linear {
    fn clone(&self) -> Self {
        Self {
            f8_weight: self.f8_weight.clone(),
            weight: std::sync::RwLock::new(self.weight.read().unwrap().clone()),
            bias: self.bias.clone(),
        }
    }
}

impl Fp8Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        if weight.dtype() == DType::F8E4M3 {
            Self {
                f8_weight: Some(weight),
                weight: std::sync::RwLock::new(None),
                bias,
            }
        } else {
            Self {
                f8_weight: None,
                weight: std::sync::RwLock::new(Some(weight)),
                bias,
            }
        }
    }

    /// Public constructor for testing.
    pub fn new_pub(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::new(weight, bias)
    }

    fn get_weight(&self) -> Result<Tensor> {
        // Fast path: cached weight (F16 for F8 source, or original dtype)
        if let Some(w) = self.weight.read().unwrap().as_ref() {
            return Ok(w.clone());
        }
        // Slow path: dequantize F8→F16 and cache
        let mut cache = self.weight.write().unwrap();
        if let Some(w) = cache.as_ref() {
            return Ok(w.clone());
        }
        let f8_w = self.f8_weight.as_ref().expect("no F8 weight and no cached weight");
        let f16_w = crate::utils::fused_ops::f8e4m3_to_f16(f8_w)?;
        *cache = Some(f16_w.clone());
        Ok(f16_w)
    }

    fn compute_dtype(&self) -> DType {
        if self.f8_weight.is_some() {
            DType::F16 // F8 models use F16 compute
        } else {
            // Non-F8: use the stored weight's dtype
            self.weight.read().unwrap().as_ref().map(|w| w.dtype()).unwrap_or(DType::F32)
        }
    }

    /// Pre-dequantize F8→F16 and cache. Call once before inference loop.
    pub fn warmup(&self) -> Result<()> {
        let _ = self.get_weight()?;
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.get_weight()?;
        let dtype = self.compute_dtype();
        let x = x.to_dtype(dtype)?;
        let w = w.to_dtype(dtype)?;
        // Use broadcast_matmul for 3D×2D: reshape x to 2D, matmul, reshape back
        let dims = x.dims().to_vec();
        let last = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x.reshape((batch, last))?;
        let y_2d = x_2d.matmul(&w.t()?)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(w.dim(0)?);
        let y = y_2d.reshape(out_dims)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(dtype)?),
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_linear_f32_roundtrip() {
        let w = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &candle_core::Device::Cpu)
            .unwrap();
        let linear = Fp8Linear::new(w, None);
        let x = Tensor::new(&[[0.5f32, 1.0]], &candle_core::Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 3]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_f8_to_f32_software_dequant_cuda() {
        let dev = match candle_core::Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Create F32 tensor, cast to F8E4M3 on CPU, move to GPU, dequant with our kernel
        let original = Tensor::new(&[0.5f32, 1.0, -0.5, 2.0], &candle_core::Device::Cpu).unwrap();
        let f8 = original.to_dtype(DType::F8E4M3).unwrap();
        let f8_gpu = f8.to_device(&dev).unwrap();
        assert_eq!(f8_gpu.dtype(), DType::F8E4M3);

        // Use our software dequant (works on all SM including A100/SM80)
        let f32_gpu = crate::utils::fused_ops::f8e4m3_to_f32(&f8_gpu).unwrap();
        assert_eq!(f32_gpu.dtype(), DType::F32);

        let vals: Vec<f32> = f32_gpu.to_vec1().unwrap();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 0.5).abs() < 0.1, "val[0]={}", vals[0]);
        assert!((vals[1] - 1.0).abs() < 0.1, "val[1]={}", vals[1]);
        assert!((vals[2] + 0.5).abs() < 0.1, "val[2]={}", vals[2]);
        assert!((vals[3] - 2.0).abs() < 0.1, "val[3]={}", vals[3]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_linear_cuda_forward() {
        let dev = match candle_core::Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Create F8 weight on GPU via our software path
        let w_f32 = Tensor::new(
            &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let w_f8 = w_f32.to_dtype(DType::F8E4M3).unwrap().to_device(&dev).unwrap();
        let linear = Fp8Linear::new(w_f8, None);

        let x = Tensor::new(&[[0.5f32, 1.0, 0.5, 1.0]], &dev).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2]);
        assert_eq!(y.dtype(), DType::F16);
    }

    #[test]
    fn test_scaled_dot_product_attention_shape() {
        let q = Tensor::randn(0f32, 1., (1, 4, 16, 32), &candle_core::Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1., (1, 4, 16, 32), &candle_core::Device::Cpu).unwrap();
        let v = Tensor::randn(0f32, 1., (1, 4, 16, 32), &candle_core::Device::Cpu).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.dims(), &[1, 4, 16, 32]);
    }

    #[test]
    fn test_scaled_dot_product_attention_different_seq() {
        let q = Tensor::randn(0f32, 1., (1, 2, 8, 16), &candle_core::Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1., (1, 2, 12, 16), &candle_core::Device::Cpu).unwrap();
        let v = Tensor::randn(0f32, 1., (1, 2, 12, 16), &candle_core::Device::Cpu).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.dims(), &[1, 2, 8, 16]); // seq follows q
    }

    #[test]
    fn test_rope_shape() {
        // rope expects pos: (batch, seq_len)
        let pos = Tensor::randn(0f32, 1., (1, 16), &candle_core::Device::Cpu).unwrap();
        let out = rope(&pos, 32, 10000).unwrap();
        // Output: (batch, seq_len, dim/2, 2, 2)
        assert_eq!(out.dims(), &[1, 16, 16, 2, 2]);
    }

    #[test]
    fn test_rope_odd_dim_errors() {
        let pos = Tensor::randn(0f32, 1., (1, 4), &candle_core::Device::Cpu).unwrap();
        assert!(rope(&pos, 33, 10000).is_err());
    }

    #[test]
    fn test_apply_rope_preserves_shape() {
        // x: (batch, heads, seq, head_dim), pe: (batch, seq, head_dim/2, 2, 2)
        let x = Tensor::randn(0f32, 1., (1, 4, 8, 32), &candle_core::Device::Cpu).unwrap();
        let pos = Tensor::randn(0f32, 1., (1, 8), &candle_core::Device::Cpu).unwrap();
        let freq_cis = rope(&pos, 32, 10000).unwrap();
        let out = apply_rope(&x, &freq_cis).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8, 32]);
    }

    #[test]
    fn test_attention_full_pipeline() {
        let q = Tensor::randn(0f32, 1., (1, 4, 8, 32), &candle_core::Device::Cpu).unwrap();
        let k = Tensor::randn(0f32, 1., (1, 4, 8, 32), &candle_core::Device::Cpu).unwrap();
        let v = Tensor::randn(0f32, 1., (1, 4, 8, 32), &candle_core::Device::Cpu).unwrap();
        let pos = Tensor::randn(0f32, 1., (1, 8), &candle_core::Device::Cpu).unwrap();
        let pe = rope(&pos, 32, 10000).unwrap();
        let out = attention(&q, &k, &v, &pe).unwrap();
        assert_eq!(out.dims(), &[1, 8, 128]); // seq_len=8, 4 heads × 32 dim = 128
    }

    #[test]
    fn test_timestep_embedding_f16() {
        let t = Tensor::new(&[0.5f32], &candle_core::Device::Cpu).unwrap();
        let emb = timestep_embedding(&t, 256, DType::F16).unwrap();
        assert_eq!(emb.dims(), &[1, 256]);
        assert_eq!(emb.dtype(), DType::F16);
    }

    #[test]
    fn test_embed_nd() {
        // EmbedNd expects input shape matching the axes: each column is a position axis.
        // For axes_dim=[16, 56, 56], input needs 3 "columns" concatenated.
        // The actual usage is: Tensor::cat(&[txt_ids, img_ids], 1) which is (batch, total_seq, num_axes)
        // but the Module impl for EmbedNd calls rope() per axis.
        let pe = EmbedNd { theta: 10000, axes_dim: vec![16, 16] };
        let ids = Tensor::zeros((1, 20, 2), DType::F32, &candle_core::Device::Cpu).unwrap();
        let out = ids.apply(&pe).unwrap();
        // Output should have proper shape
        assert_eq!(out.dim(0).unwrap(), 1);
    }
}

/// Load an Fp8Linear from VarBuilder (weights stay in native dtype).
fn fp8_linear(_in_d: usize, _out_d: usize, vb: VarBuilder) -> Result<Fp8Linear> {
    let weight = vb.get_unchecked("weight")?;
    let bias = vb.get_unchecked("bias").ok();
    Ok(Fp8Linear::new(weight, bias))
}

fn fp8_linear_b(_in_d: usize, _out_d: usize, bias: bool, vb: VarBuilder) -> Result<Fp8Linear> {
    let weight = vb.get_unchecked("weight")?;
    let bias = if bias {
        Some(vb.get_unchecked("bias")?)
    } else {
        None
    };
    Ok(Fp8Linear::new(weight, bias))
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let ws = Tensor::ones(dim, DType::F16, vb.device())?;
    Ok(LayerNorm::new_no_bias(ws, 1e-6))
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
}

fn rope(pos: &Tensor, dim: usize, theta: usize) -> Result<Tensor> {
    if dim % 2 == 1 {
        candle_core::bail!("dim {dim} is odd")
    }
    let dev = pos.device();
    let theta = theta as f64;
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, 1, inv_freq_len), dev)?;
    let inv_freq = inv_freq.to_dtype(pos.dtype())?;
    let freqs = pos.unsqueeze(2)?.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    let out = Tensor::stack(&[&cos, &sin.neg()?, &sin, &cos], 3)?;
    let (b, n, d, _ij) = out.dims4()?;
    out.reshape((b, n, d, 2, 2))
}

fn apply_rope(x: &Tensor, freq_cis: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let fr0 = freq_cis.get_on_dim(D::Minus1, 0)?;
    let fr1 = freq_cis.get_on_dim(D::Minus1, 1)?;
    (fr0.broadcast_mul(&x0)? + fr1.broadcast_mul(&x1)?)?.reshape(dims.to_vec())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, pe: &Tensor) -> Result<Tensor> {
    let q = apply_rope(q, pe)?.contiguous()?;
    let k = apply_rope(k, pe)?.contiguous()?;
    let x = scaled_dot_product_attention(&q, &k, v)?;
    x.transpose(1, 2)?.flatten_from(2)
}

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

// ── Positional Embeddings ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EmbedNd {
    theta: usize,
    axes_dim: Vec<usize>,
}

impl EmbedNd {
    pub fn new(theta: usize, axes_dim: Vec<usize>) -> Self {
        Self { theta, axes_dim }
    }
}

impl Module for EmbedNd {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let n_axes = ids.dim(D::Minus1)?;
        let mut emb = Vec::with_capacity(n_axes);
        for idx in 0..n_axes {
            let r = rope(
                &ids.get_on_dim(D::Minus1, idx)?,
                self.axes_dim[idx],
                self.theta,
            )?;
            emb.push(r)
        }
        let emb = Tensor::cat(&emb, 2)?;
        emb.unsqueeze(1)
    }
}

// ── MLP Embedder ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MlpEmbedder {
    in_layer: Fp8Linear,
    out_layer: Fp8Linear,
}

impl MlpEmbedder {
    fn new(in_sz: usize, h_sz: usize, vb: VarBuilder) -> Result<Self> {
        let in_layer = fp8_linear(in_sz, h_sz, vb.pp("in_layer"))?;
        let out_layer = fp8_linear(h_sz, h_sz, vb.pp("out_layer"))?;
        Ok(Self {
            in_layer,
            out_layer,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.out_layer.forward(&self.in_layer.forward(xs)?.silu()?)
    }
}

// ── QK Norm ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Norm weights must be in BF16 (not F8) for RmsNorm computation
        let query_norm = vb.get(dim, "query_norm.scale")?.to_dtype(DType::F16)?;
        let query_norm = RmsNorm::new(query_norm, 1e-6);
        let key_norm = vb.get(dim, "key_norm.scale")?.to_dtype(DType::F16)?;
        let key_norm = RmsNorm::new(key_norm, 1e-6);
        Ok(Self {
            query_norm,
            key_norm,
        })
    }
}

// ── Modulation ─────────────────────────────────────────────────────────────────

struct ModulationOut {
    shift: Tensor,
    scale: Tensor,
    gate: Tensor,
}

impl ModulationOut {
    fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.)?)?
            .broadcast_add(&self.shift)
    }

    fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

#[derive(Debug, Clone)]
struct Modulation1 {
    lin: Fp8Linear,
}

impl Modulation1 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = fp8_linear(dim, 3 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<ModulationOut> {
        let ys = self
            .lin
            .forward(&vec_.silu()?)?
            .unsqueeze(1)?
            .chunk(3, D::Minus1)?;
        if ys.len() != 3 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        Ok(ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct Modulation2 {
    lin: Fp8Linear,
}

impl Modulation2 {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let lin = fp8_linear(dim, 6 * dim, vb.pp("lin"))?;
        Ok(Self { lin })
    }

    fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let ys = self
            .lin
            .forward(&vec_.silu()?)?
            .unsqueeze(1)?
            .chunk(6, D::Minus1)?;
        if ys.len() != 6 {
            candle_core::bail!("unexpected len from chunk {ys:?}")
        }
        let mod1 = ModulationOut {
            shift: ys[0].clone(),
            scale: ys[1].clone(),
            gate: ys[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: ys[3].clone(),
            scale: ys[4].clone(),
            gate: ys[5].clone(),
        };
        Ok((mod1, mod2))
    }
}

// ── Self Attention ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SelfAttention {
    qkv: Fp8Linear,
    norm: QkNorm,
    proj: Fp8Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv = fp8_linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let proj = fp8_linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            norm,
            proj,
            num_heads,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let qkv = self.qkv.forward(xs)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        Ok((q, k, v))
    }
}

// ── MLP ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    lin1: Fp8Linear,
    lin2: Fp8Linear,
}

impl Mlp {
    fn new(in_sz: usize, mlp_sz: usize, vb: VarBuilder) -> Result<Self> {
        let lin1 = fp8_linear(in_sz, mlp_sz, vb.pp("0"))?;
        let lin2 = fp8_linear(mlp_sz, in_sz, vb.pp("2"))?;
        Ok(Self { lin1, lin2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.lin2.forward(&self.lin1.forward(xs)?.gelu()?)
    }
}

// ── Double Stream Block ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    img_mod: Modulation2,
    img_norm1: LayerNorm,
    img_attn: SelfAttention,
    img_norm2: LayerNorm,
    img_mlp: Mlp,
    txt_mod: Modulation2,
    txt_norm1: LayerNorm,
    txt_attn: SelfAttention,
    txt_norm2: LayerNorm,
    txt_mlp: Mlp,
}

impl DoubleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let img_mod = Modulation2::new(h_sz, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm(h_sz, vb.pp("img_norm1"))?;
        let img_attn =
            SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("img_attn"))?;
        let img_norm2 = layer_norm(h_sz, vb.pp("img_norm2"))?;
        let img_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("img_mlp"))?;
        let txt_mod = Modulation2::new(h_sz, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm(h_sz, vb.pp("txt_norm1"))?;
        let txt_attn =
            SelfAttention::new(h_sz, cfg.num_heads, cfg.qkv_bias, vb.pp("txt_attn"))?;
        let txt_norm2 = layer_norm(h_sz, vb.pp("txt_norm2"))?;
        let txt_mlp = Mlp::new(h_sz, mlp_sz, vb.pp("txt_mlp"))?;
        Ok(Self {
            img_mod,
            img_norm1,
            img_attn,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_attn,
            txt_norm2,
            txt_mlp,
        })
    }

    fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        vec_: &Tensor,
        pe: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (img_mod1, img_mod2) = self.img_mod.forward(vec_)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(vec_)?;
        let img_modulated = img.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_modulated)?;
        let (img_q, img_k, img_v) = self.img_attn.qkv(&img_modulated)?;

        let txt_modulated = txt.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_modulated)?;
        let (txt_q, txt_k, txt_v) = self.txt_attn.qkv(&txt_modulated)?;

        let q = Tensor::cat(&[txt_q, img_q], 2)?;
        let k = Tensor::cat(&[txt_k, img_k], 2)?;
        let v = Tensor::cat(&[txt_v, img_v], 2)?;

        let attn = attention(&q, &k, &v, pe)?;
        let txt_attn = attn.narrow(1, 0, txt.dim(1)?)?;
        let img_attn = attn.narrow(1, txt.dim(1)?, attn.dim(1)? - txt.dim(1)?)?;

        let img = (img + img_mod1.gate(&self.img_attn.proj.forward(&img_attn)?)?)?;
        let img = (&img
            + img_mod2.gate(
                &self
                    .img_mlp
                    .forward(&img_mod2.scale_shift(&img.apply(&self.img_norm2)?)?)?,
            )?)?;

        let txt = (txt + txt_mod1.gate(&self.txt_attn.proj.forward(&txt_attn)?)?)?;
        let txt = (&txt
            + txt_mod2.gate(
                &self
                    .txt_mlp
                    .forward(&txt_mod2.scale_shift(&txt.apply(&self.txt_norm2)?)?)?,
            )?)?;

        Ok((img, txt))
    }
}

// ── Single Stream Block ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    linear1: Fp8Linear,
    linear2: Fp8Linear,
    norm: QkNorm,
    pre_norm: LayerNorm,
    modulation: Modulation1,
    h_sz: usize,
    mlp_sz: usize,
    num_heads: usize,
}

impl SingleStreamBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h_sz = cfg.hidden_size;
        let mlp_sz = (h_sz as f64 * cfg.mlp_ratio) as usize;
        let head_dim = h_sz / cfg.num_heads;
        let linear1 = fp8_linear(h_sz, h_sz * 3 + mlp_sz, vb.pp("linear1"))?;
        let linear2 = fp8_linear(h_sz + mlp_sz, h_sz, vb.pp("linear2"))?;
        let norm = QkNorm::new(head_dim, vb.pp("norm"))?;
        let pre_norm = layer_norm(h_sz, vb.pp("pre_norm"))?;
        let modulation = Modulation1::new(h_sz, vb.pp("modulation"))?;
        Ok(Self {
            linear1,
            linear2,
            norm,
            pre_norm,
            modulation,
            h_sz,
            mlp_sz,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, vec_: &Tensor, pe: &Tensor) -> Result<Tensor> {
        let mod_ = self.modulation.forward(vec_)?;
        let x_mod = mod_.scale_shift(&xs.apply(&self.pre_norm)?)?;
        let x_mod = self.linear1.forward(&x_mod)?;
        let qkv = x_mod.narrow(D::Minus1, 0, 3 * self.h_sz)?;
        let (b, l, _khd) = qkv.dims3()?;
        let qkv = qkv.reshape((b, l, 3, self.num_heads, ()))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?;
        let mlp = x_mod.narrow(D::Minus1, 3 * self.h_sz, self.mlp_sz)?;
        let q = q.apply(&self.norm.query_norm)?;
        let k = k.apply(&self.norm.key_norm)?;
        let attn = attention(&q, &k, &v, pe)?;
        let output =
            Tensor::cat(&[attn, mlp.gelu()?], 2).and_then(|t| self.linear2.forward(&t))?;
        xs + mod_.gate(&output)
    }
}

// ── Last Layer ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LastLayer {
    norm_final: LayerNorm,
    linear: Fp8Linear,
    ada_ln_modulation: Fp8Linear,
}

impl LastLayer {
    fn new(h_sz: usize, p_sz: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let norm_final = layer_norm(h_sz, vb.pp("norm_final"))?;
        let linear = fp8_linear(h_sz, p_sz * p_sz * out_c, vb.pp("linear"))?;
        let ada_ln_modulation = fp8_linear(h_sz, 2 * h_sz, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    fn forward(&self, xs: &Tensor, vec: &Tensor) -> Result<Tensor> {
        let chunks = self
            .ada_ln_modulation
            .forward(&vec.silu()?)?
            .chunk(2, 1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);
        let xs = xs
            .apply(&self.norm_final)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)?;
        self.linear.forward(&xs)
    }
}

// ── Main Model ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Flux1Transformer {
    img_in: Fp8Linear,
    txt_in: Fp8Linear,
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double_blocks: Vec<DoubleStreamBlock>,
    single_blocks: Vec<SingleStreamBlock>,
    final_layer: LastLayer,
}

impl Flux1Transformer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let img_in = fp8_linear(cfg.in_channels, cfg.hidden_size, vb.pp("img_in"))?;
        let txt_in = fp8_linear(cfg.context_in_dim, cfg.hidden_size, vb.pp("txt_in"))?;
        let mut double_blocks = Vec::with_capacity(cfg.depth);
        let vb_d = vb.pp("double_blocks");
        for idx in 0..cfg.depth {
            let db = DoubleStreamBlock::new(cfg, vb_d.pp(idx))?;
            double_blocks.push(db)
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single_blocks);
        let vb_s = vb.pp("single_blocks");
        for idx in 0..cfg.depth_single_blocks {
            let sb = SingleStreamBlock::new(cfg, vb_s.pp(idx))?;
            single_blocks.push(sb)
        }
        let time_in = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("time_in"))?;
        let vector_in = MlpEmbedder::new(cfg.vec_in_dim, cfg.hidden_size, vb.pp("vector_in"))?;
        let guidance_in = if cfg.guidance_embed {
            let mlp = MlpEmbedder::new(256, cfg.hidden_size, vb.pp("guidance_in"))?;
            Some(mlp)
        } else {
            None
        };
        let final_layer =
            LastLayer::new(cfg.hidden_size, 1, cfg.in_channels, vb.pp("final_layer"))?;
        let pe_dim = cfg.hidden_size / cfg.num_heads;
        let pe_embedder = EmbedNd::new(cfg.theta, cfg.axes_dim.to_vec());
        let _ = pe_dim;
        Ok(Self {
            img_in,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_blocks,
            single_blocks,
            final_layer,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor> {
        if txt.rank() != 3 {
            candle_core::bail!("unexpected shape for txt {:?}", txt.shape())
        }
        if img.rank() != 3 {
            candle_core::bail!("unexpected shape for img {:?}", img.shape())
        }
        let dtype = img.dtype();
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?.to_dtype(dtype)?
        };
        let mut txt = self.txt_in.forward(txt)?;
        let mut img = self.img_in.forward(img)?;
        let vec_ = timestep_embedding(timesteps, 256, dtype)
            .and_then(|t| self.time_in.forward(&t))?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(guidance)) => {
                (vec_ + timestep_embedding(guidance, 256, dtype)
                    .and_then(|t| g_in.forward(&t))?)?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + self.vector_in.forward(y)?)?;

        // Double blocks
        for block in self.double_blocks.iter() {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?
        }
        // Single blocks
        let mut img = Tensor::cat(&[&txt, &img], 1)?;
        for block in self.single_blocks.iter() {
            img = block.forward(&img, &vec_, &pe)?;
        }
        let img = img.i((.., txt.dim(1)?..))?;
        self.final_layer.forward(&img, &vec_)
    }
}
