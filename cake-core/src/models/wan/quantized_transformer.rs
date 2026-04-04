//! Quantized Wan DiT transformer that loads from GGUF.
//!
//! Replaces all large Linear weights with QMatMul to keep the 14B model
//! (9.2 GB Q4_K_S) on a 24 GB GPU. Small tensors (biases, norms, modulation,
//! patch_embedding) are dequantized to F16.
//!
//! The forward pass is identical to `WanModel` in `vendored/model.rs`.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Tensor, D};
use log::info;
use std::path::Path;

use super::vendored::adaln::modulate;
use super::vendored::config::WanTransformerConfig;
use super::vendored::rope::{apply_wan_rope, precompute_wan_rope_3d};

const EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

// ---------------------------------------------------------------------------
// QMatMul wrapper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> candle_core::Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs32 = xs.to_dtype(DType::F32)?;
        self.inner.forward(&xs32)
    }
}

// ---------------------------------------------------------------------------
// QLinear: QMatMul + dequantized bias
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QLinear {
    weight: QMatMul,
    bias: Tensor,
}

impl QLinear {
    fn new(weight: QTensor, bias: Tensor) -> candle_core::Result<Self> {
        Ok(Self {
            weight: QMatMul::from_qtensor(weight)?,
            bias,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // QMatMul operates in F32 — keep F32 throughout to avoid truncation
        let xs32 = xs.to_dtype(DType::F32)?;
        let out = self.weight.forward(&xs32)?;
        let bias32 = self.bias.to_dtype(DType::F32)?;
        out.broadcast_add(&bias32)
    }
}

// ---------------------------------------------------------------------------
// RmsNorm (weight-only, no bias) -- matches WanRmsNorm from vendored/attention.rs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QWanRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QWanRmsNorm {
    fn from_tensor(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        Ok(x_normed.broadcast_mul(&w)?)
    }
}

// ---------------------------------------------------------------------------
// LayerNorm (weight + bias) -- matches norm3 in WanTransformerBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl QLayerNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        let b = self.bias.to_dtype(DType::F32)?;
        Ok(x.broadcast_mul(&w)?.broadcast_add(&b)?)
    }
}

// ---------------------------------------------------------------------------
// Quantized attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QWanAttention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    norm_q: QWanRmsNorm,
    norm_k: QWanRmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl QWanAttention {
    fn forward_self(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // RMSNorm before reshape (across all heads)
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Reshape to [B, S, H, D]
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, s, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, s, self.num_heads, self.head_dim))?;

        // Apply RoPE
        let q = apply_wan_rope(&q, rope_cos, rope_sin)?;
        let k = apply_wan_rope(&k, rope_cos, rope_sin)?;

        // Transpose to [B, H, S, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let y = self.sdpa(&q, &k, &v)?;

        let y = y.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&y)?)
    }

    fn forward_cross(
        &self,
        x: &Tensor,
        context: &Tensor,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let ctx_len = context.dim(1)?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, ctx_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, ctx_len, self.num_heads, self.head_dim))?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let y = self.sdpa(&q, &k, &v)?;

        let y = y.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&y)?)
    }

    fn sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // Flash attention requires F16 — skip for quantized model to stay in F32
        // (flash attn introduces F16 truncation that compounds over 30 denoising steps)
        let scale = (self.head_dim as f64).sqrt();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        Ok(att.matmul(&v)?)
    }
}

// ---------------------------------------------------------------------------
// Quantized feed-forward
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QWanFeedForward {
    linear_in: QLinear,
    linear_out: QLinear,
}

impl QWanFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_in.forward(x)?;
        let x = x.gelu_erf()?;
        Ok(self.linear_out.forward(&x)?)
    }
}

// ---------------------------------------------------------------------------
// Quantized transformer block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QWanTransformerBlock {
    self_attn: QWanAttention,
    cross_attn: QWanAttention,
    ffn: QWanFeedForward,
    norm3: QLayerNorm,
    modulation: Tensor,
}

impl QWanTransformerBlock {
    fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        timestep_proj: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        // Add per-block modulation bias to timestep projection
        let mod_params = (timestep_proj + &self.modulation)?; // [B, 6, dim]

        // Split into 6 modulation vectors
        let shift_sa = mod_params.narrow(1, 0, 1)?.squeeze(1)?;
        let scale_sa = mod_params.narrow(1, 1, 1)?.squeeze(1)?;
        let gate_sa = mod_params.narrow(1, 2, 1)?.squeeze(1)?;
        let shift_ff = mod_params.narrow(1, 3, 1)?.squeeze(1)?;
        let scale_ff = mod_params.narrow(1, 4, 1)?.squeeze(1)?;
        let gate_ff = mod_params.narrow(1, 5, 1)?.squeeze(1)?;

        // 1. Self-attention with AdaLN
        let x_mod = modulate(x, &shift_sa.unsqueeze(1)?, &scale_sa.unsqueeze(1)?, EPS)?;
        let attn_out = self.self_attn.forward_self(&x_mod, rope_cos, rope_sin)?;
        let x = (x + attn_out.broadcast_mul(&gate_sa.unsqueeze(1)?)?)?;

        // 2. Cross-attention
        let x_normed = self.norm3.forward(&x)?;
        let cross_out = self.cross_attn.forward_cross(&x_normed, context)?;
        let x = (x + cross_out)?;

        // 3. FFN with AdaLN
        let x_mod = modulate(&x, &shift_ff.unsqueeze(1)?, &scale_ff.unsqueeze(1)?, EPS)?;
        let ff_out = self.ffn.forward(&x_mod)?;
        let x = (x + ff_out.broadcast_mul(&gate_ff.unsqueeze(1)?)?)?;

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal timestep embedding (matches vendored/model.rs)
// ---------------------------------------------------------------------------

fn sinusoidal_embedding(
    timestep: &Tensor,
    dim: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let half = dim / 2;
    let max_period: f64 = 10000.0;

    let exponent: Vec<f32> = (0..half)
        .map(|i| -(max_period.ln() as f32) * (i as f32) / (half as f32))
        .collect();
    let exponent = Tensor::new(exponent.as_slice(), device)?;
    let freqs = exponent.exp()?;

    let t = timestep.to_dtype(DType::F32)?;
    let args = t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;

    // flip_sin_to_cos=True: cos first, then sin
    let embedding = Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)?;

    Ok(embedding)
}

// ---------------------------------------------------------------------------
// Helper: dequantize a QTensor to F16 on device
// ---------------------------------------------------------------------------

fn deq_f16(qt: QTensor, device: &Device) -> candle_core::Result<Tensor> {
    qt.dequantize(device)?.to_dtype(DType::F32)
}

// ---------------------------------------------------------------------------
// QuantizedWanModel
// ---------------------------------------------------------------------------

/// Quantized Wan2.2 DiT transformer loaded from GGUF.
///
/// All large weight matrices use QMatMul; small tensors (bias, norms,
/// modulation, patch_embedding) are dequantized to F16 at load time.
#[derive(Debug)]
pub struct QuantizedWanModel {
    // Patch embedding (Conv3d stored as weight + bias)
    patch_embedding_weight: Tensor,
    patch_embedding_bias: Tensor,
    // Text embedding MLP: Linear -> GELU -> Linear
    text_embed_0: QLinear,
    text_embed_2: QLinear,
    // Timestep embedding MLP: Linear -> SiLU -> Linear
    time_embed_0: QLinear,
    time_embed_2: QLinear,
    // Time projection: SiLU -> Linear(dim, dim*6)
    time_proj_1: QLinear,
    // Transformer blocks
    blocks: Vec<QWanTransformerBlock>,
    // Output head
    head_weight: QLinear,
    head_modulation: Tensor,
    // Config
    cfg: WanTransformerConfig,
}

impl QuantizedWanModel {
    /// Load the quantized Wan DiT from a GGUF file.
    pub fn load_from_gguf(path: &Path, cfg: &WanTransformerConfig, device: &Device) -> Result<Self> {
        let mut reader = std::io::BufReader::new(
            std::fs::File::open(path)
                .map_err(|e| anyhow::anyhow!("can't open GGUF {}: {e}", path.display()))?,
        );
        let ct = gguf_file::Content::read(&mut reader)
            .map_err(|e| anyhow::anyhow!("can't read GGUF {}: {e}", path.display()))?;

        Self::load_from_gguf_content(ct, &mut reader, cfg, device)
    }

    /// Load from already-parsed GGUF Content.
    pub fn load_from_gguf_content<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        cfg: &WanTransformerConfig,
        device: &Device,
    ) -> Result<Self> {
        let dim = cfg.hidden_size;

        // --- Patch embedding ---
        // [dim, in_channels, 1, 2, 2] -- small, dequantize to F16
        let patch_embedding_weight = deq_f16(
            ct.tensor(reader, "patch_embedding.weight", device)?,
            device,
        )?;
        let patch_embedding_bias = deq_f16(
            ct.tensor(reader, "patch_embedding.bias", device)?,
            device,
        )?;

        // --- Text embedding ---
        let text_embed_0 = QLinear::new(
            ct.tensor(reader, "text_embedding.0.weight", device)?,
            deq_f16(ct.tensor(reader, "text_embedding.0.bias", device)?, device)?,
        )?;
        let text_embed_2 = QLinear::new(
            ct.tensor(reader, "text_embedding.2.weight", device)?,
            deq_f16(ct.tensor(reader, "text_embedding.2.bias", device)?, device)?,
        )?;

        // --- Time embedding ---
        let time_embed_0 = QLinear::new(
            ct.tensor(reader, "time_embedding.0.weight", device)?,
            deq_f16(ct.tensor(reader, "time_embedding.0.bias", device)?, device)?,
        )?;
        let time_embed_2 = QLinear::new(
            ct.tensor(reader, "time_embedding.2.weight", device)?,
            deq_f16(ct.tensor(reader, "time_embedding.2.bias", device)?, device)?,
        )?;

        // --- Time projection ---
        let time_proj_1 = QLinear::new(
            ct.tensor(reader, "time_projection.1.weight", device)?,
            deq_f16(ct.tensor(reader, "time_projection.1.bias", device)?, device)?,
        )?;

        // --- Transformer blocks ---
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            info!("loading quantized transformer block {}/{}", i + 1, cfg.num_layers);
            let pfx = format!("blocks.{i}");

            // Self-attention
            let self_attn = Self::load_attention(&ct, reader, &format!("{pfx}.self_attn"), dim, cfg.num_attention_heads, device)?;
            // Cross-attention
            let cross_attn = Self::load_attention(&ct, reader, &format!("{pfx}.cross_attn"), dim, cfg.num_attention_heads, device)?;

            // FFN
            let ffn = QWanFeedForward {
                linear_in: QLinear::new(
                    ct.tensor(reader, &format!("{pfx}.ffn.0.weight"), device)?,
                    deq_f16(ct.tensor(reader, &format!("{pfx}.ffn.0.bias"), device)?, device)?,
                )?,
                linear_out: QLinear::new(
                    ct.tensor(reader, &format!("{pfx}.ffn.2.weight"), device)?,
                    deq_f16(ct.tensor(reader, &format!("{pfx}.ffn.2.bias"), device)?, device)?,
                )?,
            };

            // norm3: LayerNorm
            let norm3 = QLayerNorm {
                weight: deq_f16(ct.tensor(reader, &format!("{pfx}.norm3.weight"), device)?, device)?,
                bias: deq_f16(ct.tensor(reader, &format!("{pfx}.norm3.bias"), device)?, device)?,
                eps: EPS,
            };

            // Modulation: [1, 6, dim] -- small, dequantize
            let modulation = deq_f16(
                ct.tensor(reader, &format!("{pfx}.modulation"), device)?,
                device,
            )?;

            blocks.push(QWanTransformerBlock {
                self_attn,
                cross_attn,
                ffn,
                norm3,
                modulation,
            });
        }

        // --- Output head ---
        let head_weight = QLinear::new(
            ct.tensor(reader, "head.head.weight", device)?,
            deq_f16(ct.tensor(reader, "head.head.bias", device)?, device)?,
        )?;
        let head_modulation = deq_f16(
            ct.tensor(reader, "head.modulation", device)?,
            device,
        )?;

        info!(
            "Quantized WanModel loaded: {} layers, dim={}, heads={}, ffn_dim={}",
            cfg.num_layers, dim, cfg.num_attention_heads, cfg.ffn_dim,
        );

        Ok(Self {
            patch_embedding_weight,
            patch_embedding_bias,
            text_embed_0,
            text_embed_2,
            time_embed_0,
            time_embed_2,
            time_proj_1,
            blocks,
            head_weight,
            head_modulation,
            cfg: cfg.clone(),
        })
    }

    /// Load a QWanAttention from GGUF tensors under `prefix`.
    fn load_attention<R: std::io::Seek + std::io::Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        prefix: &str,
        dim: usize,
        num_heads: usize,
        device: &Device,
    ) -> Result<QWanAttention> {
        let head_dim = dim / num_heads;

        let q_proj = QLinear::new(
            ct.tensor(reader, &format!("{prefix}.q.weight"), device)?,
            deq_f16(ct.tensor(reader, &format!("{prefix}.q.bias"), device)?, device)?,
        )?;
        let k_proj = QLinear::new(
            ct.tensor(reader, &format!("{prefix}.k.weight"), device)?,
            deq_f16(ct.tensor(reader, &format!("{prefix}.k.bias"), device)?, device)?,
        )?;
        let v_proj = QLinear::new(
            ct.tensor(reader, &format!("{prefix}.v.weight"), device)?,
            deq_f16(ct.tensor(reader, &format!("{prefix}.v.bias"), device)?, device)?,
        )?;
        let o_proj = QLinear::new(
            ct.tensor(reader, &format!("{prefix}.o.weight"), device)?,
            deq_f16(ct.tensor(reader, &format!("{prefix}.o.bias"), device)?, device)?,
        )?;

        let norm_q = QWanRmsNorm::from_tensor(
            deq_f16(ct.tensor(reader, &format!("{prefix}.norm_q.weight"), device)?, device)?,
            EPS,
        );
        let norm_k = QWanRmsNorm::from_tensor(
            deq_f16(ct.tensor(reader, &format!("{prefix}.norm_k.weight"), device)?, device)?,
            EPS,
        );

        Ok(QWanAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            norm_q,
            norm_k,
            num_heads,
            head_dim,
        })
    }

    // -----------------------------------------------------------------------
    // Forward pass (identical structure to WanModel)
    // -----------------------------------------------------------------------

    /// Setup phase: patch embed, timestep embed, text embed, RoPE.
    /// Returns (hidden, temb, timestep_proj, context, rope_cos, rope_sin).
    pub fn forward_setup(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        num_frames: usize,
        _height: usize,
        _width: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let device = latents.device();
        let (_, ph, pw) = self.cfg.patch_size();

        // 1. Patch embedding via reshape + linear (Conv3d with kernel (1,2,2), stride (1,2,2))
        let (b_lat, c_lat, f_lat, h_lat, w_lat) = latents.dims5()?;
        let h_pat = h_lat / ph;
        let w_pat = w_lat / pw;
        // Reshape: [B, C, F, H, W] -> [B, C, F, H/2, 2, W/2, 2]
        let hidden = latents.reshape(&[b_lat, c_lat, f_lat, h_pat, ph, w_pat, pw])?;
        // Permute: [B, F, H/2, W/2, C, 2, 2] -> flatten last 3 dims
        let hidden = hidden.permute(vec![0, 2, 3, 5, 1, 4, 6])?;
        let hidden = hidden.reshape((b_lat, f_lat * h_pat * w_pat, c_lat * ph * pw))?;
        // Linear projection
        // All computation in F32 to avoid truncation error accumulation
        let w = self.patch_embedding_weight.to_dtype(DType::F32)?.reshape((self.cfg.hidden_size, c_lat * ph * pw))?;
        let hidden = hidden.to_dtype(DType::F32)?.broadcast_matmul(&w.t()?)?;
        let hidden = hidden.broadcast_add(&self.patch_embedding_bias.to_dtype(DType::F32)?)?;
        let b = b_lat;
        let (f, h, w) = (f_lat, h_pat, w_pat);

        // 2. Timestep embedding
        let t_emb = sinusoidal_embedding(timestep, self.cfg.freq_dim, device)?;
        let t_emb = t_emb.to_dtype(hidden.dtype())?;
        let temb = self.time_embed_0.forward(&t_emb)?.silu()?;
        let temb = self.time_embed_2.forward(&temb)?; // [B, dim]

        // 3. Timestep projection -> 6*dim for modulation
        let timestep_proj = self.time_proj_1.forward(&temb.silu()?)?; // [B, dim*6]
        let timestep_proj = timestep_proj.reshape((b, 6, self.cfg.hidden_size))?; // [B, 6, dim]

        // 4. Text embedding
        let context = context.to_dtype(hidden.dtype())?;
        let context = self.text_embed_0.forward(&context)?;
        let context = context.gelu_erf()?;
        let context = self.text_embed_2.forward(&context)?; // [B, L, dim]

        // 5. 3D RoPE
        let (t_dim, h_dim, w_dim) = self.cfg.rope_dims();
        let (rope_cos, rope_sin) = precompute_wan_rope_3d(
            f, h, w, t_dim, h_dim, w_dim, ROPE_THETA, device,
        )?;

        Ok((hidden, temb, timestep_proj, context, rope_cos, rope_sin))
    }

    /// Run transformer blocks.
    pub fn forward_blocks(
        &self,
        mut hidden: Tensor,
        context: &Tensor,
        timestep_proj: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
    ) -> Result<Tensor> {
        for block in &self.blocks {
            hidden = block.forward(&hidden, context, timestep_proj, rope_cos, rope_sin)?;
        }
        Ok(hidden)
    }

    /// Finalize: AdaLN + output projection + unpatchify.
    pub fn forward_finalize(
        &self,
        hidden: &Tensor,
        temb: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let (_, ph, pw) = self.cfg.patch_size();
        let h_pat = height / ph;
        let w_pat = width / pw;

        // Output AdaLN
        let head_mod = self.head_modulation.broadcast_add(&temb.unsqueeze(1)?)?; // [B, 2, dim]
        let shift = head_mod.narrow(1, 0, 1)?.squeeze(1)?; // [B, dim]
        let scale = head_mod.narrow(1, 1, 1)?.squeeze(1)?;

        let hidden = modulate(hidden, &shift.unsqueeze(1)?, &scale.unsqueeze(1)?, EPS)?;
        let hidden = self.head_weight.forward(&hidden)?; // [B, S, C*ph*pw]

        // Unpatchify: [B, F*H'*W', C*pt*ph*pw] -> [B, C, F, H, W]
        // Must match Python: reshape to [B, F, H', W', pt, ph, pw, C] then permute
        let b = hidden.dim(0)?;
        let c = self.cfg.out_channels;
        let (pt, _, _) = self.cfg.patch_size();
        let hidden = hidden.reshape(&[b, num_frames, h_pat, w_pat, pt, ph, pw, c])?;
        let hidden = hidden.permute(vec![0, 7, 1, 4, 2, 5, 3, 6])?;
        let hidden = hidden.reshape((b, c, num_frames * pt, h_pat * ph, w_pat * pw))?;

        Ok(hidden)
    }

    /// Full forward pass.
    pub fn forward(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let (hidden, temb, timestep_proj, context, rope_cos, rope_sin) =
            self.forward_setup(latents, timestep, context, num_frames, height, width)?;

        let hidden = self.forward_blocks(hidden, &context, &timestep_proj, &rope_cos, &rope_sin)?;

        self.forward_finalize(&hidden, &temb, num_frames, height, width)
    }
}
