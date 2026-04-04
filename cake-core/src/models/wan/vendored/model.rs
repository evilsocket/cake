use anyhow::Result;
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use log::info;

use super::adaln::{fp32_layer_norm, modulate};
use super::config::WanTransformerConfig;
use super::rope::precompute_wan_rope_3d;
use super::transformer_block::WanTransformerBlock;

const EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

/// Wan2.2 DiT transformer model.
///
/// Patch embedding -> timestep/text embedding -> N transformer blocks -> output head.
#[derive(Debug)]
pub struct WanModel {
    /// Conv3d-equivalent patch embedding: Linear(in_channels -> hidden_size)
    /// Actually Conv3d(16, 5120, (1,2,2), stride=(1,2,2)) but stored as weight+bias
    patch_embedding_weight: Tensor,
    patch_embedding_bias: Tensor,
    /// Text embedding: Linear(text_dim, dim) -> GELU -> Linear(dim, dim)
    text_embed_0: Linear,
    text_embed_2: Linear,
    /// Timestep embedding: Linear(freq_dim, dim) -> SiLU -> Linear(dim, dim)
    time_embed_0: Linear,
    time_embed_2: Linear,
    /// Time projection: SiLU -> Linear(dim, dim*6)
    time_proj_1: Linear,
    /// Transformer blocks
    blocks: Vec<WanTransformerBlock>,
    /// Output head
    head_weight: Linear,
    head_modulation: Tensor,
    /// Config
    cfg: WanTransformerConfig,
    /// Block range for distributed inference (None = all blocks)
    block_range: Option<(usize, usize)>,
}

impl WanModel {
    pub fn load(vb: VarBuilder, cfg: &WanTransformerConfig) -> Result<Self> {
        Self::load_block_range(vb, cfg, None)
    }

    pub fn load_block_range(
        vb: VarBuilder,
        cfg: &WanTransformerConfig,
        block_range: Option<(usize, usize)>,
    ) -> Result<Self> {
        let dim = cfg.hidden_size;
        let (start, end) = block_range.unwrap_or((0, cfg.num_layers));

        // Patch embedding (Conv3d stored as weight [out, in, 1, 2, 2] and bias [out])
        let pe_vb = vb.pp("patch_embedding");
        let patch_embedding_weight = pe_vb.get((dim, cfg.in_channels, 1, 2, 2), "weight")?;
        let patch_embedding_bias = pe_vb.get(dim, "bias")?;

        // Text embedding MLP
        let te_vb = vb.pp("text_embedding");
        let text_embed_0 = candle_nn::linear(cfg.text_dim, dim, te_vb.pp("0"))?;
        let text_embed_2 = candle_nn::linear(dim, dim, te_vb.pp("2"))?;

        // Timestep embedding MLP
        let ts_vb = vb.pp("time_embedding");
        let time_embed_0 = candle_nn::linear(cfg.freq_dim, dim, ts_vb.pp("0"))?;
        let time_embed_2 = candle_nn::linear(dim, dim, ts_vb.pp("2"))?;

        // Time projection
        let tp_vb = vb.pp("time_projection");
        let time_proj_1 = candle_nn::linear(dim, dim * 6, tp_vb.pp("1"))?;

        // Transformer blocks
        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(end - start);
        for i in start..end {
            info!("loading transformer block {}/{}", i + 1, cfg.num_layers);
            let block = WanTransformerBlock::load(blocks_vb.pp(i), cfg)?;
            blocks.push(block);
        }

        // Output head
        let head_vb = vb.pp("head");
        let (pt, ph, pw) = cfg.patch_size();
        let out_features = cfg.out_channels * pt * ph * pw; // 16 * 1 * 2 * 2 = 64
        let head_weight = candle_nn::linear(dim, out_features, head_vb.pp("head"))?;
        let head_modulation = head_vb.get((1, 2, dim), "modulation")?;

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
            block_range,
        })
    }

    /// Setup phase: patch embed, timestep embed, text embed, RoPE.
    /// Returns (hidden, temb, timestep_proj, context, rope_cos, rope_sin).
    pub fn forward_setup(
        &self,
        latents: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let device = latents.device();
        let (_, ph, pw) = self.cfg.patch_size();

        // 1. Patch embedding via reshape + linear (simulating Conv3d with kernel (1,2,2), stride (1,2,2))
        // latents: [B, C, F, H, W] -> reshape to [B, F, H/2, W/2, C*1*2*2] -> linear to [B, S, dim]
        let (b_lat, c_lat, f_lat, h_lat, w_lat) = latents.dims5()?;
        let h_pat = h_lat / ph;
        let w_pat = w_lat / pw;
        // Reshape: [B, C, F, H, W] -> [B, C, F, H/2, 2, W/2, 2]
        let hidden = latents.reshape(&[b_lat, c_lat, f_lat, h_pat, ph, w_pat, pw])?;
        // Permute: [B, F, H/2, W/2, C, 2, 2] -> flatten last 3 dims
        let hidden = hidden.permute(vec![0, 2, 3, 5, 1, 4, 6])?;
        let hidden = hidden.reshape((b_lat, f_lat * h_pat * w_pat, c_lat * ph * pw))?;
        // Linear projection: [B, S, C*4] -> [B, S, dim]
        // Reshape conv weight [dim, C, 1, 2, 2] -> [dim, C*4] for matmul
        let w = self.patch_embedding_weight.reshape((self.cfg.hidden_size, c_lat * ph * pw))?;
        let hidden = hidden.to_dtype(w.dtype())?.broadcast_matmul(&w.t()?)?;
        let hidden = hidden.broadcast_add(&self.patch_embedding_bias)?;
        // hidden is now [B, F*H_pat*W_pat, dim] — already the correct shape
        let b = b_lat;
        let (f, h, w) = (f_lat, h_pat, w_pat); // patched spatial dims for RoPE
        log::debug!("patch embed done: hidden={:?}, b={b}", hidden.shape());

        // 2. Timestep embedding
        let t_emb = sinusoidal_embedding(timestep, self.cfg.freq_dim, device)?;
        let t_emb = t_emb.to_dtype(hidden.dtype())?;
        let temb = self.time_embed_0.forward(&t_emb)?.silu()?;
        let temb = self.time_embed_2.forward(&temb)?; // [B, dim]
        log::debug!("time embed done: temb={:?}", temb.shape());

        // 3. Timestep projection -> 6*dim for modulation
        let timestep_proj = temb.silu()?.apply(&self.time_proj_1)?; // [B, dim*6]
        let timestep_proj = timestep_proj.reshape((b, 6, self.cfg.hidden_size))?; // [B, 6, dim]

        // 4. Text embedding (ensure context matches weight dtype)
        let context = context.to_dtype(hidden.dtype())?;
        let context = self.text_embed_0.forward(&context)?;
        let context = context.gelu_erf()?;
        let context = self.text_embed_2.forward(&context)?; // [B, L, dim]
        log::debug!("text embed done: context={:?}, timestep_proj={:?}", context.shape(), timestep_proj.shape());

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
        // height/width here are already in latent space (divided by spatial_compression=8)
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
        let (pt, _, _) = self.cfg.patch_size(); // pt=1
        // Step 1: [B, F*H'*W', C*pt*ph*pw] -> [B, F, H', W', pt, ph, pw, C]
        let hidden = hidden.reshape(&[b, num_frames, h_pat, w_pat, pt, ph, pw, c])?;
        // Step 2: permute to [B, C, F, pt, H', ph, W', pw]
        let hidden = hidden.permute(vec![0, 7, 1, 4, 2, 5, 3, 6])?;
        // Step 3: merge: flatten (F, pt) -> F*pt, (H', ph) -> H, (W', pw) -> W
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

/// Sinusoidal timestep embedding (matching diffusers' get_timestep_embedding).
fn sinusoidal_embedding(
    timestep: &Tensor,
    dim: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
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
