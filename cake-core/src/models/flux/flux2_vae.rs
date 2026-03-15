//! Custom FLUX.2 VAE decoder matching `AutoencoderKLFlux2` weight layout.
//!
//! The standard SD `AutoEncoderKL` doesn't match FLUX.2's decoder structure.
//! This implements the decoder from scratch using candle-nn primitives.

use candle_core::{Module, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::VarBuilder;

// ── ResNet Block ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2d,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2d,
    conv_shortcut: Option<nn::Conv2d>,
}

impl ResnetBlock2D {
    fn load(vb: VarBuilder, in_ch: usize, out_ch: usize, groups: usize) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig { padding: 1, ..Default::default() };
        let norm1 = nn::group_norm(groups, in_ch, 1e-6, vb.pp("norm1"))?;
        let conv1 = nn::conv2d(in_ch, out_ch, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = nn::group_norm(groups, out_ch, 1e-6, vb.pp("norm2"))?;
        let conv2 = nn::conv2d(out_ch, out_ch, 3, conv_cfg, vb.pp("conv2"))?;
        let conv_shortcut = if in_ch != out_ch {
            let sc_cfg = nn::Conv2dConfig { padding: 0, ..Default::default() };
            Some(nn::conv2d(in_ch, out_ch, 1, sc_cfg, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self { norm1, conv1, norm2, conv2, conv_shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = match &self.conv_shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        let h = self.norm1.forward(x)?;
        let h = nn::ops::silu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        let h = nn::ops::silu(&h)?;
        let h = self.conv2.forward(&h)?;
        h + residual
    }
}

// ── Attention Block ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct AttnBlock {
    group_norm: nn::GroupNorm,
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    channels: usize,
}

impl AttnBlock {
    fn load(vb: VarBuilder, channels: usize, groups: usize) -> Result<Self> {
        let group_norm = nn::group_norm(groups, channels, 1e-6, vb.pp("group_norm"))?;
        let to_q = nn::linear(channels, channels, vb.pp("to_q"))?;
        let to_k = nn::linear(channels, channels, vb.pp("to_k"))?;
        let to_v = nn::linear(channels, channels, vb.pp("to_v"))?;
        let to_out = nn::linear(channels, channels, vb.pp("to_out").pp("0"))?;
        Ok(Self { group_norm, to_q, to_k, to_v, to_out, channels })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, c, h, w) = x.dims4()?;

        let x = self.group_norm.forward(x)?;
        // Reshape to (b, c, h*w) → (b, h*w, c)
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?;

        let q = self.to_q.forward(&x)?;
        let k = self.to_k.forward(&x)?;
        let v = self.to_v.forward(&x)?;

        // Single-head attention
        let scale = 1.0 / (self.channels as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = self.to_out.forward(&out)?;
        // Reshape back: (b, h*w, c) → (b, c, h, w)
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;
        out + residual
    }
}

// ── Up Decoder Block ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct UpDecoderBlock {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<nn::Conv2d>,
}

impl UpDecoderBlock {
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        num_layers: usize,
        groups: usize,
        add_upsample: bool,
    ) -> Result<Self> {
        let vb_r = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let r_in = if i == 0 { in_ch } else { out_ch };
            resnets.push(ResnetBlock2D::load(vb_r.pp(i), r_in, out_ch, groups)?);
        }
        let upsampler = if add_upsample {
            let conv_cfg = nn::Conv2dConfig { padding: 1, ..Default::default() };
            Some(nn::conv2d(out_ch, out_ch, 3, conv_cfg, vb.pp("upsamplers").pp("0").pp("conv"))?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(up) = &self.upsampler {
            let (b, c, h, w) = x.dims4()?;
            // Nearest-neighbor 2x upsample
            x = x.upsample_nearest2d(h * 2, w * 2)?;
            x = up.forward(&x)?;
        }
        Ok(x)
    }
}

// ── Mid Block ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct MidBlock {
    resnet0: ResnetBlock2D,
    attn: AttnBlock,
    resnet1: ResnetBlock2D,
}

impl MidBlock {
    fn load(vb: VarBuilder, channels: usize, groups: usize) -> Result<Self> {
        let resnet0 = ResnetBlock2D::load(vb.pp("resnets").pp("0"), channels, channels, groups)?;
        let attn = AttnBlock::load(vb.pp("attentions").pp("0"), channels, groups)?;
        let resnet1 = ResnetBlock2D::load(vb.pp("resnets").pp("1"), channels, channels, groups)?;
        Ok(Self { resnet0, attn, resnet1 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

// ── Full Decoder ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Flux2Decoder {
    conv_in: nn::Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
}

impl Flux2Decoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let groups = 32;
        // block_out_channels = [128, 256, 512, 512], reversed for decoder
        let block_channels: Vec<(usize, usize)> = vec![
            (512, 512),   // up_block 0: in=512, out=512, upsample
            (512, 512),   // up_block 1: in=512, out=512, upsample
            (512, 256),   // up_block 2: in=512, out=256, upsample
            (256, 128),   // up_block 3: in=256, out=128, no upsample
        ];
        let num_layers = 3; // layers_per_block + 1

        let conv_cfg = nn::Conv2dConfig { padding: 1, ..Default::default() };
        let conv_in = nn::conv2d(32, 512, 3, conv_cfg, vb.pp("conv_in"))?;
        let mid_block = MidBlock::load(vb.pp("mid_block"), 512, groups)?;

        let mut up_blocks = Vec::new();
        let vb_up = vb.pp("up_blocks");
        for (i, (in_ch, out_ch)) in block_channels.iter().enumerate() {
            let add_upsample = i < 3; // first 3 blocks have upsampler
            up_blocks.push(UpDecoderBlock::load(
                vb_up.pp(i), *in_ch, *out_ch, num_layers, groups, add_upsample,
            )?);
        }

        let conv_norm_out = nn::group_norm(groups, 128, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = nn::conv2d(128, 3, 3, conv_cfg, vb.pp("conv_out"))?;

        Ok(Self { conv_in, mid_block, up_blocks, conv_norm_out, conv_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        x = self.mid_block.forward(&x)?;
        for up_block in &self.up_blocks {
            x = up_block.forward(&x)?;
        }
        let x = self.conv_norm_out.forward(&x)?;
        let x = nn::ops::silu(&x)?;
        self.conv_out.forward(&x)
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Custom FLUX.2 VAE that matches AutoencoderKLFlux2 weight layout.
#[derive(Debug)]
pub struct Flux2VAE {
    decoder: Flux2Decoder,
    post_quant_conv: nn::Conv2d,
}

impl Flux2VAE {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let decoder = Flux2Decoder::load(vb.pp("decoder"))?;
        let conv_cfg = nn::Conv2dConfig { padding: 0, ..Default::default() };
        let post_quant_conv = nn::conv2d(32, 32, 1, conv_cfg, vb.pp("post_quant_conv"))?;
        Ok(Self { decoder, post_quant_conv })
    }

    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(latent)?;
        self.decoder.forward(&x)
    }

}
