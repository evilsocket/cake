//! Custom FLUX.2 VAE decoder matching `AutoencoderKLFlux2` weight layout.
//!
//! The standard SD `AutoEncoderKL` doesn't match FLUX.2's decoder structure.
//! This implements the decoder from scratch using backend primitives.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

use crate::backends::ComputeBackend;

// ── Conv2d helper ─────────────────────────────────────────────────────────────

/// Raw conv2d storage — weight + optional bias + config.
#[derive(Debug, Clone)]
struct RawConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
}

impl RawConv2d {
    fn load(in_ch: usize, out_ch: usize, kernel: usize, padding: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_ch, in_ch, kernel, kernel), "weight")?;
        let bias = vb.get(out_ch, "bias").ok();
        Ok(Self { weight, bias, padding })
    }

    fn forward(&self, x: &Tensor, backend: &dyn ComputeBackend) -> Result<Tensor> {
        backend.conv2d(x, &self.weight, self.bias.as_ref(), self.padding, 1, 1, 1)
    }
}

// ── ResNet Block ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    conv1: RawConv2d,
    norm2_weight: Tensor,
    norm2_bias: Tensor,
    conv2: RawConv2d,
    conv_shortcut: Option<RawConv2d>,
    num_groups: usize,
    norm_eps: f32,
    backend: Arc<dyn ComputeBackend>,
}

impl ResnetBlock2D {
    /// Public load for testing.
    pub fn load_pub(vb: VarBuilder, in_ch: usize, out_ch: usize, groups: usize) -> Result<Self> {
        let backend = crate::backends::create_backend(vb.device());
        Self::load(vb, in_ch, out_ch, groups, backend)
    }

    /// Public forward for testing.
    pub fn forward_pub(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    fn load(vb: VarBuilder, in_ch: usize, out_ch: usize, groups: usize, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let norm1_weight = vb.pp("norm1").get(in_ch, "weight")?;
        let norm1_bias = vb.pp("norm1").get(in_ch, "bias")?;
        let conv1 = RawConv2d::load(in_ch, out_ch, 3, 1, vb.pp("conv1"))?;
        let norm2_weight = vb.pp("norm2").get(out_ch, "weight")?;
        let norm2_bias = vb.pp("norm2").get(out_ch, "bias")?;
        let conv2 = RawConv2d::load(out_ch, out_ch, 3, 1, vb.pp("conv2"))?;
        let conv_shortcut = if in_ch != out_ch {
            Some(RawConv2d::load(in_ch, out_ch, 1, 0, vb.pp("conv_shortcut"))?)
        } else {
            None
        };
        Ok(Self { norm1_weight, norm1_bias, conv1, norm2_weight, norm2_bias, conv2, conv_shortcut, num_groups: groups, norm_eps: 1e-6, backend })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = match &self.conv_shortcut {
            Some(sc) => sc.forward(x, &*self.backend)?,
            None => x.clone(),
        };
        let h = self.backend.group_norm(x, &self.norm1_weight, &self.norm1_bias, self.num_groups, self.norm_eps)?;
        let h = self.backend.silu(&h)?;
        let h = self.conv1.forward(&h, &*self.backend)?;
        let h = self.backend.group_norm(&h, &self.norm2_weight, &self.norm2_bias, self.num_groups, self.norm_eps)?;
        let h = self.backend.silu(&h)?;
        let h = self.conv2.forward(&h, &*self.backend)?;
        h + residual
    }
}

// ── Attention Block ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct AttnBlock {
    group_norm_weight: Tensor,
    group_norm_bias: Tensor,
    num_groups: usize,
    norm_eps: f32,
    to_q_weight: Tensor,
    to_q_bias: Option<Tensor>,
    to_k_weight: Tensor,
    to_k_bias: Option<Tensor>,
    to_v_weight: Tensor,
    to_v_bias: Option<Tensor>,
    to_out_weight: Tensor,
    to_out_bias: Option<Tensor>,
    channels: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl AttnBlock {
    fn load(vb: VarBuilder, channels: usize, groups: usize, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let group_norm_weight = vb.pp("group_norm").get(channels, "weight")?;
        let group_norm_bias = vb.pp("group_norm").get(channels, "bias")?;
        let to_q_weight = vb.pp("to_q").get((channels, channels), "weight")?;
        let to_q_bias = Some(vb.pp("to_q").get(channels, "bias")?);
        let to_k_weight = vb.pp("to_k").get((channels, channels), "weight")?;
        let to_k_bias = Some(vb.pp("to_k").get(channels, "bias")?);
        let to_v_weight = vb.pp("to_v").get((channels, channels), "weight")?;
        let to_v_bias = Some(vb.pp("to_v").get(channels, "bias")?);
        let to_out_weight = vb.pp("to_out").pp("0").get((channels, channels), "weight")?;
        let to_out_bias = Some(vb.pp("to_out").pp("0").get(channels, "bias")?);
        Ok(Self { group_norm_weight, group_norm_bias, num_groups: groups, norm_eps: 1e-6, to_q_weight, to_q_bias, to_k_weight, to_k_bias, to_v_weight, to_v_bias, to_out_weight, to_out_bias, channels, backend })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let (b, c, h, w) = x.dims4()?;

        let x = self.backend.group_norm(x, &self.group_norm_weight, &self.group_norm_bias, self.num_groups, self.norm_eps)?;
        // Reshape to (b, c, h*w) → (b, h*w, c)
        let x = x.reshape((b, c, h * w))?.transpose(1, 2)?;

        let q = self.backend.linear_forward(&x, &self.to_q_weight, self.to_q_bias.as_ref())?;
        let k = self.backend.linear_forward(&x, &self.to_k_weight, self.to_k_bias.as_ref())?;
        let v = self.backend.linear_forward(&x, &self.to_v_weight, self.to_v_bias.as_ref())?;

        // Single-head attention
        let scale = 1.0 / (self.channels as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let last_dim = attn.rank() - 1;
        let attn = self.backend.softmax(&attn, last_dim)?;
        let out = attn.matmul(&v)?;

        let out = self.backend.linear_forward(&out, &self.to_out_weight, self.to_out_bias.as_ref())?;
        // Reshape back: (b, h*w, c) → (b, c, h, w)
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;
        out + residual
    }
}

// ── Up Decoder Block ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct UpDecoderBlock {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<RawConv2d>,
    backend: Arc<dyn ComputeBackend>,
}

impl UpDecoderBlock {
    fn load(
        vb: VarBuilder,
        in_ch: usize,
        out_ch: usize,
        num_layers: usize,
        groups: usize,
        add_upsample: bool,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        let vb_r = vb.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let r_in = if i == 0 { in_ch } else { out_ch };
            resnets.push(ResnetBlock2D::load(vb_r.pp(i), r_in, out_ch, groups, backend.clone())?);
        }
        let upsampler = if add_upsample {
            Some(RawConv2d::load(out_ch, out_ch, 3, 1, vb.pp("upsamplers").pp("0").pp("conv"))?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler, backend })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for resnet in &self.resnets {
            x = resnet.forward(&x)?;
        }
        if let Some(up) = &self.upsampler {
            let (_b, _c, h, w) = x.dims4()?;
            // Nearest-neighbor 2x upsample
            x = x.upsample_nearest2d(h * 2, w * 2)?;
            x = up.forward(&x, &*self.backend)?;
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
    fn load(vb: VarBuilder, channels: usize, groups: usize, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let resnet0 = ResnetBlock2D::load(vb.pp("resnets").pp("0"), channels, channels, groups, backend.clone())?;
        let attn = AttnBlock::load(vb.pp("attentions").pp("0"), channels, groups, backend.clone())?;
        let resnet1 = ResnetBlock2D::load(vb.pp("resnets").pp("1"), channels, channels, groups, backend)?;
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
    conv_in: RawConv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out_weight: Tensor,
    conv_norm_out_bias: Tensor,
    conv_norm_out_groups: usize,
    conv_norm_out_eps: f32,
    conv_out: RawConv2d,
    backend: Arc<dyn ComputeBackend>,
}

impl Flux2Decoder {
    fn load(vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let groups = 32;
        let block_channels: Vec<(usize, usize)> = vec![
            (512, 512),   // up_block 0: in=512, out=512, upsample
            (512, 512),   // up_block 1: in=512, out=512, upsample
            (512, 256),   // up_block 2: in=512, out=256, upsample
            (256, 128),   // up_block 3: in=256, out=128, no upsample
        ];
        let num_layers = 3;

        let conv_in = RawConv2d::load(32, 512, 3, 1, vb.pp("conv_in"))?;
        let mid_block = MidBlock::load(vb.pp("mid_block"), 512, groups, backend.clone())?;

        let mut up_blocks = Vec::new();
        let vb_up = vb.pp("up_blocks");
        for (i, (in_ch, out_ch)) in block_channels.iter().enumerate() {
            let add_upsample = i < 3;
            up_blocks.push(UpDecoderBlock::load(
                vb_up.pp(i), *in_ch, *out_ch, num_layers, groups, add_upsample, backend.clone(),
            )?);
        }

        let conv_norm_out_weight = vb.pp("conv_norm_out").get(128, "weight")?;
        let conv_norm_out_bias = vb.pp("conv_norm_out").get(128, "bias")?;
        let conv_out = RawConv2d::load(128, 3, 3, 1, vb.pp("conv_out"))?;

        Ok(Self { conv_in, mid_block, up_blocks, conv_norm_out_weight, conv_norm_out_bias, conv_norm_out_groups: groups, conv_norm_out_eps: 1e-6, conv_out, backend })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x, &*self.backend)?;
        x = self.mid_block.forward(&x)?;
        for up_block in &self.up_blocks {
            x = up_block.forward(&x)?;
        }
        let x = self.backend.group_norm(&x, &self.conv_norm_out_weight, &self.conv_norm_out_bias, self.conv_norm_out_groups, self.conv_norm_out_eps)?;
        let x = self.backend.silu(&x)?;
        self.conv_out.forward(&x, &*self.backend)
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Custom FLUX.2 VAE that matches AutoencoderKLFlux2 weight layout.
#[derive(Debug)]
pub struct Flux2VAE {
    decoder: Flux2Decoder,
    post_quant_conv: RawConv2d,
}

impl Flux2VAE {
    pub fn load(vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let decoder = Flux2Decoder::load(vb.pp("decoder"), backend.clone())?;
        let post_quant_conv = RawConv2d::load(32, 32, 1, 0, vb.pp("post_quant_conv"))?;
        Ok(Self { decoder, post_quant_conv })
    }

    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(latent, &*self.decoder.backend)?;
        self.decoder.forward(&x)
    }

}
