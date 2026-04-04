// Wan 3D VAE forwarder.
//
// Layer name: "wan-vae"
// Direction flag: 0.0 = decode

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::unpack_tensors;
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use std::fmt::{Debug, Display, Formatter};
use std::path::PathBuf;

use super::vendored::config::WanVaeConfig;
use super::vendored::vae::WanVaeDecoder;

#[derive(Debug)]
pub struct WanVae {
    decoder: WanVaeDecoder,
}

impl Display for WanVae {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "WanVae (local)")
    }
}

#[async_trait]
impl Forwarder for WanVae {
    fn load(_name: String, ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        // Build VarBuilder: try ctx.var_builder first, then load from vae/ directory.
        // Prefer cake-format converted file (wan_vae_cake.safetensors) over diffusers format.
        let vb = if let Some(ref existing) = ctx.var_builder {
            existing.clone()
        } else {
            let cake_vae = ctx.data_path.join("vae/wan_vae_cake.safetensors");
            // Load VAE on CPU — 3D convolutions at F32 are too memory-intensive for GPU
            let vae_device = candle_core::Device::Cpu;
            if cake_vae.exists() {
                log::info!("loading Wan VAE from {} (on CPU)", cake_vae.display());
                unsafe { VarBuilder::from_mmaped_safetensors(&[cake_vae], DType::F32, &vae_device)? }
            } else {
                let vae_dir = ctx.data_path.join("vae");
                let weight_files = find_vae_safetensors(&vae_dir, &ctx.data_path)?;
                log::info!("loading Wan VAE from {} safetensors file(s) (on CPU)", weight_files.len());
                unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &vae_device)? }
            }
        };

        let cfg = WanVaeConfig::default();
        let decoder = WanVaeDecoder::load(vb, &cfg)?;
        Ok(Box::new(Self { decoder }))
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        anyhow::bail!("WanVae requires forward_mut for chunked temporal decoding")
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        let direction = unpacked[0].to_vec1::<f32>()?[0];
        let input = &unpacked[1].to_dtype(DType::F32)?;

        if direction == 1.0 {
            anyhow::bail!("WanVae encode not implemented (decode-only for T2V)")
        } else {
            // Move to CPU for decode to avoid GPU OOM on 3D convolutions
            let input = input.to_device(&candle_core::Device::Cpu)?;
            let result = self.decoder.decode(&input)?;
            Ok(result)
        }
    }

    fn layer_name(&self) -> &str {
        "wan-vae"
    }
}

/// Remap our internal VAE key names to diffusers-format keys.
fn remap_vae_diffusers(vb: VarBuilder<'static>) -> VarBuilder<'static> {
    vb.rename_f(|key| {
        let mut k = key.to_string();
        // conv1 -> conv_in, head.2 -> conv_out
        k = k.replace("decoder.conv1.", "decoder.conv_in.");
        k = k.replace("decoder.head.2.", "decoder.conv_out.");
        // head.0 (norm_out) -> norm_out
        k = k.replace("decoder.head.0.", "decoder.norm_out.");
        // middle.0 -> mid_block.resnets.0, middle.2 -> mid_block.resnets.1
        k = k.replace("decoder.middle.0.", "decoder.mid_block.resnets.0.");
        k = k.replace("decoder.middle.2.", "decoder.mid_block.resnets.1.");
        // middle.1 (attention) -> mid_block.attentions.0
        k = k.replace("decoder.middle.1.", "decoder.mid_block.attentions.0.");
        // upsamples.N.block.B -> up_blocks.N.resnets.B
        // upsamples.N.upsample -> up_blocks.N.upsamplers.0 (or time_upsample)
        if k.contains("decoder.upsamples.") {
            k = k.replace("decoder.upsamples.", "decoder.up_blocks.");
            k = k.replace(".block.", ".resnets.");
            k = k.replace(".upsample.", ".upsamplers.0.");
        }
        // Spatial conv in upsampler: spatial_conv -> resample.1
        k = k.replace(".spatial_conv.", ".resample.1.");
        // Conv in non-temporal upsample: .conv. -> .resample.1.
        if k.contains("upsamplers.0.conv.") {
            k = k.replace("upsamplers.0.conv.", "upsamplers.0.resample.1.");
        }
        // RmsNorm: beta -> bias (diffusers uses gamma only, no beta for this VAE)
        // Actually, the diffusers VAE norm only has gamma, not beta.
        // Our code loads gamma and beta. We may need to handle this.
        k
    })
}

/// Find VAE safetensors files: try vae/ subdir first, then root.
fn find_vae_safetensors(vae_dir: &PathBuf, root: &PathBuf) -> Result<Vec<PathBuf>> {
    for dir in [vae_dir, root] {
        if !dir.exists() {
            continue;
        }
        let files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect();
        if !files.is_empty() {
            return Ok(files);
        }
    }
    anyhow::bail!("no VAE safetensors found in {} or {}", vae_dir.display(), root.display())
}
