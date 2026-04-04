// Wan2.2 DiT transformer forwarder.
//
// Layer name: "wan-transformer"
// Supports block-range sharding: "wan-transformer.0-19"

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{pack_tensors, unpack_tensors};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::path::PathBuf;

use super::vendored::config::WanTransformerConfig;
use super::vendored::model::WanModel;
use super::quantized_transformer::QuantizedWanModel;

/// Backend: either full-precision WanModel or quantized GGUF QuantizedWanModel.
enum WanBackend {
    Full(WanModel),
    Quantized(QuantizedWanModel),
}

impl Debug for WanBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full(_) => write!(f, "WanBackend::Full"),
            Self::Quantized(_) => write!(f, "WanBackend::Quantized"),
        }
    }
}

#[derive(Debug)]
pub struct WanTransformer {
    backend: WanBackend,
    name: String,
}

impl Display for WanTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tag = match &self.backend {
            WanBackend::Full(_) => "F16",
            WanBackend::Quantized(_) => "GGUF",
        };
        write!(f, "WanTransformer[{}] ({}, local)", self.name, tag)
    }
}

#[async_trait]
impl Forwarder for WanTransformer {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        // Load config: try transformer/config.json (diffusers), config.json (original),
        // then default 14B for GGUF files (which don't ship a config).
        let cfg = {
            let transformer_config = ctx.data_path.join("transformer/config.json");
            let root_config = ctx.data_path.join("config.json");
            if transformer_config.exists() {
                log::info!("loading Wan config from {}", transformer_config.display());
                WanTransformerConfig::from_path(&transformer_config)?
            } else if root_config.exists() && !root_config.to_string_lossy().contains(".gguf") {
                log::info!("loading Wan config from {}", root_config.display());
                WanTransformerConfig::from_path(&root_config)?
            } else {
                // GGUF or no config — use 14B defaults (most GGUF models are 14B)
                log::info!("using default Wan 14B config (5120 hidden, 40 heads, 40 layers)");
                WanTransformerConfig::wan22_14b()
            }
        };

        log::info!("Wan transformer: hidden={}, heads={}, layers={}, ffn={}",
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_layers, cfg.ffn_dim);

        // Parse block range from name (e.g. "wan-transformer.0-19")
        let block_range = parse_block_range(&name);

        // Try GGUF first (quantized), then safetensors (full precision).
        let backend = if let Some(gguf_path) = find_gguf_file(&ctx.data_path) {
            log::info!("loading quantized Wan transformer from GGUF: {}", gguf_path.display());
            let model = QuantizedWanModel::load_from_gguf(&gguf_path, &cfg, &ctx.device)?;
            WanBackend::Quantized(model)
        } else {
            let vb = if let Some(ref existing) = ctx.var_builder {
                existing.clone()
            } else {
                let transformer_dir = ctx.data_path.join("transformer");
                let weight_files = find_safetensors_files(&transformer_dir)
                    .or_else(|_| find_safetensors_files(&ctx.data_path))?;
                log::info!("loading Wan transformer from {} safetensors file(s)", weight_files.len());
                let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, ctx.dtype, &ctx.device)? };
                if is_diffusers_format_path(&ctx.data_path) {
                    log::info!("detected diffusers format, remapping weight keys");
                    remap_diffusers_vb(vb)
                } else {
                    vb
                }
            };
            let model = WanModel::load_block_range(vb, &cfg, block_range)?;
            WanBackend::Full(model)
        };

        Ok(Box::new(Self { backend, name }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        // Unpack: [latents, timestep, context, num_frames_t, height_t, width_t]
        let unpacked = unpack_tensors(x)?;
        let latents = &unpacked[0];
        let timestep = &unpacked[1];
        let context = &unpacked[2];
        // Scalar metadata encoded as 1-element tensors
        let num_frames: usize = unpacked[3].to_vec1::<f32>()?[0] as usize;
        let height: usize = unpacked[4].to_vec1::<f32>()?[0] as usize;
        let width: usize = unpacked[5].to_vec1::<f32>()?[0] as usize;

        let result = match &self.backend {
            WanBackend::Full(model) => model.forward(
                latents, timestep, context, num_frames, height, width,
            )?,
            WanBackend::Quantized(model) => model.forward(
                latents, timestep, context, num_frames, height, width,
            )?,
        };
        Ok(result)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}

impl WanTransformer {
    /// Pack inputs for RPC transfer.
    pub async fn forward_packed(
        transformer: &mut Box<dyn Forwarder>,
        latents: Tensor,
        timestep: Tensor,
        context: Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let device = &ctx.device;
        let tensors = vec![
            latents,
            timestep,
            context,
            Tensor::from_slice(&[num_frames as f32], 1, device)?,
            Tensor::from_slice(&[height as f32], 1, device)?,
            Tensor::from_slice(&[width as f32], 1, device)?,
        ];
        let combined = pack_tensors(tensors, device)?;
        transformer.forward_mut(&combined, 0, 0, ctx).await
    }
}

/// Check if the model path is in diffusers format.
/// Diffusers format has transformer/config.json with "_class_name": "WanTransformer3DModel".
fn is_diffusers_format_path(data_path: &std::path::Path) -> bool {
    let config_path = data_path.join("transformer/config.json");
    if let Ok(data) = std::fs::read(&config_path) {
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) {
            return json.get("_class_name").is_some();
        }
    }
    false
}

/// Remap our internal key names to diffusers-format keys.
/// rename_f transforms the *requested* key to match the *stored* key.
/// Our code requests "text_embedding.0.weight" → we must produce "condition_embedder.text_embedder.0.weight".
fn remap_diffusers_vb(vb: VarBuilder<'static>) -> VarBuilder<'static> {
    vb.rename_f(|key| {
        let mut k = key.to_string();
        // Output head: head.head.* -> proj_out.*
        k = k.replace("head.head.", "proj_out.");
        // head.modulation -> scale_shift_table
        if k == "head.modulation" || k.starts_with("head.modulation.") {
            k = k.replace("head.modulation", "scale_shift_table");
        }
        // Block modulation -> scale_shift_table
        // e.g. blocks.0.modulation -> blocks.0.scale_shift_table
        if k.contains(".modulation") && k.starts_with("blocks.") {
            k = k.replace(".modulation", ".scale_shift_table");
        }
        // Norm: norm3 -> norm2
        k = k.replace(".norm3.", ".norm2.");
        // FFN: ffn.0. -> ffn.net.0.proj., ffn.2. -> ffn.net.2.
        k = k.replace(".ffn.0.", ".ffn.net.0.proj.");
        k = k.replace(".ffn.2.", ".ffn.net.2.");
        // Attention: self_attn -> attn1, cross_attn -> attn2
        k = k.replace(".self_attn.", ".attn1.");
        k = k.replace(".cross_attn.", ".attn2.");
        // Proj names: .q. -> .to_q., .k. -> .to_k., .v. -> .to_v., .o. -> .to_out.0.
        // Only within attn blocks (after attn1/attn2 prefix)
        if k.contains(".attn1.") || k.contains(".attn2.") {
            // Be careful not to replace norm_q/norm_k
            k = k.replace(".q.weight", ".to_q.weight");
            k = k.replace(".q.bias", ".to_q.bias");
            k = k.replace(".k.weight", ".to_k.weight");
            k = k.replace(".k.bias", ".to_k.bias");
            k = k.replace(".v.weight", ".to_v.weight");
            k = k.replace(".v.bias", ".to_v.bias");
            k = k.replace(".o.weight", ".to_out.0.weight");
            k = k.replace(".o.bias", ".to_out.0.bias");
        }
        // Embeddings: text_embedding.0.* -> condition_embedder.text_embedder.linear_1.*
        //             text_embedding.2.* -> condition_embedder.text_embedder.linear_2.*
        k = k.replace("text_embedding.0.", "condition_embedder.text_embedder.linear_1.");
        k = k.replace("text_embedding.2.", "condition_embedder.text_embedder.linear_2.");
        k = k.replace("time_embedding.0.", "condition_embedder.time_embedder.linear_1.");
        k = k.replace("time_embedding.2.", "condition_embedder.time_embedder.linear_2.");
        // time_projection.1.* -> condition_embedder.time_proj.*
        k = k.replace("time_projection.1.", "condition_embedder.time_proj.");
        k
    })
}

/// Find a .gguf file in the data path.
fn find_gguf_file(data_path: &PathBuf) -> Option<PathBuf> {
    // Check if data_path itself is a GGUF file
    if data_path.extension().map_or(false, |e| e == "gguf") && data_path.exists() {
        return Some(data_path.clone());
    }
    // Check for GGUF files in the directory
    if data_path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(data_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "gguf") {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Load a Wan DiT GGUF file into a VarBuilder.
/// The city96 GGUF format uses the same key names as our internal format,
/// so no remapping is needed.
fn load_wan_gguf<'a>(
    gguf_path: &PathBuf,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<VarBuilder<'a>> {
    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| anyhow::anyhow!("can't open GGUF file: {e}"))?;

    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("can't parse GGUF file: {e}"))?;

    log::info!(
        "GGUF: {} tensors, {} metadata entries",
        content.tensor_infos.len(),
        content.metadata.len(),
    );

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let start = std::time::Instant::now();

    // Dequantize all tensors to CPU first, then the model loader will
    // move them to GPU per-layer (avoids OOM from all 29GB on GPU at once).
    let cpu = candle_core::Device::Cpu;
    for tensor_name in content.tensor_infos.keys() {
        let qtensor = content
            .tensor(&mut file, tensor_name, &cpu)
            .map_err(|e| anyhow::anyhow!("can't load GGUF tensor '{}': {e}", tensor_name))?;

        // Dequantize to F16 on CPU
        let tensor = match qtensor.dequantize_f16(&cpu) {
            Ok(t) => t,
            Err(_) => qtensor.dequantize(&cpu)
                .map_err(|e| anyhow::anyhow!("can't dequantize '{}': {e}", tensor_name))?
                .to_dtype(DType::F16)?,
        };

        tensors.insert(tensor_name.to_string(), tensor);
    }

    log::info!(
        "GGUF: loaded {} tensors in {:.1}s",
        tensors.len(),
        start.elapsed().as_secs_f64(),
    );

    // Build VarBuilder on CPU — model loading will put tensors on CPU,
    // and we'll move the entire model to GPU after loading.
    Ok(VarBuilder::from_tensors(tensors, DType::F16, &cpu))
}

/// Find all safetensors files in a directory.
fn find_safetensors_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        anyhow::bail!("directory does not exist: {}", dir.display());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    if files.is_empty() {
        anyhow::bail!("no safetensors files found in {}", dir.display());
    }
    files.sort();
    Ok(files)
}

fn parse_block_range(name: &str) -> Option<(usize, usize)> {
    if let Some(range_str) = name.strip_prefix("wan-transformer.") {
        if let Some((start, end)) = range_str.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                return Some((s, e + 1)); // exclusive end
            }
        }
    }
    None
}
