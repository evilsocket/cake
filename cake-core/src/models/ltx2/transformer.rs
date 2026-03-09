use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::path::PathBuf;

use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};

use super::vendored::config::Ltx2TransformerConfig;
use super::vendored::model::LTXModel;

/// LTX-2 dual-stream DiT transformer Forwarder.
///
/// Supports two modes:
/// 1. Full model: layer name `"ltx2-transformer"` — runs all 48 blocks + setup + finalize
/// 2. Block range: layer name `"ltx2-transformer.N-M"` — runs blocks N through M only
///
/// Full model packed tensor format:
/// 0: video_latent  [B, T, in_channels]
/// 1: sigma         [B]
/// 2: timesteps     [B]
/// 3: positions     [B, 3, T]
/// 4: context       [B, L, cross_attention_dim]
/// 5: context_mask  [B, L]
///
/// Block range packed tensor format:
/// 0: hidden        [B, T, video_dim]
/// 1: temb          [B, 1, adaln_params, video_dim]
/// 2: pe_cos        [B, H, T, d_head/2]
/// 3: pe_sin        [B, H, T, d_head/2]
/// 4: context       [B, L, video_dim]  (already through caption projection)
/// 5: context_mask  [B, L]
/// 6: embedded_ts   [B, 1, video_dim]  (for finalize, if this shard includes it)
#[derive(Debug)]
pub struct Ltx2Transformer {
    name: String,
    model: LTXModel,
    /// true when running only a block range (not the full model)
    is_block_range: bool,
    /// Actual dtype of loaded weights (BF16 for LTX-2)
    model_dtype: DType,
}

impl std::fmt::Display for Ltx2Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

/// Parse block range from layer name like "ltx2-transformer.0-23".
/// Returns (start, end_exclusive) or None for full model.
fn parse_block_range(name: &str) -> Option<(usize, usize)> {
    let suffix = name.strip_prefix("ltx2-transformer.")?;
    let parts: Vec<&str> = suffix.split('-').collect();
    if parts.len() == 2 {
        let start: usize = parts[0].parse().ok()?;
        let end: usize = parts[1].parse().ok()?;
        Some((start, end + 1)) // inclusive to exclusive
    } else {
        None
    }
}

impl Ltx2Transformer {
    /// Load as a full model (all blocks + setup + finalize).
    pub fn load_model(ctx: &Context) -> Result<Box<dyn Forwarder>> {
        let (config, weights_path) = Self::resolve_config_and_weights(ctx)?;

        info!("Loading LTX-2 transformer from {:?}...", weights_path);

        let weight_files = find_weight_files(&weights_path)?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, ctx.dtype, &ctx.device)?
        };

        let model = LTXModel::new(config, vb)?;

        info!("LTX-2 transformer loaded!");

        Ok(Box::new(Self {
            name: "ltx2-transformer".to_string(),
            model,
            is_block_range: false,
            model_dtype: DType::BF16,
        }))
    }

    /// Load a block range (e.g., blocks 0-23).
    pub fn load_block_range(
        name: String,
        ctx: &Context,
        block_start: usize,
        block_end: usize,
    ) -> Result<Box<dyn Forwarder>> {
        let (config, weights_path) = Self::resolve_config_and_weights(ctx)?;

        info!(
            "Loading LTX-2 transformer blocks {}-{} from {:?}...",
            block_start,
            block_end - 1,
            weights_path
        );

        let weight_files = find_weight_files(&weights_path)?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, ctx.dtype, &ctx.device)?
        };

        let model = LTXModel::new_block_range(config, vb, block_start, Some(block_end))?;

        Ok(Box::new(Self {
            name,
            model,
            is_block_range: true,
            model_dtype: DType::BF16,
        }))
    }

    pub(crate) fn resolve_config_and_weights(ctx: &Context) -> Result<(Ltx2TransformerConfig, PathBuf)> {
        let ltx_args = &ctx.args.ltx_args;

        // If explicit transformer path given, use it directly
        if let Some(ref p) = ltx_args.ltx_transformer {
            let path = PathBuf::from(p);
            return Ok((Ltx2TransformerConfig::default(), path));
        }

        // Try direct path first: --model points to a directory containing transformer/
        let model_dir = PathBuf::from(&ctx.args.model);
        let direct_transformer = model_dir.join("transformer");
        if direct_transformer.is_dir() {
            let config = Self::load_config_from_dir(&direct_transformer);
            let weights = Self::find_weights_in_dir(&direct_transformer)?;
            return Ok((config, weights));
        }

        // Fall back to HF cache resolution
        let repo = ltx_args.ltx_repo();
        let cache_path = model_dir.join("hub");
        let api = if model_dir.is_dir() && cache_path.is_dir() {
            // Model-local cache (e.g., /path/to/LTX-2/hub/) — only when --model is a real directory
            ApiBuilder::from_cache(Cache::new(cache_path)).build()?
        } else {
            // Use default HF cache (~/.cache/huggingface/hub)
            let mut builder = ApiBuilder::new();
            if let Ok(token) = std::env::var("HF_TOKEN") {
                builder = builder.with_token(Some(token));
            }
            builder.build()?
        };
        let model_api = api.model(repo);

        let config = if let Ok(config_path) = model_api.get("transformer/config.json") {
            let config_str = std::fs::read_to_string(&config_path)?;
            match serde_json::from_str::<Ltx2TransformerConfig>(&config_str) {
                Ok(cfg) => {
                    info!("Loaded transformer config from {:?}", config_path);
                    cfg
                }
                Err(e) => {
                    log::warn!("Failed to parse transformer config.json: {}, using defaults", e);
                    Ltx2TransformerConfig::default()
                }
            }
        } else {
            Ltx2TransformerConfig::default()
        };

        // Resolve weights — try single file first, then find index for sharded models
        let weights_path =
            if let Ok(path) = model_api.get("transformer/diffusion_pytorch_model.safetensors") {
                path
            } else {
                // Sharded model — get the index, parse shard filenames, download each shard
                let index_path = model_api
                    .get("transformer/diffusion_pytorch_model.safetensors.index.json")?;
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)?;

                // Extract unique shard filenames from weight_map values
                let mut shard_names: Vec<String> = Vec::new();
                if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                    for v in weight_map.values() {
                        if let Some(name) = v.as_str() {
                            if !shard_names.contains(&name.to_string()) {
                                shard_names.push(name.to_string());
                            }
                        }
                    }
                }
                shard_names.sort();
                info!("Downloading {} transformer weight shards from HF...", shard_names.len());

                for shard in &shard_names {
                    let hf_path = format!("transformer/{}", shard);
                    info!("  downloading {}...", hf_path);
                    model_api.get(&hf_path)?;
                }

                // Return the directory containing the downloaded shards
                index_path.parent().unwrap().to_path_buf()
            };

        Ok((config, weights_path))
    }

    fn load_config_from_dir(dir: &PathBuf) -> Ltx2TransformerConfig {
        let config_path = dir.join("config.json");
        if config_path.exists() {
            if let Ok(s) = std::fs::read_to_string(&config_path) {
                if let Ok(cfg) = serde_json::from_str::<Ltx2TransformerConfig>(&s) {
                    info!("Loaded transformer config from {:?}", config_path);
                    return cfg;
                }
            }
        }
        info!("Using default transformer config");
        Ltx2TransformerConfig::default()
    }

    fn find_weights_in_dir(dir: &PathBuf) -> Result<PathBuf> {
        // Single file
        let single = dir.join("diffusion_pytorch_model.safetensors");
        if single.exists() {
            return Ok(single);
        }
        // Sharded — return the directory (find_weight_files will scan it)
        let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
        if index.exists() {
            return Ok(dir.clone());
        }
        // Look for any safetensors file
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if p.extension().map_or(false, |e| e == "safetensors") {
                return Ok(p);
            }
        }
        anyhow::bail!("No safetensors files found in {:?}", dir)
    }

    /// Pack tensors for full-model network transport and call the forwarder.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_packed(
        forwarder: &mut Box<dyn Forwarder>,
        video_latent: Tensor,
        sigma: Tensor,
        timesteps: Tensor,
        positions: Tensor,
        context: Tensor,
        context_mask: Tensor,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let packed = pack_tensors(
            vec![video_latent, sigma, timesteps, positions, context, context_mask],
            &ctx.device,
        )?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }

    /// Pack tensors for block-range network transport and call the forwarder.
    ///
    /// Sends pre-computed hidden states + metadata instead of raw latents.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_blocks_packed(
        forwarder: &mut Box<dyn Forwarder>,
        hidden: Tensor,
        temb: Tensor,
        pe_cos: Tensor,
        pe_sin: Tensor,
        context: Tensor,
        context_mask: Tensor,
        embedded_ts: Tensor,
        prompt_temb: Option<Tensor>,
        stg_skip_blocks: &[usize],
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let mut tensors = vec![hidden, temb, pe_cos, pe_sin, context, context_mask, embedded_ts];
        if let Some(pt) = prompt_temb {
            tensors.push(pt);
        }
        // Encode STG skip blocks as a 1D F32 tensor (block indices as floats)
        if !stg_skip_blocks.is_empty() {
            let stg_vals: Vec<f32> = stg_skip_blocks.iter().map(|&b| b as f32).collect();
            tensors.push(Tensor::new(stg_vals, &ctx.device)?);
        }
        let packed = pack_tensors(tensors, &ctx.device)?;
        // block_idx: 1 = normal block-range, 2 = block-range with STG
        let block_idx = if stg_skip_blocks.is_empty() { 1 } else { 2 };
        forwarder.forward_mut(&packed, 0, block_idx, ctx).await
    }

    /// Reference to the inner model (for master-side local execution).
    pub fn model(&self) -> &LTXModel {
        &self.model
    }
}

#[async_trait]
impl Forwarder for Ltx2Transformer {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let (config, weights_path) = Self::resolve_config_and_weights(ctx)?;

        // LTX-2 weights are natively BF16 — loading as F16 causes NaN
        let model_dtype = DType::BF16;

        let is_block_range;
        let model = if let Some((start, end)) = parse_block_range(&name) {
            info!(
                "Loading LTX-2 transformer blocks {}-{} from {:?} (dtype={:?})...",
                start,
                end - 1,
                weights_path,
                model_dtype,
            );
            is_block_range = true;
            let weight_files = find_weight_files(&weights_path)?;
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &weight_files,
                    model_dtype,
                    &ctx.device,
                )?
            };
            LTXModel::new_block_range(config, vb, start, Some(end))?
        } else {
            info!("Loading full LTX-2 transformer from {:?} (dtype={:?})...", weights_path, model_dtype);
            is_block_range = false;
            let weight_files = find_weight_files(&weights_path)?;
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &weight_files,
                    model_dtype,
                    &ctx.device,
                )?
            };
            LTXModel::new(config, vb)?
        };

        info!("LTX-2 transformer loaded!");

        Ok(Box::new(Self {
            name,
            model,
            is_block_range,
            model_dtype,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let unpacked = unpack_tensors(x)?;

        // block_idx == 1 or 2 signals block-range format (2 = with STG)
        if self.is_block_range || block_idx == 1 || block_idx == 2 {
            // Block-range format: [hidden, temb, pe_cos, pe_sin, context, context_mask, embedded_ts, prompt_temb?, stg_blocks?]
            let dt = self.model_dtype;
            let hidden = unpacked[0].to_dtype(dt)?;
            let temb = unpacked[1].to_dtype(dt)?;
            let pe_cos = unpacked[2].to_dtype(dt)?;
            let pe_sin = unpacked[3].to_dtype(dt)?;
            let context = unpacked[4].to_dtype(dt)?;
            let context_mask = unpacked[5].to_dtype(dt)?;
            // Determine how many optional tensors follow the 7 base tensors.
            // For block_idx==2 (STG), the LAST tensor is always stg_blocks.
            // Base: [hidden, temb, pe_cos, pe_sin, context, context_mask, embedded_ts] = 7
            // Optional: prompt_temb (index 7), stg_blocks (last, only when block_idx==2)
            let has_stg = block_idx == 2;
            let num_base = 7;
            let num_optional_after = unpacked.len() - num_base;
            // If STG, last optional is stg_blocks. prompt_temb exists if there's more than just stg.
            let (prompt_temb, stg_skip_blocks) = if has_stg {
                let stg_tensor = &unpacked[unpacked.len() - 1];
                let stg_vals: Vec<f32> = stg_tensor.to_vec1()?;
                let stg_blocks: Vec<usize> = stg_vals.iter().map(|&v| v as usize).collect();
                // prompt_temb at index 7 if there are 2+ optional tensors (prompt_temb + stg)
                let pt = if num_optional_after >= 2 {
                    Some(unpacked[7].to_dtype(dt)?)
                } else {
                    None
                };
                (pt, stg_blocks)
            } else {
                let pt = if unpacked.len() > 7 {
                    Some(unpacked[7].to_dtype(dt)?)
                } else {
                    None
                };
                (pt, vec![])
            };
            let embedded_ts = if unpacked.len() > 6 {
                Some(unpacked[6].to_dtype(dt)?)
            } else {
                None
            };

            info!(
                "LTX-2 transformer blocks forwarding (unpack: {}ms, hidden: {:?}{})",
                t0.elapsed().as_millis(),
                hidden.shape(),
                if stg_skip_blocks.is_empty() { String::new() } else { format!(", stg_skip={:?}", stg_skip_blocks) }
            );

            let pe = (pe_cos, pe_sin);
            let result = self.model.forward_blocks_only_with_stg(
                &hidden,
                &temb,
                &pe,
                &context,
                Some(&context_mask),
                embedded_ts.as_ref(),
                prompt_temb.as_ref(),
                &stg_skip_blocks,
            )?;

            info!("LTX-2 transformer blocks done in {}ms", t0.elapsed().as_millis());
            Ok(result)
        } else {
            // Full model format: [video_latent, sigma, timesteps, positions, context, context_mask]
            let dt = self.model_dtype;
            let video_latent = unpacked[0].to_dtype(dt)?;
            let sigma = unpacked[1].to_dtype(dt)?;
            let timesteps = unpacked[2].to_dtype(dt)?;
            let positions = unpacked[3].to_dtype(DType::F32)?;
            let context = unpacked[4].to_dtype(dt)?;
            let context_mask = unpacked[5].to_dtype(dt)?;

            info!(
                "LTX-2 transformer forwarding (unpack: {}ms, latent: {:?})",
                t0.elapsed().as_millis(),
                video_latent.shape()
            );

            let result = self.model.forward_video(
                &video_latent,
                &sigma,
                &timesteps,
                &positions,
                &context,
                Some(&context_mask),
            )?;

            info!("LTX-2 transformer done in {}ms", t0.elapsed().as_millis());
            Ok(result)
        }
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

/// Find all safetensors weight files from a path.
///
/// If `path` is a single .safetensors file, returns just that file.
/// If `path` is a directory, scans for all diffusion_pytorch_model*.safetensors files.
fn find_weight_files(path: &PathBuf) -> Result<Vec<PathBuf>> {
    // Single safetensors file
    if path.extension().map_or(false, |e| e == "safetensors") && path.exists() {
        return Ok(vec![path.clone()]);
    }

    // Directory: scan for shards
    if path.is_dir() {
        let mut shards = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("diffusion_pytorch_model")
                    && name.ends_with(".safetensors")
                    && !name.contains("index")
                {
                    shards.push(p);
                }
            }
        }
        if !shards.is_empty() {
            shards.sort();
            info!("Found {} transformer weight shards", shards.len());
            return Ok(shards);
        }
    }

    // Try parent directory scan (for paths pointing to specific shard files)
    if let Some(parent) = path.parent() {
        let mut shards = Vec::new();
        for entry in std::fs::read_dir(parent)? {
            let entry = entry?;
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("diffusion_pytorch_model")
                    && name.ends_with(".safetensors")
                    && !name.contains("index")
                {
                    shards.push(p);
                }
            }
        }
        if !shards.is_empty() {
            shards.sort();
            info!("Found {} transformer weight shards", shards.len());
            return Ok(shards);
        }
    }

    Ok(vec![path.clone()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_block_range() {
        assert_eq!(parse_block_range("ltx2-transformer"), None);
        assert_eq!(parse_block_range("ltx2-transformer.0-23"), Some((0, 24)));
        assert_eq!(parse_block_range("ltx2-transformer.24-47"), Some((24, 48)));
        assert_eq!(parse_block_range("ltx2-transformer.0-47"), Some((0, 48)));
        assert_eq!(parse_block_range("ltx2-transformer.abc"), None);
        assert_eq!(parse_block_range("ltx2-vae"), None);
    }
}
