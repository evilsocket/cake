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
/// Layer name: `"ltx2-transformer"`
///
/// Packed tensor format (for network transport):
/// 0: video_latent  [B, T, in_channels]
/// 1: sigma         [B]
/// 2: timesteps     [B]
/// 3: positions     [B, 3, T]
/// 4: context       [B, L, cross_attention_dim]
/// 5: context_mask  [B, L]
#[derive(Debug)]
pub struct Ltx2Transformer {
    name: String,
    model: LTXModel,
}

impl std::fmt::Display for Ltx2Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl Ltx2Transformer {
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
        }))
    }

    fn resolve_config_and_weights(ctx: &Context) -> Result<(Ltx2TransformerConfig, PathBuf)> {
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
        let mut cache_path = model_dir.clone();
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
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

        let weights_path =
            if let Ok(path) = model_api.get("transformer/diffusion_pytorch_model.safetensors") {
                path
            } else {
                let index_path = model_api
                    .get("transformer/diffusion_pytorch_model.safetensors.index.json")?;
                index_path
                    .parent()
                    .unwrap()
                    .join("diffusion_pytorch_model-00001-of-00002.safetensors")
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
        // Sharded — return the index file (find_weight_files will resolve shards)
        let index = dir.join("diffusion_pytorch_model.safetensors.index.json");
        if index.exists() {
            return Ok(index);
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

    /// Pack tensors for network transport and call the forwarder.
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
}

#[async_trait]
impl Forwarder for Ltx2Transformer {
    fn load(_name: String, ctx: &Context) -> Result<Box<Self>> {
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
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        // Packed: [video_latent, sigma, timesteps, positions, context, context_mask]
        let video_latent = unpacked[0].to_dtype(ctx.dtype)?;
        let sigma = unpacked[1].to_dtype(ctx.dtype)?;
        let timesteps = unpacked[2].to_dtype(ctx.dtype)?;
        let positions = unpacked[3].to_dtype(DType::F32)?;
        let context = unpacked[4].to_dtype(ctx.dtype)?;
        let context_mask = unpacked[5].to_dtype(ctx.dtype)?;

        info!("LTX-2 transformer forwarding...");

        let result = self.model.forward_video(
            &video_latent,
            &sigma,
            &timesteps,
            &positions,
            &context,
            Some(&context_mask),
        )?;

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

fn find_weight_files(path: &PathBuf) -> Result<Vec<PathBuf>> {
    if path.extension().map_or(false, |e| e == "safetensors") && path.exists() {
        return Ok(vec![path.clone()]);
    }

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
            return Ok(shards);
        }
    }

    Ok(vec![path.clone()])
}
