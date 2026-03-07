use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::path::PathBuf;

use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};

use super::gemma_encoder::{gemma3_12b_config, Gemma3TextEncoder};
use super::vendored::config::Ltx2ConnectorConfig;
use super::vendored::connector::Ltx2TextConnectors;

/// LTX-2 Gemma-3 text encoder + connector Forwarder.
///
/// Layer name: `"ltx2-gemma"`
///
/// This component handles:
/// 1. Gemma-3 text encoding (12B) — extracts all 49 hidden states, normalizes, packs
/// 2. LTX2TextConnectors — self-attention transformer with registers
///
/// Input format (packed tensors):
/// - If Gemma is loaded: `[0]` = token IDs `[B, L]` (u32), `[1]` = attention mask `[B, L]`
/// - If Gemma is NOT loaded: `[0]` = pre-computed packed embeddings `[B, L, 188160]`,
///   `[1]` = attention mask `[B, L]`
///
/// Output: `[B, seq_len, cross_attention_dim]` — context for transformer
pub struct Ltx2Gemma {
    name: String,
    connector: Option<Ltx2TextConnectors>,
    #[allow(dead_code)]
    encoder: Option<Gemma3TextEncoder>,
}

impl std::fmt::Debug for Ltx2Gemma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Gemma")
            .field("name", &self.name)
            .field("connector", &self.connector)
            .field("encoder", &self.encoder.is_some())
            .finish()
    }
}

impl std::fmt::Display for Ltx2Gemma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

/// Resolve a file from an HF repo, trying direct path first, then HF cache.
fn resolve_hf_file(repo: &str, filename: &str, model_base: &str) -> Result<PathBuf> {
    // Try direct path first: model_base/filename
    let direct = PathBuf::from(model_base).join(filename);
    if direct.exists() {
        return Ok(direct);
    }

    // Fall back to HF cache
    let mut cache_path = PathBuf::from(model_base);
    cache_path.push("hub");
    let cache = Cache::new(cache_path);
    let api = ApiBuilder::from_cache(cache).build()?;
    let model_api = api.model(repo.to_string());
    Ok(model_api.get(filename)?)
}

impl Ltx2Gemma {
    pub fn load_model(ctx: &Context) -> Result<Box<dyn Forwarder>> {
        let ltx_args = &ctx.args.ltx_args;
        let ltx_repo = ltx_args.ltx_repo();

        // Load connector weights
        let connector_path = resolve_hf_file(
            &ltx_repo,
            "connectors/diffusion_pytorch_model.safetensors",
            &ctx.args.model,
        )?;

        info!("Loading LTX-2 text connectors from {:?}...", connector_path);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[connector_path],
                ctx.dtype,
                &ctx.device,
            )?
        };

        let config = Ltx2ConnectorConfig::default();
        let connector = Ltx2TextConnectors::new(&config, false, vb)?;

        info!("LTX-2 text connectors loaded!");

        // Try to load Gemma-3 encoder
        let encoder = match Self::try_load_gemma(ctx) {
            Ok(enc) => {
                info!("Gemma-3 text encoder loaded successfully!");
                Some(enc)
            }
            Err(e) => {
                log::warn!(
                    "Gemma-3 text encoder not available: {}. \
                     Pass pre-computed packed embeddings [B, L, 188160] as input.",
                    e
                );
                None
            }
        };

        Ok(Box::new(Self {
            name: "ltx2-gemma".to_string(),
            connector: Some(connector),
            encoder,
        }))
    }

    /// Try to load the Gemma-3 12B model.
    ///
    /// Looks for model weights in the HF cache under the Gemma-3 repo.
    /// The user can set `--model` to point to a cache directory containing the model.
    fn try_load_gemma(ctx: &Context) -> Result<Gemma3TextEncoder> {
        let gemma_repo = "google/gemma-3-12b-pt";

        // Resolve model files
        let mut cache_path = PathBuf::from(&ctx.args.model);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let model_api = api.model(gemma_repo.to_string());

        // Get tokenizer
        let tokenizer_path = model_api.get("tokenizer.json")?;

        // Get model weight files (safetensors, possibly sharded)
        let config_path = model_api.get("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;

        // Parse config to get the actual model config
        let gemma_config: candle_transformers::models::gemma3::Config =
            serde_json::from_str(&config_str)
                .unwrap_or_else(|_| gemma3_12b_config());

        // Find safetensors files
        let index_path = model_api.get("model.safetensors.index.json");
        let model_paths = if let Ok(index_file) = index_path {
            // Sharded model — parse the index to find all shard files
            let index_str = std::fs::read_to_string(&index_file)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
            let weight_map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index"))?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let mut paths = Vec::new();
            for shard in &shard_files {
                paths.push(model_api.get(shard)?);
            }
            paths
        } else {
            // Single file model
            vec![model_api.get("model.safetensors")?]
        };

        Gemma3TextEncoder::load(
            &model_paths,
            &tokenizer_path,
            &gemma_config,
            ctx.dtype,
            &ctx.device,
        )
    }

    /// Encode text through the full pipeline (Gemma + connector).
    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        text_embeds: Tensor,
        text_mask: Option<Tensor>,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let mut tensors = vec![text_embeds];
        if let Some(mask) = text_mask {
            tensors.push(mask);
        }
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }
}

#[async_trait]
impl Forwarder for Ltx2Gemma {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let ltx_args = &ctx.args.ltx_args;
        let ltx_repo = ltx_args.ltx_repo();

        let connector_path = resolve_hf_file(
            &ltx_repo,
            "connectors/diffusion_pytorch_model.safetensors",
            &ctx.args.model,
        )?;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[connector_path],
                ctx.dtype,
                &ctx.device,
            )?
        };

        let config = Ltx2ConnectorConfig::default();
        let connector = Ltx2TextConnectors::new(&config, false, vb)?;

        // Try to load Gemma encoder on worker too
        let encoder = Self::try_load_gemma(ctx).ok();

        Ok(Box::new(Self {
            name,
            connector: Some(connector),
            encoder,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let connector = self
            .connector
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LTX-2 text connector not loaded"))?;

        let unpacked = unpack_tensors(x)?;
        let text_embeds = unpacked[0].to_dtype(ctx.dtype)?;
        let text_mask = if unpacked.len() > 1 {
            Some(unpacked[1].to_dtype(DType::F32)?)
        } else {
            None
        };

        info!("LTX-2 text connector forwarding...");

        // Input is already packed embeddings [B, L, 188160]
        // (either pre-computed or from Gemma encoder on the master side)
        if text_embeds.rank() == 2 {
            anyhow::bail!(
                "Expected packed Gemma embeddings [B, L, 188160], got rank-2 tensor. \
                 Use Gemma3TextEncoder::encode() on the master to produce packed embeddings."
            );
        }

        let (result, _mask) = connector.forward_video(&text_embeds, text_mask.as_ref())?;
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

use candle_core::DType;
