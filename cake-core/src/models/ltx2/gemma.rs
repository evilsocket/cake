use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::path::PathBuf;

use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};

use super::vendored::config::Ltx2ConnectorConfig;
use super::vendored::connector::Ltx2TextConnectors;

/// LTX-2 text connector Forwarder.
///
/// Layer name: `"ltx2-gemma"`
///
/// This component runs ONLY the LTX2TextConnectors (self-attention transformer
/// with registers). The Gemma-3 text encoder runs on the master GPU and sends
/// pre-computed packed embeddings here.
///
/// Input (packed tensors):
/// - `[0]` = packed Gemma embeddings `[B, L, 188160]`
/// - `[1]` = attention mask `[B, L]`
///
/// Output: `[B, seq_len, cross_attention_dim]` — context for transformer
pub struct Ltx2Gemma {
    name: String,
    connector: Option<Ltx2TextConnectors>,
}

impl std::fmt::Debug for Ltx2Gemma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ltx2Gemma")
            .field("name", &self.name)
            .field("connector", &self.connector)
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
        let is_ltx23 = ltx_args.is_ltx23();

        // Load connector weights only — Gemma encoder lives on the master
        let connector_path = resolve_hf_file(
            &ltx_repo,
            "connectors/diffusion_pytorch_model.safetensors",
            &ctx.args.model,
        )?;

        info!("Loading LTX-2{} text connectors from {:?}...",
            if is_ltx23 { ".3" } else { "" }, connector_path);

        // LTX-2 connector weights are BF16 — load as BF16 to avoid NaN
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[connector_path],
                DType::BF16,
                &ctx.device,
            )?
        };

        let config = if is_ltx23 {
            // Try loading config from connectors/config.json (created by conversion script)
            let config_path = resolve_hf_file(
                &ltx_repo,
                "connectors/config.json",
                &ctx.args.model,
            );
            match config_path {
                Ok(path) => {
                    let config_str = std::fs::read_to_string(&path)?;
                    serde_json::from_str(&config_str).unwrap_or_else(|_| Ltx2ConnectorConfig::for_ltx23())
                }
                Err(_) => Ltx2ConnectorConfig::for_ltx23(),
            }
        } else {
            Ltx2ConnectorConfig::default()
        };
        let connector = Ltx2TextConnectors::new(&config, false, vb)?;

        info!("LTX-2 text connectors loaded!");

        Ok(Box::new(Self {
            name: "ltx2-gemma".to_string(),
            connector: Some(connector),
        }))
    }

    /// Encode text through the connector pipeline.
    ///
    /// `text_embeds` should be pre-computed packed Gemma embeddings `[B, L, 188160]`.
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
        let is_ltx23 = ltx_args.is_ltx23();

        let connector_path = resolve_hf_file(
            &ltx_repo,
            "connectors/diffusion_pytorch_model.safetensors",
            &ctx.args.model,
        )?;

        // LTX-2 connector weights are BF16 — load as BF16 to avoid NaN
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[connector_path],
                DType::BF16,
                &ctx.device,
            )?
        };

        let config = if is_ltx23 {
            Ltx2ConnectorConfig::for_ltx23()
        } else {
            Ltx2ConnectorConfig::default()
        };
        let connector = Ltx2TextConnectors::new(&config, false, vb)?;

        Ok(Box::new(Self {
            name,
            connector: Some(connector),
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
        // Connector weights are BF16 — convert inputs to match
        let text_embeds = unpacked[0].to_dtype(DType::BF16)?;
        let text_mask = if unpacked.len() > 1 {
            Some(unpacked[1].to_dtype(DType::F32)?)
        } else {
            None
        };

        if text_embeds.rank() == 2 {
            anyhow::bail!(
                "Expected packed Gemma embeddings [B, L, 188160], got rank-2 tensor. \
                 Gemma encoder should run on the master and send packed embeddings."
            );
        }

        info!("LTX-2 text connector forwarding...");
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
