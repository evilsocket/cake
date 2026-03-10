use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use image::{ImageBuffer, Rgb};
use log::info;
use std::path::PathBuf;

use super::gemma::Ltx2Gemma;
use super::gemma_encoder::{gemma3_12b_config, Gemma3TextEncoder};
use super::ltx2_shardable::Ltx2Shardable;
use super::transformer::Ltx2Transformer;
use super::vae_forwarder::Ltx2Vae;
use super::vocoder::Ltx2Vocoder;
use super::vendored::config::{Ltx2SchedulerConfig, Ltx2TransformerConfig, Ltx2VaeConfig};
use super::vendored::model::LTXModel;
use super::vendored::pipeline::{
    build_video_positions, denormalize_latents, normalize_latents, pack_latents, unpack_latents,
};
use super::vendored::scheduler::{euler_step, Ltx2Scheduler};
use crate::cake::{Context, Forwarder};
use crate::models::{Generator, VideoGenerator};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;

/// LTX-2 model (19B audio+video generation).
///
/// Architecture:
/// - Asymmetric dual-stream DiT transformer (14B video + 5B audio)
/// - Gemma-3 12B text encoder
/// - Video VAE decoder (native 4K support)
/// - Audio vocoder (synchronized with video)
///
/// Supports split transformer topology for distributed inference:
/// ```yaml
/// win5090:
///   host: "worker1:10128"
///   layers:
///     - "ltx2-transformer.0-23"  # First 24 blocks (~17GB)
/// # Master keeps blocks 24-47 + connector + VAE + Gemma
/// ```
pub struct Ltx2 {
    /// Connector forwarder (runs locally on master GPU)
    gemma_connector: Box<dyn Forwarder>,
    /// Gemma-3 12B text encoder (stays on CPU permanently)
    gemma_encoder: Option<Gemma3TextEncoder>,
    /// Remote transformer forwarder (full model or block range)
    transformer: Box<dyn Forwarder>,
    /// Local transformer blocks (for split mode — the master's block range)
    local_transformer: Option<LTXModel>,
    vae: Box<dyn Forwarder>,
    #[allow(dead_code)]
    vocoder: Box<dyn Forwarder>,
    /// Per-channel latent normalization parameters (from VAE safetensors)
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    context: Context,
}

#[async_trait]
impl Generator for Ltx2 {
    type Shardable = Ltx2Shardable;
    const MODEL_NAME: &'static str = "ltx-2";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        info!("Loading LTX-2 components...");

        // Text connector (runs locally on master GPU)
        let gemma_connector: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-gemma") {
                info!("ltx2-gemma (connector) will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-gemma",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                Ltx2Gemma::load_model(context)?
            };

        // Transformer — check for full or block-range topology
        let (transformer, local_transformer) = Self::load_transformer(context).await?;

        // VAE — load locally to get latents_mean/std
        let (vae, latents_mean, latents_std): (Box<dyn Forwarder>, Vec<f32>, Vec<f32>) =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-vae") {
                info!("ltx2-vae will be served by {}", &node.host);
                let client = Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-vae",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                );
                // Remote VAE — use identity normalization as fallback
                (client, vec![0.0; 128], vec![1.0; 128])
            } else {
                Ltx2Vae::load_with_stats(context)?
            };

        // Vocoder
        let vocoder: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-vocoder") {
                info!("ltx2-vocoder will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "ltx2-vocoder",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                Ltx2Vocoder::load_model(context)?
            };

        // Gemma-3 12B encoder — stays on master CPU permanently
        let gemma_encoder = match Self::try_load_gemma_encoder(context) {
            Ok(enc) => {
                info!("Gemma-3 12B encoder loaded on master CPU — text prompts supported!");
                Some(enc)
            }
            Err(e) => {
                log::warn!(
                    "Gemma-3 encoder not available: {}. \
                     Pre-computed embeddings must be provided.",
                    e
                );
                None
            }
        };

        info!("LTX-2 components loaded");

        Ok(Some(Box::new(Self {
            gemma_connector,
            gemma_encoder,
            transformer,
            local_transformer,
            vae,
            vocoder,
            latents_mean,
            latents_std,
            context: context.clone(),
        })))
    }
}

impl Ltx2 {
    /// Load the transformer, handling both full-model and block-range topologies.
    ///
    /// Returns (remote_forwarder, local_transformer_option):
    /// - Full model on worker: (Client, None)
    /// - Full model local: (Ltx2Transformer, None)
    /// - Block range on worker: (Client for remote blocks, Some(LTXModel for local blocks))
    async fn load_transformer(
        context: &mut Context,
    ) -> Result<(Box<dyn Forwarder>, Option<LTXModel>)> {
        // Check for full transformer on a worker
        if let Some((_name, node)) = context.topology.get_node_for_layer("ltx2-transformer") {
            info!("ltx2-transformer (full) will be served by {}", &node.host);
            let client = Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    "ltx2-transformer",
                    context.args.cluster_key.as_deref(),
                )
                .await?,
            );
            return Ok((client, None));
        }

        // Check for block-range assignments
        // Find any layer name matching "ltx2-transformer.N-M"
        let block_range_layer = context
            .topology
            .all_worker_layers()
            .into_iter()
            .find(|name| name.starts_with("ltx2-transformer."));

        if let Some(ref remote_layer) = block_range_layer {
            let (_name, node) = context
                .topology
                .get_node_for_layer(remote_layer)
                .ok_or_else(|| anyhow::anyhow!("No node found for layer {}", remote_layer))?;

            info!("{} will be served by {}", remote_layer, &node.host);

            // Parse remote block range
            let suffix = remote_layer
                .strip_prefix("ltx2-transformer.")
                .unwrap();
            let parts: Vec<&str> = suffix.split('-').collect();
            let remote_start: usize = parts[0].parse()?;
            let remote_end: usize = parts[1].parse::<usize>()? + 1;

            let client: Box<dyn Forwarder> = Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    remote_layer,
                    context.args.cluster_key.as_deref(),
                )
                .await?,
            );

            // Load the remaining blocks locally on the master
            let config = Ltx2TransformerConfig::default();
            let num_layers = config.num_layers;

            // Local gets the complement of remote.
            // For split transformer, master should have first blocks (with setup).
            let (local_start, local_end) = if remote_start == 0 {
                (remote_end, num_layers)
            } else {
                (0, remote_start)
            };

            if local_start != 0 {
                log::warn!(
                    "Master has blocks {}-{} without setup. \
                     Put the HIGHER block range on the worker for best performance.",
                    local_start,
                    local_end - 1
                );
            }

            info!(
                "Loading local transformer blocks {}-{} on master GPU",
                local_start,
                local_end - 1
            );

            // Load local blocks via Ltx2Transformer resolver (handles HF cache)
            let local_model = {
                let (cfg, weights_path) =
                    Ltx2Transformer::resolve_config_and_weights(context)?;
                let weight_files = find_local_weight_files(&weights_path)?;
                // LTX-2 weights are BF16 — load as BF16 to avoid conversion artifacts
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        &weight_files,
                        DType::BF16,
                        &context.device,
                    )?
                };
                LTXModel::new_block_range(cfg, vb, local_start, Some(local_end))?
            };

            return Ok((client, Some(local_model)));
        }

        // No topology entry — load full model locally
        info!("Loading full LTX-2 transformer locally");
        let transformer = Ltx2Transformer::load_model(context)?;
        Ok((transformer, None))
    }

    /// Load Gemma-3 12B encoder on the master's CPU.
    fn try_load_gemma_encoder(ctx: &Context) -> Result<Gemma3TextEncoder> {
        use hf_hub::api::sync::ApiBuilder;
        use hf_hub::Cache;

        let gemma_repo = "google/gemma-3-12b-pt";

        // Try model-local cache first, then standard HF cache, then download with token
        let mut cache_path = PathBuf::from(&ctx.args.model);
        cache_path.push("hub");
        let api = if cache_path.exists() {
            ApiBuilder::from_cache(Cache::new(cache_path)).build()?
        } else {
            // Use default HF cache (~/.cache/huggingface/hub) with optional token
            let mut builder = ApiBuilder::new();
            if let Ok(token) = std::env::var("HF_TOKEN") {
                builder = builder.with_token(Some(token));
            }
            builder.build()?
        };
        let model_api = api.model(gemma_repo.to_string());

        let tokenizer_path = model_api.get("tokenizer.json")?;

        let config_path = model_api.get("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let gemma_config: candle_transformers::models::gemma3::Config =
            serde_json::from_str(&config_str).unwrap_or_else(|_| gemma3_12b_config());

        // Find safetensors files (handle sharded models)
        let model_paths = if let Ok(index_file) = model_api.get("model.safetensors.index.json") {
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
            vec![model_api.get("model.safetensors")?]
        };

        info!("Loading Gemma-3 12B on CPU (F32)...");
        Gemma3TextEncoder::load(
            &model_paths,
            &tokenizer_path,
            &gemma_config,
            DType::F32,
            &Device::Cpu,
        )
    }
}

/// Find weight files from a path (for local master loading).
fn find_local_weight_files(path: &PathBuf) -> Result<Vec<PathBuf>> {
    if path.extension().map_or(false, |e| e == "safetensors") && path.exists() {
        return Ok(vec![path.clone()]);
    }
    if path.is_dir() {
        let mut shards = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let p = entry?.path();
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
    if let Some(parent) = path.parent() {
        let mut shards = Vec::new();
        for entry in std::fs::read_dir(parent)? {
            let p = entry?.path();
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

#[async_trait]
impl VideoGenerator for Ltx2 {
    async fn generate_video(&mut self, args: &ImageGenerationArgs) -> Result<VideoOutput> {
        let ImageGenerationArgs {
            image_prompt: _,
            image_seed,
            guidance_scale,
            ..
        } = args;

        // Copy all ltx_args values out to avoid borrow conflicts with &mut self later
        let height = self.context.args.ltx_args.ltx_height;
        let width = self.context.args.ltx_args.ltx_width;
        let num_frames = self.context.args.ltx_args.ltx_num_frames;
        let num_steps = self.context.args.ltx_args.ltx_num_steps.unwrap_or(30);
        let frame_rate = self.context.args.ltx_args.ltx_fps;
        let stg_scale_arg = self.context.args.ltx_args.ltx_stg_scale;
        let stg_block_arg = self.context.args.ltx_args.ltx_stg_block;
        let rescale_arg = self.context.args.ltx_args.ltx_rescale;
        let guidance_scale = guidance_scale.unwrap_or(4.0) as f32;

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }

        let trans_config = Ltx2TransformerConfig::default();
        let vae_config = Ltx2VaeConfig::default();
        let sched_config = Ltx2SchedulerConfig::default();

        info!(
            "Generating LTX-2 video: {}x{}, {} frames, {} steps, guidance_scale={:.1}",
            width, height, num_frames, num_steps, guidance_scale
        );

        // 1. Encode prompt with Gemma-3 on master CPU → send packed embeddings to connector
        info!("Encoding prompt...");
        let prompt_text = if args.image_prompt.is_empty() {
            "a beautiful video"
        } else {
            &args.image_prompt
        };

        let (packed_embeds, text_mask) = if let Some(ref mut encoder) = self.gemma_encoder {
            info!("Encoding text with Gemma-3 (CPU): \"{}\"", prompt_text);
            let (embeds, mask) = encoder.encode(prompt_text)?;
            // Transfer from CPU to GPU for network serialization
            let embeds = embeds
                .to_device(&self.context.device)?
                .to_dtype(DType::BF16)?;
            let mask = mask.to_device(&self.context.device)?;
            (embeds, mask)
        } else {
            // Fallback: dummy packed embeddings (for testing without Gemma weights)
            log::warn!("Using dummy text embeddings (Gemma-3 not loaded)");
            let seq_len = 1024usize;
            let packed_dim = trans_config.caption_channels * 49; // 3840 * 49 = 188160
            let dummy = Tensor::randn(
                0f32,
                1f32,
                (1, seq_len, packed_dim),
                &self.context.device,
            )?
            .to_dtype(DType::BF16)?;
            let mask = Tensor::ones((1, seq_len), DType::F32, &self.context.device)?;
            (dummy, mask)
        };

        // Debug: log Gemma output stats before connector
        {
            let ge_f32 = packed_embeds.to_dtype(DType::F32)?.flatten_all()?;
            let ge_min: f32 = ge_f32.min(0)?.to_scalar()?;
            let ge_max: f32 = ge_f32.max(0)?.to_scalar()?;
            let ge_std: f32 = ge_f32.var(0)?.to_scalar::<f32>()?.sqrt();
            info!(
                "Gemma packed embeds (pre-connector): {:?}, min={:.4}, max={:.4}, std={:.4}",
                packed_embeds.shape(), ge_min, ge_max, ge_std
            );
        }
        // Send packed embeddings to connector (local)
        info!("Sending packed embeddings to connector...");
        let prompt_embeds = Ltx2Gemma::encode(
            &mut self.gemma_connector,
            packed_embeds,
            Some(text_mask),
            &mut self.context,
        )
        .await?
        .to_dtype(DType::BF16)?;

        let ctx_seq_len = prompt_embeds.dim(1)?;
        let context_mask = Tensor::ones((1, ctx_seq_len), DType::F32, &self.context.device)?
            .to_dtype(DType::BF16)?;

        // Debug: log prompt embedding statistics
        {
            let pe_f32 = prompt_embeds.to_dtype(DType::F32)?.flatten_all()?;
            let pe_min: f32 = pe_f32.min(0)?.to_scalar()?;
            let pe_max: f32 = pe_f32.max(0)?.to_scalar()?;
            let pe_mean: f32 = pe_f32.mean(0)?.to_scalar()?;
            info!(
                "Text connector done: {:?}, min={:.4}, max={:.4}, mean={:.4}",
                prompt_embeds.shape(), pe_min, pe_max, pe_mean
            );
        }

        // Prepare unconditional context for classifier-free guidance
        // Python diffusers encodes empty string "" through full Gemma + connector pipeline
        let do_cfg = guidance_scale > 1.0;
        let (uncond_embeds, uncond_mask) = if do_cfg {
            info!("Preparing unconditional embeddings for CFG (guidance_scale={:.1})", guidance_scale);

            let (neg_packed, neg_mask) = if let Some(ref mut encoder) = self.gemma_encoder {
                info!("Encoding empty string for unconditional embeddings...");
                let (embeds, mask) = encoder.encode("")?;
                let embeds = embeds
                    .to_device(&self.context.device)?
                    .to_dtype(DType::BF16)?;
                let mask = mask.to_device(&self.context.device)?;
                (embeds, mask)
            } else {
                // Without Gemma, use zeros as fallback
                let seq_len = 1024usize;
                let packed_dim = trans_config.caption_channels * 49;
                let dummy = Tensor::zeros(
                    (1, seq_len, packed_dim),
                    DType::BF16,
                    &self.context.device,
                )?;
                let mask = Tensor::zeros((1, seq_len), DType::F32, &self.context.device)?;
                (dummy, mask)
            };

            // Debug: log negative Gemma output
            {
                let nge_f32 = neg_packed.to_dtype(DType::F32)?.flatten_all()?;
                let nge_std: f32 = nge_f32.var(0)?.to_scalar::<f32>()?.sqrt();
                info!(
                    "Gemma uncond packed embeds std={:.4}",
                    nge_std
                );
            }
            // Run through connector (same as positive prompt)
            let neg_embeds = Ltx2Gemma::encode(
                &mut self.gemma_connector,
                neg_packed,
                Some(neg_mask),
                &mut self.context,
            )
            .await?
            .to_dtype(DType::BF16)?;

            let neg_ctx_len = neg_embeds.dim(1)?;
            let neg_ctx_mask = Tensor::ones((1, neg_ctx_len), DType::F32, &self.context.device)?
                .to_dtype(DType::BF16)?;

            {
                let ne_f32 = neg_embeds.to_dtype(DType::F32)?.flatten_all()?;
                let ne_min: f32 = ne_f32.min(0)?.to_scalar()?;
                let ne_max: f32 = ne_f32.max(0)?.to_scalar()?;
                let ne_mean: f32 = ne_f32.mean(0)?.to_scalar()?;
                info!(
                    "Unconditional embeds: {:?}, min={:.4}, max={:.4}, mean={:.4}",
                    neg_embeds.shape(), ne_min, ne_max, ne_mean
                );
                // Compare cond vs uncond (overall)
                let pe_f32 = prompt_embeds.to_dtype(DType::F32)?.flatten_all()?;
                let diff = (&pe_f32 - &ne_f32)?;
                let diff_std: f32 = diff.var(0)?.to_scalar::<f32>()?.sqrt();
                let diff_mean: f32 = diff.mean(0)?.to_scalar()?;
                info!(
                    "Cond vs uncond context diff: mean={:.6}, std={:.6}",
                    diff_mean, diff_std
                );
                // Per-position analysis: compare first 30 vs last 30 tokens
                // Python shows: first 30 diff_std=0.421, last 30 diff_std=0.009
                let pe_2d = prompt_embeds.to_dtype(DType::F32)?; // [1, L, D]
                let ne_2d = neg_embeds.to_dtype(DType::F32)?;
                let diff_2d = (&pe_2d - &ne_2d)?;
                let seq = diff_2d.dim(1)?;
                let n_check = 30.min(seq);
                let first_diff = diff_2d.narrow(1, 0, n_check)?.flatten_all()?;
                let last_diff = diff_2d.narrow(1, seq - n_check, n_check)?.flatten_all()?;
                let first_std: f32 = first_diff.var(0)?.to_scalar::<f32>()?.sqrt();
                let last_std: f32 = last_diff.var(0)?.to_scalar::<f32>()?.sqrt();
                // Per-token L2 norms
                let per_tok = diff_2d.sqr()?.sum(2)?.sqrt()?.squeeze(0)?; // [L]
                let tok_vals: Vec<f32> = per_tok.to_vec1()?;
                let nonzero = tok_vals.iter().filter(|&&v| v > 0.01).count();
                info!(
                    "  first {} tokens diff_std={:.6}, last {} diff_std={:.6}, nonzero(>0.01)={}/{}",
                    n_check, first_std, n_check, last_std, nonzero, seq
                );
            }
            (Some(neg_embeds), Some(neg_ctx_mask))
        } else {
            (None, None)
        };

        // DEBUG: optionally load Python reference connector outputs for comparison/substitution
        // Set LTX2_PYTHON_REF=/tmp/ltx2_connector_io.safetensors to enable
        let (prompt_embeds, context_mask, uncond_embeds, uncond_mask) =
            if let Ok(ref_path) = std::env::var("LTX2_PYTHON_REF") {
                info!("Loading Python reference connector outputs from {}", ref_path);
                let ref_tensors = candle_core::safetensors::load(&ref_path, &self.context.device)?;

                let py_pos = ref_tensors.get("prompt_connector_out")
                    .ok_or_else(|| anyhow::anyhow!("Missing prompt_connector_out"))?
                    .to_dtype(DType::BF16)?;
                let py_neg = ref_tensors.get("neg_connector_out")
                    .ok_or_else(|| anyhow::anyhow!("Missing neg_connector_out"))?
                    .to_dtype(DType::BF16)?;

                // Compare Rust vs Python connector outputs
                {
                    let rust_pos_f32 = prompt_embeds.to_dtype(DType::F32)?.flatten_all()?;
                    let py_pos_f32 = py_pos.to_dtype(DType::F32)?.flatten_all()?;
                    let pos_diff = (&rust_pos_f32 - &py_pos_f32)?;
                    info!("Rust vs Python connector pos: diff_std={:.6}, max_abs={:.6}",
                        pos_diff.var(0)?.to_scalar::<f32>()?.sqrt(),
                        pos_diff.abs()?.max(0)?.to_scalar::<f32>()?);
                }
                if let Some(ref rust_neg) = uncond_embeds {
                    let rust_neg_f32 = rust_neg.to_dtype(DType::F32)?.flatten_all()?;
                    let py_neg_f32 = py_neg.to_dtype(DType::F32)?.flatten_all()?;
                    let neg_diff = (&rust_neg_f32 - &py_neg_f32)?;
                    info!("Rust vs Python connector neg: diff_std={:.6}, max_abs={:.6}",
                        neg_diff.var(0)?.to_scalar::<f32>()?.sqrt(),
                        neg_diff.abs()?.max(0)?.to_scalar::<f32>()?);
                }

                // Substitute Python outputs
                info!("SUBSTITUTING Python connector outputs for this run");
                let pos_len = py_pos.dim(1)?;
                let neg_len = py_neg.dim(1)?;
                let pos_mask = Tensor::ones((1, pos_len), DType::F32, &self.context.device)?
                    .to_dtype(DType::BF16)?;
                let neg_mask = Tensor::ones((1, neg_len), DType::F32, &self.context.device)?
                    .to_dtype(DType::BF16)?;
                (py_pos, pos_mask, Some(py_neg), Some(neg_mask))
            } else {
                (prompt_embeds, context_mask, uncond_embeds, uncond_mask)
            };

        // 2. Prepare latents
        let latent_h = height / vae_config.spatial_compression_ratio;
        let latent_w = width / vae_config.spatial_compression_ratio;
        let latent_f = (num_frames - 1) / vae_config.temporal_compression_ratio + 1;
        let in_channels = trans_config.in_channels;

        // LTX-2 weights are BF16 — keep latents in BF16 throughout to avoid F16 precision loss
        let latents_5d = Tensor::randn(
            0f32,
            1f32,
            (1, in_channels, latent_f, latent_h, latent_w),
            &self.context.device,
        )?
        .to_dtype(DType::BF16)?;

        // NOTE: Python LTX2Pipeline does NOT normalize initial noise.
        // Normalization only happens when img2vid latents are provided.
        // For txt2vid, initial noise is standard normal, and only
        // denormalize_latents is applied at the end before VAE decode.
        let latents_mean =
            Tensor::new(self.latents_mean.as_slice(), &self.context.device)?;
        let latents_std =
            Tensor::new(self.latents_std.as_slice(), &self.context.device)?;

        // Pack latents: [B, C, F, H, W] -> [B, S, C] (patch_size=1)
        let mut latents = pack_latents(&latents_5d)?;

        // 3. Build video positions for RoPE
        let positions = build_video_positions(
            1, // batch_size
            latent_f,
            latent_h,
            latent_w,
            vae_config.temporal_compression_ratio,
            vae_config.spatial_compression_ratio,
            frame_rate,
            &self.context.device,
        )?;

        // 4. Prepare scheduler
        let num_tokens = latent_f * latent_h * latent_w;
        let scheduler = Ltx2Scheduler::new(sched_config);
        let sigmas = scheduler.execute(num_steps, num_tokens);

        info!(
            "Denoising: {} steps, {} tokens, sigma range {:.4}..{:.4}",
            num_steps,
            num_tokens,
            sigmas.first().unwrap_or(&0.0),
            sigmas.last().unwrap_or(&0.0),
        );

        // 5. Denoising loop
        let is_split = self.local_transformer.is_some();

        // STG config: LTX-2.3 defaults
        let stg_scale = stg_scale_arg.unwrap_or(1.0);
        let stg_block: usize = stg_block_arg.unwrap_or(28);
        let rescale_scale = rescale_arg.unwrap_or(0.7);
        let do_stg = stg_scale > 0.0;
        let stg_skip_blocks: Vec<usize> = if do_stg { vec![stg_block] } else { vec![] };

        if do_stg {
            info!(
                "STG enabled: scale={:.1}, block={}, rescale={:.2}",
                stg_scale, stg_block, rescale_scale
            );
        }

        // DEBUG: per-block diff diagnostic (cond vs uncond through local blocks)
        if is_split && do_cfg {
            let local = self.local_transformer.as_ref().unwrap();
            let sigma_test = Tensor::full(sigmas[0], (1,), &self.context.device)?
                .to_dtype(DType::BF16)?;
            let pos_f32 = positions.to_dtype(DType::F32)?;
            let lat_bf16 = latents.to_dtype(DType::BF16)?;

            // Setup for both contexts
            let ctx_cond = prompt_embeds.to_dtype(DType::BF16)?;
            let (hidden_c, temb_c, _ets_c, pe_c, ctx_proj_c, _ptc) =
                local.forward_setup(&lat_bf16, &sigma_test, &pos_f32, &ctx_cond)?;

            let uncond_ctx_t = uncond_embeds.as_ref().unwrap().to_dtype(DType::BF16)?;
            let (_hidden_u, _temb_u, _ets_u, _pe_u, ctx_proj_u, _ptu) =
                local.forward_setup(&lat_bf16, &sigma_test, &pos_f32, &uncond_ctx_t)?;

            // Caption projection diff
            let ctx_diff = (&ctx_proj_c.to_dtype(DType::F32)? - &ctx_proj_u.to_dtype(DType::F32)?)?;
            let ctx_diff_std: f32 = ctx_diff.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt();
            info!("PRE-FLIGHT: caption_projection diff: std={:.6}", ctx_diff_std);

            // Run blocks one-by-one, comparing cond vs uncond after each
            let mask_bf16 = context_mask.to_dtype(DType::BF16)?;
            let uncond_mask_bf16 = uncond_mask.as_ref().unwrap().to_dtype(DType::BF16)?;
            let mut x_c = hidden_c.clone();
            let mut x_u = hidden_c.clone(); // same initial hidden (from same latents)
            for (i, block) in local.blocks().iter().enumerate() {
                let global_idx = local.block_start() + i;
                x_c = block.forward_video_only(&x_c, &temb_c, Some(&pe_c), &ctx_proj_c, Some(&mask_bf16), None, false)?;
                x_u = block.forward_video_only(&x_u, &temb_c, Some(&pe_c), &ctx_proj_u, Some(&uncond_mask_bf16), None, false)?;

                let diff = (&x_c.to_dtype(DType::F32)? - &x_u.to_dtype(DType::F32)?)?;
                let diff_std: f32 = diff.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt();
                let pos_std: f32 = x_c.to_dtype(DType::F32)?.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt();
                info!("  block {:2}: diff_std={:.6}, pos_std={:.6}", global_idx, diff_std, pos_std);
            }

            // TEST: load Python's exact ca_query and ca_kv, run through block 0's attn2
            if let Ok(ref_path) = std::env::var("LTX2_CA_REF") {
                info!("Loading Python cross-attention reference from {}", ref_path);
                let ref_tensors = candle_core::safetensors::load(&ref_path, &self.context.device)?;

                let py_query = ref_tensors.get("ca_query").unwrap(); // [2, 2112, 4096] F32
                let py_kv = ref_tensors.get("ca_kv").unwrap(); // [2, 1024, 4096] F32
                let py_ca_out = ref_tensors.get("ca_out").unwrap(); // [2, 2112, 4096] F32

                // Run through Rust's block 0 attn2 with Python's exact inputs
                let block0 = &local.blocks()[0];
                let attn2 = block0.attn2();

                // Neg batch
                let q_neg = py_query.i(0..1)?.to_dtype(DType::BF16)?;
                let kv_neg = py_kv.i(0..1)?.to_dtype(DType::BF16)?;
                let rust_neg = attn2.forward(&q_neg, Some(&kv_neg), None, None, None)?;

                // Pos batch
                let q_pos = py_query.i(1..2)?.to_dtype(DType::BF16)?;
                let kv_pos = py_kv.i(1..2)?.to_dtype(DType::BF16)?;
                let rust_pos = attn2.forward(&q_pos, Some(&kv_pos), None, None, None)?;

                // Compare output diff
                let rust_diff = (&rust_pos.to_dtype(DType::F32)? - &rust_neg.to_dtype(DType::F32)?)?;
                let rust_diff_std: f32 = rust_diff.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt();

                let py_neg_out = py_ca_out.i(0..1)?;
                let py_pos_out = py_ca_out.i(1..2)?;
                let py_diff = (&py_pos_out - &py_neg_out)?;
                let py_diff_std: f32 = py_diff.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt();

                // Also check absolute match
                let rust_vs_py_neg = (&rust_neg.to_dtype(DType::F32)? - &py_neg_out)?;
                let neg_match_std: f32 = rust_vs_py_neg.flatten_all()?.var(0)?.to_scalar::<f32>()?.sqrt();
                let neg_match_max: f32 = rust_vs_py_neg.flatten_all()?.abs()?.max(0)?.to_scalar()?;

                info!("ATTN2 TEST: Rust ca_diff_std={:.6}, Python ca_diff_std={:.6}, ratio={:.3}",
                    rust_diff_std, py_diff_std, rust_diff_std / py_diff_std);
                info!("ATTN2 TEST: Rust vs Python neg output: diff_std={:.6}, max_abs={:.6}",
                    neg_match_std, neg_match_max);
            }
        }

        for step in 0..num_steps {
            let start_time = std::time::Instant::now();

            let sigma = sigmas[step];
            let sigma_next = sigmas[step + 1];

            let sigma_t = Tensor::full(sigma, (1,), &self.context.device)?
                .to_dtype(self.context.dtype)?;
            let timestep_t = Tensor::full(sigma, (1,), &self.context.device)?
                .to_dtype(self.context.dtype)?;

            // Conditional forward pass (no STG perturbation)
            let cond_velocity = if is_split {
                self.forward_split_transformer(
                    &latents, &sigma_t, &timestep_t, &positions,
                    &prompt_embeds, &context_mask, &[],
                ).await?
            } else {
                Ltx2Transformer::forward_packed(
                    &mut self.transformer,
                    latents.to_dtype(self.context.dtype)?,
                    sigma_t.clone(), timestep_t.clone(), positions.clone(),
                    prompt_embeds.clone(), context_mask.clone(),
                    &mut self.context,
                ).await?.to_dtype(DType::F32)?
            };

            // Apply guidance (CFG + STG)
            let mut velocity = cond_velocity.clone();

            // CFG: pred = cond + (cfg_scale - 1) * (cond - uncond)
            if do_cfg {
                let uncond_ctx = uncond_embeds.as_ref().unwrap();
                let uncond_mask = uncond_mask.as_ref().unwrap();

                let uncond_velocity = if is_split {
                    self.forward_split_transformer(
                        &latents, &sigma_t, &timestep_t, &positions,
                        uncond_ctx, uncond_mask, &[],
                    ).await?
                } else {
                    Ltx2Transformer::forward_packed(
                        &mut self.transformer,
                        latents.to_dtype(self.context.dtype)?,
                        sigma_t.clone(), timestep_t.clone(), positions.clone(),
                        uncond_ctx.clone(), uncond_mask.clone(),
                        &mut self.context,
                    ).await?.to_dtype(DType::F32)?
                };

                let cfg_diff = (&cond_velocity - &uncond_velocity)?;
                if step < 3 {
                    let diff_f32 = cfg_diff.to_dtype(DType::F32)?.flatten_all()?;
                    let diff_std: f32 = diff_f32.var(0)?.to_scalar::<f32>()?.sqrt();
                    info!("step {} CFG diff std={:.6}", step + 1, diff_std);
                }
                velocity = (&velocity + cfg_diff.affine((guidance_scale - 1.0) as f64, 0.0)?)?;
            }

            // STG: pred += stg_scale * (cond - perturbed)
            if do_stg {
                let stg_velocity = if is_split {
                    self.forward_split_transformer(
                        &latents, &sigma_t, &timestep_t, &positions,
                        &prompt_embeds, &context_mask, &stg_skip_blocks,
                    ).await?
                } else {
                    // For non-split mode, STG not yet supported
                    // (would need a separate forward_packed variant)
                    cond_velocity.clone()
                };

                let stg_diff = (&cond_velocity - &stg_velocity)?;
                if step < 3 {
                    let diff_f32 = stg_diff.to_dtype(DType::F32)?.flatten_all()?;
                    let diff_std: f32 = diff_f32.var(0)?.to_scalar::<f32>()?.sqrt();
                    info!("step {} STG diff std={:.6}", step + 1, diff_std);
                }
                velocity = (&velocity + stg_diff.affine(stg_scale as f64, 0.0)?)?;
            }

            // Rescale: prevent oversaturation from aggressive guidance
            if rescale_scale > 0.0 && (do_cfg || do_stg) {
                let cond_std: f32 = cond_velocity.to_dtype(DType::F32)?.flatten_all()?
                    .var(0)?.to_scalar::<f32>()?.sqrt();
                let pred_std: f32 = velocity.to_dtype(DType::F32)?.flatten_all()?
                    .var(0)?.to_scalar::<f32>()?.sqrt();
                if pred_std > 1e-8 {
                    let factor = rescale_scale as f64 * (cond_std / pred_std) as f64
                        + (1.0 - rescale_scale as f64);
                    velocity = velocity.affine(factor, 0.0)?;
                }
            }

            // Debug: log velocity and latent statistics
            if step < 3 || step == num_steps - 1 {
                let vel_f32 = velocity.to_dtype(DType::F32)?.flatten_all()?;
                let vel_min: f32 = vel_f32.min(0)?.to_scalar()?;
                let vel_max: f32 = vel_f32.max(0)?.to_scalar()?;
                let vel_mean: f32 = vel_f32.mean(0)?.to_scalar()?;
                let vel_std: f32 = vel_f32.var(0)?.to_scalar::<f32>()?.sqrt();
                info!(
                    "step {} velocity: min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
                    step + 1, vel_min, vel_max, vel_mean, vel_std
                );
            }

            // Euler step (keep in BF16 to match transformer weight precision)
            latents = euler_step(&latents.to_dtype(DType::F32)?, &velocity, sigma, sigma_next)?
                .to_dtype(DType::BF16)?;

            if step < 3 || step == num_steps - 1 {
                let lat_f32 = latents.to_dtype(DType::F32)?.flatten_all()?;
                let lat_min: f32 = lat_f32.min(0)?.to_scalar()?;
                let lat_max: f32 = lat_f32.max(0)?.to_scalar()?;
                let lat_mean: f32 = lat_f32.mean(0)?.to_scalar()?;
                info!(
                    "step {} latents: min={:.4}, max={:.4}, mean={:.4}",
                    step + 1, lat_min, lat_max, lat_mean
                );
            }

            let dt = start_time.elapsed().as_secs_f32();
            info!(
                "step {}/{} done, sigma={:.4}, {:.2}s",
                step + 1, num_steps, sigma, dt
            );
        }

        // 6. Unpack latents: [B, S, C] -> [B, C, F, H, W]
        let latents_5d = unpack_latents(
            &latents.to_dtype(DType::F32)?,
            latent_f,
            latent_h,
            latent_w,
        )?;

        // 7. Denormalize latents
        let latents_5d = denormalize_latents(
            &latents_5d,
            &latents_mean,
            &latents_std,
            vae_config.scaling_factor,
        )?
        .to_dtype(DType::BF16)?;

        // Debug: check latent statistics before VAE
        {
            let lat_f32 = latents_5d.to_dtype(DType::F32)?;
            let flat = lat_f32.flatten_all()?;
            let min_v: f32 = flat.min(0)?.to_scalar()?;
            let max_v: f32 = flat.max(0)?.to_scalar()?;
            let mean_v: f32 = flat.mean(0)?.to_scalar()?;
            info!(
                "Latents before VAE: shape={:?}, min={:.4}, max={:.4}, mean={:.4}",
                latents_5d.shape(), min_v, max_v, mean_v
            );
        }

        // 8. Decode with VAE
        info!("Decoding with VAE...");
        let decoded =
            Ltx2Vae::decode(&mut self.vae, latents_5d, &mut self.context).await?;

        // Debug: check decoded tensor stats
        {
            let dec_f32 = decoded.to_dtype(DType::F32)?;
            let flat = dec_f32.flatten_all()?;
            let min_v: f32 = flat.min(0)?.to_scalar()?;
            let max_v: f32 = flat.max(0)?.to_scalar()?;
            let mean_v: f32 = flat.mean(0)?.to_scalar()?;
            info!(
                "Decoded video: shape={:?}, dtype={:?}, min={:.4}, max={:.4}, mean={:.4}",
                decoded.shape(), decoded.dtype(), min_v, max_v, mean_v
            );
        }

        // 9. Convert video frames to images
        let frames = video_tensor_to_images(&decoded)?;
        info!("Generated {} frames", frames.len());

        Ok(VideoOutput::new(
            frames,
            frame_rate,
            width as u32,
            height as u32,
        ))
    }
}

impl Ltx2 {
    /// Forward pass through split transformer.
    ///
    /// Flow (master has first blocks with setup, worker has last blocks with finalize):
    /// 1. Master: setup (proj_in + adaln + caption + RoPE)
    /// 2. Master: run local blocks (0-23)
    /// 3. Send hidden states + metadata to worker
    /// 4. Worker: run remote blocks (24-47) + finalize
    /// 5. Worker returns velocity prediction
    async fn forward_split_transformer(
        &mut self,
        latents: &Tensor,
        _sigma: &Tensor,
        timestep: &Tensor,
        positions: &Tensor,
        context: &Tensor,
        context_mask: &Tensor,
        stg_skip_blocks: &[usize],
    ) -> Result<Tensor> {
        let local = self
            .local_transformer
            .as_ref()
            .expect("split mode requires local_transformer");

        // LTX-2 weights are BF16 — convert all inputs to BF16 to match
        let latents = latents.to_dtype(DType::BF16)?;
        let timestep = &timestep.to_dtype(DType::BF16)?;
        let positions = &positions.to_dtype(DType::F32)?; // RoPE always F32
        let context = &context.to_dtype(DType::BF16)?;

        // 1. Setup: proj_in + adaln + caption projection + RoPE (local)
        let (hidden, temb, embedded_ts, pe, ctx_projected, prompt_temb) =
            local.forward_setup(&latents, timestep, positions, context)?;

        // DEBUG: log caption_projection output and context diff for first few calls
        {
            static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let call = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if call < 6 {
                let ctx_f32 = ctx_projected.to_dtype(DType::F32)?.flatten_all()?;
                let ctx_std: f32 = ctx_f32.var(0)?.to_scalar::<f32>()?.sqrt();
                info!("split_transformer call {}: ctx_projected std={:.6}, stg_skip={:?}", call, ctx_std, stg_skip_blocks);
            }
        }

        // 2. Run local blocks (with STG if applicable)
        let context_mask_bf16 = context_mask.to_dtype(DType::BF16)?;
        let x = local.forward_blocks_with_stg(
            &hidden,
            &temb,
            &pe,
            &ctx_projected,
            Some(&context_mask_bf16),
            prompt_temb.as_ref(),
            stg_skip_blocks,
        )?;

        // DEBUG: log hidden state after local blocks
        {
            static LOCAL_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let call = LOCAL_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if call < 6 {
                let xf = x.to_dtype(DType::F32)?.flatten_all()?;
                let x_std: f32 = xf.var(0)?.to_scalar::<f32>()?.sqrt();
                let x_min: f32 = xf.min(0)?.to_scalar()?;
                let x_max: f32 = xf.max(0)?.to_scalar()?;
                info!("after local blocks (call {}): hidden std={:.6}, range=[{:.4},{:.4}]", call, x_std, x_min, x_max);
            }
        }

        // 3. Send to remote worker for remaining blocks + finalize
        let result = Ltx2Transformer::forward_blocks_packed(
            &mut self.transformer,
            x,
            temb,
            pe.0,
            pe.1,
            ctx_projected,
            context_mask.clone(),
            embedded_ts,
            prompt_temb,
            stg_skip_blocks,
            &mut self.context,
        )
        .await?;

        Ok(result.to_dtype(DType::F32)?)
    }
}

/// Convert a decoded video tensor `[B, C, T, H, W]` to a list of RGB images.
fn video_tensor_to_images(video: &Tensor) -> Result<Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>> {
    let mut result = Vec::new();

    let video = ((video.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .to_device(&Device::Cpu)?;

    let bsize = video.dim(0)?;
    for batch in 0..bsize {
        let batch_video = video.i(batch)?;
        let (channels, num_frames, height, width) = batch_video.dims4()?;
        if channels != 3 {
            anyhow::bail!("Expected 3 channels, got {}", channels);
        }

        for frame in 0..num_frames {
            let frame_tensor = batch_video.i((.., frame, .., ..))?;
            let frame_tensor = frame_tensor.permute((1, 2, 0))?.flatten_all()?;
            let pixels = frame_tensor.to_vec1::<u8>()?;

            let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_raw(width as u32, height as u32, pixels)
                    .ok_or_else(|| anyhow::anyhow!("Error creating image buffer"))?;
            result.push(image);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_tensor_to_images_basic() {
        let device = Device::Cpu;
        let video = Tensor::zeros((1, 3, 2, 4, 4), DType::F32, &device).unwrap();
        let frames = video_tensor_to_images(&video).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].width(), 4);
        assert_eq!(frames[0].height(), 4);
        assert_eq!(frames[0].get_pixel(0, 0)[0], 127);
    }

    #[test]
    fn test_video_tensor_to_images_clamping() {
        let device = Device::Cpu;
        let video = Tensor::full(2.0f32, (1, 3, 1, 2, 2), &device).unwrap();
        let frames = video_tensor_to_images(&video).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].get_pixel(0, 0)[0], 255);
    }

    #[test]
    fn test_video_tensor_to_images_multi_batch() {
        let device = Device::Cpu;
        let video = Tensor::zeros((2, 3, 3, 4, 4), DType::F32, &device).unwrap();
        let frames = video_tensor_to_images(&video).unwrap();
        assert_eq!(frames.len(), 6);
    }
}
