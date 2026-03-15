use crate::cake::{Context, Forwarder};
use crate::models::sd::{pack_tensors, unpack_tensors};
use async_trait::async_trait;
use candle_core::{DType, Tensor};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::fmt::{Debug, Display, Formatter};
use std::path::PathBuf;

use super::vendored::configs::get_config_by_version;
use super::vendored::ltx_transformer::LtxVideoTransformer3DModel;
use super::vendored::t2v_pipeline::TransformerConfig;

#[derive(Debug)]
pub struct LtxTransformer {
    model: LtxVideoTransformer3DModel,
}

impl Display for LtxTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ltx-transformer (local)")
    }
}

#[async_trait]
impl Forwarder for LtxTransformer {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(ctx)
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        // Packed format: [hidden_states, encoder_hidden_states, timestep,
        //                 encoder_attention_mask, video_coords,
        //                 dims_tensor(num_frames, height, width)]
        let hidden_states = unpacked[0].to_dtype(ctx.dtype)?;
        let encoder_hidden_states = unpacked[1].to_dtype(ctx.dtype)?;
        let timestep = unpacked[2].to_dtype(ctx.dtype)?;
        let encoder_attention_mask = unpacked[3].to_dtype(ctx.dtype)?;
        let video_coords = unpacked[4].to_dtype(DType::F32)?;
        let dims: Vec<f32> = unpacked[5].to_vec1()?;
        let num_frames = dims[0] as usize;
        let height = dims[1] as usize;
        let width = dims[2] as usize;

        info!("LTX transformer forwarding...");

        let result = self.model.forward(
            &hidden_states,
            &encoder_hidden_states,
            &timestep,
            Some(&encoder_attention_mask),
            num_frames,
            height,
            width,
            None,
            Some(&video_coords),
            None,
        )?;

        Ok(result)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        "ltx-transformer"
    }
}

impl LtxTransformer {
    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let ltx_args = &ctx.args.ltx_args;
        let version = &ltx_args.ltx_version;
        let config = get_config_by_version(version);

        let weights_path = if let Some(ref p) = ltx_args.ltx_transformer {
            PathBuf::from(p)
        } else {
            let repo = ltx_args.ltx_repo();
            let mut cache_path = PathBuf::from(&ctx.args.model);
            cache_path.push("hub");
            let cache = Cache::new(cache_path);
            let api = ApiBuilder::from_cache(cache).build()?;
            let model_api = api.model(repo);

            // Try single file first, then sharded
            if let Ok(path) = model_api.get("transformer/diffusion_pytorch_model.safetensors") {
                path
            } else {
                // Try sharded format
                let index_path = model_api
                    .get("transformer/diffusion_pytorch_model.safetensors.index.json")?;
                let _index: serde_json::Value =
                    serde_json::from_reader(std::fs::File::open(&index_path)?)?;
                // Just return the first shard path - loading will handle all
                index_path
                    .parent()
                    .unwrap()
                    .join("diffusion_pytorch_model-00001-of-00002.safetensors")
            }
        };

        info!(
            "Loading LTX transformer (version {}) from {:?}...",
            version, weights_path
        );

        // Handle sharded weights
        let weight_files = Self::find_weight_files(&weights_path)?;

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &weight_files,
                ctx.dtype,
                &ctx.device,
            )?
        };

        let model = LtxVideoTransformer3DModel::new(&config.transformer, vb)?;

        info!("LTX transformer loaded!");

        Ok(Box::new(Self { model }))
    }

    fn find_weight_files(path: &PathBuf) -> anyhow::Result<Vec<PathBuf>> {
        // If the path is a single safetensors file, use it
        if path.extension().map_or(false, |e| e == "safetensors") && path.exists() {
            return Ok(vec![path.clone()]);
        }

        // Check for sharded format in the same directory
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

    pub fn pipeline_config(version: &str) -> TransformerConfig {
        let config = get_config_by_version(version);
        TransformerConfig {
            in_channels: config.transformer.in_channels,
            patch_size: config.transformer.patch_size,
            patch_size_t: config.transformer.patch_size_t,
            num_layers: config.transformer.num_layers,
        }
    }

    /// Pack tensors for network transport and call the forwarder.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_packed(
        forwarder: &mut Box<dyn Forwarder>,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_attention_mask: Tensor,
        video_coords: Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        let dims = Tensor::from_vec(
            vec![num_frames as f32, height as f32, width as f32],
            3,
            &ctx.device,
        )?;
        let tensors = vec![
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            video_coords,
            dims,
        ];
        let packed = pack_tensors(tensors, &ctx.device)?;
        forwarder.forward_mut(&packed, 0, 0, ctx).await
    }
}
