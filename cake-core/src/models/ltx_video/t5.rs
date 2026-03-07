use crate::cake::{Context, Forwarder};
use async_trait::async_trait;
use candle_core::Tensor;
use candle_transformers::models::t5::{self, T5EncoderModel};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::Cache;
use log::info;
use std::fmt::{Debug, Display, Formatter};
use std::path::PathBuf;

const T5_XXL_REPO: &str = "google/t5-v1_1-xxl";

#[derive(Debug)]
pub struct LtxT5 {
    model: T5EncoderModel,
}

impl Display for LtxT5 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ltx-t5 (local)")
    }
}

#[async_trait]
impl Forwarder for LtxT5 {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(ctx)
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        anyhow::bail!("T5 encoder requires forward_mut (has KV cache)")
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("LTX T5 encoder forwarding...");
        Ok(self.model.forward(x)?)
    }

    fn layer_name(&self) -> &str {
        "ltx-t5"
    }
}

impl LtxT5 {
    /// Resolve a file from the LTX model repo or T5-XXL repo via HuggingFace cache.
    fn resolve_hf_file(
        repo: &str,
        file: &str,
        cache_dir: &str,
    ) -> anyhow::Result<PathBuf> {
        let mut cache_path = PathBuf::from(cache_dir);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let filename = api.model(repo.to_string()).get(file)?;
        Ok(filename)
    }

    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let ltx_args = &ctx.args.ltx_args;

        // Load T5 config from the LTX model repo (or T5-XXL fallback)
        let config_path = if let Some(ref p) = ltx_args.ltx_t5_config {
            PathBuf::from(p)
        } else {
            // LTX-Video ships T5 config in the main repo
            let ltx_repo = ltx_args.ltx_repo();
            Self::resolve_hf_file(&ltx_repo, "text_encoder/config.json", &ctx.args.model)
                .or_else(|_| {
                    Self::resolve_hf_file(T5_XXL_REPO, "config.json", &ctx.args.model)
                })?
        };

        info!("Loading T5 config from {:?}...", config_path);
        let config: t5::Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        // Load T5 weights (potentially sharded)
        let weight_files = if let Some(ref p) = ltx_args.ltx_t5 {
            p.split(',').map(|s| PathBuf::from(s.trim())).collect()
        } else {
            let ltx_repo = ltx_args.ltx_repo();
            Self::get_t5_weight_files(&ltx_repo, &ctx.args.model)?
        };

        info!("Loading T5 encoder from {:?}...", weight_files);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, ctx.dtype, &ctx.device)?
        };
        let model = T5EncoderModel::load(vb, &config)?;

        info!("T5 encoder loaded!");

        Ok(Box::new(Self { model }))
    }

    fn get_t5_weight_files(repo: &str, cache_dir: &str) -> anyhow::Result<Vec<PathBuf>> {
        let mut cache_path = PathBuf::from(cache_dir);
        cache_path.push("hub");
        let cache = Cache::new(cache_path);
        let api = ApiBuilder::from_cache(cache).build()?;
        let model_api = api.model(repo.to_string());

        // Try single file first
        if let Ok(path) = model_api.get("text_encoder/model.safetensors") {
            return Ok(vec![path]);
        }

        // Fall back to 2-shard format
        let shard1 = model_api.get("text_encoder/model-00001-of-00002.safetensors")?;
        let shard2 = model_api.get("text_encoder/model-00002-of-00002.safetensors")?;
        Ok(vec![shard1, shard2])
    }

    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        tokens: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        forwarder.forward_mut(&tokens, 0, 0, ctx).await
    }
}
