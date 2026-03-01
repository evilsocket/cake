//! HuggingFace Hub integration for automatic model downloading.

use std::path::PathBuf;

use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;

/// Returns true if the string looks like a HuggingFace repo ID (e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct").
pub fn looks_like_hf_repo(model: &str) -> bool {
    let parts: Vec<&str> = model.split('/').collect();
    parts.len() == 2
        && !model.starts_with('/')
        && !model.starts_with('.')
        && !model.starts_with('~')
        && !parts[0].is_empty()
        && !parts[1].is_empty()
}

/// Downloads all required model files from HuggingFace Hub.
/// Returns the local directory path containing the downloaded files.
/// Uses hf-hub's built-in caching — subsequent calls return instantly.
pub fn ensure_model_downloaded(repo_id: &str) -> Result<PathBuf> {
    log::info!(
        "model '{}' not found locally, downloading from HuggingFace Hub...",
        repo_id
    );

    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());

    // Download config.json first — validates repo access (fails fast on auth errors).
    log::info!("downloading config.json ...");
    repo.get("config.json")
        .map_err(|e| anyhow!("failed to download config.json from '{}': {}", repo_id, e))?;

    // Download tokenizer.
    log::info!("downloading tokenizer.json ...");
    repo.get("tokenizer.json")
        .map_err(|e| anyhow!("failed to download tokenizer.json from '{}': {}", repo_id, e))?;

    // Try sharded model first (model.safetensors.index.json), fall back to single file.
    let snapshot_dir = if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        log::info!("found sharded model, parsing index...");

        let index_data = std::fs::read(&index_path)?;
        let index_json: serde_json::Value = serde_json::from_slice(&index_data)?;
        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("no weight_map in model.safetensors.index.json"))?;

        let mut shard_files = std::collections::HashSet::new();
        for value in weight_map.values() {
            if let Some(file) = value.as_str() {
                shard_files.insert(file.to_string());
            }
        }

        log::info!("downloading {} shard files...", shard_files.len());
        for shard in &shard_files {
            log::info!("downloading {} ...", shard);
            repo.get(shard).map_err(|e| {
                anyhow!(
                    "failed to download shard '{}' from '{}': {}",
                    shard,
                    repo_id,
                    e
                )
            })?;
        }

        index_path.parent().unwrap().to_path_buf()
    } else {
        log::info!("downloading model.safetensors ...");
        let model_path = repo.get("model.safetensors").map_err(|e| {
            anyhow!(
                "failed to download model from '{}': no index.json and no model.safetensors found: {}",
                repo_id,
                e
            )
        })?;
        model_path.parent().unwrap().to_path_buf()
    };

    log::info!("model files ready at {}", snapshot_dir.display());
    Ok(snapshot_dir)
}
