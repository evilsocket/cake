//! HuggingFace Hub integration for automatic model downloading.

use std::collections::HashSet;
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

/// Check the HuggingFace cache for an already-downloaded complete model.
/// Returns the snapshot directory path if found with all shards present.
fn find_cached_model(repo_id: &str) -> Option<PathBuf> {
    let hf_cache = hf_cache_dir()?;

    // HF cache dirs look like "models--org--model-name"
    let cache_dir_name = format!("models--{}", repo_id.replace('/', "--"));
    let model_dir = hf_cache.join(&cache_dir_name);
    let snapshots_dir = model_dir.join("snapshots");

    if !snapshots_dir.exists() {
        return None;
    }

    // Check each snapshot (usually just one, pick the newest)
    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;

    for entry in std::fs::read_dir(&snapshots_dir).ok()?.flatten() {
        let snap_path = entry.path();
        if !snap_path.is_dir() {
            continue;
        }

        // Must have config.json
        if !snap_path.join("config.json").exists() {
            continue;
        }

        // Check model completeness
        let is_complete = if snap_path.join("model.safetensors").exists() {
            true
        } else if let Ok(index_data) =
            std::fs::read_to_string(snap_path.join("model.safetensors.index.json"))
        {
            if let Ok(index_json) = serde_json::from_str::<serde_json::Value>(&index_data) {
                if let Some(weight_map) = index_json.get("weight_map").and_then(|v| v.as_object())
                {
                    let expected: HashSet<&str> =
                        weight_map.values().filter_map(|v| v.as_str()).collect();
                    expected.iter().all(|f| snap_path.join(f).exists())
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if is_complete {
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if best.as_ref().map_or(true, |(_, t)| mtime > *t) {
                best = Some((snap_path, mtime));
            }
        }
    }

    best.map(|(p, _)| p)
}

/// Return the HuggingFace hub cache directory if it exists.
pub fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(dir) = std::env::var("HF_HOME") {
        let p = PathBuf::from(dir).join("hub");
        if p.exists() {
            return Some(p);
        }
    }
    let home = dirs::home_dir()?;
    let p = home.join(".cache/huggingface/hub");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// Downloads all required model files from HuggingFace Hub.
/// Returns the local directory path containing the downloaded files.
/// Checks local cache first — returns instantly if model already downloaded.
pub fn ensure_model_downloaded(repo_id: &str) -> Result<PathBuf> {
    // Check local cache first
    if let Some(cached_path) = find_cached_model(repo_id) {
        log::info!(
            "model '{}' found in cache at {}",
            repo_id,
            cached_path.display()
        );
        return Ok(cached_path);
    }

    log::info!(
        "downloading model '{}' from HuggingFace Hub...",
        repo_id
    );

    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());

    // Download config.json first — validates repo access (fails fast on auth errors).
    log::info!("downloading config.json ...");
    repo.download("config.json")
        .map_err(|e| anyhow!("failed to download config.json from '{}': {}", repo_id, e))?;

    // Download tokenizer.
    log::info!("downloading tokenizer.json ...");
    repo.download("tokenizer.json")
        .map_err(|e| anyhow!("failed to download tokenizer.json from '{}': {}", repo_id, e))?;

    // Try sharded model first (model.safetensors.index.json), fall back to single file.
    let snapshot_dir = if let Ok(index_path) = repo.download("model.safetensors.index.json") {
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
        for (i, shard) in shard_files.iter().enumerate() {
            log::info!("[{}/{}] downloading {} ...", i + 1, shard_files.len(), shard);
            repo.download(shard).map_err(|e| {
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
        let model_path = repo.download("model.safetensors").map_err(|e| {
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
