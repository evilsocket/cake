//! HuggingFace Hub integration for automatic model downloading.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;

/// Returns true if the string looks like a HuggingFace repo ID (e.g., "evilsocket/Qwen2.5-Coder-1.5B-Instruct").
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
            if best.as_ref().is_none_or(|(_, t)| mtime > *t) {
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

    let mut builder = ApiBuilder::new().with_progress(true);

    // Use explicit cache dir if HF_HUB_CACHE is set, to avoid hf_hub bugs
    // with env var handling (lock files created at wrong path).
    if let Ok(cache_dir) = std::env::var("HF_HUB_CACHE") {
        builder = builder.with_cache_dir(PathBuf::from(cache_dir));
    }

    let api = builder.build()?;
    let repo = api.model(repo_id.to_string());

    // Download config.json (optional — image models like FLUX don't have it).
    log::info!("downloading config.json ...");
    match repo.download("config.json") {
        Ok(_) => log::info!("config.json downloaded"),
        Err(_) => log::info!("no config.json (image/diffusion model)"),
    }

    // Download tokenizer (optional — some models like VibeVoice use external tokenizers).
    log::info!("downloading tokenizer.json ...");
    if repo.download("tokenizer.json").is_err() {
        log::info!("no tokenizer.json");
    }

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
    } else if let Ok(model_path) = repo.download("model.safetensors") {
        log::info!("downloading model.safetensors ...");
        model_path.parent().unwrap().to_path_buf()
    } else {
        // No standard model layout — list repo files and download all .safetensors
        // (e.g. FLUX repos use flux1-dev-fp8.safetensors at root)
        log::info!("no standard model layout, listing repo files...");
        let repo_info = repo.info().map_err(|e| {
            anyhow!("failed to get repo info for '{}': {}", repo_id, e)
        })?;
        let safetensor_files: Vec<&str> = repo_info.siblings.iter()
            .map(|s| s.rfilename.as_str())
            .filter(|f| f.ends_with(".safetensors") && !f.contains('/'))
            .collect();
        if safetensor_files.is_empty() {
            anyhow::bail!(
                "no .safetensors files found in '{}' — cannot download model",
                repo_id
            );
        }
        log::info!("found {} safetensor files to download", safetensor_files.len());
        let mut snapshot_dir = None;
        for (i, file) in safetensor_files.iter().enumerate() {
            log::info!("[{}/{}] downloading {} ...", i + 1, safetensor_files.len(), file);
            let path = repo.download(file).map_err(|e| {
                anyhow!("failed to download '{}' from '{}': {}", file, repo_id, e)
            })?;
            if snapshot_dir.is_none() {
                snapshot_dir = Some(path.parent().unwrap().to_path_buf());
            }
        }
        snapshot_dir.unwrap()
    };

    log::info!("model files ready at {}", snapshot_dir.display());
    Ok(snapshot_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn looks_like_hf_repo_valid() {
        assert!(looks_like_hf_repo("evilsocket/Qwen2.5-Coder-1.5B-Instruct"));
        assert!(looks_like_hf_repo("org/model"));
        assert!(looks_like_hf_repo("meta-llama/Llama-3-8B"));
    }

    #[test]
    fn looks_like_hf_repo_too_many_slashes() {
        assert!(!looks_like_hf_repo("path/to/dir"));
        assert!(!looks_like_hf_repo("a/b/c"));
    }

    #[test]
    fn looks_like_hf_repo_empty() {
        assert!(!looks_like_hf_repo(""));
    }

    #[test]
    fn looks_like_hf_repo_single_segment() {
        assert!(!looks_like_hf_repo("single"));
        assert!(!looks_like_hf_repo("model-name"));
    }

    #[test]
    fn looks_like_hf_repo_path_prefixes() {
        assert!(!looks_like_hf_repo("/absolute/path"));
        assert!(!looks_like_hf_repo("./relative/path"));
        assert!(!looks_like_hf_repo("~/home/path"));
    }

    #[test]
    fn looks_like_hf_repo_empty_parts() {
        assert!(!looks_like_hf_repo("/trailing"));
        assert!(!looks_like_hf_repo("leading/"));
    }

    // Mutex to serialize env-var-dependent tests (env vars are process-global).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn find_cached_model_complete_single() {
        let _lock = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        std::env::set_var("HF_HUB_CACHE", cache.to_str().unwrap());

        let snap = cache
            .join("models--test--model")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snap).unwrap();
        fs::write(snap.join("config.json"), "{}").unwrap();
        fs::write(snap.join("model.safetensors"), "data").unwrap();

        let result = find_cached_model("test/model");
        assert_eq!(result, Some(snap));

        std::env::remove_var("HF_HUB_CACHE");
    }

    #[test]
    fn find_cached_model_complete_sharded() {
        let _lock = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        std::env::set_var("HF_HUB_CACHE", cache.to_str().unwrap());

        let snap = cache
            .join("models--org--sharded")
            .join("snapshots")
            .join("def456");
        fs::create_dir_all(&snap).unwrap();
        fs::write(snap.join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "layer.0.weight": "shard-00001.safetensors",
                "layer.1.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            snap.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(snap.join("shard-00001.safetensors"), "data").unwrap();
        fs::write(snap.join("shard-00002.safetensors"), "data").unwrap();

        let result = find_cached_model("org/sharded");
        assert_eq!(result, Some(snap));

        std::env::remove_var("HF_HUB_CACHE");
    }

    #[test]
    fn find_cached_model_missing_shard() {
        let _lock = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        std::env::set_var("HF_HUB_CACHE", cache.to_str().unwrap());

        let snap = cache
            .join("models--org--partial")
            .join("snapshots")
            .join("ghi789");
        fs::create_dir_all(&snap).unwrap();
        fs::write(snap.join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "layer.0.weight": "shard-00001.safetensors",
                "layer.1.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            snap.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(snap.join("shard-00001.safetensors"), "data").unwrap();

        let result = find_cached_model("org/partial");
        assert!(result.is_none());

        std::env::remove_var("HF_HUB_CACHE");
    }

    #[test]
    fn find_cached_model_not_present() {
        let _lock = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path();
        std::env::set_var("HF_HUB_CACHE", cache.to_str().unwrap());

        let result = find_cached_model("nonexistent/model");
        assert!(result.is_none());

        std::env::remove_var("HF_HUB_CACHE");
    }

    #[test]
    fn hf_cache_dir_returns_path_when_set() {
        let _lock = ENV_LOCK.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", tmp.path().to_str().unwrap());

        let result = hf_cache_dir();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), tmp.path());

        std::env::remove_var("HF_HUB_CACHE");
    }
}
