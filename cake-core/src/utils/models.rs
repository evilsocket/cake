//! Scan local directories for cached/downloaded models and report their status.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;

/// Status of a locally cached model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// All safetensors shards present — model is ready for master or standalone use.
    Complete,
    /// Only a subset of shards present — typically a worker split or partial push.
    Partial {
        have: usize,
        total: usize,
    },
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelStatus::Complete => write!(f, "complete"),
            ModelStatus::Partial { have, total } => {
                write!(f, "partial ({}/{} shards)", have, total)
            }
        }
    }
}

/// A discovered local model.
#[derive(Debug, Clone)]
pub struct LocalModel {
    /// Human-readable model name (e.g. "evilsocket/Qwen2.5-Coder-1.5B-Instruct").
    pub name: String,
    /// Absolute path to the model directory.
    pub path: PathBuf,
    /// Where this model was found.
    pub source: ModelSource,
    /// Completeness status.
    pub status: ModelStatus,
    /// Total size of model files on disk (bytes).
    pub size_bytes: u64,
}

/// Where a model was discovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Downloaded via `hf-hub` (lives in `~/.cache/huggingface/hub/`).
    HuggingFaceCache,
    /// Received from a master during zero-config setup (lives in `~/.cache/cake/<hash>/`).
    ClusterCache { cluster_hash: String },
    /// A user-provided local directory.
    Local,
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSource::HuggingFaceCache => write!(f, "huggingface"),
            ModelSource::ClusterCache { cluster_hash } => {
                write!(f, "cluster ({})", cluster_hash)
            }
            ModelSource::Local => write!(f, "local"),
        }
    }
}

/// Scan all known locations for models and return a list of discovered models.
pub fn list_models() -> Result<Vec<LocalModel>> {
    let mut models = Vec::new();

    // 1. Scan HuggingFace cache
    if let Some(hf_cache) = hf_cache_dir() {
        scan_hf_cache(&hf_cache, &mut models)?;
    }

    // 2. Scan zero-config cluster cache
    if let Some(cake_cache) = cake_cache_dir() {
        scan_cake_cache(&cake_cache, &mut models)?;
    }

    // Sort: complete first, then by name
    models.sort_by(|a, b| {
        let status_ord = |s: &ModelStatus| match s {
            ModelStatus::Complete => 0,
            ModelStatus::Partial { .. } => 1,
        };
        status_ord(&a.status)
            .cmp(&status_ord(&b.status))
            .then_with(|| a.name.cmp(&b.name))
    });

    Ok(models)
}

/// Find a model by name. Supports exact match (`org/model`) or suffix match (`model`).
/// Returns `Err` if the suffix match is ambiguous (multiple models match).
pub fn find_model(name: &str) -> Result<Option<LocalModel>> {
    let models = list_models()?;

    // Exact match first
    if let Some(m) = models.iter().find(|m| m.name == name) {
        return Ok(Some(m.clone()));
    }

    // Suffix match: "Qwen3-0.6B" matches "evilsocket/Qwen3-0.6B"
    let suffix = format!("/{name}");
    let matches: Vec<_> = models.iter().filter(|m| m.name.ends_with(&suffix)).collect();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches[0].clone())),
        _ => anyhow::bail!(
            "'{}' is ambiguous, matches: {}",
            name,
            matches
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Delete a cached model from disk.
///
/// For HuggingFace cache models, removes the entire `models--org--name/` directory
/// (including all snapshots and blobs). For cluster cache models, removes the model
/// directory. Refuses to delete local models.
pub fn delete_model(model: &LocalModel) -> Result<()> {
    match model.source {
        ModelSource::HuggingFaceCache => {
            // model.path is .../snapshots/{hash}/ — go up to models--org--name/
            let hf_model_dir = model
                .path
                .parent() // snapshots/
                .and_then(|p| p.parent()) // models--org--name/
                .ok_or_else(|| anyhow::anyhow!("unexpected HF cache structure for {}", model.path.display()))?;
            std::fs::remove_dir_all(hf_model_dir)?;
        }
        ModelSource::ClusterCache { .. } => {
            std::fs::remove_dir_all(&model.path)?;
        }
        ModelSource::Local => {
            anyhow::bail!(
                "refusing to delete local model '{}' — use rm -rf manually",
                model.name
            );
        }
    }
    Ok(())
}

/// Check a single directory and return its model status, or None if it's not a model dir.
fn check_model_dir(dir: &Path) -> Option<(ModelStatus, u64)> {
    // Must have config.json to be considered a model
    if !dir.join("config.json").exists() {
        return None;
    }

    let mut total_size: u64 = 0;

    // Count config + tokenizer sizes
    for name in &["config.json", "tokenizer.json"] {
        let p = dir.join(name);
        if p.exists() {
            total_size += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        }
    }

    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        total_size += std::fs::metadata(&index_path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Sharded model — check which shards are present
        let index_data = std::fs::read_to_string(&index_path).ok()?;
        let index_json: serde_json::Value = serde_json::from_str(&index_data).ok()?;
        let weight_map = index_json.get("weight_map")?.as_object()?;

        let mut expected_shards: HashSet<String> = HashSet::new();
        for value in weight_map.values() {
            if let Some(file) = value.as_str() {
                expected_shards.insert(file.to_string());
            }
        }

        let total = expected_shards.len();
        let mut have = 0;
        for shard in &expected_shards {
            let shard_path = dir.join(shard);
            if shard_path.exists() {
                have += 1;
                total_size += std::fs::metadata(&shard_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
            }
        }

        let status = if have == total {
            ModelStatus::Complete
        } else {
            ModelStatus::Partial { have, total }
        };

        Some((status, total_size))
    } else {
        // Single safetensors file
        let single = dir.join("model.safetensors");
        if single.exists() {
            total_size += std::fs::metadata(&single)
                .map(|m| m.len())
                .unwrap_or(0);
            Some((ModelStatus::Complete, total_size))
        } else {
            // Has config but no weights at all
            Some((ModelStatus::Partial { have: 0, total: 1 }, total_size))
        }
    }
}

/// Return the HuggingFace hub cache directory if it exists.
fn hf_cache_dir() -> Option<PathBuf> {
    super::hf::hf_cache_dir()
}

/// Return the Cake cluster cache directory if it exists.
fn cake_cache_dir() -> Option<PathBuf> {
    let cache = dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("cake");
    if cache.exists() {
        Some(cache)
    } else {
        None
    }
}

/// Scan the HuggingFace hub cache for model snapshots.
fn scan_hf_cache(hf_cache: &Path, models: &mut Vec<LocalModel>) -> Result<()> {
    let entries = match std::fs::read_dir(hf_cache) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        // HF cache dirs look like "models--org--model-name"
        if !dir_name.starts_with("models--") {
            continue;
        }

        let snapshots_dir = entry.path().join("snapshots");
        if !snapshots_dir.exists() {
            continue;
        }

        // Parse model name from dir: "models--Qwen--Qwen2.5-Coder-1.5B-Instruct" → "evilsocket/Qwen2.5-Coder-1.5B-Instruct"
        let model_name = dir_name
            .strip_prefix("models--")
            .unwrap_or(&dir_name)
            .replacen("--", "/", 1);

        // Check each snapshot (usually just one)
        let snapshot_entries = match std::fs::read_dir(&snapshots_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for snap_entry in snapshot_entries.flatten() {
            let snap_path = snap_entry.path();
            if !snap_path.is_dir() {
                continue;
            }

            if let Some((status, size_bytes)) = check_model_dir(&snap_path) {
                models.push(LocalModel {
                    name: model_name.clone(),
                    path: snap_path,
                    source: ModelSource::HuggingFaceCache,
                    status,
                    size_bytes,
                });
            }
        }
    }

    Ok(())
}

/// Scan the Cake cluster cache for worker-received models.
fn scan_cake_cache(cake_cache: &Path, models: &mut Vec<LocalModel>) -> Result<()> {
    let entries = match std::fs::read_dir(cake_cache) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }

        let cluster_hash = entry.file_name().to_string_lossy().to_string();

        if let Some((status, size_bytes)) = check_model_dir(&dir) {
            // Try to extract a model name from config.json
            let name = read_model_name_from_config(&dir)
                .unwrap_or_else(|| format!("cluster:{}", &cluster_hash));

            models.push(LocalModel {
                name,
                path: dir,
                source: ModelSource::ClusterCache { cluster_hash },
                status,
                size_bytes,
            });
        }
    }

    Ok(())
}

/// Try to read a model name from config.json (e.g. _name_or_path field).
fn read_model_name_from_config(dir: &Path) -> Option<String> {
    let config = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&config).ok()?;

    // Try common fields that identify the model
    json.get("_name_or_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            json.get("model_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn check_model_dir_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(check_model_dir(tmp.path()).is_none());
    }

    #[test]
    fn check_model_dir_config_only() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let result = check_model_dir(tmp.path());
        assert!(result.is_some());
        let (status, _size) = result.unwrap();
        assert_eq!(status, ModelStatus::Partial { have: 0, total: 1 });
    }

    #[test]
    fn check_model_dir_single_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        fs::write(tmp.path().join("model.safetensors"), "fake-tensor-data").unwrap();
        let result = check_model_dir(tmp.path());
        assert!(result.is_some());
        let (status, size) = result.unwrap();
        assert_eq!(status, ModelStatus::Complete);
        assert!(size > 0);
    }

    #[test]
    fn check_model_dir_sharded_complete() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "layer.0.weight": "shard-00001.safetensors",
                "layer.1.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data1").unwrap();
        fs::write(tmp.path().join("shard-00002.safetensors"), "data2").unwrap();

        let result = check_model_dir(tmp.path());
        assert!(result.is_some());
        let (status, _size) = result.unwrap();
        assert_eq!(status, ModelStatus::Complete);
    }

    #[test]
    fn check_model_dir_sharded_missing_shard() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "layer.0.weight": "shard-00001.safetensors",
                "layer.1.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data1").unwrap();
        // shard-00002 missing

        let result = check_model_dir(tmp.path());
        assert!(result.is_some());
        let (status, _size) = result.unwrap();
        assert_eq!(status, ModelStatus::Partial { have: 1, total: 2 });
    }

    #[test]
    fn read_model_name_from_config_name_or_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "_name_or_path": "evilsocket/Qwen2.5-Coder-1.5B-Instruct"
        });
        fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();
        let result = read_model_name_from_config(tmp.path());
        assert_eq!(result, Some("evilsocket/Qwen2.5-Coder-1.5B-Instruct".to_string()));
    }

    #[test]
    fn read_model_name_from_config_model_type_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "model_type": "llama"
        });
        fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();
        let result = read_model_name_from_config(tmp.path());
        assert_eq!(result, Some("llama".to_string()));
    }

    #[test]
    fn read_model_name_from_config_missing_fields() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), r#"{"hidden_size": 1024}"#).unwrap();
        let result = read_model_name_from_config(tmp.path());
        assert!(result.is_none());
    }

    #[test]
    fn read_model_name_from_config_no_config_file() {
        let tmp = tempfile::tempdir().unwrap();
        let result = read_model_name_from_config(tmp.path());
        assert!(result.is_none());
    }

    #[test]
    fn model_status_display() {
        assert_eq!(format!("{}", ModelStatus::Complete), "complete");
        assert_eq!(
            format!("{}", ModelStatus::Partial { have: 3, total: 5 }),
            "partial (3/5 shards)"
        );
    }

    #[test]
    fn model_source_display() {
        assert_eq!(format!("{}", ModelSource::HuggingFaceCache), "huggingface");
        assert_eq!(format!("{}", ModelSource::Local), "local");
        assert_eq!(
            format!(
                "{}",
                ModelSource::ClusterCache {
                    cluster_hash: "abc".to_string()
                }
            ),
            "cluster (abc)"
        );
    }
}
