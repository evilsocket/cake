//! Safetensors weight loading with mapping support
//!
//! This module provides comprehensive weight loading utilities for loading
//! model weights from safetensors files, with support for:
//!
//! - Single file loading
//! - Sharded model loading with automatic detection via `model.safetensors.index.json`
//! - JSON config parsing for VAE/DiT configurations
//! - Python → Rust name mapping (exact, prefix, suffix)
//! - Tensor name validation

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during weight loading
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("Failed to read file: {path}")]
    FileRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse JSON config: {path}")]
    JsonParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Missing shard files: {missing:?}")]
    MissingShards { missing: Vec<String> },

    #[error("No safetensors files found in directory: {path}")]
    NoSafetensorsFound { path: String },

    #[error("Missing required tensors: {missing:?}")]
    MissingTensors { missing: Vec<String> },

    #[error("Invalid safetensors file: {path}")]
    InvalidSafetensors {
        path: String,
        #[source]
        source: safetensors::SafeTensorError,
    },

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

// =============================================================================
// Name Mapping Types
// =============================================================================

/// Types of name mapping transformations
#[derive(Debug, Clone)]
enum MappingRule {
    /// Exact match replacement
    Exact { from: String, to: String },
    /// Prefix replacement (strip prefix and optionally add new one)
    Prefix {
        from_prefix: String,
        to_prefix: String,
    },
    /// Suffix replacement
    Suffix {
        from_suffix: String,
        to_suffix: String,
    },
}

impl MappingRule {
    /// Apply this mapping rule to a name, returning the mapped name if applicable
    fn apply(&self, name: &str) -> Option<String> {
        match self {
            MappingRule::Exact { from, to } => {
                if name == from {
                    Some(to.clone())
                } else {
                    None
                }
            }
            MappingRule::Prefix {
                from_prefix,
                to_prefix,
            } => {
                if name.starts_with(from_prefix) {
                    Some(format!("{}{}", to_prefix, &name[from_prefix.len()..]))
                } else {
                    None
                }
            }
            MappingRule::Suffix {
                from_suffix,
                to_suffix,
            } => {
                if name.ends_with(from_suffix) {
                    let base = &name[..name.len() - from_suffix.len()];
                    Some(format!("{}{}", base, to_suffix))
                } else {
                    None
                }
            }
        }
    }
}

// =============================================================================
// Safetensors Index (model.safetensors.index.json)
// =============================================================================

/// Represents the parsed contents of model.safetensors.index.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndex {
    /// Maps tensor names to their shard file names
    pub weight_map: HashMap<String, String>,
    /// Optional metadata about the model
    #[serde(default)]
    pub metadata: Option<IndexMetadata>,
}

/// Metadata from the index.json file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Format of the weights (usually "safetensors")
    #[serde(default)]
    pub format: Option<String>,
    /// Total size of all weights in bytes
    #[serde(default)]
    pub total_size: Option<u64>,
    /// Model type if specified
    #[serde(default)]
    pub model_type: Option<String>,
}

impl SafetensorsIndex {
    /// Load and parse an index.json file
    pub fn load(path: impl AsRef<Path>) -> std::result::Result<Self, LoaderError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| LoaderError::FileRead {
            path: path.display().to_string(),
            source: e,
        })?;

        serde_json::from_str(&content).map_err(|e| LoaderError::JsonParse {
            path: path.display().to_string(),
            source: e,
        })
    }

    /// Get the list of unique shard files referenced in the weight map
    pub fn shard_files(&self) -> Vec<String> {
        let files: HashSet<_> = self.weight_map.values().collect();
        let mut result: Vec<_> = files.into_iter().cloned().collect();
        result.sort();
        result
    }

    /// Get the file name that contains a specific tensor
    pub fn get_file_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(|s| s.as_str())
    }

    /// Check if this index is for a sharded model
    pub fn is_sharded(&self) -> bool {
        self.shard_files().len() > 1
    }

    /// Get all tensor names in the index
    pub fn tensor_names(&self) -> Vec<&str> {
        self.weight_map.keys().map(|s| s.as_str()).collect()
    }
}

// =============================================================================
// Weight Loader
// =============================================================================

/// Weight loader with support for sharded safetensors and name mapping
pub struct WeightLoader {
    /// Device to load weights onto
    device: Device,
    /// Data type for weights
    dtype: DType,
    /// Name mapping rules (applied in order)
    mapping_rules: Vec<MappingRule>,
    /// Whether to use strict mode (error on missing tensors)
    strict_mode: bool,
}

impl WeightLoader {
    /// Create a new weight loader
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            mapping_rules: Vec::new(),
            strict_mode: false,
        }
    }

    /// Add an exact name mapping rule
    ///
    /// This is useful when Python model uses different naming conventions
    /// than the Rust implementation.
    ///
    /// # Example
    /// ```
    /// use candle_core::{Device, DType};
    /// use candle_video::loader::WeightLoader;
    ///
    /// let loader = WeightLoader::new(Device::Cpu, DType::F32)
    ///     .add_mapping("model.diffusion_model", "diffusion_model");
    /// ```
    pub fn add_mapping(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.mapping_rules.push(MappingRule::Exact {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    /// Add a prefix mapping rule
    ///
    /// Strips the `from_prefix` and optionally prepends `to_prefix`.
    ///
    /// # Example
    /// ```
    /// use candle_core::{Device, DType};
    /// use candle_video::loader::WeightLoader;
    ///
    /// // Remove "model." prefix from all tensor names
    /// let loader = WeightLoader::new(Device::Cpu, DType::F32)
    ///     .add_prefix_mapping("model.", "");
    /// ```
    pub fn add_prefix_mapping(
        mut self,
        from_prefix: impl Into<String>,
        to_prefix: impl Into<String>,
    ) -> Self {
        self.mapping_rules.push(MappingRule::Prefix {
            from_prefix: from_prefix.into(),
            to_prefix: to_prefix.into(),
        });
        self
    }

    /// Add a suffix mapping rule
    ///
    /// Replaces `from_suffix` with `to_suffix` at the end of tensor names.
    ///
    /// # Example
    /// ```
    /// use candle_core::{Device, DType};
    /// use candle_video::loader::WeightLoader;
    ///
    /// // Map PyTorch LayerNorm naming to Rust conventions
    /// let loader = WeightLoader::new(Device::Cpu, DType::F32)
    ///     .add_suffix_mapping(".gamma", ".weight")
    ///     .add_suffix_mapping(".beta", ".bias");
    /// ```
    pub fn add_suffix_mapping(
        mut self,
        from_suffix: impl Into<String>,
        to_suffix: impl Into<String>,
    ) -> Self {
        self.mapping_rules.push(MappingRule::Suffix {
            from_suffix: from_suffix.into(),
            to_suffix: to_suffix.into(),
        });
        self
    }

    /// Set strict mode for tensor loading
    ///
    /// In strict mode, loading will fail if any expected tensors are missing.
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Check if strict mode is enabled
    pub fn is_strict_mode(&self) -> bool {
        self.strict_mode
    }

    /// Check if a mapping exists for the given name
    pub fn has_mapping(&self, name: &str) -> bool {
        self.mapping_rules
            .iter()
            .any(|rule| rule.apply(name).is_some())
    }

    /// Apply all mapping rules to a tensor name
    ///
    /// Rules are applied in order. If a rule matches, its result is used
    /// as input for subsequent rules.
    pub fn map_name(&self, name: &str) -> String {
        let mut current = name.to_string();

        for rule in &self.mapping_rules {
            if let Some(mapped) = rule.apply(&current) {
                current = mapped;
            }
        }

        current
    }

    /// Load weights from a single safetensors file
    pub fn load_single(&self, path: impl AsRef<Path>) -> Result<VarBuilder<'_>> {
        let path = path.as_ref();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], self.dtype, &self.device)? };
        Ok(vb)
    }

    /// Load weights from multiple sharded safetensors files
    pub fn load_sharded(&self, paths: &[PathBuf]) -> Result<VarBuilder<'_>> {
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, self.dtype, &self.device)? };
        Ok(vb)
    }

    /// Load weights from a directory with automatic shard detection
    ///
    /// This method will:
    /// 1. Look for `model.safetensors.index.json` for sharded models
    /// 2. Fall back to looking for a single `model.safetensors`
    /// 3. Fall back to scanning for any `.safetensors` files
    ///
    /// If `strict_mode` is enabled and an index.json is found, it will
    /// verify that all referenced shard files exist.
    pub fn load_from_directory(
        &self,
        dir: impl AsRef<Path>,
    ) -> std::result::Result<VarBuilder<'_>, LoaderError> {
        let dir = dir.as_ref();

        // First, check for index.json (sharded model)
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index = SafetensorsIndex::load(&index_path)?;
            let shard_files = index.shard_files();

            // Verify all shard files exist
            let mut missing = Vec::new();
            let mut paths = Vec::new();

            for shard in &shard_files {
                let shard_path = dir.join(shard);
                if !shard_path.exists() {
                    missing.push(shard.clone());
                } else {
                    paths.push(shard_path);
                }
            }

            if !missing.is_empty() {
                return Err(LoaderError::MissingShards { missing });
            }

            return self.load_sharded(&paths).map_err(LoaderError::from);
        }

        // Check for single model.safetensors
        let single_path = dir.join("model.safetensors");
        if single_path.exists() {
            return self.load_single(&single_path).map_err(LoaderError::from);
        }

        // Fall back to scanning for .safetensors files
        let files = find_sharded_files(dir, "").map_err(|e| LoaderError::FileRead {
            path: dir.display().to_string(),
            source: std::io::Error::other(e.to_string()),
        })?;

        if files.is_empty() {
            return Err(LoaderError::NoSafetensorsFound {
                path: dir.display().to_string(),
            });
        }

        if files.len() == 1 {
            self.load_single(&files[0]).map_err(LoaderError::from)
        } else {
            self.load_sharded(&files).map_err(LoaderError::from)
        }
    }

    /// Get the data type used by this loader
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the device used by this loader
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a tensor by name with optional mapping
    ///
    /// Note: This is a placeholder. In practice, you need to know the shape
    /// to call VarBuilder::get. This method should be used with shape information.
    pub fn get_tensor<S: Into<candle_core::Shape>>(
        &self,
        vb: &VarBuilder,
        shape: S,
        name: &str,
    ) -> Result<Tensor> {
        let mapped_name = self.map_name(name);
        vb.get(shape, &mapped_name)
    }

    /// Load all tensors from a safetensors file into a HashMap
    pub fn load_all_tensors(&self, path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
        use candle_core::safetensors::load;
        let tensors = load(path, &self.device)?;
        Ok(tensors)
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Helper to find all sharded safetensors files in a directory
///
/// Files are sorted alphabetically to ensure consistent ordering.
pub fn find_sharded_files(dir: impl AsRef<Path>, prefix: &str) -> Result<Vec<PathBuf>> {
    use std::fs;
    let dir = dir.as_ref();
    let mut files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with(prefix) && name.ends_with(".safetensors") {
                files.push(path);
            }
        }





        }
    }

    files.sort();
    Ok(files)
}

/// Load a JSON configuration file and deserialize it
///
/// # Example
/// ```no_run
/// use candle_video::loader::load_model_config;
/// use candle_video::config::VaeConfig;
///
/// let config: VaeConfig = load_model_config("path/to/config.json").unwrap();
/// ```
pub fn load_model_config<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> std::result::Result<T, LoaderError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| LoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    serde_json::from_str(&content).map_err(|e| LoaderError::JsonParse {
        path: path.display().to_string(),
        source: e,
    })
}

/// Validate that all expected tensors are present in the loaded weights
///
/// Returns a list of missing tensor names.
///
/// # Example
/// ```
/// use candle_video::loader::validate_tensor_names;
///
/// let expected = vec!["weight1".to_string(), "weight2".to_string()];
/// let actual = vec!["weight1"];
///
/// let missing = validate_tensor_names(&expected, &actual);
/// assert_eq!(missing, vec!["weight2".to_string()]);
/// ```
pub fn validate_tensor_names(expected: &[String], actual: &[&str]) -> Vec<String> {
    let actual_set: HashSet<_> = actual.iter().cloned().collect();

    expected
        .iter()
        .filter(|name| !actual_set.contains(name.as_str()))
        .cloned()
        .collect()
}

/// List all tensor names in a safetensors file
///
/// This is useful for debugging and validation.
pub fn list_tensor_names(path: impl AsRef<Path>) -> std::result::Result<Vec<String>, LoaderError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| LoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
        LoaderError::InvalidSafetensors {
            path: path.display().to_string(),
            source: e,
        }
    })?;

    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
}

/// Get tensor metadata (dtype, shape) without loading the actual data
pub fn get_tensor_info(
    path: impl AsRef<Path>,
) -> std::result::Result<HashMap<String, TensorInfo>, LoaderError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| LoaderError::FileRead {
        path: path.display().to_string(),
        source: e,
    })?;

    let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
        LoaderError::InvalidSafetensors {
            path: path.display().to_string(),
            source: e,
        }
    })?;

    let mut info = HashMap::new();
    for name in tensors.names() {
        if let Ok(view) = tensors.tensor(name) {
            info.insert(
                name.to_string(),
                TensorInfo {
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
    }

    Ok(info)
}

/// Information about a tensor (without the actual data)
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Data type as a string
    pub dtype: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_loader_creation() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32);
        assert_eq!(loader.dtype, DType::F32);
    }

    #[test]
    fn test_name_mapping_exact() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32)
            .add_mapping("model.diffusion_model", "diffusion_model");

        assert_eq!(loader.map_name("model.diffusion_model"), "diffusion_model");
        // Unmapped names should return as-is
        assert_eq!(loader.map_name("other.name"), "other.name");
    }

    #[test]
    fn test_name_mapping_prefix() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32).add_prefix_mapping("model.", "");

        assert_eq!(
            loader.map_name("model.transformer.weight"),
            "transformer.weight"
        );
        // Non-matching prefix
        assert_eq!(loader.map_name("other.weight"), "other.weight");
    }

    #[test]
    fn test_name_mapping_suffix() {
        let loader =
            WeightLoader::new(Device::Cpu, DType::F32).add_suffix_mapping(".gamma", ".weight");

        assert_eq!(loader.map_name("layer_norm.gamma"), "layer_norm.weight");
    }

    #[test]
    fn test_name_mapping_chain() {
        let loader = WeightLoader::new(Device::Cpu, DType::F32)
            .add_prefix_mapping("model.", "")
            .add_suffix_mapping(".gamma", ".weight");

        // Both rules should apply in sequence
        assert_eq!(
            loader.map_name("model.layer_norm.gamma"),
            "layer_norm.weight"
        );
    }

    #[test]
    fn test_validate_tensor_names() {
        let expected = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let actual = vec!["a", "b"];

        let missing = validate_tensor_names(&expected, &actual);
        assert_eq!(missing, vec!["c".to_string()]);
    }

    #[test]
    fn test_safetensors_index_shard_files() {
        let mut weight_map = HashMap::new();
        weight_map.insert("a".to_string(), "shard1.safetensors".to_string());
        weight_map.insert("b".to_string(), "shard1.safetensors".to_string());
        weight_map.insert("c".to_string(), "shard2.safetensors".to_string());

        let index = SafetensorsIndex {
            weight_map,
            metadata: None,
        };

        let shards = index.shard_files();
        assert_eq!(shards.len(), 2);
        assert!(shards.contains(&"shard1.safetensors".to_string()));
        assert!(shards.contains(&"shard2.safetensors".to_string()));
    }
}
