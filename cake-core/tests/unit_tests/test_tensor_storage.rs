//! Tests for TensorStorageProvider and SafetensorsStorage.

use std::collections::HashMap;
use std::io::Write;

use candle_core::{DType, Device, Tensor};
use cake_core::utils::tensor_storage::{SafetensorsStorage, TensorStorageProvider};

/// Create a temporary safetensors file with given tensors.
/// Returns the temp directory (keeps the file alive).
fn create_test_safetensors(
    tensors: &[(&str, Tensor)],
) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");

    let tensor_map: HashMap<String, Tensor> = tensors
        .iter()
        .map(|(name, t)| (name.to_string(), t.clone()))
        .collect();

    candle_core::safetensors::save(&tensor_map, &path).unwrap();

    (dir, path)
}

#[test]
fn test_safetensors_storage_single_file() {
    let t1 = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu).unwrap();
    let t2 = Tensor::randn(0f32, 1.0, (16, 8), &Device::Cpu).unwrap();

    let (_dir, path) = create_test_safetensors(&[("weight_a", t1.clone()), ("weight_b", t2.clone())]);

    let storage = SafetensorsStorage::from_model_path(path.parent().unwrap()).unwrap();

    assert!(storage.has_tensor("weight_a"));
    assert!(storage.has_tensor("weight_b"));
    assert!(!storage.has_tensor("weight_c"));

    let names = storage.tensor_names();
    assert!(names.contains(&"weight_a".to_string()));
    assert!(names.contains(&"weight_b".to_string()));
}

#[test]
fn test_safetensors_read_matches_original() {
    let t = Tensor::randn(0f32, 1.0, (8, 4), &Device::Cpu).unwrap();
    let original_vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();

    let (_dir, path) = create_test_safetensors(&[("test_tensor", t)]);
    let storage = SafetensorsStorage::from_model_path(path.parent().unwrap()).unwrap();

    let data = storage.read_tensor("test_tensor").unwrap();
    assert_eq!(data.dtype, DType::F32);
    assert_eq!(data.shape, vec![8, 4]);

    // Compare raw bytes to original
    let read_tensor = Tensor::from_raw_buffer(&data.bytes, data.dtype, &data.shape, &Device::Cpu).unwrap();
    let read_vals: Vec<f32> = read_tensor.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(original_vals, read_vals);
}

#[test]
fn test_safetensors_read_nonexistent() {
    let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
    let (_dir, path) = create_test_safetensors(&[("exists", t)]);
    let storage = SafetensorsStorage::from_model_path(path.parent().unwrap()).unwrap();

    assert!(storage.read_tensor("does_not_exist").is_err());
}

#[test]
fn test_safetensors_various_dtypes() {
    let t_f32 = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
    let t_f16 = Tensor::zeros((4, 4), DType::F16, &Device::Cpu).unwrap();
    let t_bf16 = Tensor::zeros((4, 4), DType::BF16, &Device::Cpu).unwrap();

    let (_dir, path) = create_test_safetensors(&[
        ("f32_tensor", t_f32),
        ("f16_tensor", t_f16),
        ("bf16_tensor", t_bf16),
    ]);
    let storage = SafetensorsStorage::from_model_path(path.parent().unwrap()).unwrap();

    let d1 = storage.read_tensor("f32_tensor").unwrap();
    assert_eq!(d1.dtype, DType::F32);
    assert_eq!(d1.bytes.len(), 4 * 4 * 4); // 16 elements × 4 bytes

    let d2 = storage.read_tensor("f16_tensor").unwrap();
    assert_eq!(d2.dtype, DType::F16);
    assert_eq!(d2.bytes.len(), 4 * 4 * 2); // 16 elements × 2 bytes

    let d3 = storage.read_tensor("bf16_tensor").unwrap();
    assert_eq!(d3.dtype, DType::BF16);
    assert_eq!(d3.bytes.len(), 4 * 4 * 2);
}

#[test]
fn test_safetensors_multi_shard_index() {
    let dir = tempfile::tempdir().unwrap();

    // Create two shard files
    let t1 = Tensor::randn(0f32, 1.0, (4, 4), &Device::Cpu).unwrap();
    let t2 = Tensor::randn(0f32, 1.0, (8, 4), &Device::Cpu).unwrap();

    let shard1_path = dir.path().join("model-00001-of-00002.safetensors");
    let shard2_path = dir.path().join("model-00002-of-00002.safetensors");

    let mut map1 = HashMap::new();
    map1.insert("layer0.weight".to_string(), t1.clone());
    candle_core::safetensors::save(&map1, &shard1_path).unwrap();

    let mut map2 = HashMap::new();
    map2.insert("layer1.weight".to_string(), t2.clone());
    candle_core::safetensors::save(&map2, &shard2_path).unwrap();

    // Create index JSON
    let index_json = serde_json::json!({
        "weight_map": {
            "layer0.weight": "model-00001-of-00002.safetensors",
            "layer1.weight": "model-00002-of-00002.safetensors"
        }
    });
    let index_path = dir.path().join("model.safetensors.index.json");
    let mut f = std::fs::File::create(&index_path).unwrap();
    f.write_all(serde_json::to_string(&index_json).unwrap().as_bytes()).unwrap();

    let storage = SafetensorsStorage::from_model_path(dir.path()).unwrap();

    assert!(storage.has_tensor("layer0.weight"));
    assert!(storage.has_tensor("layer1.weight"));

    // Read from shard 1
    let d1 = storage.read_tensor("layer0.weight").unwrap();
    assert_eq!(d1.shape, vec![4, 4]);

    // Read from shard 2
    let d2 = storage.read_tensor("layer1.weight").unwrap();
    assert_eq!(d2.shape, vec![8, 4]);

    // Verify values match originals
    let r1 = Tensor::from_raw_buffer(&d1.bytes, d1.dtype, &d1.shape, &Device::Cpu).unwrap();
    let orig1: Vec<f32> = t1.flatten_all().unwrap().to_vec1().unwrap();
    let read1: Vec<f32> = r1.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(orig1, read1);
}
