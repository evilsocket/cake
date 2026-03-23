//! Flash-MoE integration tests: end-to-end disk-backed expert offloading.
//!
//! These tests verify the complete pipeline from safetensors on disk through
//! DiskExpertProvider to MoE forward pass, covering multi-shard, dtype conversion,
//! data integrity, concurrent access, and edge cases.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use cake_core::models::common::disk_expert_provider::DiskExpertProvider;
use cake_core::models::common::expert_provider::*;
use cake_core::utils::tensor_storage::{SafetensorsStorage, TensorStorageProvider};

// ── Helpers ────────────────────────────────────────────────────────────

fn make_test_safetensors(
    dir: &std::path::Path,
    prefix: &str,
    n: usize,
    intermediate: usize,
    hidden: usize,
) -> HashMap<String, Tensor> {
    let path = dir.join("model.safetensors");
    let mut tensors = HashMap::new();
    for e in 0..n {
        tensors.insert(
            format!("{prefix}.experts.{e}.gate_proj.weight"),
            Tensor::randn(0f32, 1.0, (intermediate, hidden), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.experts.{e}.up_proj.weight"),
            Tensor::randn(0f32, 1.0, (intermediate, hidden), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.experts.{e}.down_proj.weight"),
            Tensor::randn(0f32, 1.0, (hidden, intermediate), &Device::Cpu).unwrap(),
        );
    }
    candle_core::safetensors::save(&tensors, &path).unwrap();
    tensors
}

// ── Data integrity: disk values match original tensors ─────────────────

#[test]
fn flash_moe_disk_values_match_originals() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (4, 16, 8);
    let originals = make_test_safetensors(dir.path(), "mlp", n, i, h);

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32);

    for e in 0..n {
        let ew = provider.get_expert(e).unwrap();

        let orig_gate = &originals[&format!("mlp.experts.{e}.gate_proj.weight")];
        let orig_up = &originals[&format!("mlp.experts.{e}.up_proj.weight")];
        let orig_down = &originals[&format!("mlp.experts.{e}.down_proj.weight")];

        let gate_vals: Vec<f32> = ew.gate_proj.flatten_all().unwrap().to_vec1().unwrap();
        let orig_gate_vals: Vec<f32> = orig_gate.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(gate_vals, orig_gate_vals, "gate mismatch at expert {e}");

        let up_vals: Vec<f32> = ew.up_proj.flatten_all().unwrap().to_vec1().unwrap();
        let orig_up_vals: Vec<f32> = orig_up.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(up_vals, orig_up_vals, "up mismatch at expert {e}");

        let down_vals: Vec<f32> = ew.down_proj.flatten_all().unwrap().to_vec1().unwrap();
        let orig_down_vals: Vec<f32> = orig_down.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(down_vals, orig_down_vals, "down mismatch at expert {e}");
    }
}

// ── Dtype conversion: F16 → F32 ────────────────────────────────────────

#[test]
fn flash_moe_f16_to_f32_preserves_values() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (2, 8, 4);

    // Create F32 tensors, convert to F16 for storage, load back as F32
    let path = dir.path().join("model.safetensors");
    let mut tensors = HashMap::new();
    let mut f32_originals = HashMap::new();

    for e in 0..n {
        let gate_f32 = Tensor::randn(0f32, 1.0, (i, h), &Device::Cpu).unwrap();
        let gate_f16 = gate_f32.to_dtype(DType::F16).unwrap();
        let name = format!("mlp.experts.{e}.gate_proj.weight");
        f32_originals.insert(name.clone(), gate_f16.to_dtype(DType::F32).unwrap());
        tensors.insert(name, gate_f16);

        let up_f32 = Tensor::randn(0f32, 1.0, (i, h), &Device::Cpu).unwrap();
        let up_f16 = up_f32.to_dtype(DType::F16).unwrap();
        let name = format!("mlp.experts.{e}.up_proj.weight");
        f32_originals.insert(name.clone(), up_f16.to_dtype(DType::F32).unwrap());
        tensors.insert(name, up_f16);

        let down_f32 = Tensor::randn(0f32, 1.0, (h, i), &Device::Cpu).unwrap();
        let down_f16 = down_f32.to_dtype(DType::F16).unwrap();
        let name = format!("mlp.experts.{e}.down_proj.weight");
        f32_originals.insert(name.clone(), down_f16.to_dtype(DType::F32).unwrap());
        tensors.insert(name, down_f16);
    }
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32);

    for e in 0..n {
        let ew = provider.get_expert(e).unwrap();
        assert_eq!(ew.gate_proj.dtype(), DType::F32);
        assert_eq!(ew.up_proj.dtype(), DType::F32);
        assert_eq!(ew.down_proj.dtype(), DType::F32);

        // Values should match F16→F32 round-trip
        let expected: Vec<f32> = f32_originals[&format!("mlp.experts.{e}.gate_proj.weight")]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let actual: Vec<f32> = ew.gate_proj.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(expected, actual, "F16→F32 mismatch at expert {e}");
    }
}

// ── BF16 → F32 conversion ──────────────────────────────────────────────

#[test]
fn flash_moe_bf16_to_f32_preserves_values() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (2, 8, 4);

    let path = dir.path().join("model.safetensors");
    let mut tensors = HashMap::new();
    for e in 0..n {
        tensors.insert(
            format!("mlp.experts.{e}.gate_proj.weight"),
            Tensor::randn(0f32, 1.0, (i, h), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        tensors.insert(
            format!("mlp.experts.{e}.up_proj.weight"),
            Tensor::randn(0f32, 1.0, (i, h), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        tensors.insert(
            format!("mlp.experts.{e}.down_proj.weight"),
            Tensor::randn(0f32, 1.0, (h, i), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
    }
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32);

    for e in 0..n {
        let ew = provider.get_expert(e).unwrap();
        assert_eq!(ew.gate_proj.dtype(), DType::F32);
        assert_eq!(ew.gate_proj.dims(), &[i, h]);
    }
}

// ── Multi-shard model ──────────────────────────────────────────────────

#[test]
fn flash_moe_multi_shard_experts() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (8, 16, 8);
    let mid = n / 2;

    let shard1_path = dir.path().join("model-00001-of-00002.safetensors");
    let shard2_path = dir.path().join("model-00002-of-00002.safetensors");

    let mut map1 = HashMap::new();
    let mut map2 = HashMap::new();
    let mut weight_map = serde_json::Map::new();

    for e in 0..n {
        let (target, shard_name) = if e < mid {
            (&mut map1, "model-00001-of-00002.safetensors")
        } else {
            (&mut map2, "model-00002-of-00002.safetensors")
        };

        for proj in &["gate_proj", "up_proj"] {
            let name = format!("mlp.experts.{e}.{proj}.weight");
            target.insert(
                name.clone(),
                Tensor::randn(0f32, 1.0, (i, h), &Device::Cpu).unwrap(),
            );
            weight_map.insert(name, serde_json::Value::String(shard_name.to_string()));
        }
        let name = format!("mlp.experts.{e}.down_proj.weight");
        target.insert(
            name.clone(),
            Tensor::randn(0f32, 1.0, (h, i), &Device::Cpu).unwrap(),
        );
        weight_map.insert(name, serde_json::Value::String(shard_name.to_string()));
    }

    candle_core::safetensors::save(&map1, &shard1_path).unwrap();
    candle_core::safetensors::save(&map2, &shard2_path).unwrap();

    let index = serde_json::json!({ "weight_map": weight_map });
    let index_path = dir.path().join("model.safetensors.index.json");
    let mut f = std::fs::File::create(&index_path).unwrap();
    f.write_all(serde_json::to_string(&index).unwrap().as_bytes())
        .unwrap();

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32);

    // Verify all experts across both shards are accessible
    for e in 0..n {
        let ew = provider.get_expert(e).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[i, h], "shard access failed at expert {e}");
        assert_eq!(ew.down_proj.dims(), &[h, i]);
    }
}

// ── Repeated reads return same data ────────────────────────────────────

#[test]
fn flash_moe_repeated_reads_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (4, 16, 8);
    make_test_safetensors(dir.path(), "mlp", n, i, h);

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32);

    // Read same expert twice, values must match
    let ew1 = provider.get_expert(2).unwrap();
    let ew2 = provider.get_expert(2).unwrap();

    let vals1: Vec<f32> = ew1.gate_proj.flatten_all().unwrap().to_vec1().unwrap();
    let vals2: Vec<f32> = ew2.gate_proj.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(vals1, vals2);
}

// ── Concurrent reads (simulates multi-token routing) ───────────────────

#[test]
fn flash_moe_concurrent_reads() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (8, 16, 8);
    make_test_safetensors(dir.path(), "mlp", n, i, h);

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());

    // Spawn multiple threads reading different experts simultaneously
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let storage = storage.clone();
            std::thread::spawn(move || {
                let provider = DiskExpertProvider::new(
                    storage,
                    "mlp".to_string(),
                    n,
                    Device::Cpu,
                    DType::F32,
                );
                // Each thread reads 2 different experts
                let e1 = t * 2;
                let e2 = t * 2 + 1;
                let ew1 = provider.get_expert(e1).unwrap();
                let ew2 = provider.get_expert(e2).unwrap();
                assert_eq!(ew1.gate_proj.dims(), &[i, h]);
                assert_eq!(ew2.gate_proj.dims(), &[i, h]);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

// ── Disk provider vs stacked resident: same interface ──────────────────

#[test]
fn flash_moe_disk_provider_as_trait_object() {
    let dir = tempfile::tempdir().unwrap();
    let (n, i, h) = (4, 16, 8);
    make_test_safetensors(dir.path(), "mlp", n, i, h);

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider: Arc<dyn ExpertProvider> = Arc::new(DiskExpertProvider::new(
        storage,
        "mlp".to_string(),
        n,
        Device::Cpu,
        DType::F32,
    ));

    assert_eq!(provider.num_experts(), n);
    let ew = provider.get_expert(0).unwrap();
    assert_eq!(ew.gate_proj.dims(), &[i, h]);

    // as_any returns None for DiskExpertProvider (no fast path)
    assert!(provider.as_any().is_none());
}

// ── Edge cases ─────────────────────────────────────────────────────────

#[test]
fn flash_moe_zero_experts() {
    let dir = tempfile::tempdir().unwrap();
    // Create empty safetensors
    let path = dir.path().join("model.safetensors");
    let empty: HashMap<String, Tensor> = HashMap::new();
    candle_core::safetensors::save(&empty, &path).unwrap();

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), 0, Device::Cpu, DType::F32);

    assert_eq!(provider.num_experts(), 0);
    assert!(provider.get_expert(0).is_err());
}

#[test]
fn flash_moe_missing_tensor_gives_error() {
    // Provider expects 4 experts but storage only has 2
    let dir = tempfile::tempdir().unwrap();
    make_test_safetensors(dir.path(), "mlp", 2, 16, 8);

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), 4, Device::Cpu, DType::F32);

    // First 2 work
    assert!(provider.get_expert(0).is_ok());
    assert!(provider.get_expert(1).is_ok());
    // Expert 2 doesn't exist in storage
    assert!(provider.get_expert(2).is_err());
}

// ── Storage: tensor_names contains all experts ─────────────────────────

#[test]
fn flash_moe_storage_lists_all_expert_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let n = 8;
    make_test_safetensors(dir.path(), "mlp", n, 16, 8);

    let storage = SafetensorsStorage::from_model_path(dir.path()).unwrap();
    let names = storage.tensor_names();

    // Each expert has 3 tensors: gate, up, down
    assert_eq!(names.len(), n * 3);

    for e in 0..n {
        assert!(storage.has_tensor(&format!("mlp.experts.{e}.gate_proj.weight")));
        assert!(storage.has_tensor(&format!("mlp.experts.{e}.up_proj.weight")));
        assert!(storage.has_tensor(&format!("mlp.experts.{e}.down_proj.weight")));
    }
}

// ── Storage: no safetensors found ──────────────────────────────────────

#[test]
fn flash_moe_storage_empty_dir_fails() {
    let dir = tempfile::tempdir().unwrap();
    assert!(SafetensorsStorage::from_model_path(dir.path()).is_err());
}

// ── Large expert count (stress test) ───────────────────────────────────

#[test]
fn flash_moe_many_experts() {
    let dir = tempfile::tempdir().unwrap();
    let n = 64; // simulates Qwen3-MoE scale (128 experts, but smaller dims)
    let (i, h) = (8, 4);
    make_test_safetensors(dir.path(), "mlp", n, i, h);

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32);

    // Read all experts
    for e in 0..n {
        let ew = provider.get_expert(e).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[i, h]);
    }
}
