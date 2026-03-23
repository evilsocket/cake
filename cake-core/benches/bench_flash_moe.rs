//! Flash-MoE benchmarks: disk-backed expert loading, pread I/O, dtype conversion,
//! and end-to-end expert offload path.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use cake_core::models::common::disk_expert_provider::DiskExpertProvider;
use cake_core::models::common::expert_provider::*;
use cake_core::utils::tensor_storage::{SafetensorsStorage, TensorStorageProvider};

// ── Helpers ────────────────────────────────────────────────────────────

/// Create a safetensors file with `n` experts, each with gate/up/down projections.
fn make_expert_safetensors(
    dir: &std::path::Path,
    prefix: &str,
    n: usize,
    intermediate: usize,
    hidden: usize,
    dtype: DType,
) -> std::path::PathBuf {
    let path = dir.join("model.safetensors");
    let mut tensors = HashMap::new();
    for e in 0..n {
        tensors.insert(
            format!("{prefix}.experts.{e}.gate_proj.weight"),
            Tensor::randn(0f32, 0.01, (intermediate, hidden), &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        );
        tensors.insert(
            format!("{prefix}.experts.{e}.up_proj.weight"),
            Tensor::randn(0f32, 0.01, (intermediate, hidden), &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        );
        tensors.insert(
            format!("{prefix}.experts.{e}.down_proj.weight"),
            Tensor::randn(0f32, 0.01, (hidden, intermediate), &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        );
    }
    candle_core::safetensors::save(&tensors, &path).unwrap();
    path
}

/// Create a multi-shard safetensors model directory.
fn make_sharded_expert_safetensors(
    dir: &std::path::Path,
    prefix: &str,
    n: usize,
    intermediate: usize,
    hidden: usize,
) {
    // Split experts across 2 shards
    let mid = n / 2;
    let shard1_path = dir.join("model-00001-of-00002.safetensors");
    let shard2_path = dir.join("model-00002-of-00002.safetensors");

    let mut map1 = HashMap::new();
    let mut map2 = HashMap::new();
    let mut weight_map = serde_json::Map::new();

    for e in 0..n {
        let target = if e < mid { &mut map1 } else { &mut map2 };
        let shard_name = if e < mid {
            "model-00001-of-00002.safetensors"
        } else {
            "model-00002-of-00002.safetensors"
        };

        for proj in &["gate_proj", "up_proj"] {
            let name = format!("{prefix}.experts.{e}.{proj}.weight");
            target.insert(
                name.clone(),
                Tensor::randn(0f32, 0.01, (intermediate, hidden), &Device::Cpu).unwrap(),
            );
            weight_map.insert(name, serde_json::Value::String(shard_name.to_string()));
        }
        let name = format!("{prefix}.experts.{e}.down_proj.weight");
        target.insert(
            name.clone(),
            Tensor::randn(0f32, 0.01, (hidden, intermediate), &Device::Cpu).unwrap(),
        );
        weight_map.insert(name, serde_json::Value::String(shard_name.to_string()));
    }

    candle_core::safetensors::save(&map1, &shard1_path).unwrap();
    candle_core::safetensors::save(&map2, &shard2_path).unwrap();

    let index = serde_json::json!({ "weight_map": weight_map });
    let index_path = dir.join("model.safetensors.index.json");
    let mut f = std::fs::File::create(index_path).unwrap();
    f.write_all(serde_json::to_string(&index).unwrap().as_bytes())
        .unwrap();
}

// ── Single-expert read latency ─────────────────────────────────────────

#[divan::bench(args = [32, 64, 128])]
fn flash_moe_single_expert_read(bencher: divan::Bencher, intermediate: usize) {
    let dir = tempfile::tempdir().unwrap();
    let hidden = 32;
    let n = 16;
    make_expert_safetensors(dir.path(), "mlp", n, intermediate, hidden, DType::F32);
    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32, None);

    // Warm page cache
    for i in 0..n {
        let _ = provider.get_expert(i).unwrap();
    }

    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % n).unwrap();
        idx += 1;
    });
}

// ── Top-K expert reads (simulating MoE dispatch) ───────────────────────

#[divan::bench(args = [2, 4, 8])]
fn flash_moe_topk_experts(bencher: divan::Bencher, k: usize) {
    let dir = tempfile::tempdir().unwrap();
    let (n, intermediate, hidden) = (64, 128, 64);
    make_expert_safetensors(dir.path(), "mlp", n, intermediate, hidden, DType::F32);
    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32, None);

    // Warm page cache
    for i in 0..n {
        let _ = provider.get_expert(i).unwrap();
    }

    bencher.bench_local(|| {
        for i in 0..k {
            let _ = provider.get_expert(i).unwrap();
        }
    });
}

// ── Dtype conversion overhead ──────────────────────────────────────────

#[divan::bench(args = [64, 256])]
fn flash_moe_f16_to_f32_conversion(bencher: divan::Bencher, intermediate: usize) {
    let dir = tempfile::tempdir().unwrap();
    let (n, hidden) = (8, 32);
    // Store as F16, load as F32 — measures conversion overhead
    make_expert_safetensors(dir.path(), "mlp", n, intermediate, hidden, DType::F16);
    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32, None);

    // Warm page cache
    for i in 0..n {
        let _ = provider.get_expert(i).unwrap();
    }

    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % n).unwrap();
        idx += 1;
    });
}

#[divan::bench(args = [64, 256])]
fn flash_moe_bf16_to_f32_conversion(bencher: divan::Bencher, intermediate: usize) {
    let dir = tempfile::tempdir().unwrap();
    let (n, hidden) = (8, 32);
    make_expert_safetensors(dir.path(), "mlp", n, intermediate, hidden, DType::BF16);
    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32, None);

    for i in 0..n {
        let _ = provider.get_expert(i).unwrap();
    }

    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % n).unwrap();
        idx += 1;
    });
}

// ── Multi-shard reads ──────────────────────────────────────────────────

#[divan::bench(args = [4, 8])]
fn flash_moe_multi_shard_read(bencher: divan::Bencher, k: usize) {
    let dir = tempfile::tempdir().unwrap();
    let (n, intermediate, hidden) = (32, 64, 32);
    make_sharded_expert_safetensors(dir.path(), "mlp", n, intermediate, hidden);
    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = DiskExpertProvider::new(storage, "mlp".to_string(), n, Device::Cpu, DType::F32, None);

    // Warm page cache — experts span both shards
    for i in 0..n {
        let _ = provider.get_expert(i).unwrap();
    }

    // Read K experts crossing shard boundaries
    let start = n / 2 - k / 2;
    bencher.bench_local(|| {
        for i in start..start + k {
            let _ = provider.get_expert(i).unwrap();
        }
    });
}

// ── SafetensorsStorage indexing (cold start) ───────────────────────────

#[divan::bench(args = [16, 64, 128])]
fn flash_moe_storage_index_build(bencher: divan::Bencher, num_experts: usize) {
    let dir = tempfile::tempdir().unwrap();
    make_expert_safetensors(dir.path(), "mlp", num_experts, 64, 32, DType::F32);

    bencher.bench_local(|| {
        let _ = SafetensorsStorage::from_model_path(dir.path()).unwrap();
    });
}

// ── Disk vs resident comparison ────────────────────────────────────────

#[divan::bench(args = [8, 32])]
fn flash_moe_disk_vs_stacked_resident(bencher: divan::Bencher, num_experts: usize) {
    // Disk provider
    let dir = tempfile::tempdir().unwrap();
    let (intermediate, hidden) = (64, 32);
    make_expert_safetensors(dir.path(), "mlp", num_experts, intermediate, hidden, DType::F32);
    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let disk = DiskExpertProvider::new(
        storage,
        "mlp".to_string(),
        num_experts,
        Device::Cpu,
        DType::F32,
        None,
    );

    // Warm
    for i in 0..num_experts {
        let _ = disk.get_expert(i).unwrap();
    }

    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = disk.get_expert(idx % num_experts).unwrap();
        idx += 1;
    });
}

// ── has_tensor lookup speed ────────────────────────────────────────────

#[divan::bench(args = [64, 256])]
fn flash_moe_has_tensor_lookup(bencher: divan::Bencher, num_experts: usize) {
    let dir = tempfile::tempdir().unwrap();
    make_expert_safetensors(dir.path(), "mlp", num_experts, 64, 32, DType::F32);
    let storage = SafetensorsStorage::from_model_path(dir.path()).unwrap();

    let mut idx = 0;
    bencher.bench_local(|| {
        let name = format!("mlp.experts.{}.gate_proj.weight", idx % num_experts);
        let _ = storage.has_tensor(&name);
        idx += 1;
    });
}

// ── tensor_names enumeration ───────────────────────────────────────────

#[divan::bench(args = [16, 128])]
fn flash_moe_tensor_names_enumeration(bencher: divan::Bencher, num_experts: usize) {
    let dir = tempfile::tempdir().unwrap();
    make_expert_safetensors(dir.path(), "mlp", num_experts, 64, 32, DType::F32);
    let storage = SafetensorsStorage::from_model_path(dir.path()).unwrap();

    bencher.bench_local(|| {
        let names = storage.tensor_names();
        assert_eq!(names.len(), num_experts * 3);
    });
}
