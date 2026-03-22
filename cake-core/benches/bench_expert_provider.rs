//! Benchmarks for ExpertProvider implementations.
//! Enables iterative optimization of expert weight access patterns.

use std::sync::Arc;
use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use cake_core::models::common::expert_provider::*;
use cake_core::utils::tensor_storage::SafetensorsStorage;

fn make_stacked_provider(n: usize, i: usize, h: usize) -> StackedResidentProvider {
    let g = Tensor::randn(0f32, 0.01, (n, i, h), &Device::Cpu).unwrap();
    let u = Tensor::randn(0f32, 0.01, (n, i, h), &Device::Cpu).unwrap();
    let d = Tensor::randn(0f32, 0.01, (n, h, i), &Device::Cpu).unwrap();
    StackedResidentProvider::new(g, u, d, n)
}

fn make_individual_provider(n: usize, i: usize, h: usize) -> IndividualResidentProvider {
    let gs: Vec<Tensor> = (0..n).map(|_| Tensor::randn(0f32, 0.01, (i, h), &Device::Cpu).unwrap()).collect();
    let us: Vec<Tensor> = (0..n).map(|_| Tensor::randn(0f32, 0.01, (i, h), &Device::Cpu).unwrap()).collect();
    let ds: Vec<Tensor> = (0..n).map(|_| Tensor::randn(0f32, 0.01, (h, i), &Device::Cpu).unwrap()).collect();
    IndividualResidentProvider::new(gs, us, ds)
}

/// Create a temp safetensors file with expert tensors and return DiskProvider.
fn make_disk_provider(n: usize, i: usize, h: usize) -> (
    tempfile::TempDir,
    cake_core::models::common::disk_expert_provider::DiskExpertProvider,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");

    let mut tensors = HashMap::new();
    for e in 0..n {
        tensors.insert(
            format!("mlp.experts.{e}.gate_proj.weight"),
            Tensor::randn(0f32, 0.01, (i, h), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("mlp.experts.{e}.up_proj.weight"),
            Tensor::randn(0f32, 0.01, (i, h), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            format!("mlp.experts.{e}.down_proj.weight"),
            Tensor::randn(0f32, 0.01, (h, i), &Device::Cpu).unwrap(),
        );
    }
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let storage = Arc::new(SafetensorsStorage::from_model_path(dir.path()).unwrap());
    let provider = cake_core::models::common::disk_expert_provider::DiskExpertProvider::new(
        storage, "mlp".to_string(), n, Device::Cpu, DType::F32,
    );
    (dir, provider)
}

// ── Resident provider benchmarks ────────────────────────────────────

#[divan::bench(args = [4, 32, 128])]
fn stacked_get_expert(bencher: divan::Bencher, num_experts: usize) {
    let provider = make_stacked_provider(num_experts, 64, 32);
    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % num_experts).unwrap();
        idx += 1;
    });
}

#[divan::bench(args = [4, 32, 128])]
fn individual_get_expert(bencher: divan::Bencher, num_experts: usize) {
    let provider = make_individual_provider(num_experts, 64, 32);
    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % num_experts).unwrap();
        idx += 1;
    });
}

// ── Disk provider benchmarks ────────────────────────────────────────

#[divan::bench(args = [4, 16])]
fn disk_get_expert_warm(bencher: divan::Bencher, num_experts: usize) {
    let (_dir, provider) = make_disk_provider(num_experts, 64, 32);
    // Warm up: read each expert once to populate page cache
    for i in 0..num_experts {
        let _ = provider.get_expert(i).unwrap();
    }
    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % num_experts).unwrap();
        idx += 1;
    });
}

// ── Trait object dispatch overhead ──────────────────────────────────

#[divan::bench(args = [4, 32])]
fn trait_object_get_expert(bencher: divan::Bencher, num_experts: usize) {
    let provider: Arc<dyn ExpertProvider> = Arc::new(
        make_stacked_provider(num_experts, 64, 32)
    );
    let mut idx = 0;
    bencher.bench_local(|| {
        let _ = provider.get_expert(idx % num_experts).unwrap();
        idx += 1;
    });
}

// ── pread I/O throughput ────────────────────────────────────────────

#[divan::bench(args = [4, 8, 16])]
fn pread_k_experts(bencher: divan::Bencher, k: usize) {
    let n = 32; // total experts, read k of them per iteration
    let (_dir, provider) = make_disk_provider(n, 256, 128);
    // Warm up page cache
    for i in 0..n { let _ = provider.get_expert(i).unwrap(); }

    bencher.bench_local(|| {
        for i in 0..k {
            let _ = provider.get_expert(i).unwrap();
        }
    });
}
