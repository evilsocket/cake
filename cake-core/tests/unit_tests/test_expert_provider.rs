//! Tests for ExpertProvider trait and implementations.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use cake_core::models::common::expert_provider::*;

// ── StackedResidentProvider ─────────────────────────────────────────

#[test]
fn stacked_provider_shapes_match_config() {
    let (n, i, h) = (8, 64, 32);
    let g = Tensor::randn(0f32, 0.1, (n, i, h), &Device::Cpu).unwrap();
    let u = Tensor::randn(0f32, 0.1, (n, i, h), &Device::Cpu).unwrap();
    let d = Tensor::randn(0f32, 0.1, (n, h, i), &Device::Cpu).unwrap();

    let provider = StackedResidentProvider::new(g, u, d, n);
    assert_eq!(provider.num_experts(), 8);

    for idx in 0..n {
        let ew = provider.get_expert(idx).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[i, h]);
        assert_eq!(ew.up_proj.dims(), &[i, h]);
        assert_eq!(ew.down_proj.dims(), &[h, i]);
    }
}

#[test]
fn stacked_provider_values_match_direct_access() {
    let (n, i, h) = (4, 16, 8);
    let g = Tensor::randn(0f32, 1.0, (n, i, h), &Device::Cpu).unwrap();
    let u = Tensor::randn(0f32, 1.0, (n, i, h), &Device::Cpu).unwrap();
    let d = Tensor::randn(0f32, 1.0, (n, h, i), &Device::Cpu).unwrap();

    let provider = StackedResidentProvider::new(g.clone(), u.clone(), d.clone(), n);

    for idx in 0..n {
        let ew = provider.get_expert(idx).unwrap();
        let direct_g: Vec<f32> = g.get(idx).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let provider_g: Vec<f32> = ew.gate_proj.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(direct_g, provider_g, "gate_proj mismatch at expert {idx}");

        let direct_u: Vec<f32> = u.get(idx).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let provider_u: Vec<f32> = ew.up_proj.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(direct_u, provider_u, "up_proj mismatch at expert {idx}");
    }
}

#[test]
fn stacked_provider_as_any_returns_self() {
    let g = Tensor::zeros((2, 4, 3), DType::F32, &Device::Cpu).unwrap();
    let u = g.clone();
    let d = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu).unwrap();
    let provider = StackedResidentProvider::new(g, u, d, 2);

    let any_ref = provider.as_any();
    assert!(any_ref.is_some());
    assert!(any_ref.unwrap().downcast_ref::<StackedResidentProvider>().is_some());
}

// ── IndividualResidentProvider ───────────────────────────────────────

#[test]
fn individual_provider_shapes_match() {
    let (n, i, h) = (6, 64, 32);
    let gs: Vec<Tensor> = (0..n).map(|_| Tensor::zeros((i, h), DType::F32, &Device::Cpu).unwrap()).collect();
    let us: Vec<Tensor> = (0..n).map(|_| Tensor::zeros((i, h), DType::F32, &Device::Cpu).unwrap()).collect();
    let ds: Vec<Tensor> = (0..n).map(|_| Tensor::zeros((h, i), DType::F32, &Device::Cpu).unwrap()).collect();

    let provider = IndividualResidentProvider::new(gs, us, ds);
    assert_eq!(provider.num_experts(), n);

    for idx in 0..n {
        let ew = provider.get_expert(idx).unwrap();
        assert_eq!(ew.gate_proj.dims(), &[i, h]);
        assert_eq!(ew.down_proj.dims(), &[h, i]);
    }
}

#[test]
fn individual_provider_as_any_returns_none() {
    let gs = vec![Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap()];
    let us = vec![Tensor::zeros((4, 3), DType::F32, &Device::Cpu).unwrap()];
    let ds = vec![Tensor::zeros((3, 4), DType::F32, &Device::Cpu).unwrap()];
    let provider = IndividualResidentProvider::new(gs, us, ds);
    // IndividualResidentProvider doesn't override as_any
    assert!(provider.as_any().is_none());
}

// ── Trait object dispatch ───────────────────────────────────────────

#[test]
fn provider_trait_object_dispatch() {
    let g = Tensor::randn(0f32, 1.0, (4, 16, 8), &Device::Cpu).unwrap();
    let u = Tensor::randn(0f32, 1.0, (4, 16, 8), &Device::Cpu).unwrap();
    let d = Tensor::randn(0f32, 1.0, (4, 8, 16), &Device::Cpu).unwrap();

    let provider: Arc<dyn ExpertProvider> = Arc::new(
        StackedResidentProvider::new(g, u, d, 4)
    );

    assert_eq!(provider.num_experts(), 4);
    let ew = provider.get_expert(2).unwrap();
    assert_eq!(ew.gate_proj.dims(), &[16, 8]);
}

// ── DiskExpertProvider ──────────────────────────────────────────────

#[test]
fn disk_provider_via_mock() {
    use cake_core::utils::tensor_storage::TensorData;

    #[derive(Debug)]
    struct MockStorage(std::collections::HashMap<String, TensorData>);

    impl cake_core::utils::tensor_storage::TensorStorageProvider for MockStorage {
        fn read_tensor(&self, name: &str) -> anyhow::Result<TensorData> {
            self.0.get(name).map(|td| TensorData {
                bytes: td.bytes.clone(), dtype: td.dtype, shape: td.shape.clone(),
            }).ok_or_else(|| anyhow::anyhow!("not found: {name}"))
        }
        fn has_tensor(&self, name: &str) -> bool { self.0.contains_key(name) }
        fn tensor_names(&self) -> Vec<String> { self.0.keys().cloned().collect() }
    }

    let mut tensors = std::collections::HashMap::new();
    let (i, h) = (16, 8);
    for e in 0..3 {
        for proj in &["gate_proj", "up_proj"] {
            tensors.insert(
                format!("layer.mlp.experts.{e}.{proj}.weight"),
                TensorData { bytes: vec![0u8; i * h * 4], dtype: DType::F32, shape: vec![i, h] },
            );
        }
        tensors.insert(
            format!("layer.mlp.experts.{e}.down_proj.weight"),
            TensorData { bytes: vec![0u8; h * i * 4], dtype: DType::F32, shape: vec![h, i] },
        );
    }

    let storage: Arc<dyn cake_core::utils::tensor_storage::TensorStorageProvider> =
        Arc::new(MockStorage(tensors));
    let provider = cake_core::models::common::disk_expert_provider::DiskExpertProvider::new(
        storage, "layer.mlp".to_string(), 3, Device::Cpu, DType::F32, None,
    );

    assert_eq!(provider.num_experts(), 3);
    let ew = provider.get_expert(1).unwrap();
    assert_eq!(ew.gate_proj.dims(), &[16, 8]);
    assert_eq!(ew.down_proj.dims(), &[8, 16]);

    // Out of range
    assert!(provider.get_expert(3).is_err());
}
