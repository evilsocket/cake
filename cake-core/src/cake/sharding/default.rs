//! Default sharding strategy: TFLOPS-proportional layer assignment.

use std::collections::HashSet;
use std::path::Path;

/// Default stateless sharding strategy that assigns layers proportionally
/// to each worker's TFLOPS, capped by per-GPU VRAM.
pub struct DefaultStrategy;

impl super::Strategy for DefaultStrategy {
    fn assign_layers(
        &self,
        workers: &[&dyn super::WorkerCapacity],
        num_layers: usize,
        master_tflops: f64,
        layer_size_bytes: u64,
        master_max_layers: usize,
        layer_prefix: &str,
    ) -> Vec<(usize, Vec<String>)> {
        if workers.is_empty() || num_layers == 0 {
            return vec![];
        }

        // Include master TFLOPS in total so layers are split proportionally
        let total_tflops: f64 =
            workers.iter().map(|w| w.total_tflops()).sum::<f64>() + master_tflops;

        if total_tflops <= 0.0 {
            // No compute info — give half to workers, half to master
            let worker_layers = num_layers / 2;
            let per_worker = worker_layers / workers.len();
            let mut assignments = vec![];
            let mut offset = 0;
            for (i, _) in workers.iter().enumerate() {
                let count = if i == workers.len() - 1 {
                    worker_layers - offset
                } else {
                    per_worker
                };
                let layers: Vec<String> = (offset..offset + count)
                    .map(|l| format!("{layer_prefix}.{l}"))
                    .collect();
                assignments.push((i, layers));
                offset += count;
            }
            return assignments;
        }

        // Sort worker indices by TFLOPS descending
        let mut indices: Vec<usize> = (0..workers.len()).collect();
        indices.sort_by(|a, b| {
            workers[*b]
                .total_tflops()
                .partial_cmp(&workers[*a].total_tflops())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Total layers for all workers combined (master keeps its share)
        let workers_tflops: f64 = workers.iter().map(|w| w.total_tflops()).sum();
        let total_worker_layers =
            (workers_tflops / total_tflops * num_layers as f64).round() as usize;
        let total_worker_layers = total_worker_layers.min(num_layers);

        log::info!(
            "master: {:.1} TFLOPS — workers: {:.1} TFLOPS — assigning {} of {} layers to workers",
            master_tflops,
            workers_tflops,
            total_worker_layers,
            num_layers
        );

        let mut assignments = vec![];
        let mut offset = 0;
        let mut remaining_layers = total_worker_layers;
        let mut remaining_tflops = workers_tflops;

        for (pos, &worker_idx) in indices.iter().enumerate() {
            if remaining_layers == 0 {
                break;
            }

            let mut count = if pos == indices.len() - 1 {
                remaining_layers
            } else {
                let worker_tflops = workers[worker_idx].total_tflops();
                let proportional =
                    (worker_tflops / remaining_tflops * remaining_layers as f64).round() as usize;
                proportional.max(1).min(remaining_layers)
            };

            // Cap by per-GPU VRAM to avoid OOM on multi-GPU workers
            if layer_size_bytes > 0 {
                let max_layers = workers[worker_idx].max_layers_for_size(layer_size_bytes);
                if count > max_layers {
                    log::info!(
                        "  {} capped from {} to {} layers (VRAM limit: {} per layer)",
                        workers[worker_idx].name(),
                        count,
                        max_layers,
                        human_bytes::human_bytes(layer_size_bytes as f64)
                    );
                    count = max_layers;
                }
            }

            let layers: Vec<String> = (offset..offset + count)
                .map(|l| format!("{layer_prefix}.{l}"))
                .collect();

            assignments.push((worker_idx, layers));
            offset += count;
            remaining_layers -= count;
            remaining_tflops -= workers[worker_idx].total_tflops();
        }

        // Check if master would be left with more layers than it can hold.
        // The master keeps all layers from `offset` to `num_layers - 1`.
        let master_layers = num_layers - offset;
        if master_max_layers < usize::MAX && master_layers > master_max_layers {
            let deficit = master_layers - master_max_layers;
            log::info!(
                "master has {} local layers but can fit {} — redistributing {} to workers",
                master_layers,
                master_max_layers,
                deficit
            );

            // Try to push excess layers to workers that have spare VRAM capacity.
            let mut extra_needed = deficit;
            for (worker_idx, layers) in assignments.iter_mut() {
                if extra_needed == 0 {
                    break;
                }
                let current = layers.len();
                let max = if layer_size_bytes > 0 {
                    workers[*worker_idx].max_layers_for_size(layer_size_bytes)
                } else {
                    usize::MAX
                };
                let spare = max.saturating_sub(current);
                if spare > 0 {
                    let take = spare.min(extra_needed);
                    // Extend this worker's range (layers are at the end)
                    let new_start = offset;
                    for l in new_start..new_start + take {
                        layers.push(format!("{layer_prefix}.{l}"));
                    }
                    offset += take;
                    extra_needed -= take;
                    log::info!(
                        "  {} takes {} extra layer(s) ({} → {} total)",
                        workers[*worker_idx].name(),
                        take,
                        current,
                        layers.len()
                    );
                }
            }

            if extra_needed > 0 {
                log::warn!(
                    "cluster cannot fit all {} layers — {} layer(s) unassignable (master VRAM too small, workers full)",
                    num_layers,
                    extra_needed
                );
            }
        }

        assignments
    }
}

/// Derive the layer name prefix from config.json.
/// Returns e.g. "model.language_model.layers" for Qwen3.5, "model.layers" otherwise.
pub fn layer_prefix_for_config(config_json: &serde_json::Value) -> String {
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        for arch in archs {
            if let Some("Qwen3_5ForConditionalGeneration") = arch.as_str() {
                return "model.language_model.layers".to_string();
            }
        }
    }
    "model.layers".to_string()
}

/// Read a safetensors header and return per-tensor byte sizes.
/// The safetensors format stores an 8-byte LE header length followed by a JSON
/// object mapping tensor names to `{dtype, shape, data_offsets: [start, end]}`.
fn read_safetensors_tensor_sizes(path: &Path) -> Option<Vec<(String, u64)>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    // Sanity: headers are typically < 1 MB
    if header_len > 10 * 1024 * 1024 {
        return None;
    }
    let mut header_buf = vec![0u8; header_len];
    f.read_exact(&mut header_buf).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf).ok()?;
    let obj = header.as_object()?;
    let mut result = Vec::with_capacity(obj.len());
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        if let Some(offsets) = meta.get("data_offsets").and_then(|v| v.as_array()) {
            if offsets.len() == 2 {
                let start = offsets[0].as_u64().unwrap_or(0);
                let end = offsets[1].as_u64().unwrap_or(0);
                result.push((name.clone(), end.saturating_sub(start)));
            }
        }
    }
    Some(result)
}

/// Estimate average transformer layer size in bytes from safetensors files.
///
/// For sharded models, reads each shard's header to compute exact per-tensor
/// byte sizes, then sums only tensors matching `layer_prefix`. This excludes
/// non-layer weights (visual encoder, MTP heads, embeddings, lm_head) which
/// can be significant — e.g. Qwen3.5-27B-FP8 has ~6 GB of non-layer data.
pub fn estimate_layer_size(model_path: &Path, num_layers: usize, layer_prefix: &str) -> u64 {
    if num_layers == 0 {
        return 0;
    }

    let layer_dot = format!("{}.", layer_prefix);

    // Try sharded model first
    let index_path = model_path.join("model.safetensors.index.json");
    if let Ok(data) = std::fs::read_to_string(&index_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
            if let Some(weight_map) = json.get("weight_map").and_then(|v| v.as_object()) {
                let shards: HashSet<&str> =
                    weight_map.values().filter_map(|v| v.as_str()).collect();

                // Try reading safetensors headers for exact tensor sizes
                let mut layer_bytes: u64 = 0;
                let mut total_bytes: u64 = 0;
                let mut headers_ok = true;

                for shard in &shards {
                    let shard_path = model_path.join(shard);
                    if let Some(tensors) = read_safetensors_tensor_sizes(&shard_path) {
                        for (name, size) in &tensors {
                            total_bytes += size;
                            if name.starts_with(&layer_dot) {
                                layer_bytes += size;
                            }
                        }
                    } else {
                        headers_ok = false;
                        break;
                    }
                }

                if headers_ok && layer_bytes > 0 {
                    let non_layer = total_bytes - layer_bytes;
                    if non_layer > 0 {
                        log::info!(
                            "model weights: {} total, {} layers, {} non-layer ({:.0}% excluded)",
                            human_bytes::human_bytes(total_bytes as f64),
                            human_bytes::human_bytes(layer_bytes as f64),
                            human_bytes::human_bytes(non_layer as f64),
                            non_layer as f64 / total_bytes as f64 * 100.0,
                        );
                    }
                    return layer_bytes / num_layers as u64;
                }

                // Fallback: raw file size division
                let total: u64 = shards
                    .iter()
                    .filter_map(|s| std::fs::metadata(model_path.join(s)).ok())
                    .map(|m| m.len())
                    .sum();
                return total / num_layers as u64;
            }
        }
    }

    // Single safetensors file
    let single = model_path.join("model.safetensors");
    if let Ok(single_path) = single.canonicalize() {
        // Try header-based estimation for single file too
        if let Some(tensors) = read_safetensors_tensor_sizes(&single_path) {
            let mut layer_bytes: u64 = 0;
            for (name, size) in &tensors {
                if name.starts_with(&layer_dot) {
                    layer_bytes += size;
                }
            }
            if layer_bytes > 0 {
                return layer_bytes / num_layers as u64;
            }
        }
        // Fallback
        if let Ok(m) = std::fs::metadata(&single_path) {
            return m.len() / num_layers as u64;
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cake::sharding::discovery::{DiscoveredWorker, GpuInfo};
    use crate::cake::sharding::{Strategy, WorkerCapacity};
    use std::fs;
    use std::path::Path;

    fn make_worker(name: &str, tflops: f32, vram_gb: f64) -> DiscoveredWorker {
        DiscoveredWorker {
            name: name.to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![GpuInfo {
                name: "NVIDIA RTX Test".to_string(),
                vram_bytes: (vram_gb * 1024.0 * 1024.0 * 1024.0) as u64,
                tflops,
            }],
            backend: "cuda".to_string(),
            hostname: name.to_string(),
            os: "linux".to_string(),
        }
    }

    // Helper to call DefaultStrategy with concrete workers
    fn compute_layer_assignments<W: WorkerCapacity>(
        workers: &[W],
        num_layers: usize,
        master_tflops: f64,
        layer_size_bytes: u64,
        master_max_layers: usize,
        layer_prefix: &str,
    ) -> Vec<(usize, Vec<String>)> {
        let dyn_workers: Vec<&dyn WorkerCapacity> = workers.iter().map(|w| w as &dyn WorkerCapacity).collect();
        DefaultStrategy.assign_layers(&dyn_workers, num_layers, master_tflops, layer_size_bytes, master_max_layers, layer_prefix)
    }

    // ── layer_prefix_for_config tests ──────────────────────

    #[test]
    fn layer_prefix_for_config_qwen3_5() {
        let config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"]
        });
        assert_eq!(
            layer_prefix_for_config(&config),
            "model.language_model.layers"
        );
    }

    #[test]
    fn layer_prefix_for_config_llama() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"]
        });
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_qwen2() {
        let config = serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"]
        });
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_no_architectures() {
        let config = serde_json::json!({"hidden_size": 1024});
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_empty_architectures() {
        let config = serde_json::json!({"architectures": []});
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_multiple_architectures_qwen3_5_first() {
        let config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration", "LlamaForCausalLM"]
        });
        assert_eq!(
            layer_prefix_for_config(&config),
            "model.language_model.layers"
        );
    }

    #[test]
    fn layer_prefix_for_config_multiple_architectures_qwen3_5_second() {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM", "Qwen3_5ForConditionalGeneration"]
        });
        assert_eq!(
            layer_prefix_for_config(&config),
            "model.language_model.layers"
        );
    }

    #[test]
    fn layer_prefix_for_config_non_string_in_array() {
        let config = serde_json::json!({
            "architectures": [42, "LlamaForCausalLM"]
        });
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_architectures_not_array() {
        let config = serde_json::json!({
            "architectures": "LlamaForCausalLM"
        });
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_null_value() {
        let config = serde_json::json!(null);
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_all_supported_archs_use_model_layers() {
        for arch in &[
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
            "Phi3ForCausalLM",
            "Phi4ForCausalLM",
            "MistralForCausalLM",
            "Gemma3ForCausalLM",
            "FalconForCausalLM",
            "Olmo2ForCausalLM",
            "ExaoneForCausalLM",
            "Qwen3MoeForCausalLM",
        ] {
            let config = serde_json::json!({ "architectures": [arch] });
            assert_eq!(
                layer_prefix_for_config(&config),
                "model.layers",
                "arch {} should use model.layers",
                arch
            );
        }
    }

    #[test]
    fn test_layer_prefix_for_qwen3_5() {
        let config = serde_json::json!({"architectures": ["Qwen3_5ForConditionalGeneration"]});
        assert_eq!(layer_prefix_for_config(&config), "model.language_model.layers");
    }

    #[test]
    fn test_layer_prefix_for_standard() {
        let config = serde_json::json!({"architectures": ["LlamaForCausalLM"]});
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    // ── compute_layer_assignments tests ──────────────────────

    #[test]
    fn compute_assignments_empty_workers() {
        let workers: &[DiscoveredWorker] = &[];
        let result = compute_layer_assignments(workers, 24, 10.0, 100_000_000, usize::MAX, "model.layers");
        assert!(result.is_empty());
    }

    #[test]
    fn compute_assignments_zero_num_layers() {
        let workers = vec![make_worker("w1", 20.0, 12.0)];
        let result = compute_layer_assignments(&workers, 0, 10.0, 100_000_000, usize::MAX, "model.layers");
        assert!(result.is_empty());
    }

    #[test]
    fn compute_assignments_single_worker_equal_tflops() {
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (idx, ref layers) = result[0];
        assert_eq!(idx, 0);
        assert_eq!(layers.len(), 12);
    }

    #[test]
    fn compute_assignments_two_workers_proportional() {
        let workers = vec![
            make_worker("w1", 30.0, 24.0),
            make_worker("w2", 10.0, 24.0),
        ];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 2);

        let total_worker_layers: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert!((18..=20).contains(&total_worker_layers),
            "expected ~19 total worker layers, got {}", total_worker_layers);

        let w1_layers = result.iter().find(|(i, _)| *i == 0).map(|(_, l)| l.len()).unwrap_or(0);
        let w2_layers = result.iter().find(|(i, _)| *i == 1).map(|(_, l)| l.len()).unwrap_or(0);
        assert!(w1_layers > w2_layers,
            "w1 ({} layers) should have more than w2 ({} layers)", w1_layers, w2_layers);
    }

    #[test]
    fn compute_assignments_vram_cap() {
        let workers = vec![make_worker("small", 100.0, 1.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            1.0,
            500_000_000,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        assert!(layers.len() <= 2,
            "expected VRAM-capped to <=2 layers, got {}", layers.len());
    }

    #[test]
    fn compute_assignments_master_overflow_redistributed() {
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            100_000_000,
            4,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        assert_eq!(layers.len(), 20,
            "expected worker to get 20 layers after redistribution, got {}", layers.len());
    }

    #[test]
    fn compute_assignments_layer_name_format() {
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            10,
            10.0,
            100_000_000,
            usize::MAX,
            "model.language_model.layers",
        );
        assert!(!result.is_empty());
        let (_, ref layers) = result[0];
        for layer in layers {
            assert!(layer.starts_with("model.language_model.layers."),
                "layer name '{}' should start with 'model.language_model.layers.'", layer);
            let suffix = layer.strip_prefix("model.language_model.layers.").unwrap();
            suffix.parse::<usize>().expect("layer suffix should be a number");
        }
    }

    #[test]
    fn compute_assignments_zero_tflops_fallback() {
        let w = DiscoveredWorker {
            name: "w1".to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![],
            backend: "cpu".to_string(),
            hostname: "w1".to_string(),
            os: "linux".to_string(),
        };
        let workers = vec![w];
        let result = compute_layer_assignments(
            &workers,
            24,
            0.0,
            0,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        assert_eq!(layers.len(), 12);
    }

    #[test]
    fn compute_assignments_single_layer_model() {
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            1,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert!(total <= 1);
    }

    #[test]
    fn compute_assignments_worker_zero_vram_with_tflops() {
        let w = DiscoveredWorker {
            name: "cpu_worker".to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![],
            backend: "cpu".to_string(),
            hostname: "cpu_worker".to_string(),
            os: "linux".to_string(),
        };
        let result = compute_layer_assignments(
            &[w],
            24,
            10.0,
            500_000_000,
            usize::MAX,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn compute_assignments_very_high_tflops_low_vram() {
        let workers = vec![make_worker("fast_small", 1000.0, 0.1)];
        let result = compute_layer_assignments(
            &workers,
            24,
            1.0,
            200_000_000,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        assert!(
            layers.is_empty(),
            "expected 0 layers due to VRAM cap, got {}",
            layers.len()
        );
    }

    #[test]
    fn compute_assignments_contiguous_layer_ranges() {
        let workers = vec![
            make_worker("w1", 20.0, 24.0),
            make_worker("w2", 20.0, 24.0),
        ];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );

        let mut all_layers: Vec<usize> = Vec::new();
        for (_, layers) in &result {
            for layer in layers {
                let num: usize = layer.strip_prefix("model.layers.").unwrap().parse().unwrap();
                all_layers.push(num);
            }
        }
        all_layers.sort();

        if !all_layers.is_empty() {
            assert_eq!(all_layers[0], 0, "layers should start at 0");
            for i in 1..all_layers.len() {
                assert_eq!(
                    all_layers[i],
                    all_layers[i - 1] + 1,
                    "layers should be contiguous"
                );
            }
        }
    }

    #[test]
    fn compute_assignments_master_overflow_partial_redistribution() {
        let workers = vec![make_worker("w1", 5.0, 1.0)];
        let result = compute_layer_assignments(
            &workers,
            10,
            5.0,
            500_000_000,
            2,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert!(total <= 10);
    }

    #[test]
    fn compute_assignments_no_layer_size_constraint() {
        let workers = vec![make_worker("w1", 20.0, 0.001)];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            0,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        assert_eq!(layers.len(), 16);
    }

    #[test]
    fn compute_assignments_three_workers_sorted_by_tflops() {
        let workers = vec![
            make_worker("slow", 5.0, 24.0),
            make_worker("fast", 30.0, 24.0),
            make_worker("medium", 15.0, 24.0),
        ];
        let result = compute_layer_assignments(
            &workers,
            48,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        let fast_layers = result.iter().find(|(i, _)| *i == 1).map(|(_, l)| l.len()).unwrap_or(0);
        let medium_layers = result.iter().find(|(i, _)| *i == 2).map(|(_, l)| l.len()).unwrap_or(0);
        let slow_layers = result.iter().find(|(i, _)| *i == 0).map(|(_, l)| l.len()).unwrap_or(0);
        assert!(fast_layers >= medium_layers, "fast ({}) >= medium ({})", fast_layers, medium_layers);
        assert!(medium_layers >= slow_layers, "medium ({}) >= slow ({})", medium_layers, slow_layers);
    }

    #[test]
    fn compute_assignments_zero_tflops_multiple_workers() {
        let make_zero = |name: &str| DiscoveredWorker {
            name: name.to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![],
            backend: "cpu".to_string(),
            hostname: name.to_string(),
            os: "linux".to_string(),
        };
        let workers = vec![make_zero("w1"), make_zero("w2"), make_zero("w3")];
        let result = compute_layer_assignments(
            &workers,
            24,
            0.0,
            0,
            usize::MAX,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert_eq!(total, 12);
        for (_, layers) in &result {
            assert_eq!(layers.len(), 4);
        }
    }

    #[test]
    fn compute_assignments_all_layers_accounted_for() {
        let workers = vec![
            make_worker("w1", 15.0, 24.0),
            make_worker("w2", 25.0, 24.0),
        ];
        let num_layers = 32;
        let result = compute_layer_assignments(
            &workers,
            num_layers,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        let total_assigned: usize = result.iter().map(|(_, l)| l.len()).sum();
        let master_remainder = num_layers - total_assigned;
        assert_eq!(total_assigned + master_remainder, num_layers);
        assert!(master_remainder > 0, "master should keep some layers");
    }

    #[test]
    fn compute_assignments_multi_gpu_worker() {
        let w = DiscoveredWorker {
            name: "dual_gpu".to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![
                GpuInfo {
                    name: "NVIDIA RTX 3080".to_string(),
                    vram_bytes: 10 * 1024 * 1024 * 1024,
                    tflops: 15.0,
                },
                GpuInfo {
                    name: "NVIDIA RTX 3080".to_string(),
                    vram_bytes: 10 * 1024 * 1024 * 1024,
                    tflops: 15.0,
                },
            ],
            backend: "cuda".to_string(),
            hostname: "dual_gpu".to_string(),
            os: "linux".to_string(),
        };
        let result = compute_layer_assignments(
            &[w],
            24,
            10.0,
            1_000_000_000,
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        assert!(!layers.is_empty());
    }

    #[test]
    fn compute_assignments_master_max_layers_zero() {
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            10,
            10.0,
            100_000_000,
            0,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert_eq!(total, 10, "all layers should be assigned to workers");
    }

    #[test]
    fn compute_assignments_odd_layer_count() {
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            7,
            10.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert!(total <= 7);
        for (_, layers) in &result {
            for layer in layers {
                let num: usize = layer.strip_prefix("model.layers.").unwrap().parse().unwrap();
                assert!(num < 7, "layer index {} should be < 7", num);
            }
        }
    }

    #[test]
    fn compute_assignments_dominant_master() {
        let workers = vec![make_worker("w1", 1.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            100.0,
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert!(total <= 2, "weak worker should get very few layers, got {}", total);
    }

    // ── read_safetensors_tensor_sizes tests ───────────────────────────

    /// Build a minimal safetensors file with the given header JSON.
    fn write_fake_safetensors(path: &Path, header: &serde_json::Value) {
        use std::io::Write;
        let header_bytes = serde_json::to_vec(header).unwrap();
        let header_len = header_bytes.len() as u64;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(&header_bytes).unwrap();
        f.write_all(&vec![0u8; 1024]).unwrap();
    }

    #[test]
    fn read_safetensors_tensor_sizes_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({
            "model.layers.0.attn.weight": {
                "dtype": "F16",
                "shape": [1024, 1024],
                "data_offsets": [0, 2097152]
            },
            "model.layers.0.mlp.weight": {
                "dtype": "F16",
                "shape": [4096, 1024],
                "data_offsets": [2097152, 10485760]
            }
        });
        write_fake_safetensors(&path, &header);
        let result = read_safetensors_tensor_sizes(&path).unwrap();
        assert_eq!(result.len(), 2);
        let sizes: std::collections::HashMap<String, u64> = result.into_iter().collect();
        assert_eq!(sizes["model.layers.0.attn.weight"], 2097152);
        assert_eq!(sizes["model.layers.0.mlp.weight"], 10485760 - 2097152);
    }

    #[test]
    fn read_safetensors_tensor_sizes_skips_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "weight": { "dtype": "F16", "shape": [10], "data_offsets": [0, 20] }
        });
        write_fake_safetensors(&path, &header);
        let result = read_safetensors_tensor_sizes(&path).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "weight");
        assert_eq!(result[0].1, 20);
    }

    #[test]
    fn read_safetensors_tensor_sizes_nonexistent_file() {
        let result = read_safetensors_tensor_sizes(Path::new("/nonexistent/file.safetensors"));
        assert!(result.is_none());
    }

    #[test]
    fn read_safetensors_tensor_sizes_truncated_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        fs::write(&path, [0u8; 4]).unwrap();
        let result = read_safetensors_tensor_sizes(&path);
        assert!(result.is_none());
    }

    #[test]
    fn read_safetensors_tensor_sizes_header_too_large() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let huge_len: u64 = 20 * 1024 * 1024;
        let mut f = std::fs::File::create(&path).unwrap();
        use std::io::Write;
        f.write_all(&huge_len.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        drop(f);
        let result = read_safetensors_tensor_sizes(&path);
        assert!(result.is_none());
    }

    #[test]
    fn read_safetensors_tensor_sizes_invalid_json_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let garbage = b"this is not json!!!!";
        let header_len = garbage.len() as u64;
        let mut f = std::fs::File::create(&path).unwrap();
        use std::io::Write;
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(garbage).unwrap();
        drop(f);
        let result = read_safetensors_tensor_sizes(&path);
        assert!(result.is_none());
    }

    #[test]
    fn read_safetensors_tensor_sizes_missing_data_offsets() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({
            "weight": { "dtype": "F16", "shape": [10] }
        });
        write_fake_safetensors(&path, &header);
        let result = read_safetensors_tensor_sizes(&path).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn read_safetensors_tensor_sizes_empty_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({});
        write_fake_safetensors(&path, &header);
        let result = read_safetensors_tensor_sizes(&path).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ── estimate_layer_size tests ─────────────────────────────────────

    #[test]
    fn estimate_layer_size_zero_layers() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(estimate_layer_size(tmp.path(), 0, "model.layers"), 0);
    }

    #[test]
    fn estimate_layer_size_no_files() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(estimate_layer_size(tmp.path(), 24, "model.layers"), 0);
    }

    #[test]
    fn estimate_layer_size_single_safetensors_with_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "model.layers.0.attn.weight": {
                "dtype": "F16", "shape": [100, 10], "data_offsets": [0, 1000]
            },
            "model.layers.0.mlp.weight": {
                "dtype": "F16", "shape": [100, 10], "data_offsets": [1000, 2000]
            },
            "model.layers.1.attn.weight": {
                "dtype": "F16", "shape": [100, 10], "data_offsets": [2000, 3000]
            },
            "model.layers.1.mlp.weight": {
                "dtype": "F16", "shape": [100, 10], "data_offsets": [3000, 4000]
            },
            "model.embed_tokens.weight": {
                "dtype": "F16", "shape": [32000, 100], "data_offsets": [4000, 6404000]
            }
        });
        write_fake_safetensors(&path, &header);
        let result = estimate_layer_size(tmp.path(), 2, "model.layers");
        assert_eq!(result, 2000);
    }

    #[test]
    fn estimate_layer_size_single_safetensors_no_matching_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({
            "encoder.layers.0.weight": {
                "dtype": "F16", "shape": [10], "data_offsets": [0, 20]
            }
        });
        write_fake_safetensors(&path, &header);
        let result = estimate_layer_size(tmp.path(), 1, "model.layers");
        assert!(result > 0);
    }

    #[test]
    fn estimate_layer_size_sharded_with_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.attn.weight": "shard-00001.safetensors",
                "model.layers.0.mlp.weight": "shard-00001.safetensors",
                "model.layers.1.attn.weight": "shard-00002.safetensors",
                "model.layers.1.mlp.weight": "shard-00002.safetensors",
                "model.embed_tokens.weight": "shard-00001.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        ).unwrap();

        let shard1_header = serde_json::json!({
            "model.layers.0.attn.weight": { "dtype": "F16", "shape": [100], "data_offsets": [0, 500] },
            "model.layers.0.mlp.weight": { "dtype": "F16", "shape": [100], "data_offsets": [500, 1000] },
            "model.embed_tokens.weight": { "dtype": "F16", "shape": [32000], "data_offsets": [1000, 65000] }
        });
        write_fake_safetensors(&tmp.path().join("shard-00001.safetensors"), &shard1_header);

        let shard2_header = serde_json::json!({
            "model.layers.1.attn.weight": { "dtype": "F16", "shape": [100], "data_offsets": [0, 500] },
            "model.layers.1.mlp.weight": { "dtype": "F16", "shape": [100], "data_offsets": [500, 1000] }
        });
        write_fake_safetensors(&tmp.path().join("shard-00002.safetensors"), &shard2_header);

        let result = estimate_layer_size(tmp.path(), 2, "model.layers");
        assert_eq!(result, 1000);
    }

    #[test]
    fn estimate_layer_size_sharded_missing_shard_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.weight": "shard-00001.safetensors",
                "model.layers.1.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        ).unwrap();

        let shard1_header = serde_json::json!({
            "model.layers.0.weight": { "dtype": "F16", "shape": [100], "data_offsets": [0, 200] }
        });
        write_fake_safetensors(&tmp.path().join("shard-00001.safetensors"), &shard1_header);

        let result = estimate_layer_size(tmp.path(), 2, "model.layers");
        assert!(result > 0);
    }

    #[test]
    fn estimate_layer_size_with_language_model_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        let header = serde_json::json!({
            "model.language_model.layers.0.attn.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [0, 800]
            },
            "model.language_model.layers.1.attn.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [800, 1600]
            },
            "model.visual.layers.0.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [1600, 10000]
            }
        });
        write_fake_safetensors(&path, &header);
        let result = estimate_layer_size(tmp.path(), 2, "model.language_model.layers");
        assert_eq!(result, 800);
    }
}
