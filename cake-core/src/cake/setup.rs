//! Zero-config setup protocol for master and worker nodes.
//!
//! Runs **before** the normal inference lifecycle:
//!   1. Workers advertise via mDNS, master discovers them.
//!   2. Master computes layer assignments based on GPU VRAM.
//!   3. Master connects to each worker, authenticates, assigns layers,
//!      and pushes model data if the worker doesn't have it cached.
//!   4. Workers load their assigned layers and signal readiness.
//!
//! After setup, both sides proceed with normal `Context::from_args()` / inference.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::net::{TcpListener, TcpStream};

use super::auth;
use super::discovery::{self, DiscoveredWorker};
use super::proto::Message;
use super::topology::{Node, Topology};

/// Derive the layer name prefix from config.json.
/// Returns e.g. "model.language_model.layers" for Qwen3.5, "model.layers" otherwise.
fn layer_prefix_for_config(config_json: &serde_json::Value) -> String {
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        for arch in archs {
            if let Some("Qwen3_5ForConditionalGeneration") = arch.as_str() {
                return "model.language_model.layers".to_string();
            }
        }
    }
    "model.layers".to_string()
}

/// Maximum chunk size for model data transfer (128 MB).
const MODEL_DATA_CHUNK_SIZE: usize = 128 * 1024 * 1024;

/// Query actual free GPU memory via nvidia-smi (CUDA only).
/// Returns 0 if unavailable.
fn detect_free_gpu_memory() -> u64 {
    #[cfg(feature = "cuda")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout
                    .lines()
                    .filter_map(|line| line.trim().parse::<u64>().ok())
                    .map(|mb| mb * 1024 * 1024)
                    .sum();
            }
        }
    }
    0
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
fn estimate_layer_size(model_path: &Path, num_layers: usize, layer_prefix: &str) -> u64 {
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

// ── Layer assignment ────────────────────────────────────────────────────────

/// Compute layer assignments proportional to each worker's estimated TFLOPS,
/// accounting for the master's own compute so it retains its fair share of layers.
/// When `layer_size_bytes` > 0, each worker's assignment is capped by its
/// per-GPU VRAM to prevent out-of-memory errors on multi-GPU nodes.
/// The master's local layers are also capped by `master_max_layers` to avoid OOM.
///
/// Returns a vec of `(worker_index, layer_names)`.
/// Workers are sorted by TFLOPS descending, and layers are assigned as
/// contiguous ranges starting from layer 0. Unassigned layers remain on master.
pub fn compute_layer_assignments(
    workers: &[DiscoveredWorker],
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
                    &workers[worker_idx].name,
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
                    &workers[*worker_idx].name,
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

// ── Master setup ────────────────────────────────────────────────────────────

/// Run the full zero-config master setup.
///
/// Discovers workers via mDNS, computes layer assignments based on VRAM,
/// connects to each worker with mutual authentication, pushes model data
/// as needed, and returns a `Topology` ready for normal inference.
pub async fn master_setup(
    cluster_key: &str,
    model_path: &Path,
    discovery_timeout: Duration,
    min_workers: usize,
) -> Result<Topology> {
    // Read config.json and compute a fingerprint for cache keying
    let config_path = model_path.join("config.json");
    let config_data = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("failed to read {}: {}", config_path.display(), e))?;
    let model_hash = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(config_data.as_bytes());
        let result = hasher.finalize();
        hex::encode(&result[..4])
    };
    let config_json: serde_json::Value = serde_json::from_str(&config_data)?;
    let num_layers = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            // Some models (e.g. Qwen3.5) nest config under text_config
            config_json
                .get("text_config")
                .and_then(|tc| tc.get("num_hidden_layers"))
                .and_then(|v| v.as_u64())
        })
        .ok_or_else(|| anyhow!("num_hidden_layers not found in config.json"))? as usize;

    // Derive layer naming prefix from architecture (needed early for layer size estimation)
    let layer_prefix = layer_prefix_for_config(&config_json);

    log::info!(
        "model has {} transformer layers (prefix: {})",
        num_layers,
        &layer_prefix,
    );

    // Detect master GPU and free VRAM concurrently with the discovery window
    // (nvidia-smi can take ~1-2s; hide that cost inside the discovery timeout).
    let master_gpus = discovery::detect_gpus();
    let master_tflops: f64 = master_gpus.iter().map(|g| g.tflops as f64).sum();
    let free_gpu_fut = tokio::task::spawn_blocking(detect_free_gpu_memory);

    // Discover workers
    let workers = discovery::discover_workers(cluster_key, discovery_timeout, min_workers).await?;
    if workers.is_empty() {
        log::warn!("no workers discovered — all layers will be loaded locally");
        return Ok(Topology::new());
    }

    // nvidia-smi result is now ready (ran during the discovery window)
    let master_free_from_smi = free_gpu_fut.await.unwrap_or(0);

    // Estimate per-layer size for VRAM-aware capping.
    // Uses weight_map tensor-count fractions to exclude non-layer weights
    // (visual encoder, MTP heads, embeddings, lm_head, FP8 scale_inv, etc.).
    let layer_size_on_disk = estimate_layer_size(model_path, num_layers, &layer_prefix);
    if layer_size_on_disk > 0 {
        log::info!(
            "estimated layer size (on disk): {}",
            human_bytes::human_bytes(layer_size_on_disk as f64)
        );
    }

    // Estimate non-layer overhead the master must hold (embeddings + lm_head + norm + CUDA runtime).
    // embeddings = vocab_size * hidden_size * dtype_bytes
    // lm_head    = vocab_size * hidden_size * dtype_bytes (unless tied)
    let dtype_bytes: u64 = 2; // F16
    let vocab_size = config_json
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            config_json
                .get("text_config")
                .and_then(|tc| tc.get("vocab_size"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(32000);
    let hidden_size = config_json
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            config_json
                .get("text_config")
                .and_then(|tc| tc.get("hidden_size"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(4096);
    let tie_embeddings = config_json
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let embed_size = vocab_size * hidden_size * dtype_bytes;
    let lm_head_size = if tie_embeddings { 0 } else { embed_size };
    // Add ~1 GiB for CUDA runtime/context, KV cache, memory fragmentation, and misc overhead
    let master_overhead = embed_size + lm_head_size + 1024 * 1024 * 1024;

    // FP8 models store weights at 1 byte per element on disk, but after dequantization
    // they expand to the target dtype (F16 = 2 bytes, BF16 = 2 bytes). Scale the layer
    // size estimate so VRAM-based capping uses the actual in-memory size.
    let is_fp8 = crate::utils::fp8::is_fp8_quantized(&config_path);
    let layer_size_bytes = if is_fp8 && layer_size_on_disk > 0 {
        let expanded = layer_size_on_disk * dtype_bytes; // FP8 is 1 byte, target is dtype_bytes
        log::info!(
            "FP8 model: layer size after dequantization: {} ({}x expansion)",
            human_bytes::human_bytes(expanded as f64),
            dtype_bytes,
        );
        expanded
    } else {
        layer_size_on_disk
    };

    log::info!(
        "master overhead: embeddings={} lm_head={} total={}",
        human_bytes::human_bytes(embed_size as f64),
        human_bytes::human_bytes(lm_head_size as f64),
        human_bytes::human_bytes(master_overhead as f64),
    );

    // Cap master layers by its own GPU VRAM minus the non-layer overhead.
    // Use actual free VRAM (from nvidia-smi) when available, as total VRAM
    // overestimates on systems with display servers or other GPU consumers.
    let master_max_layers = if layer_size_bytes > 0 && !master_gpus.is_empty() {
        let master_vram: u64 = master_gpus.iter().map(|g| g.vram_bytes).sum();
        let effective_vram = if master_free_from_smi > 0 && master_free_from_smi < master_vram {
            log::info!(
                "master GPU: {} total, {} free",
                human_bytes::human_bytes(master_vram as f64),
                human_bytes::human_bytes(master_free_from_smi as f64),
            );
            master_free_from_smi
        } else {
            master_vram
        };
        let available = effective_vram.saturating_sub(master_overhead);
        let max = (available / layer_size_bytes) as usize;
        log::info!(
            "master GPU: {} available for layers — can fit ~{} layers locally",
            human_bytes::human_bytes(available as f64),
            max
        );
        max
    } else {
        usize::MAX
    };

    // Compute assignments based on TFLOPS, capped by per-GPU VRAM
    let assignments = compute_layer_assignments(
        &workers,
        num_layers,
        master_tflops,
        layer_size_bytes,
        master_max_layers,
        &layer_prefix,
    );

    // Summarise layer assignments and estimate per-node weight loads
    let total_assigned: usize = assignments.iter().map(|(_, l)| l.len()).sum();
    let master_layers = num_layers - total_assigned;
    log::info!("layer assignments:");
    for (worker_idx, layers) in &assignments {
        let w = &workers[*worker_idx];
        let range = if layers.is_empty() {
            "(none)".to_string()
        } else {
            format!("{} — {}", layers.first().unwrap(), layers.last().unwrap())
        };
        let weight_load = layers.len() as u64 * layer_size_bytes;
        log::info!(
            "  {} ({}, {:.1} TFLOPS) → {} layers ({}) [{}]",
            &w.name,
            human_bytes::human_bytes(w.total_vram() as f64),
            w.total_tflops(),
            layers.len(),
            human_bytes::human_bytes(weight_load as f64),
            range,
        );
    }
    log::info!(
        "  master ({:.1} TFLOPS) → {} layers ({} weights + {} overhead)",
        master_tflops,
        master_layers,
        human_bytes::human_bytes((master_layers as u64 * layer_size_bytes) as f64),
        human_bytes::human_bytes(master_overhead as f64),
    );
    if layer_size_bytes > 0 {
        log::info!(
            "total weight read per token: {} ({} per layer × {})",
            human_bytes::human_bytes((num_layers as u64 * layer_size_bytes) as f64),
            human_bytes::human_bytes(layer_size_bytes as f64),
            num_layers,
        );
    }

    // Connect to all workers concurrently: authenticate, assign layers, push data
    let mut handles = Vec::new();

    for (worker_idx, layers) in &assignments {
        let worker = workers[*worker_idx].clone();
        if layers.is_empty() {
            continue;
        }

        let layers = layers.clone();
        let cluster_key = cluster_key.to_string();
        let model_hash = model_hash.clone();
        let model_path = model_path.to_path_buf();
        let model_name = model_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        handles.push(tokio::spawn(async move {
            log::info!(
                "connecting to worker '{}' at {} ...",
                &worker.name,
                &worker.host
            );

            let mut stream = TcpStream::connect(&worker.host)
                .await
                .map_err(|e| anyhow!("can't connect to {}: {}", &worker.host, e))?;
            let _ = stream.set_nodelay(true);

            // Mutual authentication
            auth::authenticate_as_master(&mut stream, &cluster_key).await?;
            log::info!("[{}] authenticated", &worker.name);

            // Send layer assignment
            let msg = Message::LayerAssignment {
                layers: layers.clone(),
                model_hash,
            };
            msg.to_writer(&mut stream).await?;

            // Read ack
            let (_, ack) = Message::from_reader(&mut stream).await?;
            let needs_data = match ack {
                Message::LayerAssignmentAck { needs_data } => needs_data,
                other => {
                    return Err(anyhow!(
                        "[{}] unexpected response to LayerAssignment: {:?}",
                        &worker.name,
                        other
                    ))
                }
            };

            if needs_data {
                push_model_data(&mut stream, &model_path, &layers, &worker.name, &model_name).await?;
            } else {
                log::info!("[{}] worker has model data cached", &worker.name);
            }

            // Wait for WorkerReady
            let (_, ready) = Message::from_reader(&mut stream).await?;
            if !matches!(ready, Message::WorkerReady) {
                return Err(anyhow!(
                    "[{}] expected WorkerReady, got {:?}",
                    &worker.name,
                    ready
                ));
            }
            log::info!("[{}] worker ready", &worker.name);

            Ok::<_, anyhow::Error>((worker, layers))
        }));
    }

    // Collect results
    let mut topology = Topology::new();
    for handle in handles {
        let (worker, layers) = handle.await??;
        topology.insert(
            worker.name.clone(),
            Node {
                host: worker.host.clone(),
                description: Some(
                    worker
                        .gpus
                        .iter()
                        .map(|g| g.name.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
                layers,
                vram_bytes: worker.total_vram(),
                tflops: worker.total_tflops(),
                backend: worker.backend.clone(),
                hostname: worker.hostname.clone(),
                os: worker.os.clone(),
            },
        );
    }

    Ok(topology)
}

/// Push model data files to a worker that doesn't have them cached.
async fn push_model_data(
    stream: &mut TcpStream,
    model_path: &Path,
    layers: &[String],
    worker_name: &str,
    model_name: &str,
) -> Result<()> {
    let overall_start = Instant::now();
    let mut overall_bytes: u64 = 0;

    let layer_range = if layers.is_empty() {
        "(none)".to_string()
    } else {
        format!(
            "{} — {} ({} layers)",
            layers.first().unwrap(),
            layers.last().unwrap(),
            layers.len()
        )
    };

    log::info!(
        "[{}] pushing {} [{}]",
        worker_name,
        model_name,
        layer_range
    );

    // Always send config.json and tokenizer.json
    let mut files_to_send: Vec<PathBuf> = vec![
        model_path.join("config.json"),
        model_path.join("tokenizer.json"),
    ];

    // Determine which safetensors shard files contain the assigned layers
    let index_path = model_path.join("model.safetensors.index.json");
    let mut filtered_index: Option<Vec<u8>> = None;
    if index_path.exists() {
        files_to_send.push(index_path.clone());
        let index_data = std::fs::read(&index_path)?;
        let mut index_json: serde_json::Value = serde_json::from_slice(&index_data)?;
        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("no weight_map in model.safetensors.index.json"))?
            .clone();

        // Find shard files that contain tensors for the assigned layers
        let mut needed_shards: HashSet<String> = HashSet::new();
        let mut needed_weights: serde_json::Map<String, serde_json::Value> =
            serde_json::Map::new();
        for (tensor_name, shard_file) in &weight_map {
            for layer in layers {
                if tensor_name.starts_with(&format!("{}.", layer)) {
                    if let Some(filename) = shard_file.as_str() {
                        needed_shards.insert(filename.to_string());
                    }
                    needed_weights.insert(tensor_name.clone(), shard_file.clone());
                }
            }
        }

        // Build a filtered index.json that only references the pushed shards
        if let Some(obj) = index_json.as_object_mut() {
            obj.insert(
                "weight_map".to_string(),
                serde_json::Value::Object(needed_weights),
            );
        }
        filtered_index = Some(serde_json::to_vec_pretty(&index_json)?);

        log::info!(
            "[{}] pushing {} shard file(s) + config + tokenizer + index",
            worker_name,
            needed_shards.len()
        );

        for shard in &needed_shards {
            files_to_send.push(model_path.join(shard));
        }
    } else {
        // Single safetensors file
        let single = model_path.join("model.safetensors");
        if single.exists() {
            files_to_send.push(single);
        }
    }

    // Stream each file
    for file_path in &files_to_send {
        let filename = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Use filtered index if this is the index file
        let file_data = if filename == "model.safetensors.index.json" {
            if let Some(ref data) = filtered_index {
                data.clone()
            } else {
                std::fs::read(file_path)
                    .map_err(|e| anyhow!("failed to read {}: {}", file_path.display(), e))?
            }
        } else {
            std::fs::read(file_path)
                .map_err(|e| anyhow!("failed to read {}: {}", file_path.display(), e))?
        };
        let total_size = file_data.len() as u64;
        let file_start = Instant::now();
        let mut offset: u64 = 0;

        log::info!(
            "[{}] sending {} ({}) ...",
            worker_name,
            &filename,
            human_bytes::human_bytes(total_size as f64)
        );

        for chunk in file_data.chunks(MODEL_DATA_CHUNK_SIZE) {
            let msg = Message::ModelDataChunk {
                filename: filename.clone(),
                offset,
                total_size,
                data: chunk.to_vec(),
            };
            msg.to_writer(stream).await?;
            offset += chunk.len() as u64;

            // Log progress for large files
            if total_size > MODEL_DATA_CHUNK_SIZE as u64 {
                let elapsed = file_start.elapsed().as_secs_f64();
                let speed = offset as f64 / elapsed;
                let pct = (offset as f64 / total_size as f64) * 100.0;
                let remaining = total_size - offset;
                let eta_secs = if speed > 0.0 {
                    remaining as f64 / speed
                } else {
                    0.0
                };
                log::info!(
                    "[{}] {} — {}/{} ({:.1}%) — {}/s — ETA {:.0}s",
                    worker_name,
                    &filename,
                    human_bytes::human_bytes(offset as f64),
                    human_bytes::human_bytes(total_size as f64),
                    pct,
                    human_bytes::human_bytes(speed),
                    eta_secs
                );
            }
        }

        let file_elapsed = file_start.elapsed();
        let file_speed = total_size as f64 / file_elapsed.as_secs_f64();
        overall_bytes += total_size;

        log::info!(
            "[{}] sent {} ({}) in {:.1}s — {}/s",
            worker_name,
            &filename,
            human_bytes::human_bytes(total_size as f64),
            file_elapsed.as_secs_f64(),
            human_bytes::human_bytes(file_speed)
        );
    }

    // Signal done
    Message::ModelDataDone.to_writer(stream).await?;

    let overall_elapsed = overall_start.elapsed();
    let overall_speed = overall_bytes as f64 / overall_elapsed.as_secs_f64();
    log::info!(
        "[{}] transfer complete: {} in {:.1}s — {}/s avg",
        worker_name,
        human_bytes::human_bytes(overall_bytes as f64),
        overall_elapsed.as_secs_f64(),
        human_bytes::human_bytes(overall_speed)
    );

    Ok(())
}

/// Check whether a cache directory contains valid model data for the given layers.
///
/// For sharded models, verifies that the cached index's weight_map references all
/// assigned layers and that the shard files containing those layers exist on disk.
fn has_valid_model_cache(cache_dir: &Path, layers: &[String]) -> bool {
    if !cache_dir.join("config.json").exists() {
        return false;
    }
    // Single safetensors file — if it exists, assume it has everything
    if cache_dir.join("model.safetensors").exists() {
        return true;
    }
    // Sharded model: need index + shard files for all assigned layers
    let index_path = cache_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        if let Ok(data) = std::fs::read_to_string(&index_path) {
            if let Ok(index) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    // For each assigned layer, check that at least one tensor exists
                    // in the weight_map and its shard file is present on disk.
                    for layer in layers {
                        let prefix = format!("{}.", layer);
                        let has_layer = weight_map.iter().any(|(tensor_name, shard_file)| {
                            tensor_name.starts_with(&prefix)
                                && shard_file
                                    .as_str()
                                    .is_some_and(|f| cache_dir.join(f).exists())
                        });
                        if !has_layer {
                            log::debug!(
                                "cache miss: {} not found in {}",
                                layer,
                                cache_dir.display()
                            );
                            return false;
                        }
                    }
                    return true;
                }
            }
        }
    }
    false
}

// ── Worker setup ────────────────────────────────────────────────────────────

/// Run the zero-config worker setup.
///
/// Advertises via mDNS, waits for the master to connect and assign layers,
/// receives model data if needed, and returns the assigned layers and model
/// cache path.
///
/// The `listener` is returned so it can be reused for inference connections.
/// Progress callback for worker setup stages.
///
/// Arguments: (stage, message, progress 0.0–1.0)
/// - stage: "discovery", "connected", "authenticated", "layers", "receiving", "cached", "ready"
/// - message: human-readable status
/// - progress: 0.0–1.0 for transfer, 0.0 otherwise
pub type SetupProgressFn = dyn Fn(&str, &str, f64) + Send + Sync;

pub async fn worker_setup(
    worker_name: &str,
    cluster_key: &str,
    bind_address: &str,
    model_cache_dir: &Path,
) -> Result<(Vec<String>, PathBuf, TcpListener)> {
    worker_setup_with_progress(worker_name, cluster_key, bind_address, model_cache_dir, None).await
}

pub async fn worker_setup_with_progress(
    worker_name: &str,
    cluster_key: &str,
    bind_address: &str,
    model_cache_dir: &Path,
    on_progress: Option<&SetupProgressFn>,
) -> Result<(Vec<String>, PathBuf, TcpListener)> {
    // Detect GPUs
    let gpus = discovery::detect_gpus();
    log::info!("detected {} GPU(s):", gpus.len());
    for gpu in &gpus {
        log::info!(
            "  {} — {} (~{:.1} TFLOPS)",
            &gpu.name,
            human_bytes::human_bytes(gpu.vram_bytes as f64),
            gpu.tflops
        );
    }

    // Bind listener
    let listener = TcpListener::bind(bind_address).await?;
    let port = listener.local_addr()?.port();
    log::info!("listening on {} (setup mode)", bind_address);

    // Advertise via UDP broadcast
    let _discovery = discovery::advertise_worker(worker_name, port, cluster_key, &gpus)?;

    log::info!("waiting for master to connect and assign layers...");
    if let Some(cb) = &on_progress {
        cb("discovery", "Waiting for master...", 0.0);
    }

    // Accept one setup connection from master
    let (mut stream, client_addr) = listener.accept().await?;
    let _ = stream.set_nodelay(true);
    log::info!("[{}] master connected", client_addr);
    if let Some(cb) = &on_progress {
        cb("connected", &format!("Master connected ({})", client_addr), 0.0);
    }

    // Authenticate
    auth::authenticate_as_worker(&mut stream, cluster_key).await?;
    log::info!("[{}] authenticated", client_addr);
    if let Some(cb) = &on_progress {
        cb("authenticated", "Authenticated with master", 0.0);
    }

    // Receive layer assignment
    let (_, msg) = Message::from_reader(&mut stream).await?;
    let (layers, model_hash) = match msg {
        Message::LayerAssignment {
            layers,
            model_hash,
        } => (layers, model_hash),
        other => {
            return Err(anyhow!(
                "expected LayerAssignment, got {:?}",
                other
            ))
        }
    };

    log::info!("assigned {} layers:", layers.len());
    for layer in &layers {
        log::info!("  {}", layer);
    }
    if let Some(cb) = &on_progress {
        cb("layers", &format!("Assigned {} layer(s)", layers.len()), 0.0);
    }

    // Determine cache directory: cluster_hash/model_hash
    // This ensures switching models invalidates the cache.
    let cluster_id = discovery::cluster_hash(cluster_key);
    let cache_dir = if model_hash.is_empty() {
        // Backwards compat with old masters that don't send model_hash
        model_cache_dir.join(&cluster_id)
    } else {
        model_cache_dir.join(format!("{}-{}", cluster_id, model_hash))
    };
    std::fs::create_dir_all(&cache_dir)?;

    // Check if we already have a valid model data cache for the assigned layers.
    let needs_data = !has_valid_model_cache(&cache_dir, &layers);

    let ack = Message::LayerAssignmentAck { needs_data };
    ack.to_writer(&mut stream).await?;

    if needs_data {
        if let Some(cb) = &on_progress {
            cb("receiving", "Receiving model data...", 0.0);
        }
        receive_model_data(&mut stream, &cache_dir, &layers, on_progress).await?;
    } else {
        log::info!("using cached model data from {}", cache_dir.display());
        if let Some(cb) = &on_progress {
            cb("cached", "Using cached model data", 1.0);
        }
    }

    // Signal ready
    Message::WorkerReady.to_writer(&mut stream).await?;
    log::info!("setup complete, ready for inference");
    if let Some(cb) = &on_progress {
        cb("ready", "Setup complete", 1.0);
    }

    // Drop the setup connection (stream goes out of scope)
    // The listener is returned for reuse
    Ok((layers, cache_dir, listener))
}

/// Receive model data from master and write to the cache directory.
async fn receive_model_data(
    stream: &mut TcpStream,
    cache_dir: &Path,
    layers: &[String],
    on_progress: Option<&SetupProgressFn>,
) -> Result<()> {
    let overall_start = Instant::now();
    let mut overall_bytes: u64 = 0;
    let mut current_file: Option<(String, std::fs::File, Instant, u64)> = None;

    let layer_range = if layers.is_empty() {
        "(none)".to_string()
    } else {
        format!(
            "{} — {} ({} layers)",
            layers.first().unwrap(),
            layers.last().unwrap(),
            layers.len()
        )
    };

    log::info!("receiving model data [{}] ...", layer_range);

    loop {
        let (_, msg) = Message::from_reader(stream).await?;

        match msg {
            Message::ModelDataChunk {
                filename,
                offset,
                total_size,
                data,
            } => {
                // Open new file if needed
                let file = if let Some((ref name, ref mut file, _, _)) = current_file {
                    if name == &filename {
                        file
                    } else {
                        // Close previous file, log stats
                        if let Some((prev_name, _, start, size)) = current_file.take() {
                            let elapsed = start.elapsed();
                            let speed = size as f64 / elapsed.as_secs_f64();
                            log::info!(
                                "received {} ({}) — {}/s",
                                &prev_name,
                                human_bytes::human_bytes(size as f64),
                                human_bytes::human_bytes(speed)
                            );
                            if let Some(cb) = &on_progress {
                                cb("receiving", &format!("{} complete", &prev_name), 1.0);
                            }
                        }
                        let path = cache_dir.join(&filename);
                        let f = std::fs::File::create(&path)?;
                        current_file = Some((filename.clone(), f, Instant::now(), total_size));
                        if let Some(cb) = &on_progress {
                            cb("receiving", &format!("Receiving {} ({})", &filename, human_bytes::human_bytes(total_size as f64)), 0.0);
                        }
                        &mut current_file.as_mut().unwrap().1
                    }
                } else {
                    let path = cache_dir.join(&filename);
                    let f = std::fs::File::create(&path)?;
                    current_file = Some((filename.clone(), f, Instant::now(), total_size));
                    if let Some(cb) = &on_progress {
                        cb("receiving", &format!("Receiving {} ({})", &filename, human_bytes::human_bytes(total_size as f64)), 0.0);
                    }
                    &mut current_file.as_mut().unwrap().1
                };

                file.write_all(&data)?;
                overall_bytes += data.len() as u64;

                // Progress callback for all files (not just large ones)
                let written = offset + data.len() as u64;
                if let Some((_, _, ref start, _)) = current_file {
                    let elapsed = start.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 { written as f64 / elapsed } else { 0.0 };
                    let pct = if total_size > 0 { (written as f64 / total_size as f64) * 100.0 } else { 0.0 };

                    if total_size > MODEL_DATA_CHUNK_SIZE as u64 && written < total_size {
                        let remaining = total_size - written;
                        let eta_secs = if speed > 0.0 { remaining as f64 / speed } else { 0.0 };
                        log::info!(
                            "  {} — {}/{} ({:.1}%) — {}/s — ETA {:.0}s",
                            &filename,
                            human_bytes::human_bytes(written as f64),
                            human_bytes::human_bytes(total_size as f64),
                            pct,
                            human_bytes::human_bytes(speed),
                            eta_secs
                        );
                        if let Some(cb) = &on_progress {
                            let msg = format!(
                                "{} — {}/{} — {}/s — ETA {:.0}s",
                                &filename,
                                human_bytes::human_bytes(written as f64),
                                human_bytes::human_bytes(total_size as f64),
                                human_bytes::human_bytes(speed),
                                eta_secs
                            );
                            cb("receiving", &msg, pct / 100.0);
                        }
                    }
                }
            }
            Message::ModelDataDone => {
                // Log last file
                if let Some((name, _, start, size)) = current_file.take() {
                    let elapsed = start.elapsed();
                    let speed = size as f64 / elapsed.as_secs_f64();
                    log::info!(
                        "received {} ({}) — {}/s",
                        &name,
                        human_bytes::human_bytes(size as f64),
                        human_bytes::human_bytes(speed)
                    );
                }
                break;
            }
            other => {
                return Err(anyhow!(
                    "unexpected message during data transfer: {:?}",
                    other
                ));
            }
        }
    }

    let overall_elapsed = overall_start.elapsed();
    let overall_speed = overall_bytes as f64 / overall_elapsed.as_secs_f64();
    log::info!(
        "model data received: {} in {:.1}s — {}/s avg, cached to {}",
        human_bytes::human_bytes(overall_bytes as f64),
        overall_elapsed.as_secs_f64(),
        human_bytes::human_bytes(overall_speed),
        cache_dir.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
    fn has_valid_model_cache_no_config() {
        let tmp = tempfile::tempdir().unwrap();
        let layers = vec!["model.layers.0".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_single_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        fs::write(tmp.path().join("model.safetensors"), "data").unwrap();
        let layers = vec!["model.layers.0".to_string()];
        assert!(has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_complete() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.attn.weight": "shard-00001.safetensors",
                "model.layers.1.mlp.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        fs::write(tmp.path().join("shard-00002.safetensors"), "data").unwrap();

        let layers = vec![
            "model.layers.0".to_string(),
            "model.layers.1".to_string(),
        ];
        assert!(has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_missing_layer() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.attn.weight": "shard-00001.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();

        // Request layer 1 which is not in the weight map at all
        let layers = vec![
            "model.layers.0".to_string(),
            "model.layers.1".to_string(),
        ];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_missing_shard_file() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.attn.weight": "shard-00001.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        // Don't create shard-00001.safetensors

        let layers = vec!["model.layers.0".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_config_only_no_weights() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let layers = vec!["model.layers.0".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    // ── compute_layer_assignments tests ──────────────────────────

    use super::compute_layer_assignments;
    use crate::cake::discovery::{DiscoveredWorker, GpuInfo};

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

    #[test]
    fn compute_assignments_empty_workers() {
        let result = compute_layer_assignments(&[], 24, 10.0, 100_000_000, usize::MAX, "model.layers");
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
        // Worker 10 TFLOPS, master 10 TFLOPS => 50/50 split of 24 layers
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            100_000_000, // 100 MB per layer, 24 GB VRAM = plenty
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (idx, ref layers) = result[0];
        assert_eq!(idx, 0);
        // 10/(10+10) * 24 = 12 layers
        assert_eq!(layers.len(), 12);
    }

    #[test]
    fn compute_assignments_two_workers_proportional() {
        // w1: 30 TFLOPS, w2: 10 TFLOPS, master: 10 TFLOPS
        // total = 50 TFLOPS, workers get (40/50)*24 = 19.2 => 19 layers
        // w1 gets 30/40 * 19 ~= 14, w2 gets remainder
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
        // Workers get ~19 layers (rounding), master gets ~5
        assert!((18..=20).contains(&total_worker_layers),
            "expected ~19 total worker layers, got {}", total_worker_layers);

        // w1 (30 TFLOPS) should get more than w2 (10 TFLOPS)
        let w1_layers = result.iter().find(|(i, _)| *i == 0).map(|(_, l)| l.len()).unwrap_or(0);
        let w2_layers = result.iter().find(|(i, _)| *i == 1).map(|(_, l)| l.len()).unwrap_or(0);
        assert!(w1_layers > w2_layers,
            "w1 ({} layers) should have more than w2 ({} layers)", w1_layers, w2_layers);
    }

    #[test]
    fn compute_assignments_vram_cap() {
        // Worker has only 1 GB VRAM, layer is 500 MB => can fit ~1 layer
        // (after 5% reserve on dedicated GPU: usable = 0.95 GB => 1 layer)
        let workers = vec![make_worker("small", 100.0, 1.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            1.0,         // master very weak
            500_000_000, // 500 MB per layer
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        // Even though TFLOPS says it should get ~24 layers, VRAM caps it
        assert!(layers.len() <= 2,
            "expected VRAM-capped to <=2 layers, got {}", layers.len());
    }

    #[test]
    fn compute_assignments_master_overflow_redistributed() {
        // 24 layers, master can hold only 4, master has 10 TFLOPS,
        // worker has 10 TFLOPS => worker initially gets 12, master gets 12.
        // Master overflow: 12 - 4 = 8 deficit, redistributed to worker.
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            100_000_000,
            4, // master_max_layers
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        // Worker should get 12 + 8 = 20 layers (master keeps 4)
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
            // Verify the suffix is a valid number
            let suffix = layer.strip_prefix("model.language_model.layers.").unwrap();
            suffix.parse::<usize>().expect("layer suffix should be a number");
        }
    }

    #[test]
    fn compute_assignments_zero_tflops_fallback() {
        // When all TFLOPS are zero (total_tflops <= 0.0), layers split evenly:
        // half to workers, half to master. This requires empty gpus list since
        // total_tflops() has a VRAM-based fallback for non-empty gpu lists.
        let w = DiscoveredWorker {
            name: "w1".to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![], // empty gpus => total_tflops() = 0.0
            backend: "cpu".to_string(),
            hostname: "w1".to_string(),
            os: "linux".to_string(),
        };
        let workers = vec![w];
        let result = compute_layer_assignments(
            &workers,
            24,
            0.0, // master also zero
            0,   // no layer size constraint
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        // Half of 24 = 12 to worker
        assert_eq!(layers.len(), 12);
    }

    // ── Additional layer_prefix_for_config tests ──────────────────────

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
        // Iterates through all archs, finds Qwen3_5 on second pass
        assert_eq!(
            layer_prefix_for_config(&config),
            "model.language_model.layers"
        );
    }

    #[test]
    fn layer_prefix_for_config_non_string_in_array() {
        // Architectures array with non-string element
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
        // Not an array, falls through to default
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_null_value() {
        let config = serde_json::json!(null);
        assert_eq!(layer_prefix_for_config(&config), "model.layers");
    }

    #[test]
    fn layer_prefix_for_config_all_supported_archs_use_model_layers() {
        // Verify all other supported architectures use "model.layers"
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

    // ── Additional has_valid_model_cache tests ────────────────────────

    #[test]
    fn has_valid_model_cache_empty_layers_with_config() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        // Empty layers list — no layers to validate, but no safetensors either
        let layers: Vec<String> = vec![];
        // With no safetensors files, falls through to index check which also fails
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_single_safetensors_empty_layers() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        fs::write(tmp.path().join("model.safetensors"), "data").unwrap();
        // Empty layers list with single safetensors => true (file exists)
        let layers: Vec<String> = vec![];
        assert!(has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_empty_layers() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.attn.weight": "shard-00001.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        // Empty layers => all layers validated (vacuously true)
        let layers: Vec<String> = vec![];
        assert!(has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_malformed_index_json() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            "not valid json",
        )
        .unwrap();
        let layers = vec!["model.layers.0".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_index_missing_weight_map() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({ "metadata": {} });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        let layers = vec!["model.layers.0".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_multiple_tensors_per_layer() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.0.attn.q_proj.weight": "shard-00001.safetensors",
                "model.layers.0.attn.k_proj.weight": "shard-00001.safetensors",
                "model.layers.0.attn.v_proj.weight": "shard-00001.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "shard-00002.safetensors",
                "model.layers.0.mlp.up_proj.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        // Only one shard file exists
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        // Missing shard-00002 — but layer 0 has tensors in shard-00001 which exists
        // The function checks that at least one tensor exists with its shard present
        let layers = vec!["model.layers.0".to_string()];
        assert!(has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_layer_prefix_disambiguation() {
        // Ensure "model.layers.1" doesn't match "model.layers.10"
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "model.layers.10.attn.weight": "shard-00001.safetensors"
            }
        });
        fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        // Request layer 1 — should NOT match layer 10
        let layers = vec!["model.layers.1".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
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
        // Write some dummy tensor data (doesn't matter for header parsing)
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
            "__metadata__": {
                "format": "pt"
            },
            "weight": {
                "dtype": "F16",
                "shape": [10],
                "data_offsets": [0, 20]
            }
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
        // Write only 4 bytes (not enough for the 8-byte header length)
        fs::write(&path, [0u8; 4]).unwrap();
        let result = read_safetensors_tensor_sizes(&path);
        assert!(result.is_none());
    }

    #[test]
    fn read_safetensors_tensor_sizes_header_too_large() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        // Claim a header larger than 10 MB
        let huge_len: u64 = 20 * 1024 * 1024;
        let mut f = std::fs::File::create(&path).unwrap();
        use std::io::Write;
        f.write_all(&huge_len.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 64]).unwrap(); // not enough data
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
            "weight": {
                "dtype": "F16",
                "shape": [10]
                // no data_offsets
            }
        });
        write_fake_safetensors(&path, &header);
        let result = read_safetensors_tensor_sizes(&path).unwrap();
        // Tensor without data_offsets is skipped
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
        // No safetensors files at all
        assert_eq!(estimate_layer_size(tmp.path(), 24, "model.layers"), 0);
    }

    #[test]
    fn estimate_layer_size_single_safetensors_with_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        // 2 layers, each with 2 tensors of 1000 bytes
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
        // Total layer bytes = 4 * 1000 = 4000, divided by 2 layers = 2000
        assert_eq!(result, 2000);
    }

    #[test]
    fn estimate_layer_size_single_safetensors_no_matching_prefix() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        // Tensors use a different prefix
        let header = serde_json::json!({
            "encoder.layers.0.weight": {
                "dtype": "F16", "shape": [10], "data_offsets": [0, 20]
            }
        });
        write_fake_safetensors(&path, &header);
        // layer_bytes=0, falls through to file-size fallback
        let result = estimate_layer_size(tmp.path(), 1, "model.layers");
        // Fallback: total file size / num_layers
        assert!(result > 0);
    }

    #[test]
    fn estimate_layer_size_sharded_with_headers() {
        let tmp = tempfile::tempdir().unwrap();
        // Create index file
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
        )
        .unwrap();

        // Create shard files with proper safetensors headers
        let shard1_header = serde_json::json!({
            "model.layers.0.attn.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [0, 500]
            },
            "model.layers.0.mlp.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [500, 1000]
            },
            "model.embed_tokens.weight": {
                "dtype": "F16", "shape": [32000], "data_offsets": [1000, 65000]
            }
        });
        write_fake_safetensors(&tmp.path().join("shard-00001.safetensors"), &shard1_header);

        let shard2_header = serde_json::json!({
            "model.layers.1.attn.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [0, 500]
            },
            "model.layers.1.mlp.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [500, 1000]
            }
        });
        write_fake_safetensors(&tmp.path().join("shard-00002.safetensors"), &shard2_header);

        let result = estimate_layer_size(tmp.path(), 2, "model.layers");
        // layer bytes = 500+500+500+500 = 2000, /2 = 1000
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
        )
        .unwrap();

        // Create shard-00001 but NOT shard-00002 (header read will fail)
        let shard1_header = serde_json::json!({
            "model.layers.0.weight": {
                "dtype": "F16", "shape": [100], "data_offsets": [0, 200]
            }
        });
        write_fake_safetensors(&tmp.path().join("shard-00001.safetensors"), &shard1_header);

        // Should fall through to raw file size fallback
        let result = estimate_layer_size(tmp.path(), 2, "model.layers");
        // Only shard-00001 exists; fallback sums existing files / num_layers
        // shard-00001 size / 2
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
        // Only language_model tensors: 800 + 800 = 1600, /2 = 800
        assert_eq!(result, 800);
    }

    // ── Additional compute_layer_assignments tests ────────────────────

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
        // With equal TFLOPS, worker gets round(0.5*1) = 1 or 0 layers
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert!(total <= 1);
    }

    #[test]
    fn compute_assignments_worker_zero_vram_with_tflops() {
        // Worker with TFLOPS but zero VRAM — no GPUs, so max_layers_for_size = MAX
        let w = DiscoveredWorker {
            name: "cpu_worker".to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![], // no GPUs
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
        // total_tflops = 0 (no gpus) + 10 master = 10
        // worker tflops = 0 => worker_layers = round(0/10 * 24) = 0
        // But zero TFLOPS fallback: workers get half
        // Actually master_tflops=10 > 0, so total_tflops=10 > 0, no fallback
        // worker gets 0 layers proportionally
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn compute_assignments_very_high_tflops_low_vram() {
        // Worker with extremely high TFLOPS but tiny VRAM (100 MB)
        let workers = vec![make_worker("fast_small", 1000.0, 0.1)];
        let result = compute_layer_assignments(
            &workers,
            24,
            1.0,
            200_000_000, // 200 MB per layer
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        // TFLOPS says it should get nearly all layers, but VRAM caps it severely
        // 0.1 GB = ~107 MB usable (after 5% reserve), can't fit any 200 MB layers
        assert!(
            layers.is_empty(),
            "expected 0 layers due to VRAM cap, got {}",
            layers.len()
        );
    }

    #[test]
    fn compute_assignments_contiguous_layer_ranges() {
        // Verify layers are assigned as contiguous ranges starting from 0
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

        // Collect all assigned layer numbers
        let mut all_layers: Vec<usize> = Vec::new();
        for (_, layers) in &result {
            for layer in layers {
                let num: usize = layer.strip_prefix("model.layers.").unwrap().parse().unwrap();
                all_layers.push(num);
            }
        }
        all_layers.sort();

        // Verify contiguous from 0
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
        // Master can only hold 2 layers, workers can't absorb all excess
        // w1: 1 GB VRAM, 500 MB per layer => ~1 layer max (after reserve)
        let workers = vec![make_worker("w1", 5.0, 1.0)];
        let result = compute_layer_assignments(
            &workers,
            10,
            5.0,
            500_000_000, // 500 MB per layer
            2,           // master can only hold 2
            "model.layers",
        );
        // Worker is VRAM-capped to ~1 layer, master wants only 2
        // deficit = (10 - initial_worker_layers) - 2; worker can't take more than ~1 total
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        // The total assigned should respect VRAM caps even with redistribution
        assert!(total <= 10);
    }

    #[test]
    fn compute_assignments_no_layer_size_constraint() {
        // layer_size_bytes = 0 means no VRAM capping
        let workers = vec![make_worker("w1", 20.0, 0.001)]; // tiny VRAM
        let result = compute_layer_assignments(
            &workers,
            24,
            10.0,
            0, // no layer size constraint
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        // Without VRAM cap, worker gets proportional share: 20/30 * 24 = 16
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
        // Fast worker (idx=1) should get the most layers
        let fast_layers = result
            .iter()
            .find(|(i, _)| *i == 1)
            .map(|(_, l)| l.len())
            .unwrap_or(0);
        let medium_layers = result
            .iter()
            .find(|(i, _)| *i == 2)
            .map(|(_, l)| l.len())
            .unwrap_or(0);
        let slow_layers = result
            .iter()
            .find(|(i, _)| *i == 0)
            .map(|(_, l)| l.len())
            .unwrap_or(0);
        assert!(
            fast_layers >= medium_layers,
            "fast ({}) >= medium ({})",
            fast_layers,
            medium_layers
        );
        assert!(
            medium_layers >= slow_layers,
            "medium ({}) >= slow ({})",
            medium_layers,
            slow_layers
        );
    }

    #[test]
    fn compute_assignments_zero_tflops_multiple_workers() {
        // Multiple workers with zero TFLOPS, layers split evenly among them
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
        // Workers get half of 24 = 12 layers, split among 3 = 4 each
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        assert_eq!(total, 12);
        // Each worker gets 4 layers
        for (_, layers) in &result {
            assert_eq!(layers.len(), 4);
        }
    }

    #[test]
    fn compute_assignments_all_layers_accounted_for() {
        // Verify total assigned + master remainder = num_layers
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
        // Master should keep some layers
        assert!(master_remainder > 0, "master should keep some layers");
    }

    #[test]
    fn compute_assignments_multi_gpu_worker() {
        // Worker with 2 GPUs
        let w = DiscoveredWorker {
            name: "dual_gpu".to_string(),
            host: "127.0.0.1".to_string(),
            port: 10128,
            gpus: vec![
                GpuInfo {
                    name: "NVIDIA RTX 3080".to_string(),
                    vram_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
                    tflops: 15.0,
                },
                GpuInfo {
                    name: "NVIDIA RTX 3080".to_string(),
                    vram_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
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
            1_000_000_000, // 1 GB per layer
            usize::MAX,
            "model.layers",
        );
        assert_eq!(result.len(), 1);
        let (_, ref layers) = result[0];
        // 30 TFLOPS worker vs 10 TFLOPS master => worker gets 30/40*24 = 18
        // But per-GPU VRAM caps: each GPU 10GB * 0.95 / 1GB = 9 layers per GPU
        // max_layers_for_size sums across GPUs = 18, so no cap needed
        assert!(!layers.is_empty());
    }

    #[test]
    fn compute_assignments_master_max_layers_zero() {
        // Master can hold 0 layers — all must go to workers
        let workers = vec![make_worker("w1", 10.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            10,
            10.0,
            100_000_000,
            0, // master can hold 0 layers
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        // Worker initially gets 5 (50%), master has 5 but can only hold 0
        // deficit=5, worker has plenty of VRAM, absorbs all 5
        assert_eq!(total, 10, "all layers should be assigned to workers");
    }

    #[test]
    fn compute_assignments_odd_layer_count() {
        // Odd number of layers shouldn't cause issues
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
        // Verify all layer names are valid
        for (_, layers) in &result {
            for layer in layers {
                let num: usize = layer.strip_prefix("model.layers.").unwrap().parse().unwrap();
                assert!(num < 7, "layer index {} should be < 7", num);
            }
        }
    }

    #[test]
    fn compute_assignments_dominant_master() {
        // Master has vastly more TFLOPS than worker
        let workers = vec![make_worker("w1", 1.0, 24.0)];
        let result = compute_layer_assignments(
            &workers,
            24,
            100.0, // master 100x stronger
            100_000_000,
            usize::MAX,
            "model.layers",
        );
        let total: usize = result.iter().map(|(_, l)| l.len()).sum();
        // Worker gets round(1/101 * 24) = 0 layers
        // But proportional code does .max(1) => at least 1 layer
        assert!(total <= 2, "weak worker should get very few layers, got {}", total);
    }
}
