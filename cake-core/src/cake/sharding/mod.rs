//! Sharding, topology, protocol, client, worker, and discovery code.
//!
//! This module consolidates all distributed inference infrastructure:
//! - [`Strategy`] trait and [`DefaultStrategy`] implementation
//! - [`WorkerCapacity`] trait for abstracting worker resources
//! - Topology management, wire protocol, client/worker networking
//! - Zero-config setup (master and worker)

pub mod topology;
pub mod default;
pub mod discovery;
pub(crate) mod proto;
pub(crate) mod client;
pub(crate) mod worker;
pub mod auth;
#[cfg(feature = "master")]
pub mod api;
#[cfg(feature = "master")]
pub mod master;

pub use topology::*;
pub use default::DefaultStrategy;
pub use client::*;
pub use proto::*;
pub use worker::*;

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::net::{TcpListener, TcpStream};


/// A sharding strategy decides how to distribute transformer layers across workers.
pub trait Strategy: Send + Sync {
    /// Assign layers to workers.
    ///
    /// Returns a vec of `(worker_index, layer_names)`.
    /// Workers are sorted by TFLOPS descending, and layers are assigned as
    /// contiguous ranges starting from layer 0. Unassigned layers remain on master.
    fn assign_layers(
        &self,
        workers: &[&dyn WorkerCapacity],
        num_layers: usize,
        master_tflops: f64,
        layer_size_bytes: u64,
        master_max_layers: usize,
        layer_prefix: &str,
    ) -> Vec<(usize, Vec<String>)>;
}

/// Common interface for layer assignment — implemented by both [`discovery::DiscoveredWorker`]
/// (zero-config) and [`topology::NamedNode`] (topology file with auto-sharding).
pub trait WorkerCapacity {
    /// Worker name (used for logging).
    fn name(&self) -> &str;
    /// Total VRAM (or system RAM) in bytes.
    fn total_vram(&self) -> u64;
    /// Estimated FP16 TFLOPS for this worker.
    fn total_tflops(&self) -> f64;
    /// Maximum number of layers this worker can fit given `layer_size_bytes`.
    fn max_layers_for_size(&self, layer_size_bytes: u64) -> usize;
}

/// Compute max layers from a list of GPUs, applying per-device VRAM reserves.
///
/// - **Dedicated VRAM (CUDA)**: reserve max(5%, 768 MiB)
/// - **Unified memory (Apple Silicon)**: reserve max(28%, 6 GiB)
/// - **CPU / mobile**: reserve 20%
pub fn max_layers_for_gpus(gpus: &[discovery::GpuInfo], layer_size_bytes: u64) -> usize {
    if layer_size_bytes == 0 || gpus.is_empty() {
        return usize::MAX;
    }
    gpus.iter()
        .map(|g| {
            let name_lower = g.name.to_lowercase();
            let is_cpu = name_lower.starts_with("cpu");
            let is_unified = name_lower.contains("apple");
            let usable = if is_cpu {
                let reserve = (g.vram_bytes as f64 * 0.20) as u64;
                g.vram_bytes.saturating_sub(reserve)
            } else if is_unified {
                let min_reserve = 6u64 * 1024 * 1024 * 1024;
                let pct_reserve = (g.vram_bytes as f64 * 0.28) as u64;
                let os_reserve = pct_reserve.max(min_reserve);
                g.vram_bytes.saturating_sub(os_reserve)
            } else {
                let min_reserve = 768u64 * 1024 * 1024;
                let pct_reserve = (g.vram_bytes as f64 * 0.05) as u64;
                let reserve = pct_reserve.max(min_reserve);
                g.vram_bytes.saturating_sub(reserve)
            };
            (usable / layer_size_bytes) as usize
        })
        .sum()
}

/// Estimate TFLOPS from a list of GPUs. Falls back to a VRAM-based heuristic
/// when workers report 0 (old binaries).
pub fn estimate_tflops_for_gpus(gpus: &[discovery::GpuInfo]) -> f64 {
    let reported: f64 = gpus.iter().map(|g| g.tflops as f64).sum();
    if reported > 0.0 {
        return reported;
    }
    fn name_contains_ci(name: &str, needle: &str) -> bool {
        name.as_bytes().windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
    }
    gpus.iter()
        .map(|g| {
            let vram_gb = g.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let name = &g.name;
            if name_contains_ci(name, "nvidia")
                || name_contains_ci(name, "geforce")
                || name_contains_ci(name, "rtx")
                || name_contains_ci(name, "gtx")
                || name_contains_ci(name, "tesla")
            {
                vram_gb * 3.0
            } else if name_contains_ci(name, "apple") || name_contains_ci(name, "silicon") {
                vram_gb * 0.4
            } else {
                2.0
            }
        })
        .sum()
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
    let layer_prefix = default::layer_prefix_for_config(&config_json);

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
    let layer_size_on_disk = default::estimate_layer_size(model_path, num_layers, &layer_prefix);
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

    // Quantized models may expand in memory (e.g. FP8: 1 byte → 2 bytes for F16).
    // Use the quantization strategy's VRAM estimator.
    let quant = crate::utils::detect_quantization(&config_path);
    let layer_size_bytes = quant.estimate_layer_vram(layer_size_on_disk, dtype_bytes);
    if layer_size_bytes != layer_size_on_disk && layer_size_on_disk > 0 {
        log::info!(
            "{} model: layer size after dequantization: {} ({}x expansion)",
            quant.name(),
            human_bytes::human_bytes(layer_size_bytes as f64),
            layer_size_bytes / layer_size_on_disk,
        );
    }

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
    let dyn_workers: Vec<&dyn WorkerCapacity> = workers.iter().map(|w| w as &dyn WorkerCapacity).collect();
    let strategy = DefaultStrategy;
    let assignments = strategy.assign_layers(
        &dyn_workers,
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

    // Stream each file using chunked reads (constant 128MB memory, not full file)
    let mut read_buf = vec![0u8; MODEL_DATA_CHUNK_SIZE];
    let mut write_buf = Vec::with_capacity(MODEL_DATA_CHUNK_SIZE + 1024); // reusable write buffer

    for file_path in &files_to_send {
        let filename = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Use filtered index if this is the index file (small, keep in-memory)
        let is_index = filename == "model.safetensors.index.json";
        let small_data = if is_index {
            if let Some(ref data) = filtered_index {
                Some(data.clone())
            } else {
                Some(
                    std::fs::read(file_path)
                        .map_err(|e| anyhow!("failed to read {}: {}", file_path.display(), e))?,
                )
            }
        } else {
            None
        };

        let total_size = if let Some(ref data) = small_data {
            data.len() as u64
        } else {
            std::fs::metadata(file_path)
                .map_err(|e| anyhow!("failed to stat {}: {}", file_path.display(), e))?
                .len()
        };

        let file_start = Instant::now();
        let mut offset: u64 = 0;

        log::info!(
            "[{}] sending {} ({}) ...",
            worker_name,
            &filename,
            human_bytes::human_bytes(total_size as f64)
        );

        // Open file handle for streaming (large files only)
        let mut file_handle = if small_data.is_none() {
            Some(
                std::fs::File::open(file_path)
                    .map_err(|e| anyhow!("failed to open {}: {}", file_path.display(), e))?,
            )
        } else {
            None
        };

        while offset < total_size {
            let to_read = ((total_size - offset) as usize).min(MODEL_DATA_CHUNK_SIZE);
            // Get a slice to the raw chunk data without copying
            let raw_slice: &[u8] = if let Some(ref data) = small_data {
                // Small files (config, tokenizer, index): already in memory
                &data[offset as usize..offset as usize + to_read]
            } else {
                // Large files (safetensors): stream from disk into reusable buffer
                use std::io::Read;
                let fh = file_handle.as_mut().unwrap();
                fh.read_exact(&mut read_buf[..to_read])
                    .map_err(|e| anyhow!("read error at offset {offset}: {e}"))?;
                &read_buf[..to_read]
            };

            // Compress with zstd level 1 (only if it saves space).
            // For large chunks, probe a small sample first to avoid wasting CPU
            // on incompressible model weight data (F16/BF16 pseudo-random).
            let (data, is_compressed) = if raw_slice.len() > 4096 {
                let sample = &raw_slice[..4096.min(raw_slice.len())];
                let sample_compressed = zstd::encode_all(sample, 1)
                    .unwrap_or_else(|_| sample.to_vec());
                if sample_compressed.len() < sample.len() {
                    // Sample compresses — try the full chunk
                    let compressed_data = zstd::encode_all(raw_slice, 1)
                        .unwrap_or_else(|_| raw_slice.to_vec());
                    if compressed_data.len() < raw_slice.len() {
                        (compressed_data, true)
                    } else {
                        (raw_slice.to_vec(), false)
                    }
                } else {
                    // Sample doesn't compress — skip full compression
                    (raw_slice.to_vec(), false)
                }
            } else {
                let compressed_data = zstd::encode_all(raw_slice, 1)
                    .unwrap_or_else(|_| raw_slice.to_vec());
                if compressed_data.len() < raw_slice.len() {
                    (compressed_data, true)
                } else {
                    (raw_slice.to_vec(), false)
                }
            };

            // CRC32 checksum of wire data (after compression)
            let checksum = crc32fast::hash(&data);

            let msg = Message::ModelDataChunk {
                filename: filename.clone(),
                offset,
                total_size,
                compressed: is_compressed,
                checksum,
                data,
            };
            msg.to_writer_buf(stream, &mut write_buf).await?;
            offset += to_read as u64;

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
                compressed,
                checksum,
                data,
            } => {
                // Verify CRC32 checksum before writing
                let actual_crc = crc32fast::hash(&data);
                if actual_crc != checksum {
                    return Err(anyhow!(
                        "checksum mismatch for {} at offset {}: expected {:#x}, got {:#x}",
                        filename, offset, checksum, actual_crc
                    ));
                }

                // Decompress if compressed
                let data = if compressed {
                    zstd::decode_all(data.as_slice())
                        .map_err(|e| anyhow!("zstd decompress failed for {} at offset {}: {}", filename, offset, e))?
                } else {
                    data
                };
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

    // ── Strategy trait tests ─────────────────────────────

    #[test]
    fn test_default_strategy_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DefaultStrategy>();
    }

    #[test]
    fn test_strategy_trait_object() {
        let strategy: Box<dyn Strategy> = Box::new(DefaultStrategy);
        let _ = strategy; // just verify it compiles
    }

    #[test]
    fn test_max_layers_for_gpus_empty() {
        assert_eq!(max_layers_for_gpus(&[], 1000), usize::MAX);
    }

    #[test]
    fn test_max_layers_for_gpus_zero_layer_size() {
        let gpus = vec![discovery::GpuInfo { name: "GPU".into(), vram_bytes: 1024, tflops: 1.0 }];
        assert_eq!(max_layers_for_gpus(&gpus, 0), usize::MAX);
    }

    #[test]
    fn test_estimate_tflops_reported_vs_fallback() {
        let gpus_reported = vec![discovery::GpuInfo { name: "GPU".into(), vram_bytes: 1024, tflops: 42.0 }];
        assert!((estimate_tflops_for_gpus(&gpus_reported) - 42.0).abs() < 0.01);

        let gpus_zero = vec![discovery::GpuInfo { name: "NVIDIA RTX".into(), vram_bytes: 24 * 1024 * 1024 * 1024, tflops: 0.0 }];
        assert!(estimate_tflops_for_gpus(&gpus_zero) > 0.0); // should use fallback
    }

    // ── has_valid_model_cache tests ─────────────────────

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
        ).unwrap();
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
        ).unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();

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
        ).unwrap();

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

    #[test]
    fn has_valid_model_cache_empty_layers_with_config() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let layers: Vec<String> = vec![];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_single_safetensors_empty_layers() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        fs::write(tmp.path().join("model.safetensors"), "data").unwrap();
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
        ).unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
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
        ).unwrap();
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
        ).unwrap();
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
        ).unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        let layers = vec!["model.layers.0".to_string()];
        assert!(has_valid_model_cache(tmp.path(), &layers));
    }

    #[test]
    fn has_valid_model_cache_sharded_layer_prefix_disambiguation() {
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
        ).unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        let layers = vec!["model.layers.1".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }
}
