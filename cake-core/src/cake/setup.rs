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

/// Maximum chunk size for model data transfer (128 MB).
const MODEL_DATA_CHUNK_SIZE: usize = 128 * 1024 * 1024;

// ── Layer assignment ────────────────────────────────────────────────────────

/// Compute layer assignments proportional to each worker's estimated TFLOPS,
/// accounting for the master's own compute so it retains its fair share of layers.
///
/// Returns a vec of `(worker_index, layer_names)`.
/// Workers are sorted by TFLOPS descending, and layers are assigned as
/// contiguous ranges starting from layer 0. Unassigned layers remain on master.
pub fn compute_layer_assignments(
    workers: &[DiscoveredWorker],
    num_layers: usize,
    master_tflops: f64,
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
                .map(|l| format!("model.layers.{l}"))
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

        let count = if pos == indices.len() - 1 {
            remaining_layers
        } else {
            let worker_tflops = workers[worker_idx].total_tflops();
            let proportional =
                (worker_tflops / remaining_tflops * remaining_layers as f64).round() as usize;
            proportional.max(1).min(remaining_layers)
        };

        let layers: Vec<String> = (offset..offset + count)
            .map(|l| format!("model.layers.{l}"))
            .collect();

        assignments.push((worker_idx, layers));
        offset += count;
        remaining_layers -= count;
        remaining_tflops -= workers[worker_idx].total_tflops();
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

    log::info!("model has {} transformer layers", num_layers);

    // Discover workers
    let workers = discovery::discover_workers(cluster_key, discovery_timeout).await?;
    if workers.is_empty() {
        log::warn!("no workers discovered — all layers will be loaded locally");
        return Ok(Topology::new());
    }

    // Detect master GPU for proportional split
    let master_gpus = discovery::detect_gpus();
    let master_tflops: f64 = master_gpus.iter().map(|g| g.tflops as f64).sum();

    // Compute assignments based on TFLOPS
    let assignments = compute_layer_assignments(&workers, num_layers, master_tflops);

    log::info!("layer assignments:");
    for (worker_idx, layers) in &assignments {
        let w = &workers[*worker_idx];
        let range = if layers.is_empty() {
            "(none)".to_string()
        } else {
            format!("{} — {}", layers.first().unwrap(), layers.last().unwrap())
        };
        log::info!(
            "  {} ({}, {:.1} TFLOPS) → {} layers [{}]",
            &w.name,
            human_bytes::human_bytes(w.total_vram() as f64),
            w.total_tflops(),
            layers.len(),
            range
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

        handles.push(tokio::spawn(async move {
            log::info!(
                "connecting to worker '{}' at {} ...",
                &worker.name,
                &worker.host
            );

            let mut stream = TcpStream::connect(&worker.host)
                .await
                .map_err(|e| anyhow!("can't connect to {}: {}", &worker.host, e))?;

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
                push_model_data(&mut stream, &model_path, &layers, &worker.name).await?;
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
) -> Result<()> {
    let overall_start = Instant::now();
    let mut overall_bytes: u64 = 0;

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
                    "[{}]   {}/{} ({:.1}%) — {}/s — ETA {:.0}s",
                    worker_name,
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

/// Check whether a cache directory contains a valid model.
fn has_valid_model_cache(cache_dir: &Path) -> bool {
    if !cache_dir.join("config.json").exists() {
        return false;
    }
    // Single safetensors file
    if cache_dir.join("model.safetensors").exists() {
        return true;
    }
    // Sharded model: need index + at least one shard
    let index_path = cache_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        if let Ok(data) = std::fs::read_to_string(&index_path) {
            if let Ok(index) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    // Check that at least one referenced shard file exists
                    return weight_map.values().any(|v| {
                        v.as_str()
                            .is_some_and(|f| cache_dir.join(f).exists())
                    });
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
pub async fn worker_setup(
    worker_name: &str,
    cluster_key: &str,
    bind_address: &str,
    model_cache_dir: &Path,
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

    // Accept one setup connection from master
    let (mut stream, client_addr) = listener.accept().await?;
    log::info!("[{}] master connected", client_addr);

    // Authenticate
    auth::authenticate_as_worker(&mut stream, cluster_key).await?;
    log::info!("[{}] authenticated", client_addr);

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

    // Check if we already have a valid model data cache.
    // Needs config.json + either model.safetensors or the index + at least one shard.
    let needs_data = !has_valid_model_cache(&cache_dir);

    let ack = Message::LayerAssignmentAck { needs_data };
    ack.to_writer(&mut stream).await?;

    if needs_data {
        receive_model_data(&mut stream, &cache_dir).await?;
    } else {
        log::info!("using cached model data from {}", cache_dir.display());
    }

    // Signal ready
    Message::WorkerReady.to_writer(&mut stream).await?;
    log::info!("setup complete, ready for inference");

    // Drop the setup connection (stream goes out of scope)
    // The listener is returned for reuse
    Ok((layers, cache_dir, listener))
}

/// Receive model data from master and write to the cache directory.
async fn receive_model_data(stream: &mut TcpStream, cache_dir: &Path) -> Result<()> {
    let overall_start = Instant::now();
    let mut overall_bytes: u64 = 0;
    let mut current_file: Option<(String, std::fs::File, Instant, u64)> = None;

    log::info!("receiving model data...");

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
                        }
                        let path = cache_dir.join(&filename);
                        let f = std::fs::File::create(&path)?;
                        current_file = Some((filename.clone(), f, Instant::now(), total_size));
                        &mut current_file.as_mut().unwrap().1
                    }
                } else {
                    let path = cache_dir.join(&filename);
                    let f = std::fs::File::create(&path)?;
                    current_file = Some((filename.clone(), f, Instant::now(), total_size));
                    &mut current_file.as_mut().unwrap().1
                };

                file.write_all(&data)?;
                overall_bytes += data.len() as u64;

                // Progress for large files
                let written = offset + data.len() as u64;
                if total_size > MODEL_DATA_CHUNK_SIZE as u64 && written < total_size {
                    if let Some((_, _, ref start, _)) = current_file {
                        let elapsed = start.elapsed().as_secs_f64();
                        let speed = written as f64 / elapsed;
                        let pct = (written as f64 / total_size as f64) * 100.0;
                        let remaining = total_size - written;
                        let eta_secs = if speed > 0.0 {
                            remaining as f64 / speed
                        } else {
                            0.0
                        };
                        log::info!(
                            "  {}/{} ({:.1}%) — {}/s — ETA {:.0}s",
                            human_bytes::human_bytes(written as f64),
                            human_bytes::human_bytes(total_size as f64),
                            pct,
                            human_bytes::human_bytes(speed),
                            eta_secs
                        );
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
