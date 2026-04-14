//! Sharding, topology, protocol, client, worker, and discovery code.
//!
//! This module consolidates all distributed inference infrastructure:
//! - [`Strategy`] trait and [`DefaultStrategy`] implementation
//! - [`WorkerCapacity`] trait for abstracting worker resources
//! - Topology management, wire protocol, client/worker networking
//! - Zero-config setup (master and worker)

#[cfg(feature = "master")]
pub mod api;
pub mod auth;
pub(crate) mod client;
pub mod default;
pub mod discovery;
#[cfg(feature = "master")]
pub mod master;
pub(crate) mod proto;
pub mod topology;
pub(crate) mod worker;

pub use client::*;
pub use default::DefaultStrategy;
pub use proto::*;
pub use topology::*;
pub use worker::*;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::net::{TcpListener, TcpStream};

pub const PROTOCOL_VERSION: u32 = 1;
pub const CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyClass {
    MobileOnly,
    MixedDesktopMobile,
    DesktopHeavy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerLossAction {
    Abort,
    ReassignToMaster,
    Redistribute,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerLossPlan {
    pub action: WorkerLossAction,
    pub reason: String,
}

/// A sharding strategy decides how to distribute transformer layers across workers.
pub trait Strategy: Send + Sync {
    /// Assign layers to workers.
    ///
    /// Returns a vec of `(worker_index, layer_names)`.
    /// Workers are sorted by TFLOPS descending, and layers are assigned as
    /// contiguous ranges starting from layer 0. Unassigned layers remain on master.
    #[allow(clippy::too_many_arguments)]
    fn assign_layers(
        &self,
        workers: &[&dyn WorkerCapacity],
        num_layers: usize,
        master_tflops: f64,
        layer_size_bytes: u64,
        master_max_layers: usize,
        layer_prefix: &str,
        min_layers_per_worker: usize,
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

const DEFAULT_MOBILE_LAYER_BUDGET_MB: u64 = 1536;
const DEFAULT_MOBILE_RESERVE_PCT: u64 = 80;

fn mobile_layer_budget_bytes() -> u64 {
    std::env::var("CAKE_MOBILE_LAYER_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MOBILE_LAYER_BUDGET_MB)
        * 1024
        * 1024
}

fn mobile_reserve_pct() -> f64 {
    std::env::var("CAKE_MOBILE_RESERVE_PCT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MOBILE_RESERVE_PCT) as f64
        / 100.0
}

/// Compute max layers from a list of GPUs, applying per-device VRAM reserves.
///
/// - **Dedicated VRAM (CUDA)**: reserve max(5%, 768 MiB)
/// - **Apple desktop unified memory**: reserve max(20%, 2 GiB)
/// - **Apple mobile unified memory (iPhone/iPad)**: reserve configurable %
///   (default 80%, override via `CAKE_MOBILE_RESERVE_PCT`), then cap the
///   worker layer budget (default 1.5 GiB, override via
///   `CAKE_MOBILE_LAYER_BUDGET_MB`) to stay under iOS per-process jetsam limits
/// - **CPU / mobile**: reserve 20%
pub fn max_layers_for_gpus(gpus: &[discovery::GpuInfo], layer_size_bytes: u64) -> usize {
    if layer_size_bytes == 0 || gpus.is_empty() {
        return usize::MAX;
    }
    let mobile_cap = mobile_layer_budget_bytes();
    let mobile_reserve = mobile_reserve_pct();
    gpus.iter()
        .map(|g| {
            let name_lower = g.name.to_lowercase();
            let is_cpu = name_lower.starts_with("cpu");
            let is_apple_mobile =
                name_lower.starts_with("iphone") || name_lower.starts_with("ipad");
            let is_apple_desktop = name_lower.contains("apple");
            let usable = if is_cpu {
                let reserve = (g.vram_bytes as f64 * 0.20) as u64;
                g.vram_bytes.saturating_sub(reserve)
            } else if is_apple_mobile {
                let reserve = (g.vram_bytes as f64 * mobile_reserve) as u64;
                g.vram_bytes
                    .saturating_sub(reserve)
                    .min(mobile_cap)
            } else if is_apple_desktop {
                let min_reserve = 2u64 * 1024 * 1024 * 1024;
                let pct_reserve = (g.vram_bytes as f64 * 0.20) as u64;
                let reserve = pct_reserve.max(min_reserve);
                g.vram_bytes.saturating_sub(reserve)
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
        name.as_bytes()
            .windows(needle.len())
            .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
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

pub(crate) fn layer_plan_hash(layers: &[String]) -> String {
    use sha2::{Digest, Sha256};

    let mut sorted_layers = layers.to_vec();
    sorted_layers.sort();

    let mut hasher = Sha256::new();
    for layer in &sorted_layers {
        hasher.update(layer.as_bytes());
    }

    let result = hasher.finalize();
    hex::encode(&result[..4])
}

pub fn total_system_memory_bytes() -> u64 {
    #[cfg(unix)]
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if pages > 0 && page_size > 0 {
            return (pages as u64).saturating_mul(page_size as u64);
        }
    }

    0
}

pub fn process_resident_memory_bytes() -> u64 {
    memory_stats::memory_stats()
        .map(|stats| stats.physical_mem as u64)
        .unwrap_or(0)
}

pub fn process_peak_resident_memory_bytes() -> u64 {
    #[cfg(unix)]
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 && usage.ru_maxrss > 0 {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                return usage.ru_maxrss as u64;
            }

            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                return (usage.ru_maxrss as u64).saturating_mul(1024);
            }
        }
    }

    0
}

fn uniquify_worker_names(workers: &mut [discovery::DiscoveredWorker]) {
    use std::collections::HashMap;

    let mut counts: HashMap<String, usize> = HashMap::new();
    for worker in workers.iter() {
        let base = worker.name.trim();
        let base = if base.is_empty() {
            format!("worker@{}", worker.host)
        } else {
            base.to_string()
        };
        *counts.entry(base).or_insert(0) += 1;
    }

    for worker in workers.iter_mut() {
        let base = worker.name.trim();
        let base = if base.is_empty() {
            format!("worker@{}", worker.host)
        } else {
            base.to_string()
        };

        if counts.get(&base).copied().unwrap_or(0) > 1 {
            let unique = format!("{} ({})", base, worker.host);
            log::warn!(
                "duplicate worker name '{}' detected; using '{}' as the topology key",
                base,
                unique
            );
            worker.name = unique;
        } else {
            worker.name = base;
        }
    }
}

fn append_safetensors_fingerprint(
    hasher: &mut sha2::Sha256,
    path: &Path,
    sample_bytes: usize,
) -> Result<()> {
    use sha2::Digest;
    use std::io::{Read, Seek, SeekFrom};

    let metadata = std::fs::metadata(path)?;
    hasher.update(
        path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .as_bytes(),
    );
    hasher.update(metadata.len().to_le_bytes());

    let mut file = std::fs::File::open(path)?;
    let mut head = vec![0u8; sample_bytes.min(metadata.len() as usize)];
    file.read_exact(&mut head)?;
    hasher.update(&head);

    if metadata.len() > sample_bytes as u64 {
        let tail_len = sample_bytes.min(metadata.len() as usize);
        file.seek(SeekFrom::End(-(tail_len as i64)))?;
        let mut tail = vec![0u8; tail_len];
        file.read_exact(&mut tail)?;
        hasher.update(&tail);
    }

    Ok(())
}

pub(crate) fn compute_model_hash(model_path: &Path, config_data: &str) -> String {
    use sha2::{Digest, Sha256};

    const SAMPLE_BYTES: usize = 64 * 1024;

    let mut hasher = Sha256::new();
    hasher.update(config_data.as_bytes());

    let index_path = model_path.join("model.safetensors.index.json");
    if let Ok(index_data) = std::fs::read_to_string(&index_path) {
        hasher.update(index_data.as_bytes());
    }

    if let Ok(entries) = std::fs::read_dir(model_path) {
        let mut safetensors_paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        safetensors_paths.sort();

        for path in safetensors_paths {
            if let Err(e) = append_safetensors_fingerprint(&mut hasher, &path, SAMPLE_BYTES) {
                log::warn!("failed to fingerprint {}: {}", path.display(), e);
            }
        }
    }

    let result = hasher.finalize();
    hex::encode(&result[..4])
}

pub fn classify_topology(topology: &Topology, master_memory_bytes: u64) -> TopologyClass {
    let has_ios_worker = topology
        .iter()
        .any(|(_, node)| node.os.eq_ignore_ascii_case("ios"));
    if master_memory_bytes > 32 * 1024 * 1024 * 1024 && has_ios_worker {
        return TopologyClass::MixedDesktopMobile;
    }

    let all_desktop_heavy = !topology.is_empty()
        && topology.iter().all(|(_, node)| {
            node.os.eq_ignore_ascii_case("macos") && node.vram_bytes > 16 * 1024 * 1024 * 1024
        });
    if all_desktop_heavy {
        return TopologyClass::DesktopHeavy;
    }

    TopologyClass::MobileOnly
}

pub fn plan_worker_loss(
    topology_class: TopologyClass,
    lost_worker: &Node,
    lost_layers: usize,
    layer_size_bytes: u64,
    master_free_bytes: u64,
    survivor_free_bytes: u64,
) -> WorkerLossPlan {
    let needed_bytes = (lost_layers as u64).saturating_mul(layer_size_bytes);
    match topology_class {
        TopologyClass::MobileOnly => WorkerLossPlan {
            action: WorkerLossAction::Abort,
            reason: format!(
                "worker {} lost, mobile pool cannot absorb {} layer(s) (need {}, have {} free)",
                lost_worker.host,
                lost_layers,
                human_bytes::human_bytes(needed_bytes as f64),
                human_bytes::human_bytes(master_free_bytes as f64),
            ),
        },
        TopologyClass::MixedDesktopMobile => {
            if lost_worker.os.eq_ignore_ascii_case("ios") {
                if master_free_bytes >= needed_bytes {
                    WorkerLossPlan {
                        action: WorkerLossAction::ReassignToMaster,
                        reason: format!(
                            "worker {} lost, reassigning {} layer(s) to master (need {}, have {} free)",
                            lost_worker.host,
                            lost_layers,
                            human_bytes::human_bytes(needed_bytes as f64),
                            human_bytes::human_bytes(master_free_bytes as f64),
                        ),
                    }
                } else {
                    WorkerLossPlan {
                        action: WorkerLossAction::Abort,
                        reason: format!(
                            "worker {} lost, master cannot absorb {} layer(s) (need {}, have {} free)",
                            lost_worker.host,
                            lost_layers,
                            human_bytes::human_bytes(needed_bytes as f64),
                            human_bytes::human_bytes(master_free_bytes as f64),
                        ),
                    }
                }
            } else {
                WorkerLossPlan {
                    action: WorkerLossAction::Abort,
                    reason: format!(
                        "desktop worker {} lost in mixed topology; aborting instead of absorbing desktop-scale layer set",
                        lost_worker.host
                    ),
                }
            }
        }
        TopologyClass::DesktopHeavy => {
            if survivor_free_bytes >= needed_bytes {
                WorkerLossPlan {
                    action: WorkerLossAction::Redistribute,
                    reason: format!(
                        "worker {} lost, redistributing {} layer(s) across desktop survivors (need {}, survivors report {} free)",
                        lost_worker.host,
                        lost_layers,
                        human_bytes::human_bytes(needed_bytes as f64),
                        human_bytes::human_bytes(survivor_free_bytes as f64),
                    ),
                }
            } else {
                WorkerLossPlan {
                    action: WorkerLossAction::Abort,
                    reason: format!(
                        "worker {} lost, desktop survivors cannot absorb {} layer(s) (need {}, survivors report {} free)",
                        lost_worker.host,
                        lost_layers,
                        human_bytes::human_bytes(needed_bytes as f64),
                        human_bytes::human_bytes(survivor_free_bytes as f64),
                    ),
                }
            }
        }
    }
}

fn compatibility_rejection_reason(
    protocol_version: u32,
    build_hash: &str,
    cache_schema_version: u32,
) -> Option<String> {
    if protocol_version != PROTOCOL_VERSION {
        return Some(format!(
            "protocol_version mismatch: worker={} master={}",
            PROTOCOL_VERSION, protocol_version
        ));
    }
    if build_hash != crate::BUILD_HASH {
        return Some(format!(
            "build_hash mismatch: worker={} master={}",
            crate::BUILD_HASH,
            build_hash
        ));
    }
    if cache_schema_version != CACHE_SCHEMA_VERSION {
        return Some(format!(
            "cache_schema_version mismatch: worker={} master={}",
            CACHE_SCHEMA_VERSION, cache_schema_version
        ));
    }
    None
}

fn cache_refresh_reason(
    current_model_hash: &str,
    requested_model_hash: &str,
    current_layer_hash: &str,
    requested_layer_hash: &str,
) -> Option<String> {
    if current_model_hash != requested_model_hash {
        return Some(format!(
            "model_hash mismatch: current={} requested={}",
            current_model_hash, requested_model_hash
        ));
    }
    if current_layer_hash != requested_layer_hash {
        return Some(format!(
            "layer_plan_hash mismatch: current={} requested={}",
            current_layer_hash, requested_layer_hash
        ));
    }
    None
}

// ── Master setup ────────────────────────────────────────────────────────────

/// Run the full zero-config master setup.
/// Create a `DiscoveredWorker` with conservative defaults for workers that
/// can't be probed (old binary, unreachable, etc.).
fn manual_worker(addr: &str) -> discovery::DiscoveredWorker {
    let host_part = addr.split(':').next().unwrap_or(addr);
    let port: u16 = addr
        .split(':')
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(10128);
    discovery::DiscoveredWorker {
        name: format!("worker-{}", host_part),
        host: addr.to_string(),
        port,
        gpus: vec![discovery::GpuInfo {
            name: "manual".to_string(),
            vram_bytes: 4 * 1024 * 1024 * 1024,
            tflops: 2.0,
        }],
        backend: "metal".to_string(),
        hostname: host_part.to_string(),
        os: "ios".to_string(),
    }
}

///
/// Probe a manually-specified worker over the setup connection.
///
/// The returned stream stays authenticated and must be reused for the later
/// layer-assignment / model-push flow so the worker's one setup accept() is not
/// consumed by a separate probe connection.
async fn probe_manual_worker(
    addr: &str,
    cluster_key: &str,
) -> Result<(discovery::DiscoveredWorker, TcpStream)> {
    let host_part = addr.split(':').next().unwrap_or(addr);
    let port: u16 = addr
        .split(':')
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(10128);

    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| anyhow!("manual worker probe connect failed for {}: {}", addr, e))?;
    let _ = stream.set_nodelay(true);

    auth::authenticate_as_master(&mut stream, cluster_key)
        .await
        .map_err(|e| anyhow!("manual worker probe auth failed for {}: {}", addr, e))?;

    Message::DeviceInfoRequest.to_writer(&mut stream).await?;
    let (_, reply) = Message::from_reader(&mut stream).await?;
    let worker = match reply {
        Message::DeviceInfoResponse {
            worker_name,
            gpus,
            backend,
            hostname,
            os,
        } => discovery::DiscoveredWorker {
            name: worker_name,
            host: addr.to_string(),
            port,
            gpus,
            backend,
            hostname,
            os,
        },
        other => {
            return Err(anyhow!(
                "manual worker probe expected DeviceInfoResponse from {}, got {:?}",
                addr,
                other
            ));
        }
    };

    if worker.gpus.is_empty() {
        return Err(anyhow!(
            "manual worker probe for {} returned no GPU/device info",
            addr
        ));
    }

    log::info!(
        "manual worker '{}' at {} reports {} ({:.1} TFLOPS total)",
        worker.name,
        addr,
        human_bytes::human_bytes(worker.total_vram() as f64),
        worker.total_tflops(),
    );

    if worker.hostname != host_part {
        log::debug!(
            "manual worker {} resolved hostname '{}' (address host '{}')",
            addr,
            worker.hostname,
            host_part
        );
    }

    Ok((worker, stream))
}

/// Discovers workers via mDNS, computes layer assignments based on VRAM,
/// connects to each worker with mutual authentication, pushes model data
/// as needed, and returns a `Topology` ready for normal inference.
pub async fn master_setup(
    cluster_key: &str,
    model_path: &Path,
    discovery_timeout: Duration,
    min_workers: usize,
    min_layers_per_worker: usize,
    master_max_layers_override: Option<usize>,
    manual_workers: &[String],
) -> Result<Topology> {
    // Read config.json and compute a fingerprint for cache keying
    let config_path = model_path.join("config.json");
    let config_data = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("failed to read {}: {}", config_path.display(), e))?;
    let model_hash = compute_model_hash(model_path, &config_data);
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
        .ok_or_else(|| anyhow!("num_hidden_layers not found in config.json"))?
        as usize;

    if num_layers == 0 {
        log::warn!("model has no transformer layers — skipping worker setup");
        return Ok(Topology::new());
    }

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

    // Probe manually-specified workers FIRST so they count toward min_workers
    // and use real device capacity instead of synthetic defaults.
    let mut workers: Vec<discovery::DiscoveredWorker> = Vec::new();
    let mut manual_streams: HashMap<String, TcpStream> = HashMap::new();
    for addr in manual_workers {
        match probe_manual_worker(addr, cluster_key).await {
            Ok((worker, stream)) => {
                log::info!(
                    "probed manual worker '{}' at {} ({}, {:.1} TFLOPS)",
                    &worker.name,
                    addr,
                    human_bytes::human_bytes(worker.total_vram() as f64),
                    worker.total_tflops(),
                );
                manual_streams.insert(worker.host.clone(), stream);
                workers.push(worker);
            }
            Err(e) => {
                log::warn!(
                    "probe failed for {}, using conservative defaults: {}",
                    addr,
                    e
                );
                let worker = manual_worker(addr);
                log::info!(
                    "adding manual worker '{}' at {} (4 GiB, 2.0 TFLOPS — conservative defaults)",
                    &worker.name,
                    addr,
                );
                workers.push(worker);
            }
        }
    }

    // Discover additional workers via UDP. Skip entirely when:
    // - manual workers were specified and min_workers is satisfied (or default 0)
    // - discovery_timeout is 0
    // When --workers is set with default min_workers=0, the manual workers ARE
    // the full set — no UDP needed unless the caller explicitly asks for more.
    let have_manual = !manual_workers.is_empty();
    let need_more = if have_manual {
        min_workers > 0 && manual_workers.len() < min_workers
    } else {
        true // no manual workers → always try UDP
    };
    if need_more && discovery_timeout > Duration::ZERO {
        let adjusted_min = if min_workers > workers.len() {
            min_workers - workers.len()
        } else {
            0
        };
        let max_discovery_attempts = if workers.is_empty() { 3 } else { 1 };
        for attempt in 1..=max_discovery_attempts {
            let mut discovered =
                discovery::discover_workers(cluster_key, discovery_timeout, adjusted_min).await?;
            // Deduplicate against manual workers
            discovered.retain(|d| !workers.iter().any(|w| w.host == d.host));
            workers.extend(discovered);
            if !workers.is_empty() {
                break;
            }
            if attempt < max_discovery_attempts {
                log::info!(
                    "no workers found (attempt {}/{}), retrying in 5s...",
                    attempt,
                    max_discovery_attempts,
                );
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    } else if !need_more {
        log::info!(
            "manual workers satisfy min_workers ({}), skipping UDP discovery",
            min_workers
        );
    }

    uniquify_worker_names(&mut workers);

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
    let detected_master_max_layers = if layer_size_bytes > 0 && !master_gpus.is_empty() {
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
    let master_max_layers = master_max_layers_override.unwrap_or(detected_master_max_layers);
    if let Some(override_layers) = master_max_layers_override {
        log::info!(
            "master layer cap override enabled: {} layer(s) (auto-detected: {})",
            override_layers,
            detected_master_max_layers,
        );
    }

    // Compute assignments based on TFLOPS, capped by per-GPU VRAM
    let dyn_workers: Vec<&dyn WorkerCapacity> =
        workers.iter().map(|w| w as &dyn WorkerCapacity).collect();
    let strategy = DefaultStrategy;
    let assignments = strategy.assign_layers(
        &dyn_workers,
        num_layers,
        master_tflops,
        layer_size_bytes,
        master_max_layers,
        &layer_prefix,
        min_layers_per_worker,
    );

    // Summarise layer assignments and estimate per-node weight loads
    let total_assigned: usize = assignments.iter().map(|(_, l)| l.len()).sum();
    let master_layers = num_layers - total_assigned;
    if master_max_layers < usize::MAX && master_layers > master_max_layers {
        anyhow::bail!(
            "cluster cannot fit model: master needs {} local layer(s) after sharding but cap is {}",
            master_layers,
            master_max_layers,
        );
    }
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

    // Connect to all workers concurrently: assign layers, push data, and reuse
    // the setup stream for manual workers already probed above.
    let mut handles = Vec::new();

    for (worker_idx, layers) in &assignments {
        let worker = workers[*worker_idx].clone();
        if layers.is_empty() {
            continue;
        }

        let preconnected_stream = manual_streams.remove(&worker.host);
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
            let mut stream = if let Some(stream) = preconnected_stream {
                log::info!(
                    "reusing probed setup connection for worker '{}' at {} ...",
                    &worker.name,
                    &worker.host
                );
                stream
            } else {
                log::info!(
                    "connecting to worker '{}' at {} ...",
                    &worker.name,
                    &worker.host
                );

                // Retry TCP connection with backoff (iOS workers may need time
                // after UDP discovery before accept() is ready).
                let mut stream = {
                    let max_retries = 30;
                    let mut last_err = None;
                    let mut connected = None;
                    for attempt in 1..=max_retries {
                        match TcpStream::connect(&worker.host).await {
                            Ok(s) => {
                                connected = Some(s);
                                break;
                            }
                            Err(e) => {
                                log::warn!(
                                    "  connection attempt {}/{} to {} failed: {} — retrying in 1s",
                                    attempt,
                                    max_retries,
                                    &worker.host,
                                    e
                                );
                                last_err = Some(e);
                                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                            }
                        }
                    }
                    connected.ok_or_else(|| {
                        anyhow!(
                            "can't connect to {} after {} attempts: {}",
                            &worker.host,
                            max_retries,
                            last_err.unwrap()
                        )
                    })?
                };
                let _ = stream.set_nodelay(true);

                // Mutual authentication
                auth::authenticate_as_master(&mut stream, &cluster_key).await?;
                log::info!("[{}] authenticated", &worker.name);
                stream
            };

            // Compatibility check: verify protocol, build hash, and cache schema
            // match before proceeding with layer assignment.
            let lph = layer_plan_hash(&layers);
            let compat_msg = Message::CompatibilityCheck {
                protocol_version: PROTOCOL_VERSION,
                build_hash: crate::BUILD_HASH.to_string(),
                cache_schema_version: CACHE_SCHEMA_VERSION,
                model_hash: model_hash.clone(),
                layer_plan_hash: lph,
            };
            compat_msg.to_writer(&mut stream).await?;

            // Timeout guards against stale workers that drop the connection
            // before replying.
            let compat_timeout = std::time::Duration::from_secs(10);
            match tokio::time::timeout(compat_timeout, Message::from_reader(&mut stream)).await {
                Ok(Ok((_, Message::CompatibilityResult { accepted, reason }))) => {
                    if !accepted {
                        return Err(anyhow!(
                            "[{}] compatibility check rejected: {}",
                            &worker.name,
                            reason.unwrap_or_else(|| "unknown reason".to_string())
                        ));
                    }
                    log::info!("[{}] compatibility check passed", &worker.name);
                }
                Ok(Ok((_, other))) => {
                    return Err(anyhow!(
                        "[{}] expected CompatibilityResult, got {:?}",
                        &worker.name,
                        other
                    ));
                }
                Ok(Err(e)) => {
                    return Err(anyhow!(
                        "[{}] compatibility check read error (worker may have old binary): {}",
                        &worker.name,
                        e
                    ));
                }
                Err(_) => {
                    return Err(anyhow!(
                        "[{}] compatibility check timed out after {}s — \
                         worker likely has an old binary that does not \
                         understand CompatibilityCheck. Redeploy with \
                         scripts/deploy-ios.sh and restart the app.",
                        &worker.name,
                        compat_timeout.as_secs()
                    ));
                }
            }

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
                    ));
                }
            };

            if needs_data {
                push_model_data(&mut stream, &model_path, &layers, &worker.name, &model_name)
                    .await?;
            } else {
                log::info!("[{}] worker has model data cached", &worker.name);
            }

            // Wait for WorkerReady
            let (_, ready) = Message::from_reader(&mut stream).await?;
            let (current_rss_bytes, peak_rss_bytes) = match ready {
                Message::WorkerReady {
                    current_rss_bytes,
                    peak_rss_bytes,
                } => (current_rss_bytes, peak_rss_bytes),
                other => {
                    return Err(anyhow!(
                        "[{}] expected WorkerReady, got {:?}",
                        &worker.name,
                        other
                    ));
                }
            };
            log::info!(
                "[{}] worker ready — rss={} peak={}",
                &worker.name,
                human_bytes::human_bytes(current_rss_bytes as f64),
                human_bytes::human_bytes(peak_rss_bytes as f64),
            );

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

    log::info!("[{}] pushing {} [{}]", worker_name, model_name, layer_range);

    // Always send config.json and tokenizer.json
    let mut files_to_send: Vec<PathBuf> = vec![
        model_path.join("config.json"),
        model_path.join("tokenizer.json"),
    ];

    // Determine which safetensors data to push.
    // For sharded models, extract ONLY the assigned layers' tensors into a
    // minimal safetensors file. This reduces push size dramatically:
    // e.g. 14B model: 464 MiB (3 layers) vs 5.35 GiB (full shard).
    let index_path = model_path.join("model.safetensors.index.json");
    let mut extracted_safetensors: Option<Vec<u8>> = None;

    if index_path.exists() {
        let index_data = std::fs::read(&index_path)?;
        let index_json: serde_json::Value = serde_json::from_slice(&index_data)?;
        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("no weight_map in model.safetensors.index.json"))?;

        // Collect (tensor_name, shard_filename) for assigned layers
        let mut needed: Vec<(String, String)> = Vec::new();
        for (tensor_name, shard_file) in weight_map {
            let is_needed = layers
                .iter()
                .any(|layer| tensor_name.starts_with(&format!("{}.", layer)));
            if is_needed {
                if let Some(filename) = shard_file.as_str() {
                    needed.push((tensor_name.clone(), filename.to_string()));
                }
            }
        }

        // Extract tensors from source shards into a minimal safetensors blob
        let extracted = extract_layer_tensors(model_path, &needed)
            .map_err(|e| anyhow!("failed to extract layer tensors: {}", e))?;

        log::info!(
            "[{}] extracted {} tensors ({}) for {} layer(s) from sharded model",
            worker_name,
            needed.len(),
            human_bytes::human_bytes(extracted.len() as f64),
            layers.len(),
        );

        extracted_safetensors = Some(extracted);
    } else {
        // Single safetensors file — extract only assigned layers' tensors.
        // Without this, the full file (e.g. 4 GiB for 7B-4bit) is pushed
        // even when a worker only needs 2 layers (~280 MiB).
        let single = model_path.join("model.safetensors");
        if single.exists() {
            if layers.is_empty() {
                // No specific layers assigned — push entire file
                files_to_send.push(single);
            } else {
                // Read safetensors header to enumerate tensor names
                let header: serde_json::Value = {
                    use std::io::Read;
                    let mut file = std::fs::File::open(&single)
                        .map_err(|e| anyhow!("can't open {}: {}", single.display(), e))?;
                    let mut len_buf = [0u8; 8];
                    file.read_exact(&mut len_buf)?;
                    let header_len = u64::from_le_bytes(len_buf) as usize;
                    let mut header_buf = vec![0u8; header_len];
                    file.read_exact(&mut header_buf)?;
                    serde_json::from_slice(&header_buf)?
                };

                // Filter to only tensors belonging to assigned layers
                let mut needed: Vec<(String, String)> = Vec::new();
                if let Some(obj) = header.as_object() {
                    for (tensor_name, _) in obj {
                        if tensor_name.starts_with("__") {
                            continue; // skip __metadata__
                        }
                        let is_needed = layers
                            .iter()
                            .any(|layer| tensor_name.starts_with(&format!("{}.", layer)));
                        if is_needed {
                            needed.push((
                                tensor_name.clone(),
                                "model.safetensors".to_string(),
                            ));
                        }
                    }
                }

                if needed.is_empty() {
                    // No matching tensors — push entire file as fallback
                    log::warn!(
                        "[{}] no tensors matched assigned layers in single-file model, pushing full file",
                        worker_name,
                    );
                    files_to_send.push(single);
                } else {
                    let extracted = extract_layer_tensors(model_path, &needed)
                        .map_err(|e| {
                            anyhow!("failed to extract layer tensors from single file: {}", e)
                        })?;

                    log::info!(
                        "[{}] extracted {} tensors ({}) for {} layer(s) from single-file model",
                        worker_name,
                        needed.len(),
                        human_bytes::human_bytes(extracted.len() as f64),
                        layers.len(),
                    );

                    extracted_safetensors = Some(extracted);
                }
            }
        }
    }

    // If we extracted layer tensors, push as "model.safetensors"
    // (worker treats it as a single-file model, no index.json needed)
    if extracted_safetensors.is_some() {
        files_to_send.push(PathBuf::from("__extracted_model.safetensors__"));
    }

    // Stream each file using chunked reads (constant 128MB memory, not full file)
    let mut read_buf = vec![0u8; MODEL_DATA_CHUNK_SIZE];
    let mut write_buf = Vec::with_capacity(MODEL_DATA_CHUNK_SIZE + 1024); // reusable write buffer

    for file_path in &files_to_send {
        // Handle extracted safetensors (virtual file, not on disk)
        let is_extracted = file_path.to_str() == Some("__extracted_model.safetensors__");
        let filename = if is_extracted {
            "model.safetensors".to_string()
        } else {
            file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        };

        // Small data: keep in memory (config, tokenizer, extracted safetensors)
        let small_data = if is_extracted {
            extracted_safetensors.clone()
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
                let sample_compressed =
                    zstd::encode_all(sample, 1).unwrap_or_else(|_| sample.to_vec());
                if sample_compressed.len() < sample.len() {
                    // Sample compresses — try the full chunk
                    let compressed_data =
                        zstd::encode_all(raw_slice, 1).unwrap_or_else(|_| raw_slice.to_vec());
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
                let compressed_data =
                    zstd::encode_all(raw_slice, 1).unwrap_or_else(|_| raw_slice.to_vec());
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
) -> Result<(Vec<String>, PathBuf, TcpListener, TcpStream)> {
    worker_setup_with_progress(
        worker_name,
        cluster_key,
        bind_address,
        model_cache_dir,
        None,
    )
    .await
}

pub async fn worker_setup_with_progress(
    worker_name: &str,
    cluster_key: &str,
    bind_address: &str,
    model_cache_dir: &Path,
    on_progress: Option<&SetupProgressFn>,
) -> Result<(Vec<String>, PathBuf, TcpListener, TcpStream)> {
    // Detect GPUs
    let gpus = discovery::detect_gpus();
    let backend = discovery::detect_backend();
    let hostname = discovery::detect_hostname();
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
    // Start UDP discovery listener. On iOS without the Multicast Networking
    // entitlement, this will fail gracefully — the worker remains reachable
    // via direct TCP but won't respond to UDP broadcast queries.
    let _discovery = discovery::advertise_worker(worker_name, port, cluster_key, &gpus);

    log::info!("waiting for master to connect and assign layers...");
    if let Some(cb) = &on_progress {
        cb("discovery", "Waiting for master...", 0.0);
    }

    // Accept setup connections in a retry loop. Bad connections (port scanners,
    // nc probes, stale masters) fail auth and are discarded — the worker keeps
    // listening instead of dying on the first bad handshake.
    let (mut stream, _client_addr, layers, model_hash) = loop {
        let (mut stream, client_addr) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        log::info!("[{}] master connected", client_addr);
        if let Some(cb) = &on_progress {
            cb(
                "connected",
                &format!("Master connected ({})", client_addr),
                0.0,
            );
        }

        // Authenticate — bad connections fail here
        if let Err(e) = auth::authenticate_as_worker(&mut stream, cluster_key).await {
            log::warn!(
                "[{}] setup auth failed (bad connection?): {} — waiting for next connection",
                client_addr,
                e
            );
            if let Some(cb) = &on_progress {
                cb("discovery", "Waiting for master...", 0.0);
            }
            continue;
        }
        log::info!("[{}] authenticated", client_addr);
        if let Some(cb) = &on_progress {
            cb("authenticated", "Authenticated with master", 0.0);
        }

        // Read first message: may be DeviceInfoRequest (probe), CompatibilityCheck,
        // or LayerAssignment (old master, backwards compat).
        let first_msg = match Message::from_reader(&mut stream).await {
            Ok((_, msg)) => msg,
            Err(e) => {
                log::warn!(
                    "[{}] failed to read first message: {} — waiting for next connection",
                    client_addr,
                    e
                );
                if let Some(cb) = &on_progress {
                    cb("discovery", "Waiting for master...", 0.0);
                }
                continue;
            }
        };

        let mut next_msg = match first_msg {
            Message::DeviceInfoRequest => {
                Message::DeviceInfoResponse {
                    worker_name: worker_name.to_string(),
                    gpus: gpus.clone(),
                    backend: backend.clone(),
                    hostname: hostname.clone(),
                    os: std::env::consts::OS.to_string(),
                }
                .to_writer(&mut stream)
                .await?;
                let (_, msg) = Message::from_reader(&mut stream).await?;
                msg
            }
            other => other,
        };

        // Handle CompatibilityCheck if present (sent by masters with matching binaries)
        if let Message::CompatibilityCheck {
            protocol_version,
            ref build_hash,
            cache_schema_version,
            ..
        } = next_msg
        {
            let rejection =
                compatibility_rejection_reason(protocol_version, build_hash, cache_schema_version);
            Message::CompatibilityResult {
                accepted: rejection.is_none(),
                reason: rejection.clone(),
            }
            .to_writer(&mut stream)
            .await?;
            if let Some(reason) = rejection {
                log::warn!(
                    "[{}] compatibility rejected: {} — waiting for next connection",
                    client_addr,
                    reason
                );
                if let Some(cb) = &on_progress {
                    cb("discovery", "Waiting for master...", 0.0);
                }
                continue;
            }
            log::info!("[{}] compatibility check passed", client_addr);
            let (_, msg) = Message::from_reader(&mut stream).await?;
            next_msg = msg;
        }

        // Receive layer assignment
        match next_msg {
            Message::LayerAssignment { layers, model_hash } => {
                break (stream, client_addr, layers, model_hash);
            }
            other => {
                log::warn!(
                    "[{}] expected LayerAssignment, got {:?} — waiting for next connection",
                    client_addr,
                    other
                );
                if let Some(cb) = &on_progress {
                    cb("discovery", "Waiting for master...", 0.0);
                }
                continue;
            }
        }
    };

    log::info!("assigned {} layers:", layers.len());
    for layer in &layers {
        log::info!("  {}", layer);
    }
    if let Some(cb) = &on_progress {
        cb(
            "layers",
            &format!("Assigned {} layer(s)", layers.len()),
            0.0,
        );
    }

    let assigned_layer_hash = layer_plan_hash(&layers);

    // Determine cache directory: cluster_hash-model_hash-layer_hash
    // The layer hash ensures re-assignment of different layers invalidates
    // the cache (extracted safetensors only contain the assigned layers).
    let cluster_id = discovery::cluster_hash(cluster_key);
    let layer_hash = assigned_layer_hash;
    let cache_dir = if model_hash.is_empty() {
        model_cache_dir.join(&cluster_id)
    } else {
        model_cache_dir.join(format!("{}-{}-{}", cluster_id, model_hash, layer_hash))
    };
    std::fs::create_dir_all(&cache_dir)?;

    // Prune stale cache directories. Each unique layer assignment creates a new
    // cache dir under model_cache_dir — old ones accumulate and waste storage
    // (5+ GB on 8 GB iOS devices). Keep only the current cache dir.
    if let Some(parent) = cache_dir.parent() {
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path != cache_dir {
                    let name = path.file_name().unwrap_or_default().to_string_lossy();
                    // Only prune dirs that look like cache dirs (contain a hyphen)
                    if name.contains('-') {
                        log::info!("pruning stale cache: {}", path.display());
                        let _ = std::fs::remove_dir_all(&path);
                    }
                }
            }
        }
    }

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

    log::info!("setup transfer complete, waiting for model load before serving");
    Ok((layers, cache_dir, listener, stream))
}

pub async fn send_worker_ready(stream: &mut TcpStream) -> Result<(u64, u64)> {
    let current_rss_bytes = process_resident_memory_bytes();
    let peak_rss_bytes = process_peak_resident_memory_bytes().max(current_rss_bytes);
    Message::WorkerReady {
        current_rss_bytes,
        peak_rss_bytes,
    }
    .to_writer(stream)
    .await?;
    Ok((current_rss_bytes, peak_rss_bytes))
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

    // Clean up any leftover .partial files from interrupted transfers
    cleanup_partial_files(cache_dir);
    cleanup_stale_cache_files(cache_dir)?;

    let mut read_buf = Vec::with_capacity(MODEL_DATA_CHUNK_SIZE + 1024);
    loop {
        let (_, msg) = Message::from_reader_buf(stream, &mut read_buf).await?;

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
                        filename,
                        offset,
                        checksum,
                        actual_crc
                    ));
                }

                // Decompress if compressed
                let data = if compressed {
                    zstd::decode_all(data.as_slice()).map_err(|e| {
                        anyhow!(
                            "zstd decompress failed for {} at offset {}: {}",
                            filename,
                            offset,
                            e
                        )
                    })?
                } else {
                    data
                };
                // Open new file if needed (write to .partial suffix for atomic transfer)
                let file = if let Some((ref name, ref mut file, _, _)) = current_file {
                    if name == &filename {
                        file
                    } else {
                        // Finalize previous file: rename .partial → final name
                        if let Some((prev_name, _, start, size)) = current_file.take() {
                            let elapsed = start.elapsed();
                            let speed = size as f64 / elapsed.as_secs_f64();
                            finalize_partial_file(cache_dir, &prev_name);
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
                        let partial_path = cache_dir.join(format!("{}.partial", &filename));
                        let f = std::fs::File::create(&partial_path)?;
                        current_file = Some((filename.clone(), f, Instant::now(), total_size));
                        if let Some(cb) = &on_progress {
                            cb(
                                "receiving",
                                &format!(
                                    "Receiving {} ({})",
                                    &filename,
                                    human_bytes::human_bytes(total_size as f64)
                                ),
                                0.0,
                            );
                        }
                        &mut current_file.as_mut().unwrap().1
                    }
                } else {
                    let partial_path = cache_dir.join(format!("{}.partial", &filename));
                    let f = std::fs::File::create(&partial_path)?;
                    current_file = Some((filename.clone(), f, Instant::now(), total_size));
                    if let Some(cb) = &on_progress {
                        cb(
                            "receiving",
                            &format!(
                                "Receiving {} ({})",
                                &filename,
                                human_bytes::human_bytes(total_size as f64)
                            ),
                            0.0,
                        );
                    }
                    &mut current_file.as_mut().unwrap().1
                };

                file.write_all(&data)?;
                overall_bytes += data.len() as u64;

                // Progress callback for all files (not just large ones)
                let written = offset + data.len() as u64;
                if let Some((_, _, ref start, _)) = current_file {
                    let elapsed = start.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        written as f64 / elapsed
                    } else {
                        0.0
                    };
                    let pct = if total_size > 0 {
                        (written as f64 / total_size as f64) * 100.0
                    } else {
                        0.0
                    };

                    if total_size > MODEL_DATA_CHUNK_SIZE as u64 && written < total_size {
                        let remaining = total_size - written;
                        let eta_secs = if speed > 0.0 {
                            remaining as f64 / speed
                        } else {
                            0.0
                        };
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
                // Finalize last file: rename .partial → final name
                if let Some((name, _, start, size)) = current_file.take() {
                    let elapsed = start.elapsed();
                    let speed = size as f64 / elapsed.as_secs_f64();
                    finalize_partial_file(cache_dir, &name);
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

/// Extract specific tensors from sharded safetensors files into a single
/// minimal safetensors blob. This is the key optimization for model push:
/// instead of sending entire shard files (5+ GiB), we send only the tensors
/// needed for the assigned layers (hundreds of MiB).
fn extract_layer_tensors(
    model_path: &Path,
    needed: &[(String, String)], // (tensor_name, shard_filename)
) -> Result<Vec<u8>> {
    use std::collections::HashMap;
    use std::io::Read;

    // Group needed tensors by shard file
    let mut by_shard: HashMap<&str, Vec<&str>> = HashMap::new();
    for (tensor_name, shard_file) in needed {
        by_shard
            .entry(shard_file.as_str())
            .or_default()
            .push(tensor_name.as_str());
    }

    // Extract raw tensor data from each shard
    // Structure: tensor_name → (dtype_str, shape, raw_bytes)
    let mut tensors: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();

    for (shard_file, tensor_names) in &by_shard {
        let shard_path = model_path.join(shard_file);
        let mut file = std::fs::File::open(&shard_path)
            .map_err(|e| anyhow!("can't open shard {}: {}", shard_path.display(), e))?;

        // Read safetensors header
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header_buf = vec![0u8; header_len];
        file.read_exact(&mut header_buf)?;
        let header: serde_json::Value = serde_json::from_slice(&header_buf)?;
        let data_offset = 8 + header_len; // offset where tensor data starts in file

        for tensor_name in tensor_names {
            let info = header
                .get(*tensor_name)
                .ok_or_else(|| anyhow!("tensor {} not found in {}", tensor_name, shard_file))?;
            let dtype = info
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("F16")
                .to_string();
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let offsets = info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("no data_offsets for {}", tensor_name))?;
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let byte_len = end - start;

            // Seek and read the tensor data
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start((data_offset + start) as u64))?;
            let mut data = vec![0u8; byte_len];
            file.read_exact(&mut data)?;

            tensors.push((tensor_name.to_string(), dtype, shape, data));
        }
    }

    // Build minimal safetensors: header + concatenated tensor data
    // Sort by name for deterministic output
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let mut header_map = serde_json::Map::new();
    let mut data_blob: Vec<u8> = Vec::new();

    for (name, dtype, shape, raw) in &tensors {
        let start = data_blob.len();
        data_blob.extend_from_slice(raw);
        let end = data_blob.len();

        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String(dtype.clone()));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|&s| serde_json::json!(s)).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![serde_json::json!(start), serde_json::json!(end)]),
        );
        header_map.insert(name.clone(), serde_json::Value::Object(entry));
    }

    let header_json = serde_json::to_vec(&serde_json::Value::Object(header_map))?;
    let header_len = header_json.len() as u64;

    let mut result = Vec::with_capacity(8 + header_json.len() + data_blob.len());
    result.extend_from_slice(&header_len.to_le_bytes());
    result.extend_from_slice(&header_json);
    result.extend_from_slice(&data_blob);

    Ok(result)
}

/// Remove `.partial` files left over from interrupted transfers.
fn cleanup_partial_files(cache_dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "partial") {
                log::info!(
                    "removing leftover partial file: {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Remove stale safetensors artifacts from a previous assignment before receiving
/// fresh model data into the cache directory.
fn cleanup_stale_cache_files(cache_dir: &Path) -> Result<()> {
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                log::info!(
                    "removing stale safetensors file: {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                std::fs::remove_file(&path)?;
            }
        }
    }

    let stale_index = cache_dir.join("model.safetensors.index.json");
    if stale_index.exists() {
        log::info!("removing stale index.json from previous push");
        std::fs::remove_file(stale_index)?;
    }

    Ok(())
}

/// Rename a `.partial` file to its final name after successful transfer.
fn finalize_partial_file(cache_dir: &Path, filename: &str) {
    let partial = cache_dir.join(format!("{filename}.partial"));
    let final_path = cache_dir.join(filename);
    if partial.exists() {
        if let Err(e) = std::fs::rename(&partial, &final_path) {
            log::error!(
                "failed to finalize {}: {} — file may need manual rename",
                filename,
                e,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ── Partial file cleanup tests ───────────────────────

    #[test]
    fn cleanup_partial_files_removes_partial() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("shard.safetensors.partial"), "data").unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();
        cleanup_partial_files(tmp.path());
        assert!(!tmp.path().join("shard.safetensors.partial").exists());
        assert!(tmp.path().join("config.json").exists());
    }

    #[test]
    fn cleanup_stale_cache_files_removes_old_weights_and_index() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("model.safetensors"), "weights").unwrap();
        fs::write(tmp.path().join("model-00001-of-00002.safetensors"), "shard").unwrap();
        fs::write(tmp.path().join("model.safetensors.index.json"), "{}").unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();

        cleanup_stale_cache_files(tmp.path()).unwrap();

        assert!(!tmp.path().join("model.safetensors").exists());
        assert!(!tmp.path().join("model-00001-of-00002.safetensors").exists());
        assert!(!tmp.path().join("model.safetensors.index.json").exists());
        assert!(tmp.path().join("config.json").exists());
    }

    #[test]
    fn finalize_partial_file_renames() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("shard.safetensors.partial"), "data").unwrap();
        finalize_partial_file(tmp.path(), "shard.safetensors");
        assert!(!tmp.path().join("shard.safetensors.partial").exists());
        assert!(tmp.path().join("shard.safetensors").exists());
    }

    #[test]
    fn finalize_partial_file_noop_if_no_partial() {
        let tmp = tempfile::tempdir().unwrap();
        // Should not panic or error
        finalize_partial_file(tmp.path(), "shard.safetensors");
        assert!(!tmp.path().join("shard.safetensors").exists());
    }

    #[test]
    fn compute_model_hash_changes_when_single_file_contents_change_same_size() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");
        fs::write(
            &config_path,
            r#"{"num_hidden_layers":1,"hidden_size":64,"vocab_size":128}"#,
        )
        .unwrap();
        let config_data = fs::read_to_string(&config_path).unwrap();
        let model_path = tmp.path().join("model.safetensors");

        fs::write(&model_path, vec![0x11u8; 4096]).unwrap();
        let hash_a = compute_model_hash(tmp.path(), &config_data);

        fs::write(&model_path, vec![0x22u8; 4096]).unwrap();
        let hash_b = compute_model_hash(tmp.path(), &config_data);

        assert_ne!(hash_a, hash_b);
    }

    #[tokio::test]
    async fn master_setup_with_manual_workers_skips_discovery_delay() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("config.json"),
            serde_json::json!({
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 0,
                "hidden_size": 64,
                "vocab_size": 128,
                "tie_word_embeddings": false
            })
            .to_string(),
        )
        .unwrap();

        let manual_workers = vec!["127.0.0.1:9".to_string()];
        let start = std::time::Instant::now();
        let topology = master_setup(
            "test-cluster",
            tmp.path(),
            Duration::from_secs(2),
            0,
            3,
            None,
            &manual_workers,
        )
        .await
        .unwrap();

        assert!(topology.is_empty());
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "manual --workers path should skip UDP discovery timeout, took {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn probe_manual_worker_uses_same_setup_connection() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            auth::authenticate_as_worker(&mut socket, "probe-key")
                .await
                .unwrap();

            let (_, msg) = Message::from_reader(&mut socket).await.unwrap();
            assert!(matches!(msg, Message::DeviceInfoRequest));
            Message::DeviceInfoResponse {
                worker_name: "ipad".into(),
                gpus: vec![discovery::GpuInfo {
                    name: "Apple M3".into(),
                    vram_bytes: 8 * 1024 * 1024 * 1024,
                    tflops: 3.0,
                }],
                backend: "Metal".into(),
                hostname: "ipad-host".into(),
                os: "ios".into(),
            }
            .to_writer(&mut socket)
            .await
            .unwrap();

            let (_, followup) = Message::from_reader(&mut socket).await.unwrap();
            assert!(matches!(followup, Message::Hello));
        });

        let (worker, mut stream) = probe_manual_worker(&addr.to_string(), "probe-key")
            .await
            .unwrap();
        assert_eq!(worker.name, "ipad");
        assert_eq!(worker.hostname, "ipad-host");
        assert_eq!(worker.total_vram(), 8 * 1024 * 1024 * 1024);
        assert_eq!(worker.total_tflops(), 3.0);

        Message::Hello.to_writer(&mut stream).await.unwrap();
        server.await.unwrap();
    }

    #[test]
    fn test_compatibility_check_matching_versions() {
        let reason = compatibility_rejection_reason(
            PROTOCOL_VERSION,
            crate::BUILD_HASH,
            CACHE_SCHEMA_VERSION,
        );
        assert!(reason.is_none());
    }

    #[test]
    fn test_compatibility_check_protocol_mismatch() {
        let reason = compatibility_rejection_reason(
            PROTOCOL_VERSION + 1,
            crate::BUILD_HASH,
            CACHE_SCHEMA_VERSION,
        );
        assert!(
            reason
                .as_deref()
                .is_some_and(|msg| msg.contains("protocol_version mismatch"))
        );
    }

    #[test]
    fn test_compatibility_check_build_mismatch() {
        let reason = compatibility_rejection_reason(
            PROTOCOL_VERSION,
            "different-build",
            CACHE_SCHEMA_VERSION,
        );
        assert!(
            reason
                .as_deref()
                .is_some_and(|msg| msg.contains("build_hash mismatch"))
        );
    }

    #[test]
    fn test_compatibility_check_model_mismatch() {
        let reason = cache_refresh_reason(
            "current-model",
            "requested-model",
            "same-layer",
            "same-layer",
        );
        assert!(
            reason
                .as_deref()
                .is_some_and(|msg| msg.contains("model_hash mismatch"))
        );
    }

    #[test]
    fn test_compatibility_check_layer_mismatch() {
        let reason = cache_refresh_reason(
            "same-model",
            "same-model",
            "current-layer",
            "requested-layer",
        );
        assert!(
            reason
                .as_deref()
                .is_some_and(|msg| msg.contains("layer_plan_hash mismatch"))
        );
    }

    #[test]
    fn test_topology_class_mobile_only() {
        let mut topology = Topology::new();
        topology.insert(
            "ipad".into(),
            Node {
                host: "10.0.0.2:10128".into(),
                description: None,
                layers: vec!["model.layers.0".into()],
                vram_bytes: 4 * 1024 * 1024 * 1024,
                tflops: 2.0,
                backend: "metal".into(),
                hostname: "ipad".into(),
                os: "ios".into(),
            },
        );
        assert_eq!(
            classify_topology(&topology, 16 * 1024 * 1024 * 1024),
            TopologyClass::MobileOnly
        );
    }

    #[test]
    fn test_topology_class_mixed() {
        let mut topology = Topology::new();
        topology.insert(
            "iphone".into(),
            Node {
                host: "10.0.0.3:10128".into(),
                description: None,
                layers: vec!["model.layers.1".into()],
                vram_bytes: 6 * 1024 * 1024 * 1024,
                tflops: 2.0,
                backend: "metal".into(),
                hostname: "iphone".into(),
                os: "ios".into(),
            },
        );
        assert_eq!(
            classify_topology(&topology, 64 * 1024 * 1024 * 1024),
            TopologyClass::MixedDesktopMobile
        );
    }

    #[test]
    fn test_failure_mobile_only_aborts() {
        let lost_worker = Node {
            host: "10.0.0.2:10128".into(),
            description: None,
            layers: vec!["model.layers.0".into()],
            vram_bytes: 4 * 1024 * 1024 * 1024,
            tflops: 2.0,
            backend: "metal".into(),
            hostname: "ipad".into(),
            os: "ios".into(),
        };
        let plan = plan_worker_loss(
            TopologyClass::MobileOnly,
            &lost_worker,
            2,
            256 * 1024 * 1024,
            3 * 1024 * 1024 * 1024,
            0,
        );
        assert_eq!(plan.action, WorkerLossAction::Abort);
        assert!(plan.reason.contains("mobile pool cannot absorb"));
    }

    #[test]
    fn test_failure_mixed_reassigns() {
        let lost_worker = Node {
            host: "10.0.0.2:10128".into(),
            description: None,
            layers: vec!["model.layers.0".into()],
            vram_bytes: 4 * 1024 * 1024 * 1024,
            tflops: 2.0,
            backend: "metal".into(),
            hostname: "ipad".into(),
            os: "ios".into(),
        };
        let plan = plan_worker_loss(
            TopologyClass::MixedDesktopMobile,
            &lost_worker,
            2,
            256 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            0,
        );
        assert_eq!(plan.action, WorkerLossAction::ReassignToMaster);
        assert!(plan.reason.contains("reassigning"));
    }

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
        let gpus = vec![discovery::GpuInfo {
            name: "GPU".into(),
            vram_bytes: 1024,
            tflops: 1.0,
        }];
        assert_eq!(max_layers_for_gpus(&gpus, 0), usize::MAX);
    }

    #[test]
    fn test_estimate_tflops_reported_vs_fallback() {
        let gpus_reported = vec![discovery::GpuInfo {
            name: "GPU".into(),
            vram_bytes: 1024,
            tflops: 42.0,
        }];
        assert!((estimate_tflops_for_gpus(&gpus_reported) - 42.0).abs() < 0.01);

        let gpus_zero = vec![discovery::GpuInfo {
            name: "NVIDIA RTX".into(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            tflops: 0.0,
        }];
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
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        fs::write(tmp.path().join("shard-00002.safetensors"), "data").unwrap();

        let layers = vec!["model.layers.0".to_string(), "model.layers.1".to_string()];
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

        let layers = vec!["model.layers.0".to_string(), "model.layers.1".to_string()];
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
        )
        .unwrap();
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
        )
        .unwrap();
        fs::write(tmp.path().join("shard-00001.safetensors"), "data").unwrap();
        let layers = vec!["model.layers.1".to_string()];
        assert!(!has_valid_model_cache(tmp.path(), &layers));
    }

    // ── extract_layer_tensors single-file tests ─────────────

    /// Build a minimal valid safetensors file from named tensors.
    fn build_test_safetensors(tensors: &[(&str, &[u8])]) -> Vec<u8> {
        let mut header_map = serde_json::Map::new();
        let mut data_blob: Vec<u8> = Vec::new();

        for (name, raw) in tensors {
            let start = data_blob.len();
            data_blob.extend_from_slice(raw);
            let end = data_blob.len();

            let mut entry = serde_json::Map::new();
            entry.insert("dtype".into(), serde_json::json!("F32"));
            entry.insert("shape".into(), serde_json::json!([raw.len() / 4]));
            entry.insert(
                "data_offsets".into(),
                serde_json::json!([start, end]),
            );
            header_map.insert(name.to_string(), serde_json::Value::Object(entry));
        }

        let header_json =
            serde_json::to_vec(&serde_json::Value::Object(header_map)).unwrap();
        let header_len = header_json.len() as u64;

        let mut result = Vec::with_capacity(8 + header_json.len() + data_blob.len());
        result.extend_from_slice(&header_len.to_le_bytes());
        result.extend_from_slice(&header_json);
        result.extend_from_slice(&data_blob);
        result
    }

    #[test]
    fn extract_layer_tensors_single_file_filters_correctly() {
        let tmp = tempfile::tempdir().unwrap();

        let layer0_data: Vec<u8> = vec![0x10; 16];
        let layer1_data: Vec<u8> = vec![0x20; 16];
        let layer2_data: Vec<u8> = vec![0x30; 16];
        let embed_data: Vec<u8> = vec![0x40; 16];

        let safetensors = build_test_safetensors(&[
            ("model.layers.0.weight", &layer0_data),
            ("model.layers.1.weight", &layer1_data),
            ("model.layers.2.weight", &layer2_data),
            ("model.embed_tokens.weight", &embed_data),
        ]);

        fs::write(tmp.path().join("model.safetensors"), &safetensors).unwrap();

        // Request only layers 0 and 1
        let needed = vec![
            (
                "model.layers.0.weight".to_string(),
                "model.safetensors".to_string(),
            ),
            (
                "model.layers.1.weight".to_string(),
                "model.safetensors".to_string(),
            ),
        ];

        let extracted = extract_layer_tensors(tmp.path(), &needed).unwrap();

        // Parse the extracted blob header
        let header_len =
            u64::from_le_bytes(extracted[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&extracted[8..8 + header_len]).unwrap();
        let obj = header.as_object().unwrap();

        // Should contain only the two requested tensors
        assert!(obj.contains_key("model.layers.0.weight"));
        assert!(obj.contains_key("model.layers.1.weight"));
        assert!(!obj.contains_key("model.layers.2.weight"));
        assert!(!obj.contains_key("model.embed_tokens.weight"));
        assert_eq!(obj.len(), 2);

        // Verify total data size = 32 bytes (16 x 2 tensors)
        let data_start = 8 + header_len;
        let remaining = &extracted[data_start..];
        assert_eq!(remaining.len(), 32);

        // Verify each tensor's data matches the original
        for (name, expected_data) in [
            ("model.layers.0.weight", &layer0_data),
            ("model.layers.1.weight", &layer1_data),
        ] {
            let entry = obj.get(name).unwrap();
            let offsets = entry.get("data_offsets").unwrap().as_array().unwrap();
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            assert_eq!(&remaining[start..end], expected_data.as_slice());
        }
    }

    #[test]
    fn extract_layer_tensors_single_file_all_layers() {
        let tmp = tempfile::tempdir().unwrap();

        let layer0_data: Vec<u8> = vec![0xAA; 32];
        let layer1_data: Vec<u8> = vec![0xBB; 32];

        let safetensors = build_test_safetensors(&[
            ("model.layers.0.weight", &layer0_data),
            ("model.layers.1.weight", &layer1_data),
        ]);

        fs::write(tmp.path().join("model.safetensors"), &safetensors).unwrap();

        // Request all layers
        let needed = vec![
            (
                "model.layers.0.weight".to_string(),
                "model.safetensors".to_string(),
            ),
            (
                "model.layers.1.weight".to_string(),
                "model.safetensors".to_string(),
            ),
        ];

        let extracted = extract_layer_tensors(tmp.path(), &needed).unwrap();

        let header_len =
            u64::from_le_bytes(extracted[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&extracted[8..8 + header_len]).unwrap();
        let obj = header.as_object().unwrap();
        assert_eq!(obj.len(), 2);

        let data_start = 8 + header_len;
        assert_eq!(extracted[data_start..].len(), 64);
    }

    #[test]
    fn extract_layer_tensors_single_file_missing_tensor_errors() {
        let tmp = tempfile::tempdir().unwrap();

        let safetensors =
            build_test_safetensors(&[("model.layers.0.weight", &vec![0u8; 16])]);

        fs::write(tmp.path().join("model.safetensors"), &safetensors).unwrap();

        // Request a tensor that doesn't exist
        let needed = vec![(
            "model.layers.99.weight".to_string(),
            "model.safetensors".to_string(),
        )];

        let result = extract_layer_tensors(tmp.path(), &needed);
        assert!(result.is_err());
    }
}
