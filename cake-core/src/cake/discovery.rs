//! UDP broadcast service discovery and GPU detection for zero-config clustering.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speedy::{Readable, Writable};

/// UDP broadcast port for Cake discovery.
const DISCOVERY_PORT: u16 = 10127;

/// Magic bytes to identify Cake discovery packets.
const MAGIC: &[u8; 4] = b"CAKE";

/// Default discovery timeout.
pub const DEFAULT_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// GPU information advertised by a worker.
#[derive(Debug, Clone, Serialize, Deserialize, Readable, Writable)]
pub struct GpuInfo {
    pub name: String,
    pub vram_bytes: u64,
}

/// A worker discovered via broadcast.
#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub gpus: Vec<GpuInfo>,
}

impl DiscoveredWorker {
    /// Total VRAM across all GPUs.
    pub fn total_vram(&self) -> u64 {
        self.gpus.iter().map(|g| g.vram_bytes).sum()
    }
}

/// Compute the first 8 hex chars of SHA-256(cluster_key) for filtering.
pub fn cluster_hash(cluster_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(cluster_key.as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..4])
}

/// Detect available GPUs on this system.
pub fn detect_gpus() -> Vec<GpuInfo> {
    // Try NVIDIA first
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let gpus: Vec<GpuInfo> = stdout
                .lines()
                .filter_map(|line| {
                    let parts: Vec<&str> = line.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        let name = parts[0].trim().to_string();
                        let vram_mb: u64 = parts[1].trim().parse().ok()?;
                        Some(GpuInfo {
                            name,
                            vram_bytes: vram_mb * 1024 * 1024,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            if !gpus.is_empty() {
                return gpus;
            }
        }
    }

    // Fallback: report system memory as a single "CPU" or "Metal" device
    let name = if cfg!(target_os = "macos") {
        format!("Apple Silicon ({})", std::env::consts::ARCH)
    } else {
        format!("CPU ({})", std::env::consts::ARCH)
    };

    let vram_bytes = detect_system_memory();

    vec![GpuInfo { name, vram_bytes }]
}

/// Detect total system memory in bytes.
fn detect_system_memory() -> u64 {
    // On macOS, use sysctl for a reliable reading
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            if output.status.success() {
                let s = String::from_utf8_lossy(&output.stdout);
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes;
                }
            }
        }
    }

    // On Linux, read /proc/meminfo
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let rest = rest.trim();
                    if let Some(kb_str) = rest.strip_suffix("kB") {
                        if let Ok(kb) = kb_str.trim().parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }

    // Fallback to memory_stats
    memory_stats::memory_stats()
        .map(|s| s.physical_mem as u64)
        .unwrap_or(0)
}

// ── Discovery packet format ────────────────────────────────────────────────

/// A discovery query broadcast by the master.
#[derive(Serialize, Deserialize)]
struct DiscoveryQuery {
    cluster_hash: String,
}

/// A discovery response sent by workers (unicast back to master).
#[derive(Serialize, Deserialize)]
struct DiscoveryResponse {
    cluster_hash: String,
    worker_name: String,
    port: u16,
    gpus: Vec<GpuInfo>,
}

fn encode_packet(payload: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(4 + payload.len());
    pkt.extend_from_slice(MAGIC);
    pkt.extend_from_slice(payload);
    pkt
}

fn decode_packet(data: &[u8]) -> Option<&[u8]> {
    if data.len() > 4 && data[..4] == *MAGIC {
        Some(&data[4..])
    } else {
        None
    }
}

// ── Worker advertisement (listen for queries, respond) ─────────────────────

/// Handle for a running discovery listener.
/// Must be kept alive for the worker to respond to discovery queries.
pub struct DiscoveryListener {
    _handle: std::thread::JoinHandle<()>,
}

/// Start listening for discovery queries and responding with worker info.
///
/// Spawns a background thread. Returns a handle that must be kept alive.
pub fn advertise_worker(
    worker_name: &str,
    port: u16,
    cluster_key: &str,
    gpus: &[GpuInfo],
) -> Result<DiscoveryListener> {
    let hash = cluster_hash(cluster_key);
    let response = DiscoveryResponse {
        cluster_hash: hash.clone(),
        worker_name: worker_name.to_string(),
        port,
        gpus: gpus.to_vec(),
    };
    let response_json = serde_json::to_vec(&response)?;
    let response_pkt = encode_packet(&response_json);

    let sock = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, DISCOVERY_PORT))
        .map_err(|e| anyhow!("failed to bind discovery UDP socket on port {}: {}", DISCOVERY_PORT, e))?;
    sock.set_broadcast(true)?;
    sock.set_read_timeout(Some(Duration::from_secs(1)))?;

    log::info!(
        "listening for discovery queries on UDP port {}",
        DISCOVERY_PORT
    );

    let handle = std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match sock.recv_from(&mut buf) {
                Ok((len, src)) => {
                    if let Some(payload) = decode_packet(&buf[..len]) {
                        if let Ok(query) = serde_json::from_slice::<DiscoveryQuery>(payload) {
                            if query.cluster_hash == hash {
                                // Respond directly to the master
                                if let Err(e) = sock.send_to(&response_pkt, src) {
                                    log::warn!("failed to send discovery response to {}: {}", src, e);
                                }
                            }
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    // Normal timeout, keep listening
                }
                Err(e) => {
                    log::warn!("discovery listener error: {}", e);
                    break;
                }
            }
        }
    });

    Ok(DiscoveryListener { _handle: handle })
}

// ── Master browsing (broadcast query, collect responses) ──────────────────

/// Browse for workers on the network matching the given cluster key.
///
/// Sends periodic UDP broadcast queries, collects responses until timeout.
pub async fn discover_workers(
    cluster_key: &str,
    timeout: Duration,
) -> Result<Vec<DiscoveredWorker>> {
    let expected_hash = cluster_hash(cluster_key);

    log::info!(
        "discovering workers (timeout: {}s)...",
        timeout.as_secs()
    );

    let workers = tokio::task::spawn_blocking(move || {
        let sock = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0))
            .map_err(|e| anyhow!("failed to bind discovery socket: {}", e))?;
        sock.set_broadcast(true)?;
        sock.set_read_timeout(Some(Duration::from_millis(500)))?;

        let query = DiscoveryQuery {
            cluster_hash: expected_hash.clone(),
        };
        let query_json = serde_json::to_vec(&query)?;
        let query_pkt = encode_packet(&query_json);

        let broadcast_addr = SocketAddr::V4(SocketAddrV4::new(
            Ipv4Addr::BROADCAST,
            DISCOVERY_PORT,
        ));

        let mut workers: HashMap<String, DiscoveredWorker> = HashMap::new();
        let deadline = std::time::Instant::now() + timeout;
        let mut last_query = std::time::Instant::now() - Duration::from_secs(10);
        let query_interval = Duration::from_secs(1);
        let mut buf = [0u8; 65535];

        loop {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }

            // Send periodic broadcast queries
            if now.duration_since(last_query) >= query_interval {
                if let Err(e) = sock.send_to(&query_pkt, broadcast_addr) {
                    log::warn!("failed to send discovery broadcast: {}", e);
                }
                last_query = now;
            }

            // Listen for responses
            match sock.recv_from(&mut buf) {
                Ok((len, src)) => {
                    if let Some(payload) = decode_packet(&buf[..len]) {
                        if let Ok(resp) = serde_json::from_slice::<DiscoveryResponse>(payload) {
                            if resp.cluster_hash != expected_hash {
                                continue;
                            }

                            let src_ip = match src {
                                SocketAddr::V4(a) => a.ip().to_string(),
                                SocketAddr::V6(a) => a.ip().to_string(),
                            };
                            let host = format!("{}:{}", src_ip, resp.port);

                            if !workers.contains_key(&resp.worker_name) {
                                log::info!(
                                    "discovered worker '{}' at {} with {} GPU(s)",
                                    &resp.worker_name,
                                    &host,
                                    resp.gpus.len()
                                );

                                for gpu in &resp.gpus {
                                    log::info!(
                                        "  {} — {}",
                                        &gpu.name,
                                        human_bytes::human_bytes(gpu.vram_bytes as f64)
                                    );
                                }

                                workers.insert(resp.worker_name.clone(), DiscoveredWorker {
                                    name: resp.worker_name,
                                    host,
                                    port: resp.port,
                                    gpus: resp.gpus,
                                });
                            }
                        }
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    // Normal timeout, loop again
                }
                Err(e) => {
                    log::warn!("discovery recv error: {}", e);
                }
            }
        }

        Ok::<_, anyhow::Error>(workers)
    }).await??;

    log::info!("discovery complete: {} worker(s) found", workers.len());
    Ok(workers.into_values().collect())
}
