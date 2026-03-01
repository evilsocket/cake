//! mDNS service discovery and GPU detection for zero-config clustering.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speedy::{Readable, Writable};

/// The mDNS service type for Cake workers.
const SERVICE_TYPE: &str = "_cake._tcp.local.";

/// Default discovery timeout.
pub const DEFAULT_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// GPU information advertised by a worker.
#[derive(Debug, Clone, Serialize, Deserialize, Readable, Writable)]
pub struct GpuInfo {
    pub name: String,
    pub vram_bytes: u64,
}

/// A worker discovered via mDNS.
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

/// Compute the first 8 hex chars of SHA-256(cluster_key) for mDNS filtering.
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

/// Advertise this worker via mDNS.
///
/// Returns the `ServiceDaemon` handle — it **must** be kept alive for the
/// advertisement to persist on the network.
pub fn advertise_worker(
    instance_name: &str,
    port: u16,
    cluster_key: &str,
    gpus: &[GpuInfo],
) -> Result<ServiceDaemon> {
    let mdns = ServiceDaemon::new().map_err(|e| anyhow!("failed to create mDNS daemon: {}", e))?;

    let gpus_json = serde_json::to_string(gpus)?;
    let hash = cluster_hash(cluster_key);

    let properties: HashMap<String, String> = [
        ("cluster".to_string(), hash),
        ("gpus".to_string(), gpus_json),
        ("version".to_string(), env!("CARGO_PKG_VERSION").to_string()),
    ]
    .into_iter()
    .collect();

    let hostname = hostname::get()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let host_label = if hostname.ends_with(".local.") {
        hostname
    } else if hostname.ends_with(".local") {
        format!("{}.", hostname)
    } else {
        format!("{}.local.", hostname)
    };

    let service = ServiceInfo::new(
        SERVICE_TYPE,
        instance_name,
        &host_label,
        "",
        port,
        properties,
    )
    .map_err(|e| anyhow!("failed to create mDNS service info: {}", e))?
    .enable_addr_auto();

    mdns.register(service)
        .map_err(|e| anyhow!("failed to register mDNS service: {}", e))?;

    log::info!(
        "advertising worker '{}' via mDNS on port {}",
        instance_name,
        port
    );

    Ok(mdns)
}

/// Browse for workers on the network matching the given cluster key.
///
/// Blocks for up to `timeout`, collecting all discovered workers.
pub async fn discover_workers(
    cluster_key: &str,
    timeout: Duration,
) -> Result<Vec<DiscoveredWorker>> {
    let expected_hash = cluster_hash(cluster_key);

    log::info!(
        "discovering workers via mDNS (timeout: {}s)...",
        timeout.as_secs()
    );

    // Run the entire browse loop in a blocking thread to avoid
    // event loss between async iterations.
    let workers = tokio::task::spawn_blocking(move || {
        let mdns = ServiceDaemon::new().map_err(|e| anyhow!("failed to create mDNS daemon: {}", e))?;
        let receiver = mdns
            .browse(SERVICE_TYPE)
            .map_err(|e| anyhow!("failed to browse mDNS: {}", e))?;

        let mut workers: HashMap<String, DiscoveredWorker> = HashMap::new();
        let deadline = std::time::Instant::now() + timeout;

        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }

            let recv_timeout = remaining.min(Duration::from_secs(1));
            match receiver.recv_timeout(recv_timeout) {
                Ok(ServiceEvent::ServiceResolved(info)) => {
                    // Check cluster hash
                    let cluster = info.get_property_val_str("cluster")
                        .unwrap_or_default();
                    if cluster != expected_hash {
                        continue;
                    }

                    // Parse GPU info
                    let gpus_json = info.get_property_val_str("gpus")
                        .unwrap_or("[]");
                    let gpus: Vec<GpuInfo> = serde_json::from_str(gpus_json)
                        .unwrap_or_default();

                    let port = info.get_port();
                    // Get the first address
                    let addrs = info.get_addresses();
                    let ip = if let Some(addr) = addrs.iter().next() {
                        addr.to_string()
                    } else {
                        continue;
                    };

                    let host = format!("{}:{}", ip, port);
                    let name = info.get_fullname().to_string();

                    // Use the service instance name as the worker name
                    let instance_name = name
                        .strip_suffix(&format!(".{}", SERVICE_TYPE))
                        .unwrap_or(&name)
                        .to_string();

                    log::info!(
                        "discovered worker '{}' at {} with {} GPU(s)",
                        &instance_name,
                        &host,
                        gpus.len()
                    );

                    for gpu in &gpus {
                        log::info!(
                            "  {} — {}",
                            &gpu.name,
                            human_bytes::human_bytes(gpu.vram_bytes as f64)
                        );
                    }

                    workers.insert(instance_name.clone(), DiscoveredWorker {
                        name: instance_name,
                        host,
                        port,
                        gpus,
                    });
                }
                Ok(_) => {} // other events
                Err(_) => {} // timeout on recv, loop again
            }
        }

        let _ = mdns.shutdown();
        Ok::<_, anyhow::Error>(workers)
    }).await??;

    log::info!("discovery complete: {} worker(s) found", workers.len());
    Ok(workers.into_values().collect())
}
