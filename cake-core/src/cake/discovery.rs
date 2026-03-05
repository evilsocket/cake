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
    #[serde(default)]
    pub tflops: f32,
}

/// A worker discovered via broadcast.
#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub gpus: Vec<GpuInfo>,
    pub backend: String,
    pub hostname: String,
    pub os: String,
}

impl DiscoveredWorker {
    /// Total VRAM across all GPUs.
    pub fn total_vram(&self) -> u64 {
        self.gpus.iter().map(|g| g.vram_bytes).sum()
    }

    /// Maximum number of layers this worker can fit, based on per-GPU VRAM.
    ///
    /// For dedicated GPUs (CUDA), reserves ~5% for driver/runtime overhead
    /// (typically 200–600 MiB for CUDA context + cuBLAS workspace).
    /// For unified-memory devices (Apple Silicon), reserves 28% of total
    /// (minimum 6 GiB) for macOS + inference working memory, since model
    /// weights compete with the OS for the same physical RAM and insufficient
    /// headroom causes catastrophic memory-compressor thrashing.
    pub fn max_layers_for_size(&self, layer_size_bytes: u64) -> usize {
        if layer_size_bytes == 0 || self.gpus.is_empty() {
            return usize::MAX;
        }
        self.gpus
            .iter()
            .map(|g| {
                let name_lower = g.name.to_lowercase();
                let is_cpu = name_lower.starts_with("cpu");
                let is_unified = name_lower.contains("apple");
                let usable = if is_cpu {
                    // CPU / mobile worker: reported vram_bytes is system RAM.
                    // Reserve 20% for OS + runtime; no large fixed minimum since
                    // mobile devices may have only 2–4 GiB total.
                    let reserve = (g.vram_bytes as f64 * 0.20) as u64;
                    g.vram_bytes.saturating_sub(reserve)
                } else if is_unified {
                    // Unified memory: reserve 28% of total (min 6 GiB) for OS +
                    // Metal working memory. At 30 layers on a 36 GiB M3 Pro,
                    // only 8 GiB remained and macOS memory compressor caused
                    // 100+ sec/forward-pass thrashing; 28% keeps ~10 GiB free.
                    let min_reserve = 6u64 * 1024 * 1024 * 1024;
                    let pct_reserve = (g.vram_bytes as f64 * 0.28) as u64;
                    let os_reserve = pct_reserve.max(min_reserve);
                    g.vram_bytes.saturating_sub(os_reserve)
                } else {
                    // Dedicated VRAM: reserve max(5%, 768 MiB) for CUDA context,
                    // cuBLAS workspace, and memory fragmentation. The percentage
                    // works for large GPUs (24+ GB), but on 12 GB GPUs 5% = 600 MB
                    // leaves only ~50 MB headroom after filling layers, causing OOM.
                    let min_reserve = 768u64 * 1024 * 1024;
                    let pct_reserve = (g.vram_bytes as f64 * 0.05) as u64;
                    let reserve = pct_reserve.max(min_reserve);
                    g.vram_bytes.saturating_sub(reserve)
                };
                (usable / layer_size_bytes) as usize
            })
            .sum()
    }

    /// Total estimated TFLOPS across all GPUs.
    /// Falls back to a VRAM-based estimate when workers report 0 (old binaries).
    pub fn total_tflops(&self) -> f64 {
        let reported: f64 = self.gpus.iter().map(|g| g.tflops as f64).sum();
        if reported > 0.0 {
            return reported;
        }
        // Fallback: estimate from VRAM and device name
        self.gpus
            .iter()
            .map(|g| {
                let vram_gb = g.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let name_lower = g.name.to_lowercase();
                if name_lower.contains("nvidia")
                    || name_lower.contains("geforce")
                    || name_lower.contains("rtx")
                    || name_lower.contains("gtx")
                    || name_lower.contains("tesla")
                {
                    vram_gb * 3.0 // CUDA GPU fallback
                } else if name_lower.contains("apple") || name_lower.contains("silicon") {
                    vram_gb * 0.4 // Metal fallback
                } else {
                    2.0 // CPU fallback
                }
            })
            .sum()
    }
}

/// Compute the first 8 hex chars of SHA-256(cluster_key) for filtering.
pub fn cluster_hash(cluster_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(cluster_key.as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..4])
}

/// Detect available compute devices on this system.
///
/// Only reports NVIDIA GPUs when the `cuda` feature is compiled in,
/// and Metal on macOS when the `metal` feature is compiled in.
/// Otherwise falls back to CPU with system RAM.
pub fn detect_gpus() -> Vec<GpuInfo> {
    // Only probe NVIDIA GPUs if built with CUDA support
    #[cfg(feature = "cuda")]
    {
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,clocks.max.graphics",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let gpus: Vec<GpuInfo> = stdout
                    .lines()
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.splitn(3, ',').collect();
                        if parts.len() >= 2 {
                            let name = parts[0].trim().to_string();
                            let vram_mb: u64 = parts[1].trim().parse().ok()?;
                            let vram_gb = vram_mb as f32 / 1024.0;
                            // Estimate FP16 TFLOPS from VRAM tier and max clock
                            let tflops = if parts.len() >= 3 {
                                let clock_mhz: f32 =
                                    parts[2].trim().parse().unwrap_or(1500.0);
                                vram_gb * (clock_mhz / 1000.0) * 1.5
                            } else {
                                vram_gb * 3.0
                            };
                            Some(GpuInfo {
                                name,
                                vram_bytes: vram_mb * 1024 * 1024,
                                tflops,
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
    }

    // Report Metal on macOS/iOS when built with metal support
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    {
        let chip = detect_apple_chip().unwrap_or_else(|| format!("Apple ({})", std::env::consts::ARCH));
        let vram_bytes = detect_system_memory();
        let tflops = vram_bytes as f32 / (1024.0 * 1024.0 * 1024.0) * 0.4;
        return vec![GpuInfo {
            name: chip,
            vram_bytes,
            tflops,
        }];
    }

    // Fallback: CPU with system RAM
    #[allow(unreachable_code)]
    {
        let name = format!("CPU ({})", std::env::consts::ARCH);
        let vram_bytes = detect_system_memory();
        vec![GpuInfo {
            name,
            vram_bytes,
            tflops: 2.0,
        }]
    }
}

/// Detect the system hostname.
pub fn detect_hostname() -> String {
    if let Ok(output) = std::process::Command::new("hostname").output() {
        if output.status.success() {
            let h = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !h.is_empty() {
                return h;
            }
        }
    }
    "unknown".to_string()
}

/// Detect the compute backend description for this node.
pub fn detect_backend() -> String {
    #[cfg(feature = "cuda")]
    {
        if let Some(ver) = detect_cuda_version() {
            return ver;
        }
    }
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    {
        if let Some(chip) = detect_apple_chip() {
            return chip;
        }
        return "Metal".to_string();
    }
    #[allow(unreachable_code)]
    "CPU".to_string()
}

/// Detect the CUDA toolkit version via nvcc.
pub fn detect_cuda_version() -> Option<String> {
    // Try nvcc from PATH first.
    let output = std::process::Command::new("nvcc")
        .arg("--version")
        .output();

    // On Windows, nvcc may not be in PATH but CUDA_PATH is always set by the installer.
    // Only fall back to CUDA_PATH if the PATH lookup failed.
    #[cfg(target_os = "windows")]
    let output = output.or_else(|_| {
        let cuda_path = std::env::var("CUDA_PATH")
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "CUDA_PATH not set"))?;
        let nvcc_path = std::path::PathBuf::from(&cuda_path)
            .join("bin")
            .join("nvcc.exe");
        std::process::Command::new(nvcc_path)
            .arg("--version")
            .output()
    });

    let output = output.ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Look for "release X.Y" in output like "Cuda compilation tools, release 12.4, V12.4.131"
    for line in stdout.lines() {
        if let Some(idx) = line.find("release ") {
            let rest = &line[idx + 8..];
            let ver = rest.split(',').next().unwrap_or(rest).trim();
            if !ver.is_empty() {
                return Some(format!("CUDA {}", ver));
            }
        }
    }
    None
}

/// Detect the Apple chip model (e.g. "Apple M2 Max" on macOS, "iPad8,3" on iOS).
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn detect_apple_chip() -> Option<String> {
    // Try machdep.cpu.brand_string first (macOS), then hw.machine (iOS).
    for key in &["machdep.cpu.brand_string", "hw.machine"] {
        if let Some(val) = sysctl_string(key) {
            return Some(val);
        }
    }
    None
}

/// Read a sysctl string value using the C API (works in iOS sandbox unlike subprocess).
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_string(name: &str) -> Option<String> {
    use std::ffi::CString;
    let c_name = CString::new(name).ok()?;
    let mut len: usize = 0;
    // First call to get buffer size
    let ret = unsafe {
        libc::sysctlbyname(c_name.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0)
    };
    if ret != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    let ret = unsafe {
        libc::sysctlbyname(c_name.as_ptr(), buf.as_mut_ptr() as *mut _, &mut len, std::ptr::null_mut(), 0)
    };
    if ret != 0 {
        return None;
    }
    // Strip trailing null bytes
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8(buf).ok().filter(|s| !s.is_empty())
}

/// Read a sysctl u64 value using the C API.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_u64(name: &str) -> Option<u64> {
    use std::ffi::CString;
    let c_name = CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(c_name.as_ptr(), &mut value as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0)
    };
    if ret == 0 && value > 0 {
        Some(value)
    } else {
        None
    }
}

/// Detect total system memory in bytes.
fn detect_system_memory() -> u64 {
    // On macOS/iOS, use sysctl C API (works in iOS sandbox)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if let Some(bytes) = sysctl_u64("hw.memsize") {
            return bytes;
        }
    }

    // On Windows, get total physical memory.
    // Try PowerShell/CIM first (wmic is deprecated and removed in some Win11 builds).
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command",
                   "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(bytes) = stdout.trim().parse::<u64>() {
                    return bytes;
                }
            }
        }
    }

    // On Linux and Android, read /proc/meminfo
    // Note: Android uses target_os = "android", not "linux", so both are listed.
    #[cfg(any(target_os = "linux", target_os = "android"))]
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
    #[serde(default)]
    backend: String,
    #[serde(default)]
    hostname: String,
    #[serde(default)]
    os: String,
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
/// Dropping this handle signals the listener thread to exit.
pub struct DiscoveryListener {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    _handle: std::thread::JoinHandle<()>,
}

impl Drop for DiscoveryListener {
    fn drop(&mut self) {
        // Signal the listener thread to exit on its next recv_from timeout (~1s).
        self.stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
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
    let hostname = detect_hostname();
    let backend = detect_backend();
    let response = DiscoveryResponse {
        cluster_hash: hash.clone(),
        worker_name: worker_name.to_string(),
        port,
        gpus: gpus.to_vec(),
        backend,
        hostname,
        os: std::env::consts::OS.to_string(),
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

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_thread = stop.clone();

    let handle = std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        while !stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
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
                    // Normal timeout — loop and check stop flag
                }
                Err(e) => {
                    log::warn!("discovery listener error: {}", e);
                    break;
                }
            }
        }
        log::debug!("discovery listener thread exited");
    });

    Ok(DiscoveryListener { stop, _handle: handle })
}

// ── Interface enumeration ──────────────────────────────────────────────────

/// Get directed broadcast addresses for all local IPv4 interfaces.
/// Falls back to 255.255.255.255 if enumeration fails.
fn get_broadcast_addresses() -> Vec<Ipv4Addr> {
    let mut addrs = Vec::new();

    // Parse `ip addr` on Linux or `ifconfig` on macOS to find broadcast addresses
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("ip")
            .args(["-4", "addr", "show"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                // Lines like: "    inet 192.168.50.199/24 brd 192.168.50.255 scope global ..."
                if let Some(brd_idx) = line.find("brd ") {
                    let rest = &line[brd_idx + 4..];
                    if let Some(end) = rest.find(' ') {
                        if let Ok(ip) = rest[..end].parse::<Ipv4Addr>() {
                            if !ip.is_loopback() {
                                addrs.push(ip);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("ifconfig").output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                // Lines like: "	inet 192.168.50.32 netmask 0xffffff00 broadcast 192.168.50.255"
                if let Some(brd_idx) = line.find("broadcast ") {
                    let rest = &line[brd_idx + 10..];
                    let addr_str = rest.split_whitespace().next().unwrap_or("");
                    if let Ok(ip) = addr_str.parse::<Ipv4Addr>() {
                        if !ip.is_loopback() {
                            addrs.push(ip);
                        }
                    }
                }
            }
        }
    }

    // On Windows, parse `ipconfig` to compute broadcast addresses from IP + subnet mask.
    // ipconfig groups lines under adapter headers (non-indented). Indented lines contain
    // the IPv4 address and subnet mask. We reset last_ip on adapter boundaries so a
    // stale IP from a previous adapter can't pair with the wrong subnet mask.
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("ipconfig").output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut last_ip: Option<Ipv4Addr> = None;
            for line in stdout.lines() {
                // Adapter headers have no leading whitespace; detail lines are indented.
                // Reset state on adapter boundaries to prevent cross-adapter mispairing.
                if !line.starts_with(' ') && !line.starts_with('\t') {
                    last_ip = None;
                }

                let trimmed = line.trim();
                // Match lines containing an IPv4 address value (x.x.x.x after a colon).
                // This is locale-independent: we look for ": <ipv4>" on any line and
                // distinguish address vs mask by checking if it looks like a mask
                // (starts with 255.).
                if let Some(colon_idx) = trimmed.rfind(':') {
                    let value = trimmed[colon_idx + 1..].trim();
                    if let Ok(ip) = value.parse::<Ipv4Addr>() {
                        let octets = ip.octets();
                        if octets[0] == 255 {
                            // This is a subnet mask — pair with last_ip
                            if let Some(addr) = last_ip.take() {
                                let ip_bits = u32::from(addr);
                                let mask_bits = u32::from(ip);
                                let brd = (ip_bits & mask_bits) | (!mask_bits);
                                let brd_addr = Ipv4Addr::from(brd);
                                if !brd_addr.is_loopback() {
                                    addrs.push(brd_addr);
                                }
                            }
                        } else if !ip.is_loopback() {
                            // This is an IPv4 address
                            last_ip = Some(ip);
                        }
                    }
                }
            }
        }
    }

    // Always include the limited broadcast as a fallback
    addrs.push(Ipv4Addr::BROADCAST);
    addrs.dedup();
    addrs
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

        // Collect broadcast addresses: directed subnet broadcasts are more
        // reliable than 255.255.255.255 which may not cross interfaces.
        let broadcast_addrs = get_broadcast_addresses();

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

            // Send periodic broadcast queries to all known broadcast addresses
            if now.duration_since(last_query) >= query_interval {
                for addr in &broadcast_addrs {
                    let dest = SocketAddr::V4(SocketAddrV4::new(*addr, DISCOVERY_PORT));
                    let _ = sock.send_to(&query_pkt, dest);
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
                                        "  {} — {} (~{:.1} TFLOPS)",
                                        &gpu.name,
                                        human_bytes::human_bytes(gpu.vram_bytes as f64),
                                        gpu.tflops
                                    );
                                }

                                workers.insert(resp.worker_name.clone(), DiscoveredWorker {
                                    name: resp.worker_name,
                                    host,
                                    port: resp.port,
                                    gpus: resp.gpus,
                                    backend: resp.backend,
                                    hostname: resp.hostname,
                                    os: resp.os,
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
