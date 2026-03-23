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
    pub fn max_layers_for_size(&self, layer_size_bytes: u64) -> usize {
        super::max_layers_for_gpus(&self.gpus, layer_size_bytes)
    }

    /// Total estimated TFLOPS across all GPUs.
    pub fn total_tflops(&self) -> f64 {
        super::estimate_tflops_for_gpus(&self.gpus)
    }
}

impl super::WorkerCapacity for DiscoveredWorker {
    fn name(&self) -> &str {
        &self.name
    }
    fn total_vram(&self) -> u64 {
        self.total_vram()
    }
    fn total_tflops(&self) -> f64 {
        self.total_tflops()
    }
    fn max_layers_for_size(&self, layer_size_bytes: u64) -> usize {
        self.max_layers_for_size(layer_size_bytes)
    }
}

/// Compute the first 8 hex chars of SHA-256(cluster_key) for filtering.
pub fn cluster_hash(cluster_key: &str) -> String {
    let result = Sha256::digest(cluster_key.as_bytes());
    // Manual hex encode of first 4 bytes avoids heap allocation from hex::encode
    let mut s = String::with_capacity(8);
    for &b in &result[..4] {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
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
    #[cfg(unix)]
    {
        let mut buf = [0u8; 256];
        let ret =
            unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
        if ret == 0 {
            let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            if let Ok(name) = std::str::from_utf8(&buf[..len]) {
                if !name.is_empty() {
                    return name.to_string();
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = std::process::Command::new("hostname").output() {
            if output.status.success() {
                let h = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !h.is_empty() {
                    return h;
                }
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

/// Detect the CUDA toolkit version by reading version files from the CUDA install directory.
///
/// Checks `$CUDA_HOME`, `$CUDA_PATH`, and `/usr/local/cuda` for `version.json` (CUDA 11+)
/// or `version.txt` (older). No subprocess spawning required.
pub fn detect_cuda_version() -> Option<String> {
    let cuda_home = std::env::var("CUDA_HOME").ok();
    let cuda_path = std::env::var("CUDA_PATH").ok();

    let candidates: Vec<std::path::PathBuf> = cuda_home
        .iter()
        .chain(cuda_path.iter())
        .map(std::path::PathBuf::from)
        .chain(std::iter::once(std::path::PathBuf::from("/usr/local/cuda")))
        .collect();

    detect_cuda_version_from_dirs(&candidates)
}

/// Search for CUDA version info in the given directories (testable inner function).
fn detect_cuda_version_from_dirs(candidates: &[std::path::PathBuf]) -> Option<String> {
    for base in candidates {
        // Try version.json first (modern CUDA 11+)
        let json_path = base.join("version.json");
        if let Ok(content) = std::fs::read_to_string(&json_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(ver) = val.pointer("/cuda/version").and_then(|v| v.as_str()) {
                    let short: String =
                        ver.splitn(3, '.').take(2).collect::<Vec<_>>().join(".");
                    return Some(format!("CUDA {}", short));
                }
            }
        }
        // Try version.txt fallback (older CUDA)
        let txt_path = base.join("version.txt");
        if let Ok(content) = std::fs::read_to_string(&txt_path) {
            if let Some(rest) = content.strip_prefix("CUDA Version ") {
                let ver = rest.trim();
                let short: String =
                    ver.splitn(3, '.').take(2).collect::<Vec<_>>().join(".");
                return Some(format!("CUDA {}", short));
            }
        }
    }

    None
}

/// Detect the Apple chip model (e.g. "Apple M2 Max" on macOS, "iPad8,3" on iOS).
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(dead_code)] // called only with feature = "metal"
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
#[allow(dead_code)]
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

pub fn encode_packet(payload: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::with_capacity(4 + payload.len());
    pkt.extend_from_slice(MAGIC);
    pkt.extend_from_slice(payload);
    pkt
}

pub fn decode_packet(data: &[u8]) -> Option<&[u8]> {
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

    // On Unix (excluding Android), use libc::getifaddrs to enumerate interfaces without
    // spawning a subprocess. getifaddrs is not available below Android API 24, and Android
    // is a mobile-client platform where broadcast discovery is not needed.
    // The broadcast address field differs by platform:
    //   Linux:   ifa_ifu   (union of ifa_broadaddr / ifa_dstaddr)
    //   macOS / iOS:  ifa_dstaddr
    #[cfg(all(unix, not(target_os = "android")))]
    {
        let mut ifaddrs_ptr: *mut libc::ifaddrs = std::ptr::null_mut();
        if unsafe { libc::getifaddrs(&mut ifaddrs_ptr) } == 0 {
            let mut cursor = ifaddrs_ptr;
            while !cursor.is_null() {
                let ifa = unsafe { &*cursor };
                if !ifa.ifa_addr.is_null() {
                    let family = unsafe { (*ifa.ifa_addr).sa_family } as i32;
                    if family == libc::AF_INET
                        && (ifa.ifa_flags & libc::IFF_BROADCAST as libc::c_uint) != 0
                    {
                        // Get the broadcast address pointer (platform-specific field name).
                        #[cfg(target_os = "linux")]
                        let brd_ptr = ifa.ifa_ifu;
                        #[cfg(not(target_os = "linux"))]
                        let brd_ptr = ifa.ifa_dstaddr;

                        if !brd_ptr.is_null() {
                            let brd_sa =
                                unsafe { &*(brd_ptr as *const libc::sockaddr_in) };
                            let ip = Ipv4Addr::from(u32::from_be(brd_sa.sin_addr.s_addr));
                            if !ip.is_loopback() && !addrs.contains(&ip) {
                                addrs.push(ip);
                            }
                        }
                    }
                }
                cursor = unsafe { (*cursor).ifa_next };
            }
            unsafe { libc::freeifaddrs(ifaddrs_ptr) };
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
/// Sends periodic UDP broadcast queries, collects responses until `timeout`.
/// If `min_workers` is non-zero, stops as soon as that many distinct workers
/// have been discovered (even if time remains).
pub async fn discover_workers(
    cluster_key: &str,
    timeout: Duration,
    min_workers: usize,
) -> Result<Vec<DiscoveredWorker>> {
    let expected_hash = cluster_hash(cluster_key);

    if min_workers > 0 {
        log::info!(
            "discovering workers (timeout: {}s, stop at {} worker(s))...",
            timeout.as_secs(),
            min_workers,
        );
    } else {
        log::info!(
            "discovering workers (timeout: {}s)...",
            timeout.as_secs()
        );
    }

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

                                if min_workers > 0 && workers.len() >= min_workers {
                                    log::info!(
                                        "reached min-workers ({}), stopping discovery early",
                                        min_workers
                                    );
                                    break;
                                }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── cluster_hash ─────────────────────────────────────────────

    #[test]
    fn test_cluster_hash_deterministic() {
        let h1 = cluster_hash("my-secret");
        let h2 = cluster_hash("my-secret");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_cluster_hash_different_keys() {
        let h1 = cluster_hash("key-a");
        let h2 = cluster_hash("key-b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cluster_hash_length() {
        // First 4 bytes of SHA-256 = 8 hex chars
        let h = cluster_hash("anything");
        assert_eq!(h.len(), 8);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_cluster_hash_known_value() {
        // SHA-256("test") = 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
        // First 4 bytes = 9f86d081, hex = "9f86d081"
        let h = cluster_hash("test");
        assert_eq!(h, "9f86d081");
    }

    #[test]
    fn test_cluster_hash_empty_key() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let h = cluster_hash("");
        assert_eq!(h, "e3b0c442");
    }

    // ── encode_packet / decode_packet ────────────────────────────

    #[test]
    fn test_packet_roundtrip() {
        let payload = b"hello world";
        let pkt = encode_packet(payload);
        let decoded = decode_packet(&pkt);
        assert_eq!(decoded, Some(&payload[..]));
    }

    #[test]
    fn test_decode_packet_too_short() {
        // 4 bytes of magic only, no payload — len == 4, not > 4
        assert_eq!(decode_packet(MAGIC), None);
    }

    #[test]
    fn test_decode_packet_wrong_magic() {
        let bad = b"BADXpayload";
        assert_eq!(decode_packet(bad), None);
    }

    #[test]
    fn test_decode_packet_empty() {
        assert_eq!(decode_packet(&[]), None);
    }

    // ── DiscoveredWorker::total_vram ─────────────────────────────

    fn make_worker(gpus: Vec<GpuInfo>) -> DiscoveredWorker {
        DiscoveredWorker {
            name: "test".into(),
            host: "127.0.0.1:10128".into(),
            port: 10128,
            gpus,
            backend: "test".into(),
            hostname: "test-host".into(),
            os: "linux".into(),
        }
    }

    #[test]
    fn test_total_vram_single_gpu() {
        let w = make_worker(vec![GpuInfo {
            name: "NVIDIA RTX 3080".into(),
            vram_bytes: 16 * 1024 * 1024 * 1024,
            tflops: 30.0,
        }]);
        assert_eq!(w.total_vram(), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_total_vram_multi_gpu() {
        let w = make_worker(vec![
            GpuInfo { name: "TITAN X".into(), vram_bytes: 12 * 1024 * 1024 * 1024, tflops: 10.0 },
            GpuInfo { name: "TITAN X".into(), vram_bytes: 12 * 1024 * 1024 * 1024, tflops: 10.0 },
        ]);
        assert_eq!(w.total_vram(), 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_total_vram_empty() {
        let w = make_worker(vec![]);
        assert_eq!(w.total_vram(), 0);
    }

    // ── max_layers_for_size ──────────────────────────────────────

    #[test]
    fn test_max_layers_zero_layer_size() {
        let w = make_worker(vec![GpuInfo {
            name: "NVIDIA RTX 3080".into(),
            vram_bytes: 16 * 1024 * 1024 * 1024,
            tflops: 30.0,
        }]);
        assert_eq!(w.max_layers_for_size(0), usize::MAX);
    }

    #[test]
    fn test_max_layers_no_gpus() {
        let w = make_worker(vec![]);
        assert_eq!(w.max_layers_for_size(100_000_000), usize::MAX);
    }

    #[test]
    fn test_max_layers_dedicated_gpu() {
        // 12 GB VRAM, 768 MiB reserve (max of 5%=614 MiB and 768 MiB)
        let vram = 12u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![GpuInfo {
            name: "NVIDIA TITAN X".into(),
            vram_bytes: vram,
            tflops: 10.0,
        }]);
        let layer_size = 500u64 * 1024 * 1024; // 500 MiB per layer
        let reserve = 768u64 * 1024 * 1024;
        let usable = vram - reserve;
        let expected = (usable / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    #[test]
    fn test_max_layers_large_dedicated_gpu() {
        // 24 GB VRAM — 5% = 1.2 GiB > 768 MiB, so 5% used
        let vram = 24u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![GpuInfo {
            name: "NVIDIA RTX 4090".into(),
            vram_bytes: vram,
            tflops: 80.0,
        }]);
        let layer_size = 1024u64 * 1024 * 1024; // 1 GiB per layer
        let pct_reserve = (vram as f64 * 0.05) as u64;
        let usable = vram - pct_reserve;
        let expected = (usable / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    #[test]
    fn test_max_layers_apple_unified() {
        // 36 GB unified — 28% = 10.08 GiB > 6 GiB min, so 28% used
        let vram = 36u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![GpuInfo {
            name: "Apple M3 Pro".into(),
            vram_bytes: vram,
            tflops: 14.0,
        }]);
        let layer_size = 1024u64 * 1024 * 1024;
        let pct_reserve = (vram as f64 * 0.28) as u64;
        let usable = vram - pct_reserve;
        let expected = (usable / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    #[test]
    fn test_max_layers_apple_small_memory() {
        // 8 GB unified — 28% = 2.24 GiB < 6 GiB min, so 6 GiB used
        let vram = 8u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![GpuInfo {
            name: "Apple M1".into(),
            vram_bytes: vram,
            tflops: 3.0,
        }]);
        let layer_size = 500u64 * 1024 * 1024;
        let min_reserve = 6u64 * 1024 * 1024 * 1024;
        let usable = vram - min_reserve;
        let expected = (usable / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    #[test]
    fn test_max_layers_cpu() {
        // CPU device: 20% reserve
        let vram = 16u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![GpuInfo {
            name: "CPU (aarch64)".into(),
            vram_bytes: vram,
            tflops: 2.0,
        }]);
        let layer_size = 500u64 * 1024 * 1024;
        let reserve = (vram as f64 * 0.20) as u64;
        let usable = vram - reserve;
        let expected = (usable / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    #[test]
    fn test_max_layers_multi_gpu_sums() {
        // Two 12 GB GPUs — each contributes independently
        let vram = 12u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![
            GpuInfo { name: "NVIDIA TITAN X".into(), vram_bytes: vram, tflops: 10.0 },
            GpuInfo { name: "NVIDIA TITAN X".into(), vram_bytes: vram, tflops: 10.0 },
        ]);
        let layer_size = 500u64 * 1024 * 1024;
        let reserve = 768u64 * 1024 * 1024;
        let usable_per = vram - reserve;
        let expected = 2 * (usable_per / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    // ── total_tflops ─────────────────────────────────────────────

    #[test]
    fn test_total_tflops_reported() {
        let w = make_worker(vec![
            GpuInfo { name: "RTX 3080".into(), vram_bytes: 16 * 1024 * 1024 * 1024, tflops: 30.0 },
        ]);
        assert!((w.total_tflops() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_fallback_nvidia() {
        let w = make_worker(vec![GpuInfo {
            name: "NVIDIA GeForce RTX 3080".into(),
            vram_bytes: 16 * 1024 * 1024 * 1024,
            tflops: 0.0, // old binary, no tflops reported
        }]);
        // Fallback: vram_gb * 3.0 = 16 * 3 = 48
        assert!((w.total_tflops() - 48.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_fallback_apple() {
        let w = make_worker(vec![GpuInfo {
            name: "Apple M3 Pro".into(),
            vram_bytes: 36 * 1024 * 1024 * 1024,
            tflops: 0.0,
        }]);
        // Fallback: vram_gb * 0.4 = 36 * 0.4 = 14.4
        assert!((w.total_tflops() - 14.4).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_fallback_cpu() {
        let w = make_worker(vec![GpuInfo {
            name: "unknown device".into(),
            vram_bytes: 32 * 1024 * 1024 * 1024,
            tflops: 0.0,
        }]);
        // Fallback: 2.0 flat for unknown
        assert!((w.total_tflops() - 2.0).abs() < 0.01);
    }

    // ── encode_packet / decode_packet: additional edge cases ────

    #[test]
    fn test_packet_roundtrip_empty_payload() {
        // Empty payload: MAGIC + 0 bytes of payload = 4 bytes total
        // decode requires len > 4, so empty payload should return None
        let pkt = encode_packet(&[]);
        assert_eq!(pkt.len(), 4);
        assert_eq!(decode_packet(&pkt), None, "exactly 4 bytes (magic only, no payload) should return None");
    }

    #[test]
    fn test_packet_roundtrip_single_byte_payload() {
        let pkt = encode_packet(&[0x42]);
        assert_eq!(pkt.len(), 5);
        let decoded = decode_packet(&pkt);
        assert_eq!(decoded, Some(&[0x42][..]));
    }

    #[test]
    fn test_packet_roundtrip_binary_payload() {
        let payload: Vec<u8> = (0..=255).collect();
        let pkt = encode_packet(&payload);
        assert_eq!(pkt.len(), 4 + 256);
        let decoded = decode_packet(&pkt).unwrap();
        assert_eq!(decoded, payload.as_slice());
    }

    #[test]
    fn test_decode_packet_partial_magic() {
        // Only first 3 bytes of magic
        assert_eq!(decode_packet(b"CAK"), None);
        assert_eq!(decode_packet(b"CA"), None);
        assert_eq!(decode_packet(b"C"), None);
    }

    #[test]
    fn test_encode_packet_preserves_magic() {
        let pkt = encode_packet(b"test");
        assert_eq!(&pkt[..4], b"CAKE");
        assert_eq!(&pkt[4..], b"test");
    }

    #[test]
    fn test_decode_packet_case_sensitive_magic() {
        // "cake" (lowercase) should not match "CAKE"
        let mut pkt = vec![b'c', b'a', b'k', b'e', 0x01];
        assert_eq!(decode_packet(&pkt), None);
        // Fix the magic
        pkt[..4].copy_from_slice(b"CAKE");
        assert!(decode_packet(&pkt).is_some());
    }

    // ── JSON roundtrip for discovery structs ────────────────────

    #[test]
    fn test_discovery_query_json_roundtrip() {
        let query = DiscoveryQuery {
            cluster_hash: "abcd1234".into(),
        };
        let json = serde_json::to_vec(&query).unwrap();
        let decoded: DiscoveryQuery = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.cluster_hash, "abcd1234");
    }

    #[test]
    fn test_discovery_response_json_roundtrip() {
        let resp = DiscoveryResponse {
            cluster_hash: "9f86d081".into(),
            worker_name: "worker-1".into(),
            port: 10128,
            gpus: vec![GpuInfo {
                name: "NVIDIA RTX 3080".into(),
                vram_bytes: 16 * 1024 * 1024 * 1024,
                tflops: 30.0,
            }],
            backend: "CUDA 12.4".into(),
            hostname: "blade".into(),
            os: "linux".into(),
        };
        let json = serde_json::to_vec(&resp).unwrap();
        let decoded: DiscoveryResponse = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.cluster_hash, "9f86d081");
        assert_eq!(decoded.worker_name, "worker-1");
        assert_eq!(decoded.port, 10128);
        assert_eq!(decoded.gpus.len(), 1);
        assert_eq!(decoded.gpus[0].name, "NVIDIA RTX 3080");
        assert_eq!(decoded.gpus[0].vram_bytes, 16 * 1024 * 1024 * 1024);
        assert!((decoded.gpus[0].tflops - 30.0).abs() < 0.01);
        assert_eq!(decoded.backend, "CUDA 12.4");
        assert_eq!(decoded.hostname, "blade");
        assert_eq!(decoded.os, "linux");
    }

    #[test]
    fn test_discovery_response_defaults_for_old_fields() {
        // Simulate response from an old binary that doesn't send backend/hostname/os
        let json = r#"{
            "cluster_hash": "abcd1234",
            "worker_name": "old-worker",
            "port": 10128,
            "gpus": []
        }"#;
        let resp: DiscoveryResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.backend, "");
        assert_eq!(resp.hostname, "");
        assert_eq!(resp.os, "");
    }

    #[test]
    fn test_discovery_response_multi_gpu() {
        let resp = DiscoveryResponse {
            cluster_hash: "hash".into(),
            worker_name: "multi-gpu".into(),
            port: 10128,
            gpus: vec![
                GpuInfo { name: "TITAN X".into(), vram_bytes: 12 * 1024 * 1024 * 1024, tflops: 10.0 },
                GpuInfo { name: "TITAN X".into(), vram_bytes: 12 * 1024 * 1024 * 1024, tflops: 10.0 },
            ],
            backend: "CUDA 12.4".into(),
            hostname: "bahamut".into(),
            os: "linux".into(),
        };
        let json = serde_json::to_vec(&resp).unwrap();
        let decoded: DiscoveryResponse = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.gpus.len(), 2);
    }

    // ── Full packet encode/decode with JSON payload ─────────────

    #[test]
    fn test_full_query_packet_roundtrip() {
        let query = DiscoveryQuery {
            cluster_hash: cluster_hash("my-key"),
        };
        let json = serde_json::to_vec(&query).unwrap();
        let pkt = encode_packet(&json);

        let payload = decode_packet(&pkt).unwrap();
        let decoded: DiscoveryQuery = serde_json::from_slice(payload).unwrap();
        assert_eq!(decoded.cluster_hash, cluster_hash("my-key"));
    }

    #[test]
    fn test_full_response_packet_roundtrip() {
        let resp = DiscoveryResponse {
            cluster_hash: cluster_hash("secret"),
            worker_name: "w1".into(),
            port: 9999,
            gpus: vec![GpuInfo { name: "CPU (x86_64)".into(), vram_bytes: 8_000_000_000, tflops: 2.0 }],
            backend: "CPU".into(),
            hostname: "test-host".into(),
            os: "linux".into(),
        };
        let json = serde_json::to_vec(&resp).unwrap();
        let pkt = encode_packet(&json);

        let payload = decode_packet(&pkt).unwrap();
        let decoded: DiscoveryResponse = serde_json::from_slice(payload).unwrap();
        assert_eq!(decoded.worker_name, "w1");
        assert_eq!(decoded.port, 9999);
    }

    // ── detect_gpus (CPU fallback path) ─────────────────────────

    #[test]
    fn test_detect_gpus_returns_nonempty() {
        // Without cuda/metal features in test builds, should fall back to CPU
        let gpus = detect_gpus();
        assert!(!gpus.is_empty(), "detect_gpus should always return at least one device");
    }

    #[test]
    fn test_detect_gpus_cpu_fallback_has_positive_vram() {
        let gpus = detect_gpus();
        for gpu in &gpus {
            assert!(gpu.vram_bytes > 0, "detected GPU should report positive VRAM: {}", gpu.name);
        }
    }

    #[test]
    fn test_detect_gpus_cpu_fallback_has_name() {
        let gpus = detect_gpus();
        for gpu in &gpus {
            assert!(!gpu.name.is_empty(), "GPU name should not be empty");
        }
    }

    // ── detect_hostname ─────────────────────────────────────────

    #[test]
    fn test_detect_hostname_nonempty() {
        let hostname = detect_hostname();
        assert!(!hostname.is_empty());
        assert_ne!(hostname, "unknown", "should detect a real hostname in test environment");
    }

    // ── detect_backend ──────────────────────────────────────────

    #[test]
    fn test_detect_backend_nonempty() {
        let backend = detect_backend();
        assert!(!backend.is_empty());
    }

    // ── GpuInfo serialization ───────────────────────────────────

    #[test]
    fn test_gpu_info_json_roundtrip() {
        let gpu = GpuInfo {
            name: "Test GPU".into(),
            vram_bytes: 1024 * 1024 * 1024,
            tflops: 15.5,
        };
        let json = serde_json::to_string(&gpu).unwrap();
        let decoded: GpuInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Test GPU");
        assert_eq!(decoded.vram_bytes, 1024 * 1024 * 1024);
        assert!((decoded.tflops - 15.5).abs() < 0.01);
    }

    #[test]
    fn test_gpu_info_tflops_default() {
        // Old JSON without tflops field should default to 0.0
        let json = r#"{"name": "Old GPU", "vram_bytes": 1000000}"#;
        let gpu: GpuInfo = serde_json::from_str(json).unwrap();
        assert!((gpu.tflops - 0.0).abs() < 0.01);
    }

    // ── total_tflops: multi-GPU and edge cases ──────────────────

    #[test]
    fn test_total_tflops_multi_gpu_reported() {
        let w = make_worker(vec![
            GpuInfo { name: "GPU A".into(), vram_bytes: 12 * 1024 * 1024 * 1024, tflops: 10.0 },
            GpuInfo { name: "GPU B".into(), vram_bytes: 12 * 1024 * 1024 * 1024, tflops: 15.0 },
        ]);
        assert!((w.total_tflops() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_empty_gpus() {
        let w = make_worker(vec![]);
        assert!((w.total_tflops() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_mixed_reported_and_zero() {
        // If any GPU reports non-zero tflops, the sum of reported values is used
        let w = make_worker(vec![
            GpuInfo { name: "RTX 3080".into(), vram_bytes: 16 * 1024 * 1024 * 1024, tflops: 30.0 },
            GpuInfo { name: "CPU".into(), vram_bytes: 8 * 1024 * 1024 * 1024, tflops: 0.0 },
        ]);
        // total reported = 30.0 > 0 => returns 30.0 (not fallback)
        assert!((w.total_tflops() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_fallback_geforce() {
        let w = make_worker(vec![GpuInfo {
            name: "GeForce RTX 2080".into(),
            vram_bytes: 8 * 1024 * 1024 * 1024,
            tflops: 0.0,
        }]);
        // Fallback: 8 * 3.0 = 24
        assert!((w.total_tflops() - 24.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_fallback_tesla() {
        let w = make_worker(vec![GpuInfo {
            name: "Tesla V100".into(),
            vram_bytes: 16 * 1024 * 1024 * 1024,
            tflops: 0.0,
        }]);
        // Fallback: 16 * 3.0 = 48
        assert!((w.total_tflops() - 48.0).abs() < 0.01);
    }

    #[test]
    fn test_total_tflops_fallback_silicon() {
        let w = make_worker(vec![GpuInfo {
            name: "Apple Silicon M2".into(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            tflops: 0.0,
        }]);
        // Fallback: 24 * 0.4 = 9.6
        assert!((w.total_tflops() - 9.6).abs() < 0.01);
    }

    // ── max_layers_for_size: additional edge cases ──────────────

    #[test]
    fn test_max_layers_layer_larger_than_vram() {
        // Layer bigger than usable VRAM: should return 0
        let vram = 1024u64 * 1024 * 1024; // 1 GiB
        let w = make_worker(vec![GpuInfo {
            name: "NVIDIA Small GPU".into(),
            vram_bytes: vram,
            tflops: 5.0,
        }]);
        // After 768 MiB reserve, only ~256 MiB usable, but layer is 2 GiB
        let layer_size = 2u64 * 1024 * 1024 * 1024;
        assert_eq!(w.max_layers_for_size(layer_size), 0);
    }

    #[test]
    fn test_max_layers_apple_tiny_vram_below_min_reserve() {
        // Apple device with less VRAM than the 6 GiB minimum reserve
        let vram = 4u64 * 1024 * 1024 * 1024; // 4 GiB
        let w = make_worker(vec![GpuInfo {
            name: "Apple M1 (4GB)".into(),
            vram_bytes: vram,
            tflops: 1.0,
        }]);
        let layer_size = 100u64 * 1024 * 1024;
        // min_reserve = 6 GiB > vram, so usable = saturating_sub = 0
        assert_eq!(w.max_layers_for_size(layer_size), 0);
    }

    #[test]
    fn test_max_layers_cpu_small_memory() {
        // CPU with 2 GiB (mobile-like)
        let vram = 2u64 * 1024 * 1024 * 1024;
        let w = make_worker(vec![GpuInfo {
            name: "CPU (aarch64)".into(),
            vram_bytes: vram,
            tflops: 1.0,
        }]);
        let layer_size = 100u64 * 1024 * 1024;
        // 20% reserve = 409 MiB, usable = ~1638 MiB
        let reserve = (vram as f64 * 0.20) as u64;
        let usable = vram - reserve;
        let expected = (usable / layer_size) as usize;
        assert_eq!(w.max_layers_for_size(layer_size), expected);
    }

    // ── DiscoveredWorker construction ───────────────────────────

    #[test]
    fn test_discovered_worker_total_vram_consistency() {
        let w = make_worker(vec![
            GpuInfo { name: "A".into(), vram_bytes: 100, tflops: 1.0 },
            GpuInfo { name: "B".into(), vram_bytes: 200, tflops: 2.0 },
            GpuInfo { name: "C".into(), vram_bytes: 300, tflops: 3.0 },
        ]);
        assert_eq!(w.total_vram(), 600);
    }

    // ── cluster_hash: unicode and special characters ────────────

    #[test]
    fn test_cluster_hash_unicode_key() {
        let h = cluster_hash("schluessel-\u{00FC}ber-alles");
        assert_eq!(h.len(), 8);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_cluster_hash_long_key() {
        let long_key = "a".repeat(10000);
        let h = cluster_hash(&long_key);
        assert_eq!(h.len(), 8);
    }

    #[test]
    fn test_cluster_hash_whitespace_matters() {
        let h1 = cluster_hash("key");
        let h2 = cluster_hash("key ");
        assert_ne!(h1, h2);
    }

    // ── detect_hostname (additional) ──────────────────────────

    #[test]
    fn test_detect_hostname_no_null_bytes() {
        let h = detect_hostname();
        assert!(!h.contains('\0'));
    }

    #[test]
    fn test_detect_hostname_valid_utf8() {
        let h = detect_hostname();
        assert!(h.chars().all(|c| c.is_ascii_alphanumeric()
            || c == '-'
            || c == '.'
            || c == '_'));
    }

    // ── detect_cuda_version ─────────────────────────────────────

    #[test]
    fn test_detect_cuda_version_from_version_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("version.json"),
            r#"{"cuda": {"name": "CUDA SDK", "version": "12.4.1"}}"#,
        )
        .unwrap();
        let dirs = vec![dir.path().to_path_buf()];
        assert_eq!(
            detect_cuda_version_from_dirs(&dirs),
            Some("CUDA 12.4".to_string())
        );
    }

    #[test]
    fn test_detect_cuda_version_from_version_txt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("version.txt"), "CUDA Version 11.8.0\n").unwrap();
        let dirs = vec![dir.path().to_path_buf()];
        assert_eq!(
            detect_cuda_version_from_dirs(&dirs),
            Some("CUDA 11.8".to_string())
        );
    }

    #[test]
    fn test_detect_cuda_version_missing_dir() {
        let dirs = vec![std::path::PathBuf::from("/nonexistent/path/cuda-99.9")];
        assert_eq!(detect_cuda_version_from_dirs(&dirs), None);
    }

    #[test]
    fn test_detect_cuda_version_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("version.json"), "not valid json {{{").unwrap();
        let dirs = vec![dir.path().to_path_buf()];
        // Should not panic, returns None
        assert_eq!(detect_cuda_version_from_dirs(&dirs), None);
    }

    #[test]
    fn test_detect_cuda_version_json_preferred_over_txt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("version.json"),
            r#"{"cuda": {"version": "12.6.0"}}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("version.txt"), "CUDA Version 11.0.0\n").unwrap();
        let dirs = vec![dir.path().to_path_buf()];
        // version.json should take priority
        assert_eq!(
            detect_cuda_version_from_dirs(&dirs),
            Some("CUDA 12.6".to_string())
        );
    }

    // ── get_broadcast_addresses ─────────────────────────────────

    #[test]
    fn test_broadcast_addresses_nonempty() {
        let addrs = get_broadcast_addresses();
        assert!(!addrs.is_empty());
    }

    #[test]
    fn test_broadcast_addresses_contains_fallback() {
        let addrs = get_broadcast_addresses();
        assert!(addrs.contains(&Ipv4Addr::BROADCAST));
    }

    #[test]
    fn test_broadcast_addresses_no_loopback() {
        let addrs = get_broadcast_addresses();
        for addr in &addrs {
            if *addr != Ipv4Addr::BROADCAST {
                assert!(
                    !addr.is_loopback(),
                    "broadcast list should not contain loopback address: {}",
                    addr
                );
            }
        }
    }
}
