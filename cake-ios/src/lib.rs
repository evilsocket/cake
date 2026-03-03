//! This is a small library that wraps cake-core and exposes it as an API to the Swift side of things on iOS.
uniffi::setup_scaffolding!();

use std::io::Write;
use std::sync::OnceLock;

use cake_core::{
    cake::{self, Context, Mode, Topology, Worker},
    Args, ModelType, TextModelArch,
};

/// Check if this device's GPU supports simdgroup_matrix operations (requires A13+/M1+).
/// Returns false for A12X and older chips where candle Metal kernels will fail.
fn metal_supports_simdgroup_matrix() -> bool {
    // Query hw.machine to get device identifier (e.g. "iPad8,3", "iPhone12,1")
    let machine = {
        let name = std::ffi::CString::new("hw.machine").unwrap();
        let mut size: libc::size_t = 0;
        unsafe {
            libc::sysctlbyname(name.as_ptr(), std::ptr::null_mut(), &mut size, std::ptr::null_mut(), 0);
        }
        if size == 0 {
            return false;
        }
        let mut buf = vec![0u8; size];
        let ret = unsafe {
            libc::sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as *mut _, &mut size, std::ptr::null_mut(), 0)
        };
        if ret != 0 {
            return false;
        }
        // Remove null terminator
        if let Some(pos) = buf.iter().position(|&b| b == 0) {
            buf.truncate(pos);
        }
        String::from_utf8_lossy(&buf).to_string()
    };

    log_ios(&format!("[cake-ios] hw.machine: {}", machine));

    // Parse device identifier like "iPad8,3" → family="iPad", generation=8
    // simdgroup_matrix requires Apple GPU Family 6+ (A13 Bionic or later)
    //
    // iPhone: iPhone12,x = A13, iPhone13,x = A14, iPhone14,x = A15, etc.
    // iPad:   iPad8,x = A12X, iPad11,x = A12, iPad13,x = M1, iPad14,x = M2
    //         iPad12,x = A14, iPad16,x = M4
    // iPod:   all A12 or older
    let alpha_end = machine.find(|c: char| !c.is_alphabetic()).unwrap_or(machine.len());
    let family = &machine[..alpha_end];
    let rest = &machine[alpha_end..];
    let generation: u32 = rest.split(',').next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let supported = match family {
        "iPhone" => generation >= 12,  // iPhone12,x = A13
        "iPad" => {
            // iPad8,x = A12X (NOT supported)
            // iPad11,x = A12 (NOT supported)
            // iPad12,x = A14 (supported)
            // iPad13,x = M1 (supported)
            // iPad14,x = M2 (supported)
            // iPad16,x = M4 (supported)
            generation >= 12
        }
        _ => false,
    };

    log_ios(&format!("[cake-ios] Metal simdgroup_matrix support: {} (family={}, gen={})", supported, family, generation));
    supported
}

/// Path to the iOS log file for debugging (readable via device file access).
static LOG_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

/// Log a message to both stderr and the iOS log file.
fn log_ios(msg: &str) {
    eprintln!("{}", msg);
    if let Some(path) = LOG_PATH.get() {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", msg);
        }
    }
}

/// Start a worker node that joins a cluster via discovery.
/// Returns an error string if startup fails, or empty string on clean exit.
///
/// - `name`: Worker name (e.g. device name)
/// - `model`: HuggingFace model ID (e.g. "Qwen/Qwen2.5-Coder-1.5B-Instruct")
/// - `cluster_key`: Shared secret for cluster discovery and authentication
#[uniffi::export]
pub fn start_worker(name: String, model: String, cluster_key: String) -> String {
    // Use try_init to avoid panic if called multiple times
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    // iOS sandbox: set HOME and HF cache to writable tmp directories.
    let tmp = std::env::temp_dir();

    // Initialize log file (fresh each run)
    let log_path = tmp.join("cake-worker.log");
    let _ = std::fs::remove_file(&log_path);
    let _ = LOG_PATH.set(log_path);

    log_ios("[cake-ios] start_worker called");
    log_ios(&format!("[cake-ios]   name: {name}"));
    log_ios(&format!("[cake-ios]   model: {model}"));
    log_ios(&format!("[cake-ios]   cluster_key: {}", if cluster_key.is_empty() { "(none)" } else { "(set)" }));

    let ios_home = tmp.join("cake-home");
    let hf_cache = tmp.join("huggingface").join("hub");
    let cake_cache = tmp.join("cake");
    for dir in [&ios_home, &hf_cache, &cake_cache] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            let msg = format!("failed to create dir {}: {}", dir.display(), e);
            log_ios(&format!("[cake-ios] {msg}"));
            return msg;
        }
    }
    std::env::set_var("HOME", ios_home.to_string_lossy().as_ref());
    std::env::set_var("HF_HUB_CACHE", hf_cache.to_string_lossy().as_ref());
    log_ios(&format!("[cake-ios] HOME={}", ios_home.display()));
    log_ios(&format!("[cake-ios] HF_HUB_CACHE={}", hf_cache.display()));

    let address = "0.0.0.0:10128".to_string();
    let use_cluster = !cluster_key.is_empty();

    log_ios("[cake-ios] starting tokio runtime...");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            if use_cluster {
                // Zero-config mode: wait for master to discover us and push model data.
                log_ios("[cake-ios] zero-config mode: waiting for master discovery...");
                run_zero_config_worker(&name, &cluster_key, &address, &cake_cache, &model).await
            } else {
                // Direct mode: download model and start worker.
                log_ios("[cake-ios] direct mode: loading model...");
                run_direct_worker(&name, &model, &address).await
            }
        })
}

/// Zero-config worker: advertise via mDNS, wait for master assignment, receive model data.
async fn run_zero_config_worker(
    name: &str,
    cluster_key: &str,
    address: &str,
    cache_dir: &std::path::Path,
    _model: &str,
) -> String {
    // Wait for master to discover us and send model data
    let (layers, model_path, listener) = match cake::setup::worker_setup(
        name,
        cluster_key,
        address,
        cache_dir,
    )
    .await
    {
        Ok(result) => result,
        Err(e) => return format!("worker setup failed: {}", e),
    };

    log_ios(&format!("[cake-ios] master assigned layers: {:?}", layers));
    log_ios(&format!("[cake-ios] model path: {}", model_path.display()));

    // Verify model files exist
    let config_path = model_path.join("config.json");
    let model_file = model_path.join("model.safetensors");
    log_ios(&format!("[cake-ios] config.json exists: {}", config_path.exists()));
    log_ios(&format!("[cake-ios] model.safetensors exists: {} ({})",
        model_file.exists(),
        model_file.metadata().map(|m| format!("{} bytes", m.len())).unwrap_or_else(|e| format!("err: {e}"))
    ));

    // Build topology from assigned layers
    log_ios("[cake-ios] building topology...");
    let mut topology = Topology::new();
    topology.insert(
        name.to_string(),
        cake::Node {
            host: address.to_string(),
            description: None,
            layers: layers,
            vram_bytes: 0,
            tflops: 0.0,
            backend: String::new(),
            hostname: String::new(),
            os: String::new(),
        },
    );

    let force_cpu = !metal_supports_simdgroup_matrix();
    if force_cpu {
        log_ios("[cake-ios] Metal not supported on this GPU, using CPU backend");
    }

    let args = Args {
        address: address.to_string(),
        mode: Mode::Worker,
        name: Some(name.to_string()),
        model: model_path.to_string_lossy().to_string(),
        model_type: ModelType::TextModel,
        topology_override: Some(topology),
        cpu: force_cpu,
        cluster_key: Some(cluster_key.to_string()),
        ..Default::default()
    };

    log_ios(&format!("[cake-ios] creating Context::from_args (cpu={})...", force_cpu));

    // Install a panic hook that writes to our log file before aborting
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("[cake-ios] PANIC: {}", info);
        log_ios(&msg);
    }));

    let ctx_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Context::from_args(args)
    }));

    // Restore previous panic hook
    std::panic::set_hook(prev_hook);

    let mut ctx = match ctx_result {
        Ok(Ok(ctx)) => {
            log_ios(&format!("[cake-ios] context created, device={:?}", ctx.device));
            ctx
        }
        Ok(Err(e)) => {
            let msg = format!("context creation failed: {}", e);
            log_ios(&format!("[cake-ios] ERROR: {}", msg));
            return msg;
        }
        Err(panic_info) => {
            let msg = format!("context creation panicked: {:?}", panic_info.downcast_ref::<String>().map(|s| s.as_str()).or_else(|| panic_info.downcast_ref::<&str>().copied()).unwrap_or("unknown"));
            log_ios(&format!("[cake-ios] PANIC: {}", msg));
            return msg;
        }
    };

    // Pass the pre-bound listener from setup
    log_ios("[cake-ios] setting listener_override...");
    *ctx.listener_override.lock().unwrap() = Some(listener);

    log_ios("[cake-ios] entering run_text_worker...");
    run_text_worker(&mut ctx).await
}

/// Direct worker: download model from HF and start serving.
async fn run_direct_worker(name: &str, model: &str, address: &str) -> String {
    let force_cpu = !metal_supports_simdgroup_matrix();
    let args = Args {
        address: address.to_string(),
        mode: Mode::Worker,
        name: Some(name.to_string()),
        model: model.to_string(),
        model_type: ModelType::TextModel,
        cpu: force_cpu,
        ..Default::default()
    };

    log_ios("[cake-ios] creating context...");

    let mut ctx = match Context::from_args(args) {
        Ok(ctx) => {
            log_ios(&format!("[cake-ios] context created, device={:?}", ctx.device));
            ctx
        }
        Err(e) => return format!("context creation failed: {}", e),
    };

    run_text_worker(&mut ctx).await
}

async fn run_text_worker(ctx: &mut Context) -> String {
    log_ios(&format!("[cake-ios] text model arch: {:?}", ctx.text_model_arch));

    match ctx.text_model_arch {
        #[cfg(feature = "qwen3_5")]
        TextModelArch::Qwen3_5 => {
            log_ios("[cake-ios] creating Qwen3.5 worker...");
            let mut worker =
                match Worker::<cake_core::models::qwen3_5::Qwen3_5>::new(ctx).await {
                    Ok(w) => {
                        log_ios("[cake-ios] Qwen3.5 worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        let msg = format!("Qwen3.5 worker creation failed: {}", e);
                        log_ios(&format!("[cake-ios] ERROR: {}", msg));
                        return msg;
                    }
                };

            log_ios("[cake-ios] Qwen3.5 worker.run() starting...");
            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => {
                    let msg = format!("worker error: {}", e);
                    log_ios(&format!("[cake-ios] ERROR: {}", msg));
                    msg
                }
            }
        }
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            log_ios("[cake-ios] creating Qwen2 worker...");
            let mut worker =
                match Worker::<cake_core::models::qwen2::Qwen2>::new(ctx).await {
                    Ok(w) => {
                        log_ios("[cake-ios] Qwen2 worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        let msg = format!("Qwen2 worker creation failed: {}", e);
                        log_ios(&format!("[cake-ios] ERROR: {}", msg));
                        return msg;
                    }
                };

            log_ios("[cake-ios] Qwen2 worker.run() starting...");
            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => {
                    let msg = format!("worker error: {}", e);
                    log_ios(&format!("[cake-ios] ERROR: {}", msg));
                    msg
                }
            }
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            log_ios("[cake-ios] creating LLaMA worker...");
            let mut worker =
                match Worker::<cake_core::models::llama3::LLama>::new(ctx).await {
                    Ok(w) => {
                        log_ios("[cake-ios] LLaMA worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        let msg = format!("LLaMA worker creation failed: {}", e);
                        log_ios(&format!("[cake-ios] ERROR: {}", msg));
                        return msg;
                    }
                };

            log_ios("[cake-ios] LLaMA worker.run() starting...");
            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => {
                    let msg = format!("worker error: {}", e);
                    log_ios(&format!("[cake-ios] ERROR: {}", msg));
                    msg
                }
            }
        }
        #[allow(unreachable_patterns)]
        _ => {
            format!("no text model feature enabled for architecture {:?}", ctx.text_model_arch)
        }
    }
}
