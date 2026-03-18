//! cake-mobile: cross-platform (iOS + Android) worker library backed by cake-core.
//! Exposes UniFFI functions callable from Kotlin (via JNI on Android) or Swift/ObjC (via static lib on iOS).
uniffi::setup_scaffolding!("cake_mobile");

use std::sync::Mutex;
#[cfg(not(target_os = "android"))]
use std::sync::OnceLock;

use cake_core::{
    cake::{self, Context, Mode, Topology, Worker},
    Args, ModelType, TextModelArch,
};

// ---------------------------------------------------------------------------
// Stop signal
// ---------------------------------------------------------------------------

static STOP_TX: Mutex<Option<tokio::sync::watch::Sender<bool>>> = Mutex::new(None);

#[uniffi::export]
pub fn stop_worker() {
    log_mobile("[cake-mobile] stop_worker called");
    update_status("stopping", "Stopping...", 0.0);
    if let Ok(mut guard) = STOP_TX.lock() {
        if let Some(tx) = guard.take() {
            let _ = tx.send(true);
        }
    }
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

static WORKER_STATUS: Mutex<String> = Mutex::new(String::new());

fn update_status(stage: &str, message: &str, progress: f64) {
    let json = format!(
        r#"{{"stage":"{}","message":"{}","progress":{:.4}}}"#,
        stage,
        message.replace('\\', "\\\\").replace('"', "\\\""),
        progress
    );
    log_mobile(&format!("[cake-mobile] status: {}", json));
    if let Ok(mut s) = WORKER_STATUS.lock() {
        *s = json;
    }
}

fn update_serving_status(model_name: &str, layers_range: &str, backend: &str) {
    fn esc(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }
    let json = format!(
        r#"{{"stage":"serving","message":"Ready — serving inference","progress":1.0000,"model":"{}","layers":"{}","backend":"{}"}}"#,
        esc(model_name),
        esc(layers_range),
        esc(backend),
    );
    log_mobile(&format!("[cake-mobile] status: {}", json));
    if let Ok(mut s) = WORKER_STATUS.lock() {
        *s = json;
    }
}

#[uniffi::export]
pub fn get_worker_status() -> String {
    WORKER_STATUS.lock().map(|s| s.clone()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Cache directory (Android sets this before calling start_worker)
// ---------------------------------------------------------------------------

#[cfg(target_os = "android")]
static ANDROID_CACHE_DIR: Mutex<String> = Mutex::new(String::new());

/// On Android, call this with the app's cacheDir path before start_worker.
/// No-op on iOS (sandbox paths are determined automatically).
#[uniffi::export]
pub fn set_cache_dir(path: String) {
    #[cfg(target_os = "android")]
    {
        log_mobile(&format!("[cake-mobile] set_cache_dir: {}", path));
        if let Ok(mut d) = ANDROID_CACHE_DIR.lock() {
            *d = path;
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = path;
    }
}

fn get_cache_dir() -> std::path::PathBuf {
    #[cfg(target_os = "android")]
    {
        let dir = ANDROID_CACHE_DIR.lock().map(|d| d.clone()).unwrap_or_default();
        if dir.is_empty() {
            std::path::PathBuf::from("/data/local/tmp/cake")
        } else {
            std::path::PathBuf::from(dir)
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        // iOS sandbox: use $TMPDIR
        std::env::temp_dir().join("huggingface").join("hub")
    }
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "android"))]
static LOG_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

fn log_mobile(msg: &str) {
    #[cfg(target_os = "android")]
    {
        log::info!("{}", msg);
    }
    #[cfg(not(target_os = "android"))]
    {
        use std::io::Write;
        eprintln!("{}", msg);
        if let Some(path) = LOG_PATH.get() {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{}", msg);
            }
        }
    }
}

fn init_logging() {
    #[cfg(target_os = "android")]
    {
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Info)
                .with_tag("cake-mobile"),
        );
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init();
    }
}

// ---------------------------------------------------------------------------
// GPU / backend detection
// ---------------------------------------------------------------------------

fn should_force_cpu() -> bool {
    #[cfg(target_os = "android")]
    {
        // Android: CPU-only for now
        true
    }
    #[cfg(target_os = "ios")]
    {
        !metal_supports_simdgroup_matrix()
    }
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        false
    }
}

#[cfg(target_os = "ios")]
fn metal_supports_simdgroup_matrix() -> bool {
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
    if let Some(pos) = buf.iter().position(|&b| b == 0) {
        buf.truncate(pos);
    }
    let machine = String::from_utf8_lossy(&buf).to_string();
    log_mobile(&format!("[cake-mobile] hw.machine: {}", machine));

    let alpha_end = machine.find(|c: char| !c.is_alphabetic()).unwrap_or(machine.len());
    let family = &machine[..alpha_end];
    let rest = &machine[alpha_end..];
    let generation: u32 = rest.split(',').next().and_then(|s| s.parse().ok()).unwrap_or(0);

    let supported = match family {
        "iPhone" => generation >= 12,
        "iPad" => generation >= 12,
        _ => false,
    };
    log_mobile(&format!("[cake-mobile] Metal simdgroup_matrix support: {} (family={}, gen={})", supported, family, generation));
    supported
}

// ---------------------------------------------------------------------------
// Layer range helper
// ---------------------------------------------------------------------------

fn layers_to_range(layers: &[String]) -> String {
    if layers.is_empty() {
        return "none".to_string();
    }
    let first_num = layers.first().and_then(|l| l.rsplit('.').next()).and_then(|n| n.parse::<usize>().ok());
    let last_num = layers.last().and_then(|l| l.rsplit('.').next()).and_then(|n| n.parse::<usize>().ok());
    match (first_num, last_num) {
        (Some(f), Some(l)) if f == l => format!("layer {}", f),
        (Some(f), Some(l)) => format!("layers {}–{} ({} total)", f, l, layers.len()),
        _ => format!("{} layer(s)", layers.len()),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Start a worker node that joins a cluster via discovery.
/// Returns an error string if startup fails, or empty string on clean exit.
#[uniffi::export]
pub fn start_worker(name: String, model: String, cluster_key: String) -> String {
    init_logging();

    // Set up directories
    #[cfg(target_os = "ios")]
    {
        let tmp = std::env::temp_dir();
        let log_path = tmp.join("cake-mobile.log");
        let _ = std::fs::remove_file(&log_path);
        let _ = LOG_PATH.set(log_path);

        let ios_home = tmp.join("cake-home");
        let hf_cache = tmp.join("huggingface").join("hub");
        let cake_cache = tmp.join("cake");
        for dir in [&ios_home, &hf_cache, &cake_cache] {
            if let Err(e) = std::fs::create_dir_all(dir) {
                let msg = format!("failed to create dir {}: {}", dir.display(), e);
                log_mobile(&format!("[cake-mobile] {msg}"));
                return msg;
            }
        }
        std::env::set_var("HOME", ios_home.to_string_lossy().as_ref());
        std::env::set_var("HF_HUB_CACHE", hf_cache.to_string_lossy().as_ref());
    }

    #[cfg(target_os = "android")]
    {
        let cache_dir = get_cache_dir();
        let hf_cache = cache_dir.join("huggingface").join("hub");
        let cake_cache = cache_dir.join("cake");
        let android_home = cache_dir.join("cake-home");
        for dir in [&hf_cache, &cake_cache, &android_home] {
            if let Err(e) = std::fs::create_dir_all(dir) {
                let msg = format!("failed to create dir {}: {}", dir.display(), e);
                log_mobile(&format!("[cake-mobile] {msg}"));
                return msg;
            }
        }
        std::env::set_var("HOME", android_home.to_string_lossy().as_ref());
        std::env::set_var("HF_HUB_CACHE", hf_cache.to_string_lossy().as_ref());
    }

    log_mobile("[cake-mobile] start_worker called");
    log_mobile(&format!("[cake-mobile]   name: {name}"));
    log_mobile(&format!("[cake-mobile]   model: {model}"));
    log_mobile(&format!("[cake-mobile]   cluster_key: {}", if cluster_key.is_empty() { "(none)" } else { "(set)" }));

    let address = "0.0.0.0:10128".to_string();
    let use_cluster = !cluster_key.is_empty();

    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    *STOP_TX.lock().unwrap() = Some(stop_tx);

    update_status("starting", "Initializing...", 0.0);
    log_mobile("[cake-mobile] starting tokio runtime...");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let result = rt.block_on(async {
        let worker_fut = async {
            if use_cluster {
                log_mobile("[cake-mobile] zero-config mode: waiting for master discovery...");
                let cache_dir = get_cache_dir().parent().map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("/tmp"));
                let cake_cache = cache_dir.join("cake");
                run_zero_config_worker(&name, &cluster_key, &address, &cake_cache, &model).await
            } else {
                log_mobile("[cake-mobile] direct mode: loading model...");
                run_direct_worker(&name, &model, &address).await
            }
        };

        tokio::select! {
            result = worker_fut => result,
            _ = stop_rx.wait_for(|v| *v) => {
                log_mobile("[cake-mobile] worker stopped by request");
                update_status("idle", "Stopped", 0.0);
                String::new()
            }
        }
    });

    log_mobile("[cake-mobile] shutting down tokio runtime...");
    rt.shutdown_timeout(std::time::Duration::from_secs(2));
    log_mobile("[cake-mobile] runtime shut down, port released");

    result
}

// ---------------------------------------------------------------------------
// Worker implementations (identical to cake-ios)
// ---------------------------------------------------------------------------

async fn run_zero_config_worker(
    name: &str,
    cluster_key: &str,
    address: &str,
    cache_dir: &std::path::Path,
    _model: &str,
) -> String {
    let progress_cb = |stage: &str, message: &str, progress: f64| {
        update_status(stage, message, progress);
    };

    let (layers, model_path, listener) = match cake::sharding::worker_setup_with_progress(
        name,
        cluster_key,
        address,
        cache_dir,
        Some(&progress_cb),
    )
    .await
    {
        Ok(result) => result,
        Err(e) => {
            update_status("error", &format!("Setup failed: {}", e), 0.0);
            return format!("worker setup failed: {}", e);
        }
    };

    log_mobile(&format!("[cake-mobile] master assigned layers: {:?}", layers));
    log_mobile(&format!("[cake-mobile] model path: {}", model_path.display()));

    let mut topology = Topology::new();
    topology.insert(
        name.to_string(),
        cake::Node {
            host: address.to_string(),
            description: None,
            layers: layers.clone(),
            vram_bytes: 0,
            tflops: 0.0,
            backend: String::new(),
            hostname: String::new(),
            os: String::new(),
        },
    );

    let force_cpu = should_force_cpu();
    if force_cpu {
        log_mobile("[cake-mobile] GPU not supported on this device, using CPU backend");
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

    update_status("loading", "Loading model weights...", 0.0);
    log_mobile(&format!("[cake-mobile] creating Context::from_args (cpu={})...", force_cpu));

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("[cake-mobile] PANIC: {}", info);
        log_mobile(&msg);
    }));

    let ctx_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Context::from_args(args)
    }));

    std::panic::set_hook(prev_hook);

    let mut ctx = match ctx_result {
        Ok(Ok(ctx)) => {
            log_mobile(&format!("[cake-mobile] context created, device={:?}", ctx.device));
            ctx
        }
        Ok(Err(e)) => {
            let msg = format!("context creation failed: {}", e);
            log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
            update_status("error", &msg, 0.0);
            return msg;
        }
        Err(panic_info) => {
            let msg = format!(
                "context creation panicked: {:?}",
                panic_info.downcast_ref::<String>().map(|s| s.as_str())
                    .or_else(|| panic_info.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown")
            );
            log_mobile(&format!("[cake-mobile] PANIC: {}", msg));
            update_status("error", &msg, 0.0);
            return msg;
        }
    };

    *ctx.listener_override.lock().unwrap() = Some(listener);

    let model_display = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    let layers_display = layers_to_range(&layers);
    let backend_display = if force_cpu { "CPU".to_string() } else { "Metal".to_string() };
    update_serving_status(&model_display, &layers_display, &backend_display);
    log_mobile("[cake-mobile] entering run_text_worker...");
    run_text_worker(&mut ctx).await
}

async fn run_direct_worker(name: &str, model: &str, address: &str) -> String {
    let force_cpu = should_force_cpu();
    let args = Args {
        address: address.to_string(),
        mode: Mode::Worker,
        name: Some(name.to_string()),
        model: model.to_string(),
        model_type: ModelType::TextModel,
        cpu: force_cpu,
        ..Default::default()
    };

    update_status("loading", "Downloading model...", 0.0);
    log_mobile("[cake-mobile] creating context...");

    let mut ctx = match Context::from_args(args) {
        Ok(ctx) => {
            log_mobile(&format!("[cake-mobile] context created, device={:?}", ctx.device));
            ctx
        }
        Err(e) => {
            update_status("error", &format!("Failed: {}", e), 0.0);
            return format!("context creation failed: {}", e);
        }
    };

    update_status("serving", "Ready — serving inference", 1.0);
    run_text_worker(&mut ctx).await
}

// ---------------------------------------------------------------------------
// Plain C exports — used by Kotlin/Native cinterop on iOS
// (UniFFI handles Android via JNI/JNA)
// ---------------------------------------------------------------------------

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// # Safety
/// `name`, `model`, and `cluster_key` must be valid, non-null, NUL-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn cake_start_worker(
    name: *const c_char,
    model: *const c_char,
    cluster_key: *const c_char,
) -> *mut c_char {
    let name = CStr::from_ptr(name).to_string_lossy().into_owned();
    let model = CStr::from_ptr(model).to_string_lossy().into_owned();
    let key = CStr::from_ptr(cluster_key).to_string_lossy().into_owned();
    let result = start_worker(name, model, key);
    CString::new(result).unwrap_or_default().into_raw()
}

#[no_mangle]
pub extern "C" fn cake_stop_worker() {
    stop_worker();
}

#[no_mangle]
pub extern "C" fn cake_get_worker_status() -> *mut c_char {
    CString::new(get_worker_status()).unwrap_or_default().into_raw()
}

/// # Safety
/// `path` must be a valid, non-null, NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn cake_set_cache_dir(path: *const c_char) {
    let path = CStr::from_ptr(path).to_string_lossy().into_owned();
    set_cache_dir(path);
}

/// # Safety
/// `s` must be either null or a pointer previously returned by `cake_start_worker` / `cake_get_worker_status`.
#[no_mangle]
pub unsafe extern "C" fn cake_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

async fn run_text_worker(ctx: &mut Context) -> String {
    log_mobile(&format!("[cake-mobile] text model arch: {:?}", ctx.text_model_arch));

    match ctx.text_model_arch {
        #[cfg(feature = "qwen3_5")]
        TextModelArch::Qwen3_5 => {
            log_mobile("[cake-mobile] creating Qwen3.5 worker...");
            let mut worker = match Worker::<cake_core::models::qwen3_5::Qwen3_5>::new(ctx).await {
                Ok(w) => {
                    log_mobile("[cake-mobile] Qwen3.5 worker ready on 0.0.0.0:10128");
                    w
                }
                Err(e) => {
                    let msg = format!("Qwen3.5 worker creation failed: {}", e);
                    log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
                    return msg;
                }
            };
            log_mobile("[cake-mobile] Qwen3.5 worker.run() starting...");
            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => {
                    let msg = format!("worker error: {}", e);
                    log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
                    msg
                }
            }
        }
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            log_mobile("[cake-mobile] creating Qwen2 worker...");
            let mut worker = match Worker::<cake_core::models::qwen2::Qwen2>::new(ctx).await {
                Ok(w) => {
                    log_mobile("[cake-mobile] Qwen2 worker ready on 0.0.0.0:10128");
                    w
                }
                Err(e) => {
                    let msg = format!("Qwen2 worker creation failed: {}", e);
                    log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
                    return msg;
                }
            };
            log_mobile("[cake-mobile] Qwen2 worker.run() starting...");
            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => {
                    let msg = format!("worker error: {}", e);
                    log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
                    msg
                }
            }
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            log_mobile("[cake-mobile] creating LLaMA worker...");
            let mut worker = match Worker::<cake_core::models::llama3::LLama>::new(ctx).await {
                Ok(w) => {
                    log_mobile("[cake-mobile] LLaMA worker ready on 0.0.0.0:10128");
                    w
                }
                Err(e) => {
                    let msg = format!("LLaMA worker creation failed: {}", e);
                    log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
                    return msg;
                }
            };
            log_mobile("[cake-mobile] LLaMA worker.run() starting...");
            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => {
                    let msg = format!("worker error: {}", e);
                    log_mobile(&format!("[cake-mobile] ERROR: {}", msg));
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
