//! This is a small library that wraps cake-core and exposes it as an API to the Swift side of things on iOS.
uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args, ModelType, TextModelArch,
};

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

    eprintln!("[cake-ios] start_worker called");
    eprintln!("[cake-ios]   name: {name}");
    eprintln!("[cake-ios]   model: {model}");
    eprintln!("[cake-ios]   cluster_key: {}", if cluster_key.is_empty() { "(none)" } else { "(set)" });

    // Set HF cache to an iOS-writable directory (app sandbox tmp).
    // The hf_hub crate will create subdirectories as needed.
    let hf_cache = std::env::temp_dir().join("huggingface").join("hub");
    if let Err(e) = std::fs::create_dir_all(&hf_cache) {
        let msg = format!("failed to create HF cache dir: {}", e);
        eprintln!("[cake-ios] {msg}");
        return msg;
    }
    std::env::set_var("HF_HUB_CACHE", hf_cache.to_string_lossy().as_ref());
    eprintln!("[cake-ios] HF_HUB_CACHE={}", hf_cache.display());

    let args = Args {
        address: "0.0.0.0:10128".to_string(),
        mode: Mode::Worker,
        name: Some(name),
        model,
        model_type: ModelType::TextModel,
        cluster_key: if cluster_key.is_empty() { None } else { Some(cluster_key) },
        ..Default::default()
    };

    eprintln!("[cake-ios] creating context...");

    let mut ctx = match Context::from_args(args) {
        Ok(ctx) => {
            eprintln!("[cake-ios] context created, device={:?}", ctx.device);
            ctx
        }
        Err(e) => {
            let msg = format!("context creation failed: {}", e);
            eprintln!("[cake-ios] {msg}");
            return msg;
        }
    };

    eprintln!("[cake-ios] starting tokio runtime...");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            run_text_worker(&mut ctx).await
        })
}

async fn run_text_worker(ctx: &mut Context) -> String {
    eprintln!("[cake-ios] text model arch: {:?}", ctx.text_model_arch);

    match ctx.text_model_arch {
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            eprintln!("[cake-ios] creating Qwen2 worker...");
            let mut worker =
                match Worker::<cake_core::models::qwen2::Qwen2>::new(ctx).await {
                    Ok(w) => {
                        eprintln!("[cake-ios] Qwen2 worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        return format!("Qwen2 worker creation failed: {}", e);
                    }
                };

            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => format!("worker error: {}", e),
            }
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            eprintln!("[cake-ios] creating LLaMA worker...");
            let mut worker =
                match Worker::<cake_core::models::llama3::LLama>::new(ctx).await {
                    Ok(w) => {
                        eprintln!("[cake-ios] LLaMA worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        return format!("LLaMA worker creation failed: {}", e);
                    }
                };

            match worker.run().await {
                Ok(_) => String::new(),
                Err(e) => format!("worker error: {}", e),
            }
        }
        #[allow(unreachable_patterns)]
        _ => {
            format!("no text model feature enabled for architecture {:?}", ctx.text_model_arch)
        }
    }
}
