//! This is a small library that wraps cake-core and exposes it as an API to the Swift side of things on iOS.
uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args, ModelType, TextModelArch,
};

/// Start a worker node that joins a cluster via discovery.
///
/// - `name`: Worker name (e.g. device name)
/// - `model`: HuggingFace model ID (e.g. "Qwen/Qwen2.5-Coder-1.5B-Instruct")
/// - `cluster_key`: Shared secret for cluster discovery and authentication
#[uniffi::export]
pub fn start_worker(name: String, model: String, cluster_key: String) {
    // Use try_init to avoid panic if called multiple times
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();

    log::info!("[cake-ios] start_worker called");
    log::info!("[cake-ios]   name: {name}");
    log::info!("[cake-ios]   model: {model}");
    log::info!("[cake-ios]   cluster_key: {}", if cluster_key.is_empty() { "(none)" } else { "(set)" });

    let args = Args {
        address: "0.0.0.0:10128".to_string(),
        mode: Mode::Worker,
        name: Some(name),
        model,
        model_type: ModelType::TextModel,
        cluster_key: if cluster_key.is_empty() { None } else { Some(cluster_key) },
        ..Default::default()
    };

    log::info!("[cake-ios] creating context...");

    let mut ctx = match Context::from_args(args) {
        Ok(ctx) => {
            log::info!("[cake-ios] context created, device={:?}", ctx.device);
            ctx
        }
        Err(e) => {
            log::error!("[cake-ios] context creation failed: {}", e);
            return;
        }
    };

    log::info!("[cake-ios] starting tokio runtime...");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            run_text_worker(&mut ctx).await;
        })
}

async fn run_text_worker(ctx: &mut Context) {
    log::info!("[cake-ios] text model arch: {:?}", ctx.text_model_arch);

    match ctx.text_model_arch {
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            log::info!("[cake-ios] creating Qwen2 worker...");
            let mut worker =
                match Worker::<cake_core::models::qwen2::Qwen2>::new(ctx).await {
                    Ok(w) => {
                        log::info!("[cake-ios] Qwen2 worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        log::error!("[cake-ios] Qwen2 worker creation failed: {}", e);
                        return;
                    }
                };

            match worker.run().await {
                Ok(_) => log::info!("[cake-ios] worker exited"),
                Err(e) => log::error!("[cake-ios] worker error: {}", e),
            }
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            log::info!("[cake-ios] creating LLaMA worker...");
            let mut worker =
                match Worker::<cake_core::models::llama3::LLama>::new(ctx).await {
                    Ok(w) => {
                        log::info!("[cake-ios] LLaMA worker ready on 0.0.0.0:10128");
                        w
                    }
                    Err(e) => {
                        log::error!("[cake-ios] LLaMA worker creation failed: {}", e);
                        return;
                    }
                };

            match worker.run().await {
                Ok(_) => log::info!("[cake-ios] worker exited"),
                Err(e) => log::error!("[cake-ios] worker error: {}", e),
            }
        }
        #[allow(unreachable_patterns)]
        _ => {
            log::error!("[cake-ios] no text model feature enabled for architecture {:?}", ctx.text_model_arch);
        }
    }
}
