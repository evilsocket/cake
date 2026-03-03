//! This is a small library that wraps cake-core and exposes it as an API to the Swift side of things on iOS.
uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args, ModelType, TextModelArch,
};

#[uniffi::export]
pub fn start_worker(name: String, model_path: String, topology_path: String, model_type: String) {
    // Use try_init to avoid panic if called multiple times
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .try_init();

    log::info!("[cake-ios] start_worker called");
    log::info!("[cake-ios]   name: {name}");
    log::info!("[cake-ios]   model_path: {model_path}");
    log::info!("[cake-ios]   topology_path: {topology_path}");
    log::info!("[cake-ios]   model_type: {model_type}");

    let model_type_arg = match model_type.as_str() {
        "text" => ModelType::TextModel,
        "image" => ModelType::ImageModel,
        _ => {
            log::error!("[cake-ios] unrecognized model type: {model_type}");
            return;
        }
    };

    let args = Args {
        address: "0.0.0.0:10128".to_string(),
        mode: Mode::Worker,
        name: Some(name),
        model: model_path,
        topology: Some(topology_path),
        model_type: model_type_arg,
        ..Default::default()
    };

    log::info!("[cake-ios] creating context...");

    let mut ctx = match Context::from_args(args) {
        Ok(ctx) => {
            log::info!("[cake-ios] context created successfully, device={:?}", ctx.device);
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
            match model_type.as_str() {
                "text" => run_text_worker(&mut ctx).await,
                "image" => {
                    let mut worker = match Worker::<cake_core::models::sd::SD>::new(&mut ctx).await
                    {
                        Ok(w) => w,
                        Err(e) => {
                            log::error!("[cake-ios] SD worker creation failed: {}", e);
                            return;
                        }
                    };

                    log::info!("[cake-ios] running SD worker...");

                    match worker.run().await {
                        Ok(_) => log::info!("[cake-ios] SD worker exited"),
                        Err(e) => log::error!("[cake-ios] SD worker error: {}", e),
                    }
                }
                _ => {
                    log::error!("[cake-ios] unrecognized model type: {model_type}");
                }
            }
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
                        log::info!("[cake-ios] Qwen2 worker created successfully");
                        w
                    }
                    Err(e) => {
                        log::error!("[cake-ios] Qwen2 worker creation failed: {}", e);
                        return;
                    }
                };

            log::info!("[cake-ios] running Qwen2 worker on 0.0.0.0:10128...");

            match worker.run().await {
                Ok(_) => log::info!("[cake-ios] Qwen2 worker exited"),
                Err(e) => log::error!("[cake-ios] Qwen2 worker error: {}", e),
            }
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            log::info!("[cake-ios] creating LLaMA worker...");
            let mut worker =
                match Worker::<cake_core::models::llama3::LLama>::new(ctx).await {
                    Ok(w) => {
                        log::info!("[cake-ios] LLaMA worker created successfully");
                        w
                    }
                    Err(e) => {
                        log::error!("[cake-ios] LLaMA worker creation failed: {}", e);
                        return;
                    }
                };

            log::info!("[cake-ios] running LLaMA worker on 0.0.0.0:10128...");

            match worker.run().await {
                Ok(_) => log::info!("[cake-ios] LLaMA worker exited"),
                Err(e) => log::error!("[cake-ios] LLaMA worker error: {}", e),
            }
        }
        #[allow(unreachable_patterns)]
        _ => {
            log::error!("[cake-ios] no text model feature enabled for architecture {:?}", ctx.text_model_arch);
        }
    }
}
