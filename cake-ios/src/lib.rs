//! This is a small library that wraps cake-core and exposes it as an API to the Swift side of things on iOS.
uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args, ModelType,
};

#[uniffi::export]
pub fn start_worker(name: String, model_path: String, topology_path: String, model_type: String) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    log::debug!("@ creating context");

    log::debug!("@ model type: {model_type}");

    let model_type_arg = match model_type.as_str() {
        "text" => ModelType::TextModel,
        "image" => ModelType::ImageModel,
        _ => panic!("Unrecognized model type"),
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

    let mut ctx = match Context::from_args(args) {
        Ok(ctx) => ctx,
        Err(e) => {
            log::error!("ERROR: {}", e);
            return;
        }
    };

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            log::debug!("@ creating worker");

            match model_type.as_str() {
                "text" => {
                    let mut worker =
                        match Worker::<cake_core::models::llama3::LLama>::new(&mut ctx).await {
                            Ok(w) => w,
                            Err(e) => {
                                log::error!("ERROR: {}", e);
                                return;
                            }
                        };

                    log::info!("@ running worker for text model...");

                    match worker.run().await {
                        Ok(_) => log::info!("worker exiting"),
                        Err(e) => {
                            log::error!("ERROR: {}", e);
                        }
                    }
                }
                "image" => {
                    let mut worker = match Worker::<cake_core::models::sd::SD>::new(&mut ctx).await
                    {
                        Ok(w) => w,
                        Err(e) => {
                            log::error!("ERROR: {}", e);
                            return;
                        }
                    };

                    log::info!("@ running worker for image model...");

                    match worker.run().await {
                        Ok(_) => log::info!("worker exiting"),
                        Err(e) => {
                            log::error!("ERROR: {}", e);
                        }
                    }
                }
                _ => {
                    log::error!("ERROR: unrecognized model type");
                }
            }
        })
}
