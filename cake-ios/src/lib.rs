//! This is a small library that wraps cake-core and exposes it as an API to the Swift side of things on iOS.
uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args,
};

#[uniffi::export]
pub fn start_worker(name: String, model_path: String, topology_path: String) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    log::debug!("@ creating context");

    let args = Args {
        address: "0.0.0.0:10128".to_string(),
        mode: Mode::Worker,
        name: Some(name),
        model: model_path,
        topology: topology_path,
        ..Default::default()
    };

    let ctx = match Context::from_args(args) {
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

            let mut worker = match Worker::<cake_core::model::Transformer>::new(ctx).await {
                Ok(w) => w,
                Err(e) => {
                    log::error!("ERROR: {}", e);
                    return;
                }
            };

            log::debug!("@ running worker");

            match worker.run().await {
                Ok(_) => log::info!("worker exiting"),
                Err(e) => {
                    log::error!("ERROR: {}", e);
                }
            }
        })
}
