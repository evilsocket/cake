uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args,
};

#[uniffi::export]
pub async fn start_worker(name: String, model_path: String, topology_path: String) {
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "debug");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut args = Args::default();

    args.address = "0.0.0.0:10128".to_string();
    args.mode = Mode::Worker;
    args.name = Some(name);
    args.model = model_path;
    args.topology = topology_path;

    let ctx = match Context::from_args(args) {
        Ok(ctx) => ctx,
        Err(e) => {
            log::error!("ERROR: {}", e);
            return;
        }
    };

    let mut worker = match Worker::new(ctx).await {
        Ok(w) => w,
        Err(e) => {
            log::error!("ERROR: {}", e);
            return;
        }
    };

    match worker.run().await {
        Ok(_) => log::info!("worker exiting"),
        Err(e) => {
            log::error!("ERROR: {}", e);
        }
    }
}
