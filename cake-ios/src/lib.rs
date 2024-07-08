uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args,
};

/*
https://github.com/huggingface/candle/issues/2322

#[uniffi::export]
pub fn test_metal() {
    let device = metal::Device::all().swap_remove(0);

    println!("device: {:?}", &device);

    println!(
        "MTLResourceOptions::StorageModeManaged = 0x{:x}",
        metal::MTLResourceOptions::StorageModeManaged
    );

    let seed = device.new_buffer_with_data(
        [299792458].as_ptr() as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    println!("seed: {:?}", &seed);
}

 */

#[uniffi::export]
pub async fn start_worker(name: String, model_path: String, topology_path: String) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    log::debug!("@ creating context");

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

    log::debug!("@ creating worker");

    let mut worker = match Worker::new(ctx).await {
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
}
