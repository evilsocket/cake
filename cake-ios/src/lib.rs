uniffi::setup_scaffolding!();

use cake_core::{
    cake::{Context, Mode, Worker},
    Args,
};

#[uniffi::export]
pub async fn start_worker(name: String, model_path: String, topology_path: String) {
    println!(
        "name={} model={} topology={}",
        &name, &model_path, &topology_path
    );

    let mut args = Args::default();

    args.address = "0.0.0.0:10128".to_string();
    args.mode = Mode::Worker;
    args.name = Some(name);
    args.model = model_path;
    args.topology = topology_path;

    Worker::new(Context::from_args(args).expect("can't create context"))
        .await
        .expect("can't create worker")
        .run()
        .await
        .expect("can't run worker");
}
