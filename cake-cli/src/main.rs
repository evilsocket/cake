//! This is the cake command line utility.

use cake_core::{
    cake::{Context, Master, Mode, Worker},
    Args, ModelType,
};

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    // parse command line
    let args = Args::parse();

    // setup logging
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error,actix_server=warn");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    // setup context
    let mut ctx = Context::from_args(args)?;

    // run either in master or worker mode depending on command line
    let ret = match ctx.args.mode {
        Mode::Master => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        Mode::Worker => match ctx.args.model_type {
            ModelType::TextModel => {
                Worker::<cake_core::models::llama3::LLama>::new(&mut ctx)
                    .await?
                    .run()
                    .await
            }
            ModelType::ImageModel => {
                Worker::<cake_core::models::sd::SD>::new(&mut ctx)
                    .await?
                    .run()
                    .await
            }
        },
    };

    if ret.is_err() {
        // we were possibly streaming text, add a newline before reporting the error
        println!();
        return ret;
    }

    Ok(())
}
