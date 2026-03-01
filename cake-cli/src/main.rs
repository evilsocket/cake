//! This is the cake command line utility.

use cake_core::{
    cake::{Context, Mode, Worker},
    Args, ModelType, TextModelArch,
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
        Mode::Master => run_master(ctx).await,
        Mode::Worker => run_worker(&mut ctx).await,
    };

    if ret.is_err() {
        // we were possibly streaming text, add a newline before reporting the error
        println!();
        return ret;
    }

    Ok(())
}

#[cfg(feature = "master")]
async fn run_master(ctx: Context) -> Result<()> {
    use cake_core::cake::Master;

    match ctx.text_model_arch {
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            Master::<cake_core::models::qwen2::Qwen2, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[allow(unreachable_patterns)]
        _ => anyhow::bail!(
            "no text model feature enabled for architecture {:?}",
            ctx.text_model_arch
        ),
    }
}

#[cfg(not(feature = "master"))]
async fn run_master(_ctx: Context) -> Result<()> {
    anyhow::bail!("master feature not enabled")
}

async fn run_worker(ctx: &mut Context) -> Result<()> {
    match ctx.args.model_type {
        ModelType::TextModel => match ctx.text_model_arch {
            #[cfg(feature = "qwen2")]
            TextModelArch::Qwen2 => {
                Worker::<cake_core::models::qwen2::Qwen2>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "llama")]
            TextModelArch::Llama | TextModelArch::Auto => {
                Worker::<cake_core::models::llama3::LLama>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!(
                "no text model feature enabled for architecture {:?}",
                ctx.text_model_arch
            ),
        },
        ModelType::ImageModel => {
            Worker::<cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
    }
}
