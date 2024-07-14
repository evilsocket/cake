use std::io::Write;

use cake_core::{
    cake::{Context, Master, Mode, Worker},
    Args,
};

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    let ctx = Context::from_args(args)?;

    let ret = match ctx.args.mode {
        Mode::Master => {
            Master::<cake_core::model::LLama>::new(ctx)
                .await?
                .generate(|data| {
                    if data.is_empty() {
                        println!();
                    } else {
                        print!("{data}")
                    }
                    std::io::stdout().flush().unwrap();
                })
                .await
        }
        Mode::Worker => {
            Worker::<cake_core::model::Transformer>::new(ctx)
                .await?
                .run()
                .await
        }
    };

    if ret.is_err() {
        println!(); // we were streaming text
        return ret;
    }

    Ok(())
}
