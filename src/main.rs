#[macro_use]
extern crate anyhow;

use std::io::Write;

use cake::{Context, Master, Mode, Worker};

use anyhow::Result;
use clap::Parser;

mod cake;
mod model;
mod utils;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Mode.
    #[arg(long, default_value_t, value_enum)]
    mode: Mode,
    /// Binding address and port if in worker mode.
    #[arg(long, default_value = "127.0.0.1:10128")]
    address: String,
    /// Llama3 model data path.
    #[arg(long, default_value = "./data/Meta-Llama-3-8B/")]
    model: String,
    /// Topology JSON file.
    #[arg(long, default_value = "./data/topology.json")]
    topology: String,
    /// The initial prompt.
    #[arg(long, default_value = "Hi! I am ")]
    prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 100)]
    sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,
    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,
    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

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

    match ctx.args.mode {
        Mode::Master => {
            Master::new(ctx)
                .await?
                .generate(|data| {
                    if data.is_empty() {
                        println!();
                    } else {
                        print!("{data}")
                    }
                    std::io::stdout().flush().unwrap();
                })
                .await?;
        }
        Mode::Worker => {
            Worker::new(ctx).await?.run().await?;
        }
    }

    Ok(())
}
