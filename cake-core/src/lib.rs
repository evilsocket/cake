#[macro_use]
extern crate anyhow;

use cake::Mode;

use clap::Parser;

pub mod cake;
pub mod model;
pub mod utils;

#[derive(Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// GPU device index.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Mode.
    #[arg(long, default_value_t, value_enum)]
    pub mode: Mode,
    /// Worker name.
    #[arg(long)]
    pub name: Option<String>,
    /// Binding address and port if in worker mode.
    #[arg(long, default_value = "127.0.0.1:10128")]
    pub address: String,
    /// Llama3 model data path.
    #[arg(long, default_value = "./cake-data/Meta-Llama-3-8B/")]
    pub model: String,
    /// Topology file.
    #[arg(long, default_value = "./cake-data/topology.yml")]
    pub topology: String,
    /// The initial prompt.
    #[arg(long, default_value = "Hi! I am ")]
    pub prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 100)]
    pub sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Only sample among the top K samples.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    pub repeat_last_n: usize,
    /// Use different dtype than f16
    #[arg(long)]
    pub dtype: Option<String>,
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,
}
