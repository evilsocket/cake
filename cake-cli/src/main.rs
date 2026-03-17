//! This is the cake command line utility.

use std::path::PathBuf;
use std::time::Duration;

mod chat;

use cake_core::{
    cake::{self, Context, Mode, Worker},
    utils, Args, ImageModelArch, ModelType, TextModelArch,
};

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cake", author, version, about = "Distributed LLM inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run as master node with OpenAI-compatible API
    Master {
        #[command(flatten)]
        args: Args,
    },
    /// Run as worker node
    Worker {
        #[command(flatten)]
        args: Args,
    },
    /// Download a model from HuggingFace Hub
    Download {
        /// HuggingFace repo ID (e.g., Qwen/Qwen2.5-Coder-1.5B-Instruct)
        model: String,
    },
    /// List locally available models and their status
    Models,
    /// Interactive chat with the cluster
    Chat {
        /// Master API endpoint
        #[arg(long, default_value = "http://localhost:8086")]
        server: String,
    },
    /// Split a model into per-worker bundles
    Split {
        /// Input model path
        #[arg(long)]
        model_path: String,
        /// Topology file
        #[arg(long)]
        topology: String,
        /// Worker name (or omit for all workers)
        #[arg(long)]
        worker: Option<String>,
        /// Output folder
        #[arg(long)]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // setup logging
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error,actix_server=warn");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    match cli.command {
        Commands::Models => {
            let models = utils::models::list_models()?;
            if models.is_empty() {
                println!("No models found.");
                println!();
                println!("Download a model with:");
                println!("  cake download <org/model-name>");
            } else {
                println!(
                    "{:<50} {:<15} {:<15} SOURCE",
                    "MODEL", "STATUS", "SIZE"
                );
                println!("{}", "-".repeat(95));
                for m in &models {
                    let size = human_bytes::human_bytes(m.size_bytes as f64);
                    println!(
                        "{:<50} {:<15} {:<15} {}",
                        &m.name, m.status, size, m.source,
                    );
                }
                println!();
                println!("{} model(s) found.", models.len());
            }
            Ok(())
        }
        Commands::Chat { server } => {
            chat::run(&server).await
        }
        Commands::Download { model } => {
            if utils::hf::looks_like_hf_repo(&model) {
                let path = utils::hf::ensure_model_downloaded(&model)?;
                println!("model downloaded to {}", path.display());
            } else {
                anyhow::bail!("'{}' does not look like a HuggingFace repo ID (expected format: org/model-name)", model);
            }
            Ok(())
        }
        Commands::Split {
            model_path,
            topology,
            worker,
            output,
        } => {
            utils::split::split_model(
                &std::path::PathBuf::from(&model_path),
                &topology,
                worker.as_deref(),
                &std::path::PathBuf::from(&output),
            )
        }
        Commands::Master { mut args } => {
            args.mode = Mode::Master;

            // Zero-config: discover workers, assign layers, push model data
            if args.cluster_key.is_some() && args.topology.is_none() {
                let model_path = resolve_model_path(&args.model)?;
                let timeout = Duration::from_secs(args.discovery_timeout);
                let topology = cake::setup::master_setup(
                    args.cluster_key.as_ref().unwrap(),
                    &model_path,
                    timeout,
                    args.min_workers,
                )
                .await?;
                args.topology_override = Some(topology);
            }

            let ctx = Context::from_args(args)?;
            let ret = run_master(ctx).await;
            if ret.is_err() {
                println!();
            }
            ret
        }
        Commands::Worker { mut args } => {
            args.mode = Mode::Worker;

            // Zero-config: wait for master assignment + model data
            let listener_override = if args.cluster_key.is_some() && args.topology.is_none() {
                if args.name.is_none() {
                    args.name = Some("worker".to_string());
                }
                let worker_name = args.name.as_deref().unwrap();
                let cache_dir = cache_base_dir();
                let (layers, model_path, listener) = cake::setup::worker_setup(
                    worker_name,
                    args.cluster_key.as_ref().unwrap(),
                    &args.address,
                    &cache_dir,
                )
                .await?;
                args.model = model_path.to_string_lossy().to_string();
                args.topology_override = Some(build_worker_topology(
                    worker_name,
                    &args.address,
                    &layers,
                ));
                Some(listener)
            } else {
                None
            };

            let mut ctx = Context::from_args(args)?;
            if let Some(listener) = listener_override {
                *ctx.listener_override.lock().unwrap() = Some(listener);
            }
            let ret = run_worker(&mut ctx).await;
            if ret.is_err() {
                println!();
            }
            ret
        }
    }
}

#[cfg(feature = "master")]
async fn run_master(ctx: Context) -> Result<()> {
    use cake_core::cake::Master;

    // Image model dispatch — early return to avoid duplicating text arch arms
    if ctx.args.model_type == ModelType::ImageModel {
        return run_master_image(ctx).await;
    }

    // Audio (TTS) model dispatch
    if ctx.args.model_type == ModelType::AudioModel {
        return run_master_audio(ctx).await;
    }

    match ctx.text_model_arch {
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            Master::<cake_core::models::qwen2::Qwen2, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3_5")]
        TextModelArch::Qwen3_5 => {
            Master::<cake_core::models::qwen3_5::Qwen3_5, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3")]
        TextModelArch::Qwen3 => {
            Master::<cake_core::models::qwen3::Qwen3, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3_moe")]
        TextModelArch::Qwen3Moe => {
            Master::<cake_core::models::qwen3_moe::Qwen3Moe, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3_5_moe")]
        TextModelArch::Qwen3_5Moe => {
            Master::<cake_core::models::qwen3_5_moe::Qwen3_5Moe, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "phi4")]
        TextModelArch::Phi4 => {
            Master::<cake_core::models::phi4::Phi4, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "mistral")]
        TextModelArch::Mistral => {
            Master::<cake_core::models::mistral::Mistral, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "gemma3")]
        TextModelArch::Gemma3 => {
            Master::<cake_core::models::gemma3::Gemma3, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "falcon3")]
        TextModelArch::Falcon3 => {
            Master::<cake_core::models::falcon3::Falcon3, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "olmo2")]
        TextModelArch::OLMo2 => {
            Master::<cake_core::models::olmo2::OLMo2, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "exaone4")]
        TextModelArch::EXAONE4 => {
            Master::<cake_core::models::exaone4::EXAONE4, cake_core::models::sd::SD>::new(ctx)
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

#[cfg(feature = "master")]
async fn run_master_audio(ctx: Context) -> Result<()> {
    #[cfg(feature = "vibevoice")]
    {
        use cake_core::models::vibevoice;
        use std::path::Path;

        // Resolve model path via HuggingFace cache
        let model_path = utils::hf::ensure_model_downloaded(&ctx.args.model)?;
        let config_path = model_path.join("config.json");
        let weights_path = model_path.join("model.safetensors");

        println!("[VibeVoice] Loading model from {}", model_path.display());

        let model = vibevoice::VibeVoiceTTS::load(
            &config_path,
            &weights_path,
            &ctx.device,
        )?;

        let prompt = &ctx.args.prompt;
        println!("[VibeVoice] Generating speech for: \"{}\"", prompt);

        // Tokenize — VibeVoice uses Qwen2.5 tokenizer
        let tokenizer = {
            let local = model_path.join("tokenizer.json");
            if local.exists() {
                tokenizers::Tokenizer::from_file(&local)
                    .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
            } else {
                // VibeVoice doesn't ship a tokenizer — use Qwen2.5's
                println!("[VibeVoice] Downloading Qwen2.5 tokenizer...");
                let qwen_path = utils::hf::ensure_model_downloaded("Qwen/Qwen2.5-0.5B")?;
                tokenizers::Tokenizer::from_file(qwen_path.join("tokenizer.json"))
                    .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
            }
        };

        let encoding = tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let token_ids = encoding.get_ids();
        println!("[VibeVoice] Tokenized: {} tokens", token_ids.len());

        let max_frames = ctx.args.max_audio_frames;
        let samples = model.generate(token_ids, max_frames)?;

        let output_path = Path::new(&ctx.args.audio_output);
        vibevoice::save_wav(&samples, output_path, 24000)?;
        println!(
            "[VibeVoice] Audio saved to {} ({:.1}s, {} samples)",
            output_path.display(),
            samples.len() as f64 / 24000.0,
            samples.len()
        );

        Ok(())
    }
}

#[cfg(feature = "master")]
async fn run_master_image(ctx: Context) -> Result<()> {
    use cake_core::cake::Master;

    // Use LLama as dummy TG — it's never loaded for ImageModel.
    match ctx.args.image_model_arch {
        ImageModelArch::SD => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::sd::SD>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "flux")]
        ImageModelArch::Flux => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::flux::FluxGen>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "flux")]
        ImageModelArch::Flux1 => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::flux::Flux1Gen>::new(ctx)
                .await?
                .run()
                .await
        }
        #[allow(unreachable_patterns)]
        _ => anyhow::bail!(
            "no image model feature enabled for architecture {:?}",
            ctx.args.image_model_arch
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
            #[cfg(feature = "qwen3_5")]
            TextModelArch::Qwen3_5 => {
                Worker::<cake_core::models::qwen3_5::Qwen3_5>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "qwen3")]
            TextModelArch::Qwen3 => {
                Worker::<cake_core::models::qwen3::Qwen3>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "qwen3_moe")]
            TextModelArch::Qwen3Moe => {
                Worker::<cake_core::models::qwen3_moe::Qwen3Moe>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "qwen3_5_moe")]
            TextModelArch::Qwen3_5Moe => {
                Worker::<cake_core::models::qwen3_5_moe::Qwen3_5Moe>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "phi4")]
            TextModelArch::Phi4 => {
                Worker::<cake_core::models::phi4::Phi4>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "mistral")]
            TextModelArch::Mistral => {
                Worker::<cake_core::models::mistral::Mistral>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "gemma3")]
            TextModelArch::Gemma3 => {
                Worker::<cake_core::models::gemma3::Gemma3>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "falcon3")]
            TextModelArch::Falcon3 => {
                Worker::<cake_core::models::falcon3::Falcon3>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "olmo2")]
            TextModelArch::OLMo2 => {
                Worker::<cake_core::models::olmo2::OLMo2>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "exaone4")]
            TextModelArch::EXAONE4 => {
                Worker::<cake_core::models::exaone4::EXAONE4>::new(ctx)
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
        ModelType::ImageModel => match ctx.args.image_model_arch {
            ImageModelArch::SD => {
                Worker::<cake_core::models::sd::SD>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "flux")]
            ImageModelArch::Flux => {
                Worker::<cake_core::models::flux::FluxGen>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(feature = "flux")]
            ImageModelArch::Flux1 => {
                Worker::<cake_core::models::flux::Flux1Gen>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[allow(unreachable_patterns)]
            _ => anyhow::bail!(
                "no image model feature enabled for architecture {:?}",
                ctx.args.image_model_arch
            ),
        }
        ModelType::AudioModel => {
            anyhow::bail!("AudioModel workers not yet supported; run TTS on master")
        }
    }
}

/// Resolve a model path, downloading from HuggingFace if it looks like a repo ID.
fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let path = PathBuf::from(model);
    if path.exists() {
        Ok(path)
    } else if utils::hf::looks_like_hf_repo(model) {
        utils::hf::ensure_model_downloaded(model)
    } else {
        anyhow::bail!("model path does not exist: {}", path.display())
    }
}

/// Build a minimal Topology for a single worker from assigned layers.
fn build_worker_topology(
    worker_name: &str,
    address: &str,
    layers: &[String],
) -> cake::Topology {
    let mut topology = cake::Topology::new();
    topology.insert(
        worker_name.to_string(),
        cake::Node {
            host: address.to_string(),
            description: None,
            layers: layers.to_vec(),
            vram_bytes: 0,
            tflops: 0.0,
            backend: String::new(),
            hostname: String::new(),
            os: String::new(),
        },
    );
    topology
}

/// Return the base cache directory for zero-config model data.
fn cache_base_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("cake")
}
