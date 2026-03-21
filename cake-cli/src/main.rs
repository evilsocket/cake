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
        /// HuggingFace repo ID (e.g., evilsocket/Qwen2.5-Coder-1.5B-Instruct)
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
            if let (Some(key), None) = (&args.cluster_key, &args.topology) {
                let model_path = resolve_model_path(&args.model)?;
                let timeout = Duration::from_secs(args.discovery_timeout);
                let topology = cake::sharding::master_setup(
                    key,
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
            let listener_override = if let (Some(key), None) = (&args.cluster_key, &args.topology) {
                if args.name.is_none() {
                    args.name = Some("worker".to_string());
                }
                let worker_name = args.name.as_deref().unwrap();
                let cache_dir = cache_base_dir();
                let (layers, model_path, listener) = cake::sharding::worker_setup(
                    worker_name,
                    key,
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

    // LuxTTS: TTS model using TextModel dispatch for sharding
    #[cfg(feature = "luxtts")]
    if ctx.text_model_arch == TextModelArch::LuxTTS {
        return run_master_luxtts(ctx).await;
    }

    match ctx.text_model_arch {
        #[cfg(feature = "qwen2")]
        TextModelArch::Qwen2 => {
            Master::<cake_core::models::qwen2::Qwen2>::new(ctx).await?.run().await
        }
        #[cfg(feature = "qwen3_5")]
        TextModelArch::Qwen3_5 => {
            Master::<cake_core::models::qwen3_5::Qwen3_5>::new(ctx).await?.run().await
        }
        #[cfg(feature = "qwen3")]
        TextModelArch::Qwen3 => {
            Master::<cake_core::models::qwen3::Qwen3>::new(ctx).await?.run().await
        }
        #[cfg(feature = "qwen3_moe")]
        TextModelArch::Qwen3Moe => {
            Master::<cake_core::models::qwen3_moe::Qwen3Moe>::new(ctx).await?.run().await
        }
        #[cfg(feature = "qwen3_5_moe")]
        TextModelArch::Qwen3_5Moe => {
            Master::<cake_core::models::qwen3_5_moe::Qwen3_5Moe>::new(ctx).await?.run().await
        }
        #[cfg(feature = "phi4")]
        TextModelArch::Phi4 => {
            Master::<cake_core::models::phi4::Phi4>::new(ctx).await?.run().await
        }
        #[cfg(feature = "mistral")]
        TextModelArch::Mistral => {
            Master::<cake_core::models::mistral::Mistral>::new(ctx).await?.run().await
        }
        #[cfg(feature = "gemma3")]
        TextModelArch::Gemma3 => {
            Master::<cake_core::models::gemma3::Gemma3>::new(ctx).await?.run().await
        }
        #[cfg(feature = "falcon3")]
        TextModelArch::Falcon3 => {
            Master::<cake_core::models::falcon3::Falcon3>::new(ctx).await?.run().await
        }
        #[cfg(feature = "olmo2")]
        TextModelArch::OLMo2 => {
            Master::<cake_core::models::olmo2::OLMo2>::new(ctx).await?.run().await
        }
        #[cfg(feature = "exaone4")]
        TextModelArch::EXAONE4 => {
            Master::<cake_core::models::exaone4::EXAONE4>::new(ctx).await?.run().await
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            Master::<cake_core::models::llama3::LLama>::new(ctx).await?.run().await
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
        let model_path = utils::hf::ensure_model_downloaded(&ctx.args.model)?;
        let config_path = model_path.join("config.json");

        // Detect model variant from config.json model_type
        let config_str = std::fs::read_to_string(&config_path)?;
        let model_type: String = serde_json::from_str::<serde_json::Value>(&config_str)?
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("vibevoice_streaming")
            .to_string();

        println!("[VibeVoice] Model type: {model_type}");

        match model_type.as_str() {
            "vibevoice" => {
                // VibeVoice-1.5B (non-streaming)
                run_vibevoice_1_5b(ctx, &model_path, &config_path).await
            }
            _ => {
                // VibeVoice-Realtime-0.5B (streaming)
                run_vibevoice_0_5b(ctx, &model_path, &config_path).await
            }
        }
    }
    #[cfg(not(feature = "vibevoice"))]
    {
        let _ = ctx;
        anyhow::bail!("vibevoice feature not enabled")
    }
}

#[cfg(feature = "master")]
#[cfg(feature = "vibevoice")]
async fn run_vibevoice_0_5b(
    ctx: Context,
    model_path: &std::path::Path,
    config_path: &std::path::Path,
) -> Result<()> {
    use cake_core::models::vibevoice;

    let weights_path = model_path.join("model.safetensors");
    println!("[VibeVoice-0.5B] Loading from {}", model_path.display());

    let model = vibevoice::VibeVoiceTTS::load(
        config_path,
        &weights_path,
        &ctx.device,
        Some(ctx.args.tts_diffusion_steps),
        &ctx.topology,
        ctx.args.cluster_key.as_deref(),
    )
    .await?;

    let prompt = &ctx.args.prompt;
    println!("[VibeVoice-0.5B] Generating speech for: \"{}\"", prompt);

    let tokenizer = {
        let local = model_path.join("tokenizer.json");
        if local.exists() {
            tokenizers::Tokenizer::from_file(&local)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
        } else {
            println!("[VibeVoice-0.5B] Downloading Qwen2.5 tokenizer...");
            let qwen_path = utils::hf::ensure_model_downloaded("Qwen/Qwen2.5-0.5B")?;
            tokenizers::Tokenizer::from_file(qwen_path.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
        }
    };

    let text_with_newline = format!("{}\n", prompt.trim());
    let encoding = tokenizer
        .encode(text_with_newline.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let token_ids = encoding.get_ids();
    println!("[VibeVoice-0.5B] Tokenized: {} tokens", token_ids.len());

    let voice_path = ctx
        .args
        .voice_prompt
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--voice-prompt required for TTS"))?;
    let voice_prompt = vibevoice::VoicePrompt::load_f32(
        std::path::Path::new(voice_path),
        &ctx.device,
    )?;

    let mut model = model;
    let samples = model.generate(
        token_ids,
        &voice_prompt,
        ctx.args.max_audio_frames,
        ctx.args.tts_cfg_scale,
    )
    .await?;

    let output_path = std::path::Path::new(&ctx.args.audio_output);
    vibevoice::save_wav(&samples, output_path, 24000)?;
    println!(
        "[VibeVoice-0.5B] Audio saved to {} ({:.1}s, {} samples)",
        output_path.display(),
        samples.len() as f64 / 24000.0,
        samples.len()
    );
    Ok(())
}

#[cfg(feature = "master")]
#[cfg(feature = "vibevoice")]
async fn run_vibevoice_1_5b(
    ctx: Context,
    model_path: &std::path::Path,
    config_path: &std::path::Path,
) -> Result<()> {
    use cake_core::models::vibevoice;
    use cake_core::models::vibevoice::config_1_5b::*;

    println!("[VibeVoice-1.5B] Loading from {}", model_path.display());

    // Collect weight shard paths
    let mut weight_paths: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors") && name.starts_with("model") {
            weight_paths.push(entry.path());
        }
    }
    weight_paths.sort();
    println!(
        "[VibeVoice-1.5B] Found {} weight shards",
        weight_paths.len()
    );

    let mut model = vibevoice::VibeVoice1_5B::load(
        config_path,
        &weight_paths,
        &ctx.device,
        Some(ctx.args.tts_diffusion_steps),
        &ctx.topology,
        ctx.args.cluster_key.as_deref(),
    )
    .await?;

    let prompt = &ctx.args.prompt;
    println!("[VibeVoice-1.5B] Generating speech for: \"{}\"", prompt);

    // Load tokenizer (Qwen2.5-1.5B)
    let tokenizer = {
        let local = model_path.join("tokenizer.json");
        if local.exists() {
            tokenizers::Tokenizer::from_file(&local)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
        } else {
            println!("[VibeVoice-1.5B] Downloading Qwen2.5-1.5B tokenizer...");
            let qwen_path = utils::hf::ensure_model_downloaded("Qwen/Qwen2.5-1.5B")?;
            tokenizers::Tokenizer::from_file(qwen_path.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
        }
    };

    // Load voice reference audio
    let voice_path = ctx
        .args
        .voice_prompt
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--voice-prompt required (path to .wav voice reference)"))?;
    println!("[VibeVoice-1.5B] Loading voice: {}", voice_path);

    // Read WAV file and encode voice reference
    let voice_audio = load_wav_mono_24k(std::path::Path::new(voice_path))?;
    let (_acoustic_features, voice_embeds) = model.encode_voice_from_samples(&voice_audio)?;
    let num_speech_frames = voice_embeds.dim(1)?;
    println!(
        "[VibeVoice-1.5B] Voice reference: {} frames",
        num_speech_frames
    );

    // Build input token sequence matching VibeVoiceProcessor
    let system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
    let voice_section = format!(
        " Voice input:\n Speaker 0:{start}{diffusion}{end}\n",
        start = "<|vision_start|>",
        diffusion = "<|vision_pad|>".repeat(num_speech_frames),
        end = "<|vision_end|>"
    );
    let text_section = format!(
        " Text input:\n Speaker 0: {text}\n Speech output:\n{start}",
        text = prompt.trim(),
        start = "<|vision_start|>"
    );

    let full_input = format!("{}{}{}", system_prompt, voice_section, text_section);
    let encoding = tokenizer
        .encode(full_input.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("[VibeVoice-1.5B] Input: {} tokens", token_ids.len());

    // Build speech_input_mask (positions where speech_diffusion_id should be replaced)
    let speech_input_mask: Vec<bool> = token_ids
        .iter()
        .map(|&t| t == SPEECH_DIFFUSION_ID)
        .collect();
    let speech_count: usize = speech_input_mask.iter().filter(|&&m| m).count();
    println!(
        "[VibeVoice-1.5B] Speech positions: {} (voice frames: {})",
        speech_count, num_speech_frames
    );

    // Generate
    let max_tokens = ctx.args.max_audio_frames * 3; // ~3 tokens per speech frame
    let samples = model.generate(
        &token_ids,
        &speech_input_mask,
        &voice_embeds,
        max_tokens,
        ctx.args.tts_cfg_scale,
    )
    .await?;

    let output_path = std::path::Path::new(&ctx.args.audio_output);
    vibevoice::save_wav(&samples, output_path, 24000)?;
    println!(
        "[VibeVoice-1.5B] Audio saved to {} ({:.1}s, {} samples)",
        output_path.display(),
        samples.len() as f64 / 24000.0,
        samples.len()
    );
    Ok(())
}

/// Load a WAV file as mono 24kHz f32 samples.
#[cfg(feature = "master")]
#[cfg(feature = "vibevoice")]
fn load_wav_mono_24k(path: &std::path::Path) -> Result<Vec<f32>> {
    let samples = cake_core::utils::wav::load_wav_mono(path, 24000)?;
    println!(
        "[WAV] Loaded {:.1}s audio ({} samples, 24kHz mono)",
        samples.len() as f64 / 24000.0,
        samples.len(),
    );
    Ok(samples)
}

#[cfg(feature = "master")]
#[cfg(feature = "luxtts")]
async fn run_master_luxtts(mut ctx: Context) -> Result<()> {
    use cake_core::models::luxtts;
    use cake_core::models::Generator;

    println!("[LuxTTS] Loading model from {}...", ctx.data_path.display());

    let mut model = luxtts::LuxTTS::load(&mut ctx)
        .await?
        .ok_or_else(|| anyhow::anyhow!("failed to load LuxTTS model"))?;

    let prompt = ctx.args.prompt.clone();
    println!("[LuxTTS] Generating speech for: \"{}\"", prompt);

    // Load reference audio if provided
    let reference_audio = if let Some(ref ref_path) = ctx.args.tts_reference_audio {
        println!("[LuxTTS] Loading reference audio: {}", ref_path);
        Some(load_wav_mono_24k(std::path::Path::new(ref_path))?)
    } else {
        None
    };

    let samples: Vec<f32> = model
        .generate_speech(
            &prompt,
            reference_audio.as_deref(),
            ctx.args.tts_t_shift,
            ctx.args.tts_cfg_scale,
            ctx.args.tts_diffusion_steps.min(10), // LuxTTS uses 4 steps by default
            ctx.args.tts_speed,
        )
        .await?;

    let output_path = std::path::Path::new(&ctx.args.audio_output);
    luxtts::save_wav(&samples, output_path, 48000)?;
    println!(
        "[LuxTTS] Audio saved to {} ({:.1}s, {} samples @ 48kHz)",
        output_path.display(),
        samples.len() as f64 / 48000.0,
        samples.len()
    );
    Ok(())
}

#[cfg(feature = "master")]
async fn run_master_image(ctx: Context) -> Result<()> {
    use cake_core::cake::Master;

    match ctx.args.image_model_arch {
        ImageModelArch::SD => {
            Master::<cake_core::models::sd::SD>::new(ctx).await?.run().await
        }
        #[cfg(feature = "flux")]
        ImageModelArch::Flux => {
            Master::<cake_core::models::flux::FluxGen>::new(ctx).await?.run().await
        }
        #[cfg(feature = "flux")]
        ImageModelArch::Flux1 => {
            Master::<cake_core::models::flux::Flux1Gen>::new(ctx).await?.run().await
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
            #[cfg(feature = "luxtts")]
            TextModelArch::LuxTTS => {
                Worker::<cake_core::models::luxtts::LuxTTS>::new(ctx)
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
            // VibeVoice LM layers are standard Transformer blocks — use Qwen2 worker
            // (Transformer::load is architecture-agnostic, works for any model's layers)
            #[cfg(feature = "qwen2")]
            {
                Worker::<cake_core::models::qwen2::Qwen2>::new(ctx)
                    .await?
                    .run()
                    .await
            }
            #[cfg(not(feature = "qwen2"))]
            anyhow::bail!("AudioModel workers require the qwen2 feature")
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
