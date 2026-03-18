//! This is the cake command line utility.

use std::path::PathBuf;
use std::time::Duration;

mod chat;

use cake_core::{
    cake::{self, Context, Mode, Worker},
    models::NoAudio,
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
            Master::<cake_core::models::qwen2::Qwen2, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3_5")]
        TextModelArch::Qwen3_5 => {
            Master::<cake_core::models::qwen3_5::Qwen3_5, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3")]
        TextModelArch::Qwen3 => {
            Master::<cake_core::models::qwen3::Qwen3, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3_moe")]
        TextModelArch::Qwen3Moe => {
            Master::<cake_core::models::qwen3_moe::Qwen3Moe, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "qwen3_5_moe")]
        TextModelArch::Qwen3_5Moe => {
            Master::<cake_core::models::qwen3_5_moe::Qwen3_5Moe, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "phi4")]
        TextModelArch::Phi4 => {
            Master::<cake_core::models::phi4::Phi4, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "mistral")]
        TextModelArch::Mistral => {
            Master::<cake_core::models::mistral::Mistral, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "gemma3")]
        TextModelArch::Gemma3 => {
            Master::<cake_core::models::gemma3::Gemma3, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "falcon3")]
        TextModelArch::Falcon3 => {
            Master::<cake_core::models::falcon3::Falcon3, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "olmo2")]
        TextModelArch::OLMo2 => {
            Master::<cake_core::models::olmo2::OLMo2, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "exaone4")]
        TextModelArch::EXAONE4 => {
            Master::<cake_core::models::exaone4::EXAONE4, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "llama")]
        TextModelArch::Llama | TextModelArch::Auto => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::sd::SD, NoAudio>::new(ctx)
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
    )?;

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
    )?;

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
    )?;

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
    )?;

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
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    // Parse WAV header
    if buf.len() < 44 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }

    // Find data chunk
    let mut pos = 12;
    let mut data_start = 0;
    let mut data_size = 0u32;
    let mut channels = 1u16;
    let mut sample_rate = 24000u32;
    let mut bits_per_sample = 16u16;

    while pos + 8 <= buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]);
        if chunk_id == b"fmt " {
            channels = u16::from_le_bytes([buf[pos + 10], buf[pos + 11]]);
            sample_rate = u32::from_le_bytes([buf[pos + 12], buf[pos + 13], buf[pos + 14], buf[pos + 15]]);
            bits_per_sample = u16::from_le_bytes([buf[pos + 22], buf[pos + 23]]);
        } else if chunk_id == b"data" {
            data_start = pos + 8;
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size as usize;
        if pos % 2 != 0 {
            pos += 1; // WAV chunks are word-aligned
        }
    }

    if data_start == 0 {
        anyhow::bail!("No data chunk in WAV file");
    }

    // Convert to f32 samples
    let mut samples = Vec::new();
    let data = &buf[data_start..data_start + data_size as usize];
    match bits_per_sample {
        16 => {
            for chunk in data.chunks(2) {
                if chunk.len() == 2 {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    samples.push(s as f32 / 32768.0);
                }
            }
        }
        32 => {
            for chunk in data.chunks(4) {
                if chunk.len() == 4 {
                    let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    samples.push(s);
                }
            }
        }
        _ => anyhow::bail!("Unsupported bits_per_sample: {}", bits_per_sample),
    }

    // Mix to mono if stereo
    if channels == 2 {
        let mono: Vec<f32> = samples.chunks(2).map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0).collect();
        samples = mono;
    }

    // Resample to 24kHz if needed (simple linear interpolation)
    if sample_rate != 24000 {
        let ratio = 24000.0 / sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let s0 = samples.get(idx).copied().unwrap_or(0.0);
            let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
            resampled.push(s0 + frac * (s1 - s0));
        }
        samples = resampled;
    }

    println!(
        "[WAV] Loaded {:.1}s audio ({} samples, {}ch, {}Hz, {}bit)",
        samples.len() as f64 / 24000.0,
        samples.len(),
        channels,
        sample_rate,
        bits_per_sample
    );
    Ok(samples)
}

#[cfg(feature = "master")]
async fn run_master_image(ctx: Context) -> Result<()> {
    use cake_core::cake::Master;

    // Use LLama as dummy TG, NoAudio as dummy AG — they're never loaded for ImageModel.
    match ctx.args.image_model_arch {
        ImageModelArch::SD => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::sd::SD, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "flux")]
        ImageModelArch::Flux => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::flux::FluxGen, NoAudio>::new(ctx)
                .await?
                .run()
                .await
        }
        #[cfg(feature = "flux")]
        ImageModelArch::Flux1 => {
            Master::<cake_core::models::llama3::LLama, cake_core::models::flux::Flux1Gen, NoAudio>::new(ctx)
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
