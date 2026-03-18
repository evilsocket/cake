# Voice Generation (TTS)

Cake supports text-to-speech synthesis with voice cloning using Microsoft's VibeVoice models.

## Supported Models

| Model | Parameters | VRAM | Max Duration | Speakers | Voice Ref |
|-------|-----------|------|-------------|----------|-----------|
| [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | ~3B (BF16) | ~7 GB | ~90 min | Up to 4 | .wav file |
| [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | ~0.5B (F32) | ~3 GB | ~10 min | 1 | .safetensors preset |

Both models generate 24kHz mono audio using a next-token diffusion framework: a Qwen2.5 LLM predicts context, then a diffusion head generates acoustic latents decoded by a VAE to a waveform.

## VibeVoice-1.5B

The recommended model. Supports multi-speaker voice cloning from raw .wav files.

### Basic Usage

```sh
cake master --model-type audio-model --model microsoft/VibeVoice-1.5B \
  --voice-prompt voice_reference.wav \
  --prompt "Hello world, this is a test of the voice cloning system." \
  --audio-output output.wav
```

The model and tokenizer are downloaded automatically from HuggingFace on first run.

### Voice Reference

Provide a .wav file of the target speaker (any sample rate, mono or stereo — automatically converted to 24kHz mono). Longer references (10-30 seconds) produce better voice cloning. The audio is encoded through the acoustic tokenizer to create a voice embedding.

Pre-made voice presets are available in the [VibeVoice community repo](https://github.com/vibevoice-community/VibeVoice/tree/main/demo/voices):

```sh
# Download a voice preset
wget https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices/en-Alice_woman.wav

# Use it
cake master --model-type audio-model --model microsoft/VibeVoice-1.5B \
  --voice-prompt en-Alice_woman.wav \
  --prompt "Your text here" --audio-output output.wav
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--voice-prompt` | (required) | Path to .wav voice reference file |
| `--prompt` | (required) | Text to synthesize |
| `--audio-output` | `output.wav` | Output WAV file path |
| `--tts-diffusion-steps` | 10 | Diffusion steps per speech frame (higher = better quality, slower) |
| `--tts-cfg-scale` | 1.5 | Classifier-free guidance scale (1.0-3.0) |
| `--max-audio-frames` | 150 | Maximum speech frames (~133ms each at 7.5Hz) |

### Architecture

VibeVoice-1.5B uses autoregressive next-token prediction with three special tokens:
- `speech_start` — marks the beginning of a speech segment
- `speech_diffusion` — triggers diffusion-based acoustic latent generation
- `speech_end` — marks the end of a speech segment

Each `speech_diffusion` token generates one acoustic frame (64-dim latent) which is decoded to ~133ms of 24kHz audio. The model uses classifier-free guidance with a negative prompt for quality.

The feedback loop encodes each generated audio chunk back through both the acoustic and semantic encoders, combining the features as input for the next token prediction.

## VibeVoice-Realtime-0.5B

A lightweight streaming variant optimized for real-time TTS. Uses a split LM architecture (4-layer base + 20-layer TTS) and pre-computed voice prompt KV caches instead of raw audio encoding.

### Usage

```sh
cake master --model-type audio-model --model microsoft/VibeVoice-Realtime-0.5B \
  --voice-prompt voice_preset.safetensors \
  --prompt "Hello world" --audio-output output.wav \
  --tts-cfg-scale 1.5
```

Voice presets for the 0.5B model are `.safetensors` files containing pre-computed KV caches. These can be found in the [VibeVoice community repo](https://github.com/vibevoice-community/VibeVoice/tree/main/demo/voices/streaming_model) as `.pt` files (PyTorch format) and converted to safetensors.

## API Endpoint

When running with `--api`, the audio endpoint is available at `/v1/audio/speech`:

```sh
# Start the API server
cake master --model-type audio-model --model microsoft/VibeVoice-1.5B \
  --voice-prompt voice_reference.wav --api 0.0.0.0:8080

# Generate speech
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from the API"}' \
  -o output.wav
```

Voice cloning via API (base64-encoded WAV reference):

```sh
VOICE_B64=$(base64 -w0 voice_reference.wav)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"input\": \"Hello with a cloned voice.\", \"voice_data\": \"$VOICE_B64\"}" \
  -o output.wav
```

See the full [REST API Reference](api.md) for all parameters and response formats.

## Model Detection

The model variant (1.5B vs 0.5B) is auto-detected from `config.json`:
- `model_type: "vibevoice"` — VibeVoice-1.5B
- `model_type: "vibevoice_streaming"` — VibeVoice-Realtime-0.5B
