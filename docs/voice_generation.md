# Voice Generation (TTS)

Cake supports text-to-speech synthesis with voice cloning.

## Supported Models

| Model | Parameters | VRAM | Speed | Architecture | Voice Ref |
|-------|-----------|------|-------|-------------|-----------|
| [LuxTTS](https://huggingface.co/YatharthS/LuxTTS) | ~123M | <1 GB | 150x realtime (GPU) | Zipformer + flow matching | .wav file (optional) |
| [VibeVoice-1.5B](https://huggingface.co/evilsocket/VibeVoice-1.5B) | ~3B (BF16) | ~7 GB | ~1x realtime | Qwen2.5 LM + diffusion | .wav file |
| [VibeVoice-Realtime-0.5B](https://huggingface.co/evilsocket/VibeVoice-Realtime-0.5B) | ~0.5B (F32) | ~3 GB | ~3x realtime | Qwen2.5 LM + diffusion | .safetensors preset |

## LuxTTS

A lightweight ZipVoice-based TTS model using Zipformer encoder + flow matching decoder with a 4-step Euler solver. Produces 48kHz audio at 150x realtime on GPU. Supports distributed inference — the 16 FM decoder layers can be sharded across workers.

Original model: [`YatharthS/LuxTTS`](https://huggingface.co/YatharthS/LuxTTS) (PyTorch). Pre-converted safetensors for Cake: [`evilsocket/luxtts`](https://huggingface.co/evilsocket/luxtts).

### Getting the Model

Pre-converted safetensors weights are available at [`evilsocket/luxtts`](https://huggingface.co/evilsocket/luxtts):

```sh
cake pull evilsocket/luxtts
```

#### Converting from Original Weights

If you prefer to convert from the original PyTorch weights ([`YatharthS/LuxTTS`](https://huggingface.co/YatharthS/LuxTTS)):

```sh
cake pull YatharthS/LuxTTS

python scripts/convert_luxtts.py \
  --model-dir ~/.cache/huggingface/hub/models--YatharthS--LuxTTS/snapshots/*/  \
  --output-dir /path/to/luxtts-converted
```

The conversion script produces `model.safetensors`, `vocos.safetensors`, `config.json`, and `tokens.txt`.

### Tokenization

LuxTTS uses IPA phoneme tokens. The built-in tokenizer provides a basic rule-based English-to-IPA fallback. For best quality, pre-compute IPA token IDs using an external phonemizer (e.g. espeak-ng via `piper_phonemize`) and pass them via `--tts-token-ids`:

```sh
# Generate IPA token IDs (example using Python)
python -c "
from piper_phonemize import phonemize_espeak
tokens_map = {}
with open('tokens.txt') as f:
    for line in f:
        tok, idx = line.strip().rsplit('\t', 1)
        tokens_map[tok] = int(idx)
ipa = phonemize_espeak('Hello world', 'en-us')[0][0]
ids = [tokens_map.get(c, 0) for c in '^' + ipa + '\$']
print(' '.join(str(i) for i in ids))
" > token_ids.txt

# Generate audio using pre-computed tokens
cake run evilsocket/luxtts \
  --prompt "Hello world" \
  --tts-token-ids token_ids.txt \
  --tts-speed 0.16 \
  --tts-diffusion-steps 4 \
  --audio-output output.wav \
  --dtype f32
```

### Basic Usage (built-in tokenizer)

```sh
cake run evilsocket/luxtts \
  --prompt "Hello world, this is a test." \
  --audio-output output.wav \
  --tts-speed 0.16 \
  --tts-diffusion-steps 4 \
  --dtype f32
```

### Voice Cloning

Provide a 24kHz mono WAV reference audio for voice cloning:

```sh
cake run evilsocket/luxtts \
  --prompt "Hello world" \
  --tts-reference-audio voice_sample.wav \
  --audio-output output.wav \
  --tts-speed 0.16 --tts-diffusion-steps 4 --dtype f32
```

### Distributed Inference

The FM decoder layers can be split across workers:

```yaml
# topology-luxtts.yml
worker1:
  host: "192.168.1.100:10128"
  layers:
    - "fm_decoder.layers.0"
    - "fm_decoder.layers.1"
    - "fm_decoder.layers.2"
    - "fm_decoder.layers.3"
    - "fm_decoder.layers.4"
    - "fm_decoder.layers.5"
    - "fm_decoder.layers.6"
    - "fm_decoder.layers.7"
# Master keeps fm_decoder.layers.8-15 + text encoder + vocoder
```

```sh
# Worker
cake run --model evilsocket/luxtts --name worker1 \
  --topology topology-luxtts.yml --address 0.0.0.0:10128

# Master
cake run evilsocket/luxtts --topology topology-luxtts.yml \
  --prompt "Hello world" --audio-output output.wav \
  --tts-speed 0.16 --tts-diffusion-steps 4 --dtype f32
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | (required) | Text to synthesize |
| `--audio-output` | `output.wav` | Output WAV file path |
| `--tts-diffusion-steps` | 10 | Euler solver steps (4 recommended for distilled LuxTTS) |
| `--tts-speed` | 1.0 | Speed factor (lower = longer audio; 0.16 ≈ 6 frames/token) |
| `--tts-t-shift` | 1.0 | Time shift for Euler solver schedule |
| `--tts-reference-audio` | - | Path to 24kHz mono WAV for voice cloning |
| `--tts-token-ids` | - | Pre-computed IPA token IDs file (space-separated) |
| `--dtype` | f16 | Use `f32` for best quality on CPU |

### Architecture

LuxTTS consists of three components:

1. **Text encoder** (4 Zipformer layers, dim=192) — converts IPA phoneme tokens to acoustic features. Always runs on master.
2. **FM decoder** (16 Zipformer layers across 5 multi-resolution stacks, dim=512) — flow matching decoder that denoises random noise into mel features over 4 Euler steps. **Shardable** across workers.
3. **Vocos vocoder** (ConvNeXt backbone + ISTFT head) — converts mel features to 24kHz waveform, then upsampled to 48kHz. Always runs on master.

The FM decoder uses a U-Net-style multi-resolution structure with downsampling factors [1, 2, 4, 2, 1] and layer counts [2, 2, 4, 4, 4] = 16 total layers.

---

## VibeVoice-1.5B

The recommended model. Supports multi-speaker voice cloning from raw .wav files.

### Basic Usage

```sh
cake run evilsocket/VibeVoice-1.5B --model-type audio-model \
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
cake run evilsocket/VibeVoice-1.5B --model-type audio-model \
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
cake run evilsocket/VibeVoice-Realtime-0.5B --model-type audio-model \
  --voice-prompt voice_preset.safetensors \
  --prompt "Hello world" --audio-output output.wav \
  --tts-cfg-scale 1.5
```

Voice presets for the 0.5B model are `.safetensors` files containing pre-computed KV caches. These can be found in the [VibeVoice community repo](https://github.com/vibevoice-community/VibeVoice/tree/main/demo/voices/streaming_model) as `.pt` files (PyTorch format) and converted to safetensors.

## API Endpoint

When using `cake serve`, the audio endpoint is available at `/v1/audio/speech`:

```sh
# Start the API server
cake serve evilsocket/VibeVoice-1.5B --model-type audio-model \
  --voice-prompt voice_reference.wav

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
