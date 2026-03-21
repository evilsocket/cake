# REST API

Cake exposes an OpenAI-compatible REST API when running with the `--api` flag. The same server handles text chat, image generation, and audio/TTS — only endpoints for the loaded model type return results; all others return `404` with a JSON error.

```sh
# Text model
cake master --model evilsocket/Qwen2.5-Coder-1.5B-Instruct --api 0.0.0.0:8080

# Image model
cake master --model-type image-model --image-model-arch flux1 --api 0.0.0.0:8080

# Audio model
cake master --model-type audio-model --model evilsocket/VibeVoice-1.5B \
  --voice-prompt voice.wav --api 0.0.0.0:8080
```

## Endpoints

| Endpoint | Method | Model Type | Description |
|----------|--------|------------|-------------|
| `/v1/chat/completions` | POST | Text | Chat completion (OpenAI-compatible) |
| `/api/v1/chat/completions` | POST | Text | Alias for the above |
| `/v1/audio/speech` | POST | Audio | Text-to-speech generation |
| `/v1/images/generations` | POST | Image | Image generation (OpenAI-compatible) |
| `/api/v1/image` | POST | Image | Image generation (legacy format) |
| `/v1/models` | GET | Any | List loaded models |
| `/api/v1/topology` | GET | Any | Cluster topology as JSON |
| `/` | GET | Any | Web UI |

## Chat Completion

**`POST /v1/chat/completions`**

OpenAI-compatible chat completion with optional streaming.

### Request

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Why is the sky blue?"}
    ],
    "stream": true,
    "max_tokens": 4096
}'
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `messages` | array | Yes | - | Chat messages, each with `role` (`system`, `user`, `assistant`) and `content` |
| `stream` | bool | No | `false` | Enable Server-Sent Events streaming |
| `max_tokens` | int | No | `2048` | Maximum tokens to generate |
| `model` | string | No | - | Ignored (uses the loaded model) |
| `temperature` | float | No | - | Sampling temperature |

### Response (blocking)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "evilsocket/Qwen2.5-Coder-1.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "The sky appears blue because..." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 42,
    "total_tokens": 44
  }
}
```

### Response (streaming)

With `"stream": true`, the response is `text/event-stream` with SSE chunks:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"...","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Error (no text model loaded)

```json
HTTP 404
{"error": "No text model loaded"}
```

## Audio Speech

**`POST /v1/audio/speech`**

Generate speech from text using VibeVoice TTS. Returns audio bytes directly.

### Request

```sh
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test.",
    "voice_path": "/path/to/voice_reference.wav",
    "response_format": "wav"
}' -o output.wav
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | Yes | - | Text to synthesize |
| `model` | string | No | - | Ignored (uses the loaded model) |
| `voice` | string | No | - | Voice name (reserved for future use) |
| `voice_data` | string | No | - | Base64-encoded WAV bytes for voice cloning |
| `voice_path` | string | No | - | Server-side path to voice prompt file |
| `response_format` | string | No | `"wav"` | `"wav"` (16-bit PCM WAV) or `"pcm"` (raw f32 little-endian) |
| `cfg_scale` | float | No | `1.5` | Classifier-free guidance scale (1.0-3.0) |
| `max_frames` | int | No | `150` | Max speech frames (~133ms each at 7.5Hz) |
| `diffusion_steps` | int | No | `10` | Diffusion steps per frame (higher = better, slower) |

### Voice Cloning via API

To clone a voice without server-side files, base64-encode the WAV reference and send it in `voice_data`:

```sh
VOICE_B64=$(base64 -w0 voice_reference.wav)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"Hello from a cloned voice.\",
    \"voice_data\": \"$VOICE_B64\"
}" -o output.wav
```

### Response

- **`response_format: "wav"`** — `Content-Type: audio/wav`, binary WAV body (16-bit PCM, 24kHz, mono)
- **`response_format: "pcm"`** — `Content-Type: audio/pcm`, raw f32 little-endian samples at 24kHz

### Error (no audio model loaded)

```json
HTTP 404
{"error": "No audio model loaded"}
```

## Image Generation (OpenAI-compatible)

**`POST /v1/images/generations`**

Generate images from text prompts. By default returns raw PNG bytes with `Content-Type: image/png`.

### Request

```sh
# Raw PNG (default) — pipe to file or viewer
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A photorealistic landscape at golden hour"}' \
  -o landscape.png

# Base64 JSON (OpenAI-compatible)
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A landscape", "response_format": "b64_json"}'
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of the image |
| `n` | int | No | `1` | Number of images (currently always generates 1) |
| `size` | string | No | - | Reserved (size controlled by `--flux-width`/`--flux-height`) |
| `response_format` | string | No | `"png"` | `"png"` (raw image/png) or `"b64_json"` (OpenAI JSON envelope) |

### Response (`"png"`, default)

`Content-Type: image/png` — raw PNG bytes.

### Response (`"b64_json"`)

```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUh..."
    }
  ]
}
```

### Error (no image model loaded)

```json
HTTP 404
{"error": "No image model loaded"}
```

## Image Generation (Legacy)

**`POST /api/v1/image`**

Original image generation endpoint. Accepts SD/FLUX generation arguments directly.

### Request

```sh
# Raw PNG
curl http://localhost:8080/api/v1/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_args": {"sd-image-prompt": "An old man at seaside"},
    "response_format": "png"
}' -o output.png

# Base64 JSON (default, backwards-compatible)
curl http://localhost:8080/api/v1/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_args": {
      "sd-image-prompt": "An old man sitting on the chair at seaside",
      "sd-num-samples": 1,
      "sd-image-seed": 2439383
    }
}'
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_args` | object | Yes | - | SD/FLUX generation arguments |
| `response_format` | string | No | `"b64_json"` | `"b64_json"` (backwards-compatible) or `"png"` (raw image/png) |

### Response (`"b64_json"`, default)

```json
{
  "images": ["iVBORw0KGgoAAAANSUh..."]
}
```

### Response (`"png"`)

`Content-Type: image/png` — raw PNG bytes.

## List Models

**`GET /v1/models`**

Returns the currently loaded model(s) in OpenAI-compatible format.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "evilsocket/Qwen2.5-Coder-1.5B-Instruct",
      "object": "model",
      "owned_by": "cake"
    }
  ]
}
```

When no model is loaded, `data` is an empty array.

## Cluster Topology

**`GET /api/v1/topology`**

Returns detailed cluster topology including master/worker nodes, layer assignments, VRAM usage, and per-tensor metadata.

Protected by `--ui-auth` if configured.

## Error Handling

All endpoints return structured JSON errors:

| Status | Condition |
|--------|-----------|
| `404` | Endpoint requires a model type that isn't loaded |
| `400` | Malformed request body or invalid parameters |
| `500` | Internal inference error |

Error response format:

```json
{"error": "description of what went wrong"}
```

## Using with OpenAI Client Libraries

The chat endpoint is fully compatible with OpenAI client libraries. Image and audio endpoints return raw binary by default for efficiency — use `response_format: "b64_json"` if you need the OpenAI JSON envelope.

### Python (openai)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

# Chat (streaming)
response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Image (b64_json for OpenAI client compatibility)
response = client.images.generate(
    prompt="A rusty robot on a beach",
    response_format="b64_json",
)
print(response.data[0].b64_json[:50])
```

### Python (requests) — raw binary

```python
import requests

# Image — raw PNG
resp = requests.post("http://localhost:8080/v1/images/generations",
    json={"prompt": "A cat"})
with open("cat.png", "wb") as f:
    f.write(resp.content)

# Audio — raw WAV
resp = requests.post("http://localhost:8080/v1/audio/speech",
    json={"input": "Hello world"})
with open("speech.wav", "wb") as f:
    f.write(resp.content)
```

### curl

```sh
# Image — save PNG directly
curl -s http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A rusty robot"}' \
  -o robot.png

# Audio — save WAV directly
curl -s http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from Cake TTS"}' \
  -o speech.wav

# Audio — play immediately (requires ffplay)
curl -s http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Streaming audio", "response_format": "pcm"}' \
  | ffplay -f f32le -ar 24000 -ac 1 -
```
