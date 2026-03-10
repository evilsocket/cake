# Usage

## Downloading Models

You can pass a HuggingFace repo ID as the `--model` argument and Cake will download the model automatically (with progress bars). Files are cached in `~/.cache/huggingface/hub/` — subsequent runs skip the download.

```sh
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --api 0.0.0.0:8080
```

To pre-download a model without running inference:

```sh
cake download Qwen/Qwen2.5-Coder-1.5B-Instruct
```

For gated models (like LLaMA 3), set the `HF_TOKEN` environment variable with your HuggingFace token.

You can also pass a local path to a model directory:

```sh
cake master --model /path/to/Meta-Llama-3-8B --api 0.0.0.0:8080
```

## Single Prompt

To quickly test a model with a single prompt (no API server):

```sh
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt "Why is the sky blue?"
```

## Listing Local Models

List all models available locally (downloaded from HuggingFace or received from a master):

```sh
cake models
```

This scans `~/.cache/huggingface/hub/` and `~/.cache/cake/` and shows each model's status (`complete` or `partial`), size, and source.

## Web UI

When running with the `--api` flag, Cake serves a web interface with two tabs:

```sh
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --api 0.0.0.0:8080
```

Open `http://localhost:8080` in your browser.

**Chat tab** — Interactive chat with the model. Supports streaming responses, shows token count and tokens/second throughput, and renders markdown in responses.

**Cluster tab** — Visualizes the distributed inference topology. Shows master and worker nodes, VRAM usage, layer assignments, and per-tensor details. Auto-refreshes every 10 seconds.

To protect the web UI with basic auth:

```sh
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --api 0.0.0.0:8080 --ui-auth user:pass
```

## TUI Chat

Cake includes a terminal-based chat client that connects to a running API server:

```sh
cake chat --server http://localhost:8080
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `Tab` | Switch between Chat and Cluster tabs |
| `Enter` | Send message |
| `Shift+Enter` | Insert newline |
| `Esc` / `Ctrl+C` | Quit |
| `PageUp` / `PageDown` | Scroll |

The Chat tab shows streaming responses with tokens/second stats. The Cluster tab displays topology info, VRAM usage, and layer distribution across nodes.

## API

Cake exposes an OpenAI-compatible REST API when running with `--api`:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/v1/chat/completions` | POST | Chat completion (streaming or blocking) |
| `/api/v1/chat/completions` | POST | Same as above (alternative path) |
| `/v1/models` | GET | List available models |
| `/api/v1/topology` | GET | Cluster topology as JSON |
| `/api/v1/image` | POST | Image generation (see [Stable Diffusion](stable_diffusion.md)) |

### Chat Completion

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

The request body accepts:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | array | Chat messages with `role` and `content` |
| `stream` | bool | Enable SSE streaming (default: false) |
| `max_tokens` | int | Maximum tokens to generate |
| `temperature` | float | Sampling temperature |

Streaming responses use Server-Sent Events (SSE) format, compatible with the OpenAI API.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | - | Model path or HuggingFace repo ID |
| `--prompt` | - | Single prompt for CLI inference |
| `--system-prompt` | `"You are a helpful AI assistant."` | System prompt |
| `--api` | - | API bind address (e.g. `0.0.0.0:8080`) |
| `--topology` | - | Topology file for manual cluster setup |
| `-n` / `--sample-len` | `2048` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature (0 = greedy) |
| `--top-k` | - | Top-K sampling |
| `--top-p` | - | Nucleus sampling cutoff |
| `--repeat-penalty` | `1.1` | Repeat token penalty (1.0 = none) |
| `--repeat-last-n` | `128` | Context window for repeat penalty |
| `--seed` | `299792458` | Random seed |
| `--device` | `0` | GPU device index |
| `--cpu` | `false` | Force CPU inference |
| `--dtype` | - | Override dtype (default: f16) |
| `--text-model-arch` | `auto` | Force model architecture (`auto`, `llama`, `qwen2`, `qwen3`, `qwen3-moe`, `qwen3-5`, `phi4`, `mistral`, `gemma3`, `falcon3`, `ol-mo2`, `exaone4`) |
| `--cluster-key` | - | Zero-config cluster key (or `CAKE_CLUSTER_KEY` env) |
| `--discovery-timeout` | `10` | Worker discovery timeout in seconds |
| `--ui-auth` | - | Basic auth for web UI (`user:pass`) |
