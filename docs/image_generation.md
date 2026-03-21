# Image Generation

Cake supports image generation with Stable Diffusion and FLUX models.

## Supported Models

| Model | Architecture | VRAM | Resolution | Steps | Quality |
|-------|-------------|------|-----------|-------|---------|
| Stable Diffusion 1.5 | SD | ~4 GB | 512x512 | 20-50 | Good |
| Stable Diffusion 2.1 | SD | ~5 GB | 768x768 | 20-50 | Good |
| Stable Diffusion XL | SD | ~7 GB | 1024x1024 | 20-50 | High |
| SDXL Turbo | SD | ~7 GB | 512x512 | 1-4 | Good (fast) |
| FLUX.2-klein-4B | FLUX | ~8 GB | 512x512 | 4 | Good (fast) |
| FLUX.1-dev (FP8) | FLUX | ~12 GB | up to 1024x1024 | 20 | Excellent |

## FLUX

### FLUX.1-dev (FP8)

High-quality 12B parameter flow-matching transformer. Runs in FP8 precision on a single GPU with 16GB+ VRAM. Models are downloaded automatically from HuggingFace.

```sh
cake run evilsocket/flux1-dev --model-type image-model --image-model-arch flux1 \
  --sd-image-prompt "a photorealistic landscape at golden hour, dramatic clouds" \
  --flux-height 768 --flux-width 1024 \
  --image-output landscape.png
```

### FLUX.2-klein-4B

Faster 4B variant, 4 denoising steps, best at 512x512:

```sh
cake run black-forest-labs/FLUX.2-klein-4B --model-type image-model --image-model-arch flux \
  --model black-forest-labs/FLUX.2-klein-4B \
  --sd-image-prompt "a fluffy orange cat sitting on a wooden table"
```

### FLUX Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--flux-height` | 1024 | Image height in pixels |
| `--flux-width` | 1024 | Image width in pixels |
| `--flux-steps` | 20 | Denoising steps (FLUX.2-klein uses 4) |
| `--flux-guidance` | 3.5 | CFG guidance scale |
| `--image-output` | `output.png` | Output file path (PNG) |

## Stable Diffusion

### Single Node

```sh
cake run evilsocket/flux1-dev --model-type image-model \
  --sd-image-prompt "An old man sitting on the chair at seaside" \
  --sd-version xl --sd-num-samples 1 --sd-image-seed 2439383
```

### Distributed Generation

Define the model parts in `topology.yml`:

```yaml
gpu_worker:
  host: 192.168.1.2:10128
  description: NVIDIA RTX 4090 24GB
  layers:
  - unet

macbook:
  host: 192.168.1.3:10128
  description: Macbook M2
  layers:
  - clip
  - vae
```

Start worker and master:

```sh
# Worker
cake run --model /path/to/hf/cache \
  --name gpu_worker --model-type image-model \
  --topology topology.yml --address 0.0.0.0:10128

# Master with API
cake serve /path/to/hf/cache \
  --model-type image-model --topology topology.yml
```

### API Endpoints

**OpenAI-compatible** (`/v1/images/generations`) — returns raw PNG by default:

```sh
curl http://master-ip:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "An old man sitting on the chair at seaside"}' \
  -o seaside.png
```

For base64 JSON (OpenAI client compatibility), add `"response_format": "b64_json"`.

**Legacy** (`/api/v1/image`):

```sh
curl http://master-ip:8080/api/v1/image \
  -H "Content-Type: application/json" \
  -d '{
    "image_args": {
      "sd-image-prompt": "An old man sitting on the chair at seaside",
      "sd-num-samples": 1,
      "sd-image-seed": 2439383
    }
}'
```

See the full [REST API Reference](api.md) for details.

### SD Versions

| Flag Value | Model |
|------------|-------|
| `v1-5` (default) | Stable Diffusion 1.5 |
| `v2-1` | Stable Diffusion 2.1 |
| `xl` | Stable Diffusion XL |
| `turbo` | SDXL Turbo |

### SD Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sd-version` | `v1-5` | SD version to use |
| `--sd-image-prompt` | (required) | Text prompt for image generation |
| `--sd-num-samples` | 1 | Number of images to generate |
| `--sd-image-seed` | random | Seed for reproducibility |
