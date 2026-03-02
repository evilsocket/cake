# Stable Diffusion Image Generation

Cake supports distributed image generation with Stable Diffusion 1.5, 2.1, XL, and XL Turbo.

## Topology

Define the model parts in `topology.yml`:

```yaml
wsl2_on_windows:
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

## Running

Start a worker node:

```sh
cake worker --model /path/to/hf/cache \
            --name wsl2_on_windows \
            --model-type image-model \
            --topology topology.yml \
            --address 0.0.0.0:10128
```

Start the master node with the API:

```sh
cake master --model /path/to/hf/cache \
            --api 0.0.0.0:8080 \
            --model-type image-model \
            --topology topology.yml
```

Model files are downloaded from HuggingFace automatically if not found in the local cache directory.

## Generating Images

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

## SD Versions

Switch between SD versions using the `--sd-version` flag:

| Flag Value | Model |
|------------|-------|
| `v1-5` (default) | Stable Diffusion 1.5 |
| `v2-1` | Stable Diffusion 2.1 |
| `xl` | Stable Diffusion XL |
| `turbo` | SDXL Turbo |

Additional SD arguments are available — see `cake master --help` for the full list.
