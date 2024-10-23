<p align="center">
  <small>Join the project community on our server!</small>
  <br/><br/>
  <a href="https://discord.gg/https://discord.gg/btZpkp45gQ" target="_blank" title="Join our community!">
    <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/btZpkp45gQ"/>
  </a>
</p>
<hr/>


`Cake` is a Rust framework for distributed inference of large models like [LLama3](https://x.com/evilsocket/status/1812110504531259900) and [Stable Diffusion](https://x.com/crynuxai/status/1822085290543960216) based on [Candle](https://github.com/huggingface/candle). The goal of the project is being able to run big (70B+) models by repurposing consumer hardware into an heterogeneous cluster of iOS, Android, macOS, Linux and Windows devices, effectively leveraging [planned obsolescence](https://en.wikipedia.org/wiki/Planned_obsolescence) as a tool to make AI more accessible and democratic.

<p align="center">
  <strong>
  ⚠ This is experimental code that's being actively developed and changed very quickly, expect bugs ⚠
  </strong>
</p>

The idea is to shard the transformer blocks to multiple devices in order to be able to run the inference on models that wouldn't normally fit in the GPU memory of a single device. Inferences over contiguous transformer blocks on the same worker are batched in order to minimize latency due to data transfer.

## Support

| OS                           | Architectures | Acceleration | Status |
|----------------------------------|------------------|------------------|------------------|
| GNU/Linux                 | arm, arm64, x86_64 | -                | ✅ |
| GNU/Linux                 | arm, arm64, x86_64 | CUDA                | ✅ |
| GNU/Linux                 | arm, arm64, x86_64 | BLAS                | ✅ |
| Windows                | x86_64 | BLAS                | [untested](https://github.com/evilsocket/cake/issues/7) |
| Windows                | x86_64 | CUDA                |  ✅ |
| macOS                 | x86_64 | -                | ✅ |
| macOS                 | aarch64 | -                | ✅ |
| macOS                 | aarch64 | Metal                | ✅ |
| Android                | arm, arm64, x86_64 | - | ✅ |
| Android                | arm, arm64, x86_64 | CUDA | [untested](https://docs.nvidia.com/gameworks/content/technologies/mobile/cuda_android_main.htm) |
| iOS / iPadOS                 | aarch64 | -                | ✅ |
| iOS / iPadOS                 | aarch64 | Metal                | 🛠️  [90% done, WIP](https://github.com/huggingface/candle/issues/2322) |
| Web                 | - | WebGPU                | [in theory possible, not done](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) |

CUDA >= 12.2 is required for CUDA accelerated systems.

## Compile

With [Rust installed](https://www.rust-lang.org/tools/install), you can build the core library and the CLI utilities with different accelerations.

Without acceleration (will use CPU):

```sh
cargo build --release
```

With Metal acceleration for Apple Silicon:

```sh
cargo build --release --features metal
```

With CUDA acceleration:

```sh
cargo build --release --features cuda
```

To generate the iOS bindings in the app that can then be [compiled and deployed via XCode](https://github.com/evilsocket/cake/tree/main/cake-ios-worker-app):

```sh
make ios
```

## Using

Run a worker node:

```sh
cake-cli --model /path/to/Meta-Llama-3-8B \ # model path, read below on how to optimize model size for workers
         --mode worker \                    # run as worker
         --name worker0 \                   # worker name in topology file
         --topology topology.yml \          # topology
         --address 0.0.0.0:10128            # bind address
```

Run a master node with an OpenAI compatible REST API:

```sh
cake-cli --model /path/to/Meta-Llama-3-8B \ # model path
         --api 0.0.0.0:8080               \ # API bind address
         --topology topology.yml            # topology file
```

You can also omit the topology file to load the entire model in a single instance of cake:

```sh
cake-cli --model /path/to/Meta-Llama-3-8B \ # model path
         --api 0.0.0.0:8080                # API bind address
```

### Topology

The `topology.yml` determines which layers are served by which worker (you can find a list of all the layers of a model in its [tensor index file](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/model.safetensors.index.json)):

```yaml
linux_server_1:
  host: 'linux_server.host:10128'
  description: 'NVIDIA Titan X Pascal (12GB)'
  layers:
    - 'model.layers.0-5'

linux_server_2:
  host: 'linux_server2.host:10128'
  description: 'NVIDIA GeForce 3080 (10GB)'
  layers:
    - 'model.layers.6-16'

iphone:
  host: 'iphone.host:10128'
  description: 'iPhone 15 Pro Max'
  layers:
    - 'model.layers.17'

ipad:
  host: 'ipad.host:10128'
  description: 'iPad'
  layers:
    - 'model.layers.18-19'

macbook:
  host: 'macbook.host:10128'
  description: 'M1 Max'
  layers:
    - 'model.layers.20-31'
```

You can now interact with the cluster by:

```sh
curl http://master-ip:8080/api/v1/chat/completions \                                                                                                                           ~
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Why is the sky blue?"
        }
    ]
}'
```

### Splitting the Model

As a memory and disk space optimization, you might want to give the worker only the data it actually needs from the model instead of the whole folder, in which case you can use the `cake-split-model` utility. For instance to generate a smaller version of the llama3 safetensors, you can:

```sh
cake-split-model --model-path path/to/Meta-Llama-3-8B \ # source model to split
                 --topology path/to/topology.yml \      # topology file
                 --output output-folder-name            # output folder where all the workers data bundles will be saved
```

This will create a smaller folder with only the required layers tensors and the topology file for the specific worker. Remember to also copy other model contents (config.json, tokenizer.json, etc) in the worker bundle before deploying it.

### Stable Diffusion Image Generation

Define the model parts inside `topology.yml`:

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

Run a worker node:

```sh
cake-cli --model /path/to/hf/cache \        # The cache dir for huggingface models
         --mode worker \                    # run as worker
         --name wsl2_on_windows \           # worker name in topology file
         --model-type image-model \         # use image-model for SD, text-model or skip for LLM
         --topology topology.yml \          # topology
         --address 0.0.0.0:10128            # bind address
```

The model could be switched between SD1.5, SD2.1, SDXL and SDXL Turbo by specifying [more command line arguments](./cake-core/src/lib.rs).

The model files will be downloaded from Huggingface automatically if not found in the local cache directory.

Run a master node with REST API:

```sh
cake-cli --model /path/to/hf/cache \        # The cache dir for huggingface models
         --api 0.0.0.0:8080 \               # API bind address
         --model-type image-model \         # use image-model for SD, text-model or skip for LLM
         --topology topology.yml            # topology file
```

Generate images using the cluster:

```sh
curl http://master-ip:8080/api/v1/image \                                                                                                                           ~
  -H "Content-Type: application/json" \
  -d '{
    "image_args": {
      "sd-image-prompt": "An old man sitting on the chair at seaside",
      "sd-num-samples": 1,
      "sd-image-seed": 2439383
    }
}'
```

More control arguments could be found [inside the codes](./cake-core/src/lib.rs).


## Using w/ Docker
Splitting the Model:

```sh
docker run --rm -v /path/to/data:/data ghcr.io/evilsocket/cake \
  cake-split-model --model-path /data/Meta-Llama-3-8B \   # source model to split
                   --topology /data/topology.yml \        # topology file
                   --output /data/output-folder-name      # output folder
```

Run a worker node:

```sh
docker run --rm --network host -v /path/to/data:/data ghcr.io/evilsocket/cake \
  cake-cli --model /data/Meta-Llama-3-8B \    # model path
           --mode worker \                    # run as worker
           --name worker0 \                   # worker name in topology file
           --topology /data/topology.yml \    # topology
           --address 0.0.0.0:10128            # bind address
```

Run a master node with an OpenAI compatible REST API:

```sh
docker run --rm --network host -v /path/to/data:/data ghcr.io/evilsocket/cake \
  cake-cli --model /data/Meta-Llama-3-8B \    # model path
           --api 0.0.0.0:8080 \               # API bind address
           --topology /data/topology.yml      # topology file
```


## License

Released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.
