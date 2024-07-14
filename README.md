<p align="center">
  <small>Join the project community on our server!</small>
  <br/><br/>
  <a href="https://discord.gg/https://discord.gg/btZpkp45gQ" target="_blank" title="Join our community!">
    <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/btZpkp45gQ"/>
  </a>
</p>
<hr/>

`Cake` is a pure Rust implementation of the [LLama3 distributed inference](https://x.com/evilsocket/status/1812110504531259900) based on [Candle](https://github.com/huggingface/candle). The goal of the project is being able to run big (70B+) models by repurposing consumer hardware into an heterogeneous cluster of iOS, macOS, Linux and Windows devices.

**This is experimental code**.

The idea is to shard the transformer blocks to multiple devices in order to be able to run the inference on models that wouldn't normally fit in the GPU memory of a single device. Inferences over contiguous transformer blocks on the same worker are batched in order to minimize latency due to data transfer.

Run a worker node:

```sh
cake-cli --model /path/to/Meta-Llama-3-8B \ # model path, read below on how to optimize model size for workers
         --mode worker \                    # run as worker
         --name worker0 \                   # worker name in topology file
         --topology topology.yml \          # topology
         --address 0.0.0.0:10128            # bind address
```

Run a master node:

```sh
cake-cli --model /path/to/Meta-Llama-3-8B \
         --topology topology.yml
```

Where `topology.yaml` determines which layers are served by whom:

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

As a memory and disk space optimization, you might want to give the worker only the data it actually needs from the model instead of the whole folder, in which case you can use the `cake-split-model` utility. For instance to generate a smaller version of the llama3 safetensors, you can:

```sh
cake-split-model --model-path path/to/Meta-Llama-3-8B \ # source model to split
                 --topology path/to/topology.yml \      # topology file
                 --output output-folder-name            # output folder where all the workers data bundles will be saved
```

This will create a smaller folder with only the required layers tensors and the topology file for the specific worker. Remember to also copy other model contents (config.json, tokenizer.json, etc) in the worker bundle before deploying it.

## Support

| OS                           | Architectures | Acceleration | Status |
|:----------------------------------:|:------------------:|:------------------:|:------------------:|
| GNU/Linux                 | arm, arm64, x86_64 | -                | :heavy_check_mark: |
| GNU/Linux                 | arm, arm64, x86_64 | CUDA                | :heavy_check_mark: |
| GNU/Linux                 | arm, arm64, x86_64 | BLAS                | :heavy_check_mark: |
| macOS                 | intel | -                | :heavy_check_mark: |
| macOS                 | aarch64 | -                | :heavy_check_mark: |
| macOS                 | aarch64 | Metal                | :heavy_check_mark: |
| Android                | arm, arm64, x86_64 | - | :heavy_check_mark: |
| Android                | arm, arm64, x86_64 | CUDA | untested |
| iOS / iPadOS                 | aarch64 | -                | :heavy_check_mark: |
| iOS / iPadOS                 | aarch64 | Metal                | [90% done, WIP](https://github.com/huggingface/candle/issues/2322) |
| Web                 | - | WebGPU                | [in theory possible, not done](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) |

## License

Released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.