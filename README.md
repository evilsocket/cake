`llama3-cake` is a pure Rust implementation of the llama3 LLM distributed inference based on [Candle](https://github.com/huggingface/candle).

**This is experimental code**.

The idea is to shard the transformer blocks to multiple devices in order to be able to run the inference on models that wouldn't normally fit in the GPU memory of a single device. Inferences over contiguous transformer blocks on the same worker are batched in order to minimize latency due to data transfer.

Run a worker node:

```bash
llama3-cake --model /path/to/Meta-Llama-3-8B --mode worker --name worker0 --topology topology.yml --address 0.0.0.0:10128
```

Run a master node:

```bash
llama3-cake --model /path/to/Meta-Llama-3-8B --topology topology.yml
```

Where `topology.yaml` determines which layers are served by whom:

```yaml
worker0:
  host: 'linux-server.local:10128'
  description: 'NVIDIA Titan X Pascal (12GB)'
  layers:
    - 'model.layers.0'
    - 'model.layers.1'
    - 'model.layers.2'
    - 'model.layers.3'
    - 'model.layers.4'
    - 'model.layers.5'
    - 'model.layers.6'
    - 'model.layers.7'
    - 'model.layers.8'
    - 'model.layers.9'
    - 'model.layers.10'
    - 'model.layers.11'
    - 'model.layers.12'
    - 'model.layers.13'
    - 'model.layers.14'
    - 'model.layers.15'

worker1:
  host: 'apple-server.local:10128'
  description: 'Apple M1 Max (64GB)'
  layers:
    - 'model.layers.16'
    - 'model.layers.17'
    - 'model.layers.18'
    - 'model.layers.19'
    - 'model.layers.20'
    - 'model.layers.21'
    - 'model.layers.22'
    - 'model.layers.23'
    - 'model.layers.24'
    - 'model.layers.25'
    - 'model.layers.26'
    - 'model.layers.27'
    - 'model.layers.28'
    - 'model.layers.29'
    - 'model.layers.30'
    - 'model.layers.31'
```

## License

Released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.