This is a Rust implementation of the llama3 model distributed inference. The idea is to shard the transformer blocks to multiple devices in order to be able to run the inference on models that wouldn't normally fit in the GPU memory of a single device. Inferences over contiguous block on the same worker are batched in order to minimize latency due to data transfer.

This is **experimental code**.

Run a worker node:

```
llama3-cake --model /path/to/Meta-Llama-3-8B --mode worker --topology worker-topology.json --address 0.0.0.0:10128
```

Run a master node:


```
llama3-cake --model /path/to/Meta-Llama-3-8B --topology master-topology.json --address 0.0.0.0:10128
```

Where:

`worker-topology.json`

```json
{
    "model.layers.16": "localhost:10128",
    "model.layers.17": "localhost:10128",
    "model.layers.18": "localhost:10128",
    "model.layers.19": "localhost:10128",
    "model.layers.20": "localhost:10128",
    "model.layers.21": "localhost:10128",
    "model.layers.22": "localhost:10128",
    "model.layers.23": "localhost:10128",
    "model.layers.24": "localhost:10128",
    "model.layers.25": "localhost:10128",
    "model.layers.26": "localhost:10128",
    "model.layers.27": "localhost:10128",
    "model.layers.28": "localhost:10128",
    "model.layers.29": "localhost:10128",
    "model.layers.30": "localhost:10128",
    "model.layers.31": "localhost:10128"
}
```

`master-topology.json`

```json
{
    "model.layers.0": "worker1-hostname:10128",
    "model.layers.1": "worker1-hostname:10128",
    "model.layers.2": "worker1-hostname:10128",
    "model.layers.3": "worker1-hostname:10128",
    "model.layers.4": "worker1-hostname:10128",
    "model.layers.5": "worker1-hostname:10128",
    "model.layers.6": "worker1-hostname:10128",
    "model.layers.7": "worker1-hostname:10128",
    "model.layers.8": "worker1-hostname:10128",
    "model.layers.9": "worker1-hostname:10128",
    "model.layers.10": "worker1-hostname:10128",
    "model.layers.11": "worker1-hostname:10128",
    "model.layers.12": "worker1-hostname:10128",
    "model.layers.13": "worker1-hostname:10128",
    "model.layers.14": "worker1-hostname:10128",
    "model.layers.15": "worker1-hostname:10128",
    "model.layers.25": "worker2-hostname:10128",
    "model.layers.26": "worker2-hostname:10128",
    "model.layers.27": "worker2-hostname:10128",
    "model.layers.28": "worker2-hostname:10128",
    "model.layers.29": "worker2-hostname:10128",
    "model.layers.30": "worker2-hostname:10128",
    "model.layers.31": "worker2-hostname:10128"
}
```

## License

Nerve is released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.

[![Star History Chart](https://api.star-history.com/svg?repos=evilsocket/nerve&type=Date)](https://star-history.com/#evilsocket/nerve&Date)
