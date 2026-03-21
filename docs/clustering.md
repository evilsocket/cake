# Distributed Inference

Cake shards transformer blocks across multiple devices so you can run models that don't fit on a single GPU. Contiguous blocks on the same worker are batched to minimize network latency.

## Zero-Config Cluster (mDNS Discovery)

The simplest way to set up a cluster. Start workers and a master with the same `--cluster-key`:

```sh
# On any machine — start a worker (no model data or topology needed)
cake worker --cluster-key mysecret --name gpu-server-1

# On another machine
cake worker --cluster-key mysecret --name macbook

# On the master (has the model)
cake serve /path/to/Meta-Llama-3-8B \
            --cluster-key mysecret
```

The cluster key can also be set via the `CAKE_CLUSTER_KEY` environment variable.

### What happens automatically

1. Each worker detects its GPUs (name and VRAM via `nvidia-smi`, or system memory for Metal/CPU) and advertises itself on the local network via mDNS.
2. The master discovers all workers that share the same cluster key.
3. Master and each worker perform mutual authentication (see below).
4. The master assigns transformer layers to workers proportionally to their total GPU VRAM — machines with more memory get more layers.
5. If a worker doesn't have the model data cached locally, the master streams the required safetensors files over TCP. Transfer speed is logged on both sides. Workers cache received data in `~/.cache/cake/` for future runs.
6. Once all workers are ready, inference starts normally.

If no workers are discovered within the timeout (default 10 seconds, configurable with `--discovery-timeout`), the master loads all layers locally and serves the API anyway.

### Security model

The `--cluster-key` enables mutual HMAC-SHA256 challenge-response authentication. When a connection is established, both sides prove they possess the same key through a two-round-trip nonce exchange *before* any protocol data is transmitted — an unauthenticated peer never sees valid Cake messages.

This prevents unauthorized nodes from joining the cluster or injecting/intercepting inference data. However, the traffic itself is **not encrypted** — the authentication ensures both parties are legitimate, but the tensor data and model weights are sent in plaintext. This is appropriate for trusted local networks (home lab, office LAN, VPN). For untrusted networks, use a VPN or SSH tunnel to provide transport encryption.

The cluster key is hashed (SHA-256, first 8 hex chars) before being included in mDNS advertisements, so the key itself is never broadcast — only a short hash used for filtering during discovery.

## Manual Topology

For full control over layer placement, use a topology file.

### Running the cluster

Run a worker node:

```sh
cake worker --model /path/to/Meta-Llama-3-8B \
            --name worker0 \
            --topology topology.yml \
            --address 0.0.0.0:10128
```

Run a master node with an OpenAI-compatible REST API:

```sh
cake serve /path/to/Meta-Llama-3-8B \
            --topology topology.yml
```

You can also omit the topology file to load the entire model in a single instance:

```sh
cake serve /path/to/Meta-Llama-3-8B
```

### Topology file format

The `topology.yml` determines which layers are served by which worker. You can find a list of all the layers of a model in its [tensor index file](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/model.safetensors.index.json):

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

## Splitting the Model

As a memory and disk space optimization, you can give each worker only the data it needs instead of the whole model. Use the `split` subcommand to generate per-worker bundles:

```sh
cake split --model-path path/to/Meta-Llama-3-8B \
           --topology path/to/topology.yml \
           --output output-folder-name
```

This creates a smaller folder with only the required layer tensors and the topology file for the specific worker. Remember to also copy other model contents (`config.json`, `tokenizer.json`, etc.) into the worker bundle before deploying it.
