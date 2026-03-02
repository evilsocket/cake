# Docker (Linux/NVIDIA)

For Linux systems with NVIDIA GPUs, you can run Cake via Docker. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) is required.

## Building the Image

```sh
docker build -t cake .
```

Set `CUDA_COMPUTE_CAP` to match your GPU ([common values](https://developer.nvidia.com/cuda-gpus): 75 for RTX 2000, 80 for A100, 86 for RTX 3000, 89 for RTX 4000, 90 for H100):

```sh
docker build -t cake --build-arg CUDA_COMPUTE_CAP=86 .
```

## Running Standalone

Load the entire model in a single container (no cluster):

```sh
docker run --rm --gpus all \
  -v /path/to/model:/model:ro \
  -p 8080:8080 \
  cake master --model /model --api 0.0.0.0:8080
```

## Multi-Worker Cluster

A `docker-compose.yml` is provided as an example for running a multi-worker cluster. Create a `topology-docker.yml` mapping layers to the `worker-1` / `worker-2` service names, place your model data in `./cake-data/`, and run:

```sh
docker compose up --build
```

## macOS Note

Docker on macOS cannot access Metal GPUs. For Apple Silicon, build and run natively:

```sh
cargo build --release --features metal
```
