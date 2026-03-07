# LTX-2 Distributed Video Generation Runbook

## Architecture

```
Linux Master (4090 24GB)             Windows Worker (5090 32GB)
├── ltx2-gemma (connector, 2.7GB)    ├── ltx2-transformer (36GB BF16)
├── Gemma-3 12B encoder (24GB, CPU)  └── serves via TCP :10128
├── ltx2-vae (~400MB)
└── ltx2-vocoder (~200MB)
```

VRAM note: the BF16 transformer is 36GB, the 5090 has 32GB. Candle loads
via mmap — overflow goes to system RAM via CUDA unified memory. It will
work but with some performance hit from page faults during forward pass.

## Step 1: Copy transformer weights to Windows

The worker ONLY needs the `transformer/` directory (36GB).

```bash
# Resolve the actual snapshot directory (symlinks)
SRC=$(readlink -f ~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/transformer/)

# Copy to Windows — adjust user@IP and destination path
scp -r $SRC user@WINDOWS_IP:C:/cake-models/Lightricks/LTX-2/transformer/
```

On Windows, the directory should look like:
```
C:\cake-models\Lightricks\LTX-2\transformer\
├── config.json
├── diffusion_pytorch_model.safetensors.index.json
├── diffusion_pytorch_model-00001-of-00008.safetensors
├── diffusion_pytorch_model-00002-of-00008.safetensors
├── ...
└── diffusion_pytorch_model-00008-of-00008.safetensors
```

36GB over 10GbE ~ 5 minutes.

## Step 2: Edit topology

```bash
# Replace WINDOWS_IP with the actual Windows machine IP
sed -i 's/WINDOWS_IP/192.168.1.XXX/' topology-ltx2.yml
```

## Step 3: Build on both machines

Linux:
```bash
cargo build --release --features cuda
```

Windows (PowerShell):
```powershell
cargo build --release --features cuda
```

## Step 4: Start Windows worker

```powershell
.\target\release\cake.exe worker `
  --model C:\cake-models\Lightricks\LTX-2 `
  --name win5090 `
  --topology topology-ltx2.yml `
  --address 0.0.0.0:10128 `
  --image-model-arch ltx2 `
  --ltx-version 2
```

The `--model` path should be the directory that CONTAINS `transformer/`.
Wait for: `Worker ready, listening on 0.0.0.0:10128`

If Windows firewall blocks it:
```powershell
netsh advfirewall firewall add rule name="cake" dir=in action=allow protocol=tcp localport=10128
```

## Step 5: Start Linux master

```bash
./target/release/cake master \
  --model ~/.cache/huggingface \
  --topology topology-ltx2.yml \
  --image-model-arch ltx2 \
  --ltx-version 2 \
  --prompt "a cat walking on the beach at sunset" \
  --ltx-height 512 \
  --ltx-width 704 \
  --ltx-num-frames 41 \
  --ltx-num-steps 30
```

## Expected log flow

1. Master loads connector (2.7GB GPU) + Gemma-3 (24GB, likely CPU) + VAE + vocoder
2. Master connects to Windows worker for ltx2-transformer
3. Text encoding: Gemma-3 encodes prompt → connector transforms → context embeddings
4. Denoising loop (30 steps): pack tensors → TCP to worker → transformer forward → TCP back
5. VAE decode locally → video frames
6. Output: AVI file

## Troubleshooting

**OOM on 5090**: The 36GB BF16 weights exceed 32GB VRAM. CUDA unified memory
should handle overflow to system RAM. If it crashes, reduce resolution:
`--ltx-height 384 --ltx-width 512 --ltx-num-frames 21`

**Worker can't find weights**: `--model` must point to the directory containing
`transformer/`. The code resolves `transformer/diffusion_pytorch_model.safetensors`
or the sharded index from that path.

**Connection timeout**: Verify both machines can reach each other on port 10128.
Test with: `nc -zv WINDOWS_IP 10128`

**Gemma-3 not loading**: Gemma is gated on HuggingFace. The HF token must be
saved at `~/.cache/huggingface/token` on the master. Already done.
