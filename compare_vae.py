"""Decode latents with Python VAE and compare to Rust output."""
import json
import torch
import numpy as np
from PIL import Image

# Load latents saved by Rust
print("Loading latents...")
with open("videos/latents_pre_vae.json", "rb") as f:
    shape, flat = json.load(f)
latents = torch.tensor(flat, dtype=torch.float32).reshape(shape)
print(f"  Latents shape: {latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")

# Load LTX-2 VAE only (skip text encoder to save VRAM)
print("Loading LTX-2 VAE...")
from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
vae = AutoencoderKLLTX2Video.from_pretrained(
    "Lightricks/LTX-2",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    cache_dir="/home/a/.cache/huggingface",
)
vae = vae.to("cuda:0")
vae.eval()

# Decode
print("Decoding with Python VAE...")
with torch.no_grad():
    latents_bf16 = latents.to(dtype=torch.bfloat16, device="cuda:0")
    decoded = vae.decode(latents_bf16, return_dict=False)[0]

print(f"  Decoded shape: {decoded.shape}, min={decoded.float().min():.4f}, max={decoded.float().max():.4f}, mean={decoded.float().mean():.4f}")

# Save frame 0 and frame 20
decoded_f32 = decoded.float().cpu()
for fidx in [0, 20]:
    frame = decoded_f32[0, :, fidx]  # [3, H, W]
    frame = ((frame.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    frame = frame.permute(1, 2, 0).numpy()  # [H, W, 3]
    Image.fromarray(frame).save(f"videos/python_vae_frame_{fidx:04d}.png")
    print(f"  Saved videos/python_vae_frame_{fidx:04d}.png")

# Also load Rust frames for comparison
for fidx in [0, 20]:
    rust_img = np.array(Image.open(f"videos/frames/frame_{fidx:04d}.png"))
    py_img = np.array(Image.open(f"videos/python_vae_frame_{fidx:04d}.png"))
    diff = np.abs(rust_img.astype(float) - py_img.astype(float))
    print(f"  Frame {fidx} diff: mean={diff.mean():.2f}, max={diff.max():.0f}")

print("Done!")
