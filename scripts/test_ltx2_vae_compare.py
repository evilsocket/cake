"""
Compare LTX-2 VAE decode between Python and Rust.
Generates random latents, saves them, decodes with Python VAE,
saves the output for comparison.
"""
import torch
import numpy as np
from safetensors.torch import save_file, load_file

# Generate test latents matching Rust output dimensions
# From Rust: [1, 128, 2, 12, 16] (for 9 frames, 384x512)
torch.manual_seed(42)
latent_channels = 128
latent_f = 2
latent_h = 12
latent_w = 16

# Create latents similar to what the denoiser produces
latents = torch.randn(1, latent_channels, latent_f, latent_h, latent_w, dtype=torch.float32) * 0.8

# Save latents for Rust to use
save_file({"latents": latents}, "/tmp/test_latents.safetensors")
print(f"Test latents: shape={latents.shape}, min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")

# Load VAE
from diffusers import AutoencoderKLLTX2Video
vae = AutoencoderKLLTX2Video.from_pretrained(
    "Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.bfloat16
)
vae = vae.cuda()
vae.eval()

# Denormalize (same as pipeline does)
latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).cuda().to(torch.bfloat16)
latents_std = vae.latents_std.view(1, -1, 1, 1, 1).cuda().to(torch.bfloat16)
latents_bf16 = latents.cuda().to(torch.bfloat16)
denormed = latents_bf16 * latents_std + latents_mean

print(f"\nDenormalized: min={denormed.float().min():.4f}, max={denormed.float().max():.4f}, mean={denormed.float().mean():.4f}")

# Decode
with torch.no_grad():
    decoded = vae.decode(denormed, return_dict=False)[0]

decoded_f32 = decoded.float()
print(f"Decoded: shape={decoded.shape}, dtype={decoded.dtype}")
print(f"  min={decoded_f32.min():.4f}, max={decoded_f32.max():.4f}, mean={decoded_f32.mean():.4f}, std={decoded_f32.std():.4f}")

# Save first frame
frame = decoded[0, :, 0, :, :]  # [C, H, W]
frame = ((frame.float().clamp(-1, 1) + 1) * 127.5).byte()
frame = frame.permute(1, 2, 0).cpu().numpy()

from PIL import Image
img = Image.fromarray(frame)
img.save("/tmp/ltx2_python_vae_test.png")
print(f"\nSaved Python VAE output to /tmp/ltx2_python_vae_test.png")

# Also try decoding WITHOUT denormalization to see the raw effect
with torch.no_grad():
    decoded_raw = vae.decode(latents_bf16.cuda(), return_dict=False)[0]

decoded_raw_f32 = decoded_raw.float()
print(f"\nDecoded (no denorm): shape={decoded_raw.shape}")
print(f"  min={decoded_raw_f32.min():.4f}, max={decoded_raw_f32.max():.4f}, mean={decoded_raw_f32.mean():.4f}")

# Check the conv_in and conv_out dimensions
print(f"\nVAE decoder architecture:")
print(f"  conv_in: {vae.decoder.conv_in}")
if hasattr(vae.decoder, 'up_blocks'):
    for i, block in enumerate(vae.decoder.up_blocks):
        print(f"  up_block[{i}]: {type(block).__name__}, channels={getattr(block, 'in_channels', '?')}->{getattr(block, 'out_channels', '?')}")
print(f"  conv_out: {vae.decoder.conv_out}")
