"""Check LTX-2 pipeline config without running encode."""
import torch

print("Loading LTX-2 pipeline (text_encoder=None to skip Gemma)...")
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
    cache_dir="/home/a/.cache/huggingface",
    text_encoder=None,
    tokenizer=None,
)

print("\n=== VAE Config ===")
vc = pipe.vae.config
print(f"  spatial_compression_ratio: {pipe.vae.spatial_compression_ratio}")
print(f"  temporal_compression_ratio: {pipe.vae.temporal_compression_ratio}")
print(f"  scaling_factor: {vc.scaling_factor}")
# Check all vae config keys
for k, v in vc.items():
    if 'latent' in k.lower() or 'mean' in k.lower() or 'std' in k.lower() or 'scaling' in k.lower():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] (len={len(v)})")
        else:
            print(f"  {k}: {v}")

print("\n=== Scheduler Config ===")
sc = pipe.scheduler.config
for k, v in sc.items():
    print(f"  {k}: {v}")

print("\n=== Scheduler Sigmas ===")
height, width, num_frames = 512, 704, 41
latent_f = (num_frames - 1) // pipe.vae.temporal_compression_ratio + 1
latent_h = height // pipe.vae.spatial_compression_ratio
latent_w = width // pipe.vae.spatial_compression_ratio
num_tokens = latent_f * latent_h * latent_w
print(f"  num_tokens = {num_tokens}")

pipe.scheduler.set_timesteps(30, device="cpu", n_tokens=num_tokens)
sigmas = pipe.scheduler.sigmas
timesteps = pipe.scheduler.timesteps
print(f"  Sigmas ({len(sigmas)} values):")
for i, s in enumerate(sigmas.tolist()):
    print(f"    [{i:2d}] {s:.6f}")
print(f"  Timesteps ({len(timesteps)} values): {timesteps.tolist()[:5]}...")

# Check how the pipeline normalizes latents
print("\n=== Pipeline latent normalization ===")
import inspect
src = inspect.getsource(pipe.__class__.__call__)
for i, line in enumerate(src.split('\n')):
    l = line.strip()
    if 'normalize' in l.lower() or 'latent_mean' in l.lower() or 'latent_std' in l.lower() or 'pack_latent' in l.lower():
        print(f"  Line {i}: {l}")

# Check how timestep is computed
for i, line in enumerate(src.split('\n')):
    l = line.strip()
    if 'timestep' in l.lower() and ('sigma' in l.lower() or '1.0' in l or '1 -' in l):
        print(f"  Line {i}: {l}")

print("\nDone!")
