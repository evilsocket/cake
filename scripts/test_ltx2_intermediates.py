"""
Capture intermediate tensor stats from the Python LTX-2 pipeline to compare with Rust.
"""
import torch
import time
import sys

WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 9
NUM_STEPS = 5
GUIDANCE = 3.0
PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"

from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

# 1. Patch _pack_text_embeds
original_pack = pipe._pack_text_embeds

def patched_pack(*args, **kwargs):
    result = original_pack(*args, **kwargs)
    flat = result.float().flatten()
    nonzero = flat[flat.abs() > 1e-8]
    print(f"\n=== _pack_text_embeds output ===")
    print(f"  shape={result.shape}, dtype={result.dtype}")
    print(f"  all: min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}, std={flat.std():.6f}")
    if len(nonzero) > 0:
        print(f"  nonzero ({len(nonzero)}/{len(flat)}): min={nonzero.min():.6f}, max={nonzero.max():.6f}, std={nonzero.std():.6f}")
    # Check hidden state stats from first positional arg
    if len(args) > 0:
        ths = args[0]
        print(f"  input: {ths.shape}, dtype={ths.dtype}")
        if ths.dim() == 4:
            for l in [0, 24, 47, 48]:
                if l < ths.shape[-1]:
                    layer = ths[0, :, :, l].float().flatten()
                    nonz = layer[layer.abs() > 1e-8]
                    if len(nonz) > 0:
                        print(f"    layer {l}: std={nonz.std():.4f}, min={nonz.min():.4f}, max={nonz.max():.4f}")
    return result

pipe._pack_text_embeds = patched_pack

# 2. Patch connectors
connectors = pipe.connectors
if connectors is not None:
    # Find text_proj_in and video_connector
    print(f"\nConnectors type: {type(connectors).__name__}")
    for name, mod in connectors.named_children():
        print(f"  {name}: {type(mod).__name__}")

    # Patch text_proj_in
    if hasattr(connectors, 'text_proj_in'):
        original_proj = connectors.text_proj_in.forward

        def patched_proj(*args, **kwargs):
            result = original_proj(*args, **kwargs)
            flat = result.float().flatten()
            nonzero = flat[flat.abs() > 1e-8]
            print(f"\n=== text_proj_in output ===")
            print(f"  shape={result.shape}")
            print(f"  all: min={flat.min():.6f}, max={flat.max():.6f}, std={flat.std():.6f}")
            if len(nonzero) > 0:
                print(f"  nonzero ({len(nonzero)}/{len(flat)}): std={nonzero.std():.6f}")
            return result

        connectors.text_proj_in.forward = patched_proj

    # Patch video_connector
    if hasattr(connectors, 'video_connector'):
        vc = connectors.video_connector
        original_vc = vc.forward

        def patched_vc(*args, **kwargs):
            result = original_vc(*args, **kwargs)
            if isinstance(result, tuple):
                emb = result[0]
            else:
                emb = result
            flat = emb.float().flatten()
            nonzero = flat[flat.abs() > 1e-8]
            print(f"\n=== video_connector output ===")
            print(f"  shape={emb.shape}")
            print(f"  all: min={flat.min():.6f}, max={flat.max():.6f}, std={flat.std():.6f}")
            if len(nonzero) > 0:
                print(f"  nonzero ({len(nonzero)}/{len(flat)}): std={nonzero.std():.6f}")
            return result

        connectors.video_connector.forward = patched_vc

    # Patch full connectors forward
    original_conn_fwd = connectors.forward

    def patched_conn_fwd(*args, **kwargs):
        result = original_conn_fwd(*args, **kwargs)
        if isinstance(result, tuple):
            emb = result[0]
            mask = result[1] if len(result) > 1 else None
        else:
            emb = result
            mask = None
        flat = emb.float().flatten()
        print(f"\n=== connectors.forward output ===")
        print(f"  shape={emb.shape}")
        print(f"  all: min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}, std={flat.std():.6f}")
        if mask is not None:
            print(f"  mask: shape={mask.shape}, sum={mask.float().sum():.0f}")
        return result

    connectors.forward = patched_conn_fwd

# 3. Patch caption_projection
if hasattr(pipe.transformer, 'caption_projection') and pipe.transformer.caption_projection is not None:
    original_caption = pipe.transformer.caption_projection.forward

    def patched_caption(*args, **kwargs):
        result = original_caption(*args, **kwargs)
        flat = result.float().flatten()
        print(f"\n=== caption_projection output ===")
        print(f"  shape={result.shape}")
        print(f"  min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}, std={flat.std():.6f}")
        return result

    pipe.transformer.caption_projection.forward = patched_caption
else:
    print("No caption_projection found")

# Callback for denoiser
def callback(pipe_obj, step_idx, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    if step_idx < 3:
        flat = latents.float().flatten()
        print(f"\n  step {step_idx+1}: latents min={flat.min():.4f}, max={flat.max():.4f}, "
              f"mean={flat.mean():.4f}, std={flat.std():.4f}")
    return callback_kwargs

print("\nRunning pipeline...")
result = pipe(
    prompt=PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_STEPS,
    guidance_scale=GUIDANCE,
    callback_on_step_end=callback,
    output_type="pt",
)

print(f"\n=== Final output ===")
video = result.frames
flat = video.float().flatten()
print(f"  shape={video.shape}, min={flat.min():.4f}, max={flat.max():.4f}, mean={flat.mean():.4f}, std={flat.std():.4f}")
