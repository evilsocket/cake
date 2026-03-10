"""
Run the official Python diffusers LTX-2 pipeline to verify the model works.
Uses sequential CPU offloading to fit on a single 4090.
"""
import torch
import time
import sys
import gc

# Use small resolution for speed
WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 9  # minimum
NUM_STEPS = 15
GUIDANCE = 3.0
PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"

print(f"Testing LTX-2 Python pipeline")
print(f"Resolution: {WIDTH}x{HEIGHT}, frames={NUM_FRAMES}, steps={NUM_STEPS}, guidance={GUIDANCE}")
print(f"Prompt: {PROMPT}")

try:
    from diffusers import LTX2Pipeline
except ImportError:
    print("ERROR: diffusers LTX2Pipeline not available. Need diffusers >= 0.37.0")
    sys.exit(1)

print("\nLoading pipeline with sequential CPU offloading...")
t0 = time.time()

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
)
# Sequential CPU offload moves one layer at a time to GPU — uses less VRAM
pipe.enable_sequential_cpu_offload()

print(f"Pipeline loaded in {time.time()-t0:.1f}s")

# Monkey-patch to capture intermediate values
original_pack = pipe._pack_text_embeds.__func__ if hasattr(pipe._pack_text_embeds, '__func__') else None

# Use a callback to inspect intermediates
def callback(pipe_obj, step_idx, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    if step_idx < 3 or step_idx == NUM_STEPS - 1:
        flat = latents.float().flatten()
        print(f"  step {step_idx+1}: latents shape={latents.shape}, "
              f"min={flat.min():.4f}, max={flat.max():.4f}, "
              f"mean={flat.mean():.4f}, std={flat.std():.4f}")
    return callback_kwargs

print("\nRunning pipeline...")
t0 = time.time()

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

dt = time.time() - t0
print(f"\nPipeline completed in {dt:.1f}s")

# Analyze output
video = result.frames  # should be tensor
if hasattr(video, 'shape'):
    print(f"Output shape: {video.shape}, dtype={video.dtype}")
    flat = video.float().flatten()
    print(f"Output stats: min={flat.min():.4f}, max={flat.max():.4f}, "
          f"mean={flat.mean():.4f}, std={flat.std():.4f}")

# Save first frame as PNG for visual inspection
try:
    if hasattr(video, 'shape'):
        # output_type="pt" gives [B, F, C, H, W]
        frame = video[0, 0]  # first batch, first frame: [C, H, W]
        if frame.shape[0] == 3:
            # Already [0, 1] from pipeline
            frame = (frame.float().clamp(0, 1) * 255).byte()
            frame = frame.permute(1, 2, 0)  # [H, W, C]

        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame.cpu().numpy())
        img.save("/tmp/ltx2_python_test.png")
        print(f"\nSaved first frame to /tmp/ltx2_python_test.png")
except Exception as e:
    print(f"Could not save frame: {e}")

print("\nDone!")
