"""
Capture CFG diff from Python LTX-2 pipeline by patching the scheduler step.
"""
import torch
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 9
PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"

# Patch the pipeline's __call__ denoising loop via a callback
step_data = []

def capture_callback(pipe_obj, step_index, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")
    if latents is not None:
        flat = latents.float().flatten()
        print(f"Step {step_index}: latents min={flat.min():.4f}, max={flat.max():.4f}, std={flat.std():.6f}")
    return callback_kwargs

# Patch the actual transformer forward to capture CFG diff
import types

_orig_call = pipe.transformer.__class__.__call__

def _patched_call(self, *args, **kwargs):
    result = _orig_call(self, *args, **kwargs)

    # Get the video output
    if isinstance(result, tuple):
        video_out = result[0]
    else:
        video_out = result

    if video_out is not None and video_out.shape[0] == 2:
        uncond = video_out[0:1].float()
        cond = video_out[1:2].float()
        diff = cond - uncond
        print(f"  CFG batch: cond_std={cond.flatten().std():.6f}, uncond_std={uncond.flatten().std():.6f}, diff_std={diff.flatten().std():.6f}")

    return result

pipe.transformer.__class__.__call__ = _patched_call

print("Running pipeline...")
result = pipe(
    prompt=PROMPT,
    negative_prompt="",
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    num_inference_steps=5,
    guidance_scale=3.0,
    output_type="pt",
    callback_on_step_end=capture_callback,
    callback_on_step_end_tensor_inputs=["latents"],
)
print("Done!")
