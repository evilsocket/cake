"""
Save exact transformer inputs from Python to compare with Rust.
"""
import torch
import numpy as np
from safetensors.torch import save_file

from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 9
PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"

# Capture transformer inputs by monkey-patching
transformer_inputs = {}

original_transformer_forward = pipe.transformer.forward.__wrapped__ if hasattr(pipe.transformer.forward, '__wrapped__') else pipe.transformer.forward

def patched_transformer(*args, **kwargs):
    # Save all inputs
    transformer_inputs['hidden_states'] = kwargs.get('hidden_states', args[0] if args else None)
    transformer_inputs['encoder_hidden_states'] = kwargs.get('encoder_hidden_states')
    transformer_inputs['timestep'] = kwargs.get('timestep')
    transformer_inputs['encoder_attention_mask'] = kwargs.get('encoder_attention_mask')
    transformer_inputs['image_rotary_emb'] = kwargs.get('image_rotary_emb')

    # Print input stats
    for k, v in transformer_inputs.items():
        if v is not None and hasattr(v, 'shape'):
            flat = v.float().flatten()
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={flat.min():.6f}, max={flat.max():.6f}, std={flat.std():.6f}")
        elif v is not None and isinstance(v, tuple):
            for i, t in enumerate(v):
                if hasattr(t, 'shape'):
                    flat = t.float().flatten()
                    print(f"  {k}[{i}]: shape={t.shape}, dtype={t.dtype}, std={flat.std():.6f}")

    # Call original
    result = original_transformer_forward(*args, **kwargs)

    # Save output
    if hasattr(result, 'sample'):
        out = result.sample
    elif isinstance(result, tuple):
        out = result[0]
    else:
        out = result

    if hasattr(out, 'shape'):
        flat = out.float().flatten()
        print(f"  OUTPUT: shape={out.shape}, dtype={out.dtype}, min={flat.min():.6f}, max={flat.max():.6f}, std={flat.std():.6f}")
        transformer_inputs['output'] = out.cpu()

    return result

pipe.transformer.forward = patched_transformer

# Also capture sigma/timestep from the denoising loop
step_count = [0]
original_step = pipe.scheduler.step

def patched_step(model_output, timestep, sample, **kwargs):
    step_count[0] += 1
    if step_count[0] <= 2:
        print(f"\n--- Scheduler step {step_count[0]} ---")
        print(f"  timestep={timestep}")
        flat = model_output.float().flatten()
        print(f"  model_output: shape={model_output.shape}, min={flat.min():.6f}, max={flat.max():.6f}, std={flat.std():.6f}")
        flat_s = sample.float().flatten()
        print(f"  sample: shape={sample.shape}, min={flat_s.min():.6f}, max={flat_s.max():.6f}, std={flat_s.std():.6f}")

    result = original_step(model_output, timestep, sample, **kwargs)

    if step_count[0] <= 2:
        prev = result.prev_sample
        flat_p = prev.float().flatten()
        print(f"  prev_sample: min={flat_p.min():.6f}, max={flat_p.max():.6f}, std={flat_p.std():.6f}")

    return result

pipe.scheduler.step = patched_step

print("Running pipeline with transformer instrumentation (1 step only)...")
result = pipe(
    prompt=PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    num_inference_steps=2,
    guidance_scale=3.0,
    output_type="pt",
)

# Save the captured tensors
print("\nSaving captured tensors...")
to_save = {}
for k, v in transformer_inputs.items():
    if v is not None and hasattr(v, 'shape'):
        to_save[k] = v.float().cpu().contiguous()
    elif isinstance(v, tuple):
        for i, t in enumerate(v):
            if hasattr(t, 'shape'):
                to_save[f"{k}_{i}"] = t.float().cpu().contiguous()

save_file(to_save, "/tmp/ltx2_transformer_inputs.safetensors")
print(f"Saved {len(to_save)} tensors to /tmp/ltx2_transformer_inputs.safetensors")
for k, v in to_save.items():
    print(f"  {k}: {v.shape}")
