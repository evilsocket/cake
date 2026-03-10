"""
Capture the CFG diff (cond_velocity - uncond_velocity) from the Python LTX-2 pipeline.
This directly compares with the Rust CFG diff diagnostic.
"""
import torch
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 9
PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"

# Monkey-patch the transformer to capture cond/uncond velocities separately
call_count = [0]
original_forward = pipe.transformer.__class__.forward

def patched_forward(self, hidden_states, *args, **kwargs):
    call_count[0] += 1
    result = original_forward(self, hidden_states, *args, **kwargs)

    if hasattr(result, 'sample'):
        out = result.sample
    elif isinstance(result, tuple):
        out = result[0]
    else:
        out = result

    # Check if this is a batched CFG call (batch_size=2)
    if out.shape[0] == 2:
        uncond = out[0:1]
        cond = out[1:2]
        diff = (cond - uncond).float()
        diff_std = diff.flatten().std().item()
        cond_std = cond.float().flatten().std().item()
        uncond_std = uncond.float().flatten().std().item()
        print(f"\n--- Transformer call {call_count[0]} (CFG batch) ---")
        print(f"  cond velocity: std={cond_std:.6f}")
        print(f"  uncond velocity: std={uncond_std:.6f}")
        print(f"  CFG diff (cond - uncond): std={diff_std:.6f}")
        print(f"  diff / cond ratio: {diff_std / (cond_std + 1e-8):.4f}")
    elif out.shape[0] == 1:
        out_std = out.float().flatten().std().item()
        print(f"\n--- Transformer call {call_count[0]} (single) ---")
        print(f"  velocity: std={out_std:.6f}")

    return result

pipe.transformer.__class__.forward = patched_forward

# Also capture the context embeddings
original_encode = pipe.encode_prompt

def patched_encode(*args, **kwargs):
    result = original_encode(*args, **kwargs)
    # result is (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask)
    if len(result) >= 4 and result[0] is not None:
        pe = result[0]
        ne = result[2] if result[2] is not None else None
        print(f"\nPrompt embeds: shape={pe.shape}, std={pe.float().flatten().std():.6f}")
        if ne is not None:
            print(f"Negative embeds: shape={ne.shape}, std={ne.float().flatten().std():.6f}")
            diff = (pe - ne).float()
            print(f"Embed diff (prompt - negative): std={diff.flatten().std():.6f}")
    return result

pipe.encode_prompt = patched_encode

print("Running LTX-2 pipeline with CFG diff instrumentation...")
print(f"Prompt: {PROMPT}")
print(f"Resolution: {WIDTH}x{HEIGHT}, frames: {NUM_FRAMES}")

result = pipe(
    prompt=PROMPT,
    negative_prompt="",
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    num_inference_steps=5,
    guidance_scale=3.0,
    output_type="pt",
)

print("\nDone!")
