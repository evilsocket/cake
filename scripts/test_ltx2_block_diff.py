"""
Measure the hidden state diff between cond and uncond at each block boundary.
Uses register_forward_hook to work with sequential CPU offload.
"""
import torch
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"
WIDTH = 704
HEIGHT = 512
NUM_FRAMES = 41

# Register hooks on transformer blocks
block_call_count = [0]

def make_block_hook(block_idx):
    def hook(module, input, output):
        block_call_count[0] += 1
        video_out = output[0] if isinstance(output, tuple) else output
        b = video_out.shape[0]
        if b == 2 and block_call_count[0] <= 48:
            neg = video_out[0:1].float()
            pos = video_out[1:2].float()
            diff = pos - neg
            diff_std = diff.flatten().std().item()
            pos_std = pos.flatten().std().item()
            print(f"  block {block_idx:2d}: diff_std={diff_std:.6f}, pos_std={pos_std:.6f}")
    return hook

for i, block in enumerate(pipe.transformer.transformer_blocks):
    block.register_forward_hook(make_block_hook(i))

# Hook on proj_out
def proj_out_hook(module, input, output):
    b = output.shape[0]
    if b == 2:
        neg = output[0:1].float()
        pos = output[1:2].float()
        diff = pos - neg
        print(f"  proj_out (velocity): diff_std={diff.flatten().std():.6f}")

pipe.transformer.proj_out.register_forward_hook(proj_out_hook)

# Hook on caption_projection
if hasattr(pipe.transformer, 'caption_projection') and pipe.transformer.caption_projection is not None:
    def cap_proj_hook(module, input, output):
        b = output.shape[0]
        if b == 2:
            neg = output[0:1].float()
            pos = output[1:2].float()
            diff = pos - neg
            print(f"\n  caption_projection: diff_std={diff.flatten().std():.6f}")
    pipe.transformer.caption_projection.register_forward_hook(cap_proj_hook)

print("Running pipeline with per-block diff tracking...")
result = pipe(
    prompt=PROMPT,
    negative_prompt="",
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    num_inference_steps=2,
    guidance_scale=4.0,
    output_type="pt",
)
print("\nDone!")
