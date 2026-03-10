"""
Test: what happens to per-block diff when audio stream is zeroed out?
This simulates what Rust does (skipping audio entirely).
"""
import torch
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"
WIDTH = 704
HEIGHT = 512
NUM_FRAMES = 41

# Monkey-patch each block to zero out audio contribution
def make_block_patch(original_forward, block_idx):
    def patched_forward(*args, **kwargs):
        # Call original
        video_out, audio_out = original_forward(*args, **kwargs)
        return video_out, audio_out
    return patched_forward

# Option 1: Zero out audio-to-video cross attention by patching blocks
# The a2v contribution is: hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states
# Let's patch audio_to_video_attn to return zeros
for i, block in enumerate(pipe.transformer.transformer_blocks):
    orig_a2v = block.audio_to_video_attn
    orig_v2a = block.video_to_audio_attn

    class ZeroAttn(torch.nn.Module):
        def forward(self, *args, **kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            return torch.zeros_like(hs)

    block.audio_to_video_attn = ZeroAttn()
    block.video_to_audio_attn = ZeroAttn()

# Track per-block diffs
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
            print(f"  block {block_idx:2d}: diff_std={diff_std:.6f}")
    return hook

for i, block in enumerate(pipe.transformer.transformer_blocks):
    block.register_forward_hook(make_block_hook(i))

# Hook proj_out
def proj_out_hook(module, input, output):
    b = output.shape[0]
    if b == 2:
        neg = output[0:1].float()
        pos = output[1:2].float()
        diff = pos - neg
        print(f"  proj_out (velocity): diff_std={diff.flatten().std():.6f}")
pipe.transformer.proj_out.register_forward_hook(proj_out_hook)

print("Running pipeline WITHOUT audio cross-attention...")
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
