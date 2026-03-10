"""
Save block 0 cross-attention inputs/outputs for direct comparison with Rust.
Uses register_forward_hook to work with sequential CPU offload.
"""
import torch
from safetensors.torch import save_file
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"
WIDTH = 704
HEIGHT = 512
NUM_FRAMES = 41

captured = {}

# Hook on block 0 to capture input/output
block0_call = [0]

def block0_hook(module, input, output):
    block0_call[0] += 1
    if block0_call[0] > 1:
        return

    # input is a tuple of args
    hidden_states = input[0]  # First positional arg
    video_out = output[0] if isinstance(output, tuple) else output
    b = video_out.shape[0]

    print(f"\n  Block 0 hook: input={hidden_states.shape}, output={video_out.shape}, batch={b}")

    if b == 2:
        neg_in = hidden_states[0].float()
        pos_in = hidden_states[1].float()
        in_diff = pos_in - neg_in
        print(f"    input  diff_std={in_diff.std():.6f} (should be ~0)")
        print(f"    input  neg_std={neg_in.std():.6f}, pos_std={pos_in.std():.6f}")

        neg_out = video_out[0].float()
        pos_out = video_out[1].float()
        out_diff = pos_out - neg_out
        print(f"    output diff_std={out_diff.std():.6f}")
        print(f"    output neg_std={neg_out.std():.6f}, pos_std={pos_out.std():.6f}")

        captured["block0_in_neg"] = hidden_states[0:1].float().cpu().contiguous()
        captured["block0_in_pos"] = hidden_states[1:2].float().cpu().contiguous()
        captured["block0_out_neg"] = video_out[0:1].float().cpu().contiguous()
        captured["block0_out_pos"] = video_out[1:2].float().cpu().contiguous()

pipe.transformer.transformer_blocks[0].register_forward_hook(block0_hook)

# Hook on cross-attention (attn2) of block 0
attn2_call = [0]

def attn2_hook(module, input, output):
    attn2_call[0] += 1
    if attn2_call[0] > 1:
        return
    # output is the cross-attention result
    b = output.shape[0]
    print(f"\n  attn2 hook: output={output.shape}, batch={b}")
    if b == 2:
        neg = output[0].float()
        pos = output[1].float()
        diff = pos - neg
        print(f"    ca_out neg_std={neg.std():.6f}, pos_std={pos.std():.6f}")
        print(f"    ca_out diff_std={diff.std():.6f}")
        captured["block0_ca_out_neg"] = output[0:1].float().cpu().contiguous()
        captured["block0_ca_out_pos"] = output[1:2].float().cpu().contiguous()

pipe.transformer.transformer_blocks[0].attn2.register_forward_hook(attn2_hook)

# Hook on self-attention (attn1) of block 0
attn1_call = [0]

def attn1_hook(module, input, output):
    attn1_call[0] += 1
    if attn1_call[0] > 1:
        return
    b = output.shape[0]
    if b == 2:
        neg = output[0].float()
        pos = output[1].float()
        diff = pos - neg
        print(f"\n  attn1 hook (self-attn): output={output.shape}")
        print(f"    sa_out neg_std={neg.std():.6f}, pos_std={pos.std():.6f}")
        print(f"    sa_out diff_std={diff.std():.6f} (should be ~0)")
        captured["block0_sa_out_neg"] = output[0:1].float().cpu().contiguous()
        captured["block0_sa_out_pos"] = output[1:2].float().cpu().contiguous()

pipe.transformer.transformer_blocks[0].attn1.register_forward_hook(attn1_hook)

# Hook on FFN of block 0
ff_call = [0]

def ff_hook(module, input, output):
    ff_call[0] += 1
    if ff_call[0] > 1:
        return
    b = output.shape[0]
    if b == 2:
        neg = output[0].float()
        pos = output[1].float()
        diff = pos - neg
        print(f"\n  ff hook: output={output.shape}")
        print(f"    ff_out diff_std={diff.std():.6f}")
        captured["block0_ff_out_neg"] = output[0:1].float().cpu().contiguous()
        captured["block0_ff_out_pos"] = output[1:2].float().cpu().contiguous()

pipe.transformer.transformer_blocks[0].ff.register_forward_hook(ff_hook)

print("Running pipeline...")
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

out_path = "/tmp/ltx2_block0_ca.safetensors"
print(f"\nSaving {len(captured)} tensors to {out_path}")
save_file(captured, out_path)
for k, v in captured.items():
    print(f"  {k}: {v.shape}")
print("\nDone!")
