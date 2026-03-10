"""
Save exact block 0 full inputs/outputs for Rust comparison.
Captures: hidden_states (in/out), temb, context, mask — everything the block needs.
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

# Hook on block 0 to capture ALL inputs and output
block0_call = [0]

def block0_pre_hook(module, args, kwargs):
    block0_call[0] += 1
    if block0_call[0] > 1:
        return

    print(f"\n  Block 0 pre-hook: {len(args)} args, {list(kwargs.keys())} kwargs")

    # The block forward signature:
    # forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, ...)
    # Let's capture from args
    if len(args) >= 1:
        hs = args[0]
        print(f"    hidden_states: {hs.shape}, dtype={hs.dtype}")
        captured["block0_hidden_in"] = hs.float().cpu().contiguous()
    if len(args) >= 2:
        enc = args[1]
        if enc is not None:
            print(f"    encoder_hidden_states: {enc.shape}")
            captured["block0_context"] = enc.float().cpu().contiguous()
    if len(args) >= 3:
        temb = args[2]
        if temb is not None:
            print(f"    temb: {temb.shape}")
            captured["block0_temb"] = temb.float().cpu().contiguous()
    if len(args) >= 4:
        rope = args[3]
        if rope is not None:
            if isinstance(rope, tuple):
                print(f"    image_rotary_emb: tuple of {len(rope)}")
                for i, r in enumerate(rope):
                    if isinstance(r, torch.Tensor):
                        print(f"      [{i}]: {r.shape}")
                        captured[f"block0_rope_{i}"] = r.float().cpu().contiguous()
            else:
                print(f"    image_rotary_emb: {rope.shape}")

    # Check kwargs
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"    kwarg {k}: {v.shape}")
            captured[f"block0_kwarg_{k}"] = v.float().cpu().contiguous()
        elif v is not None:
            print(f"    kwarg {k}: {type(v).__name__} = {v}")

pipe.transformer.transformer_blocks[0].register_forward_pre_hook(block0_pre_hook, with_kwargs=True)

def block0_hook(module, input, output):
    if block0_call[0] > 1:
        return
    video_out = output[0] if isinstance(output, tuple) else output
    print(f"\n  Block 0 output: {video_out.shape}")
    captured["block0_hidden_out"] = video_out.float().cpu().contiguous()

    if video_out.shape[0] == 2:
        neg = video_out[0].float()
        pos = video_out[1].float()
        diff = pos - neg
        print(f"    diff_std={diff.flatten().std():.6f}")

pipe.transformer.transformer_blocks[0].register_forward_hook(block0_hook)

# Also capture attention_mask from the transformer's forward
orig_forward = pipe.transformer.forward.__wrapped__ if hasattr(pipe.transformer.forward, '__wrapped__') else None

# Hook on the full transformer to see attention_mask
xformer_call = [0]
def xformer_pre_hook(module, args, kwargs):
    xformer_call[0] += 1
    if xformer_call[0] > 1:
        return
    print(f"\n  Transformer pre-hook: {len(args)} args, {list(kwargs.keys())} kwargs")
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"    kwarg {k}: {v.shape}, dtype={v.dtype}")
            if 'mask' in k.lower():
                print(f"      unique: {v.unique().tolist()[:5]}, sum={v.sum():.1f}")
                captured[f"xformer_{k}"] = v.float().cpu().contiguous()

pipe.transformer.register_forward_pre_hook(xformer_pre_hook, with_kwargs=True)

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

out_path = "/tmp/ltx2_block0_full.safetensors"
print(f"\nSaving {len(captured)} tensors to {out_path}")
save_file(captured, out_path)
for k, v in captured.items():
    print(f"  {k}: {v.shape}")
print("\nDone!")
