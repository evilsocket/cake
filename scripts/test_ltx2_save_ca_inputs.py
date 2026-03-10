"""
Save block 0 cross-attention exact inputs for Rust comparison.
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

# Hook on attn2 (cross-attention) of block 0 to capture inputs
attn2_call = [0]

def attn2_pre_hook(module, args, kwargs):
    attn2_call[0] += 1
    if attn2_call[0] > 1:
        return

    # LTX2AudioVideoAttnProcessor.__call__ takes:
    # attn, hidden_states, encoder_hidden_states, ...
    # But via register_forward_pre_hook, we get the args to attn2.forward()
    # which calls the processor. Let's capture what we can.
    print(f"\n  attn2 pre-hook: {len(args)} args, {len(kwargs)} kwargs")
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            print(f"    arg[{i}]: {a.shape}")

    # The forward signature is:
    # forward(hidden_states, encoder_hidden_states=None, attention_mask=None, ...)
    if len(args) >= 1:
        hs = args[0]
        print(f"    hidden_states (query): {hs.shape}, dtype={hs.dtype}")
        captured["ca_query"] = hs.float().cpu().contiguous()
    if len(args) >= 2:
        enc = args[1]
        if enc is not None:
            print(f"    encoder_hidden_states: {enc.shape}, dtype={enc.dtype}")
            captured["ca_kv"] = enc.float().cpu().contiguous()
    if 'encoder_hidden_states' in kwargs and kwargs['encoder_hidden_states'] is not None:
        enc = kwargs['encoder_hidden_states']
        print(f"    encoder_hidden_states (kwarg): {enc.shape}")
        captured["ca_kv"] = enc.float().cpu().contiguous()
    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
        mask = kwargs['attention_mask']
        print(f"    attention_mask: {mask.shape}")
        captured["ca_mask"] = mask.float().cpu().contiguous()

pipe.transformer.transformer_blocks[0].attn2.register_forward_pre_hook(attn2_pre_hook, with_kwargs=True)

# Also capture attn2 output
def attn2_hook(module, input, output):
    if attn2_call[0] > 1:
        return
    b = output.shape[0]
    if b == 2:
        captured["ca_out"] = output.float().cpu().contiguous()
        neg = output[0].float()
        pos = output[1].float()
        diff = pos - neg
        print(f"    ca_out: {output.shape}, diff_std={diff.std():.6f}")

pipe.transformer.transformer_blocks[0].attn2.register_forward_hook(attn2_hook)

# Capture FFN output
ff_call = [0]
def ff_hook(module, input, output):
    ff_call[0] += 1
    if ff_call[0] > 1:
        return
    if output.shape[0] == 2:
        captured["ff_out"] = output.float().cpu().contiguous()
        diff = output[1].float() - output[0].float()
        print(f"    ff_out: {output.shape}, diff_std={diff.std():.6f}")

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

out_path = "/tmp/ltx2_block0_ca_inputs.safetensors"
print(f"\nSaving {len(captured)} tensors to {out_path}")
save_file(captured, out_path)
for k, v in captured.items():
    print(f"  {k}: {v.shape}")
print("\nDone!")
