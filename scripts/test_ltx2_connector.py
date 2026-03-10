"""
Test LTX-2 connector pipeline: verify that the Python connector produces
meaningful differentiation between different prompts.

This tests the hypothesis that the muddy output is due to the connector
not differentiating between prompts.
"""

import torch
import numpy as np
from safetensors import safe_open
from pathlib import Path

# Load LTX-2 connector weights
CONNECTOR_PATH = Path.home() / ".cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a/connectors/diffusion_pytorch_model.safetensors"

if not CONNECTOR_PATH.exists():
    # Try alternate path
    import glob
    candidates = glob.glob(str(Path.home() / ".cache/huggingface/**/Lightricks--LTX-2/**/connectors/diffusion_pytorch_model.safetensors"), recursive=True)
    if candidates:
        CONNECTOR_PATH = Path(candidates[0])
    else:
        raise FileNotFoundError("Cannot find LTX-2 connector weights")

print(f"Loading connector from: {CONNECTOR_PATH}")

# List all keys and shapes
st = safe_open(str(CONNECTOR_PATH), framework="pt", device="cuda")
keys = sorted(st.keys())
print(f"\nTotal keys: {len(keys)}")
for k in keys:
    t = st.get_tensor(k)
    print(f"  {k}: {t.shape} {t.dtype} min={t.float().min():.4f} max={t.float().max():.4f} std={t.float().std():.4f}")

# Load key weights
text_proj_in_w = st.get_tensor("text_proj_in.weight")  # [3840, 188160]
registers = st.get_tensor("video_connector.learnable_registers")  # [128, 3840]

print(f"\ntext_proj_in weight: {text_proj_in_w.shape}")
print(f"  mean={text_proj_in_w.float().mean():.6f}")
print(f"  std={text_proj_in_w.float().std():.6f}")
print(f"  min={text_proj_in_w.float().min():.6f}")
print(f"  max={text_proj_in_w.float().max():.6f}")

print(f"\nregisters: {registers.shape}")
print(f"  mean={registers.float().mean():.6f}")
print(f"  std={registers.float().std():.6f}")

# Test: what happens when we project random input vs zeros
# Simulating V1 normalization output: values in [-8, 8] range with some structure
torch.manual_seed(42)

# Simulate a "real" packed embedding (like from Gemma)
seq_len = 256
packed_dim = 188160  # 3840 * 49
batch = 1

# Create a "real" input (normalized Gemma output)
real_input = torch.randn(batch, seq_len, packed_dim, device="cuda", dtype=torch.bfloat16) * 0.5

# Create "empty" input (what empty string encoding might look like)
empty_input = torch.randn(batch, seq_len, packed_dim, device="cuda", dtype=torch.bfloat16) * 0.5

# Project both through text_proj_in
real_proj = real_input.float() @ text_proj_in_w.float().t()  # [1, 256, 3840]
empty_proj = empty_input.float() @ text_proj_in_w.float().t()

diff = (real_proj - empty_proj)

print(f"\nProjected real: shape={real_proj.shape}")
print(f"  mean={real_proj.mean():.6f}, std={real_proj.std():.6f}")
print(f"Projected empty: shape={empty_proj.shape}")
print(f"  mean={empty_proj.mean():.6f}, std={empty_proj.std():.6f}")
print(f"Diff: mean={diff.mean():.6f}, std={diff.std():.6f}")

# Now test with the ACTUAL scale of V1 normalized embeddings
# V1: (x - mean) / (max - min) * 8.0
# With Gemma hidden state explosion at later layers (std~1700),
# the normalized values should be around [-4, 4] for typical values
# But with 256 positions, ~80% might be padding (zeros)

# Let's see what scale the projection expects
# For a well-behaved linear layer, the output std should be roughly
# input_std * weight_std * sqrt(input_dim)
w_std = text_proj_in_w.float().std().item()
input_std = 0.01  # The logged value from Rust was std=0.0105
expected_output_std = input_std * w_std * np.sqrt(packed_dim)
print(f"\nExpected output behavior:")
print(f"  weight std={w_std:.6f}")
print(f"  input std (from Rust log)={input_std}")
print(f"  expected output std = {input_std} * {w_std:.6f} * sqrt({packed_dim}) = {expected_output_std:.6f}")

# Test with actual-scale inputs
small_input = torch.randn(batch, seq_len, packed_dim, device="cuda", dtype=torch.float32) * input_std
small_proj = small_input @ text_proj_in_w.float().t()
print(f"\nWith actual-scale input (std={input_std}):")
print(f"  proj mean={small_proj.mean():.6f}, std={small_proj.std():.6f}")

# What about mask behavior?
# If all tokens are valid (mask=1), registers should NOT be used
# If most tokens are padding (mask=0), registers replace them
# With short prompts (~20 tokens out of 256), ~92% are registers

num_valid = 20
mask = torch.zeros(batch, seq_len, device="cuda")
mask[:, -num_valid:] = 1.0  # Left padding: valid tokens at the end

print(f"\nMask: {num_valid}/{seq_len} valid tokens ({100*num_valid/seq_len:.1f}%)")
print(f"  With {seq_len-num_valid} register tokens, connector output is dominated by registers")
print(f"  Register std={registers.float().std():.4f}")
print(f"  Register/project_std ratio = {registers.float().std().item() / max(small_proj.std().item(), 1e-8):.1f}x")

# Try the diffusers implementation directly
try:
    from diffusers.models.transformers.ltx2_transformer_3d import LTX2TextConnectors
    print("\n\nDiffusers LTX2TextConnectors available - running reference comparison")

    # Load config
    import json
    config_path = CONNECTOR_PATH.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Config: {json.dumps(config, indent=2)[:500]}")
except ImportError:
    print("\nDiffusers LTX2TextConnectors not available in this version")
    print("Try: pip install diffusers>=0.37.0")
