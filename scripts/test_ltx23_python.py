#!/usr/bin/env python3
"""Quick test: run LTX-2.3 transformer on a single step and check output.

Uses the converted diffusers-format weights to verify they produce
meaningful velocity predictions.
"""

import torch
from safetensors.torch import load_file
import json
import math
import sys

MODEL_DIR = "/home/a/cake-data/LTX-2.3"

def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    """Standard sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def main():
    # Load config
    with open(f"{MODEL_DIR}/transformer/config.json") as f:
        config = json.load(f)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Load a subset of weights
    print(f"\nLoading transformer weights...")
    weights = load_file(f"{MODEL_DIR}/transformer/diffusion_pytorch_model.safetensors")

    # Check proj_in
    proj_in_w = weights["proj_in.weight"]
    proj_in_b = weights["proj_in.bias"]
    print(f"proj_in: weight={proj_in_w.shape}, bias={proj_in_b.shape}")

    # Check scale_shift_table (final modulation)
    sst = weights["scale_shift_table"]
    print(f"Final scale_shift_table: {sst.shape}, values: {sst.float().mean():.4f} ± {sst.float().std():.4f}")

    # Check block 0 scale_shift_table
    block_sst = weights["transformer_blocks.0.scale_shift_table"]
    print(f"Block 0 scale_shift_table: {block_sst.shape}")
    for i in range(block_sst.shape[0]):
        row = block_sst[i].float()
        print(f"  row {i}: mean={row.mean():.4f}, std={row.std():.4f}")

    # Check time_embed
    te_l1_w = weights["time_embed.emb.timestep_embedder.linear_1.weight"]
    te_l2_w = weights["time_embed.emb.timestep_embedder.linear_2.weight"]
    te_lin_w = weights["time_embed.linear.weight"]
    print(f"\ntime_embed: l1={te_l1_w.shape}, l2={te_l2_w.shape}, linear={te_lin_w.shape}")

    # Test: run time_embed on sigma=1.0 (timestep=1000)
    ts = torch.tensor([1000.0])
    t_emb = sinusoidal_timestep_embedding(ts, 256)  # [1, 256]
    print(f"Sinusoidal embedding: {t_emb.shape}, range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")

    # Through timestep MLP
    t_emb_bf16 = t_emb.to(torch.bfloat16)
    te_l1_w_bf16 = te_l1_w
    te_l1_b_bf16 = weights["time_embed.emb.timestep_embedder.linear_1.bias"]
    h = torch.nn.functional.linear(t_emb_bf16, te_l1_w_bf16, te_l1_b_bf16)
    h = torch.nn.functional.silu(h)
    te_l2_b_bf16 = weights["time_embed.emb.timestep_embedder.linear_2.bias"]
    h = torch.nn.functional.linear(h, te_l2_w.to(torch.bfloat16), te_l2_b_bf16)
    print(f"After timestep MLP: {h.shape}, range=[{h.float().min():.4f}, {h.float().max():.4f}], std={h.float().std():.4f}")

    # Through SiLU + final linear
    h_silu = torch.nn.functional.silu(h)
    te_lin_b = weights["time_embed.linear.bias"]
    temb = torch.nn.functional.linear(h_silu, te_lin_w.to(torch.bfloat16), te_lin_b)
    print(f"Full time_embed output: {temb.shape}, range=[{temb.float().min():.4f}, {temb.float().max():.4f}], std={temb.float().std():.4f}")
    # Reshape: [1, 36864] -> [1, 1, 9, 4096]
    temb_r = temb.reshape(1, 1, 9, 4096)
    for i in range(9):
        row = temb_r[0, 0, i].float()
        print(f"  temb row {i}: mean={row.mean():.4f}, std={row.std():.4f}")

    # Quick test: proj_in on random noise
    noise = torch.randn(1, 16, 128, dtype=torch.bfloat16)  # small test [B, S, C]
    h = torch.nn.functional.linear(noise, proj_in_w.to(torch.bfloat16), proj_in_b.to(torch.bfloat16))
    print(f"\nproj_in(noise): {h.shape}, range=[{h.float().min():.4f}, {h.float().max():.4f}], std={h.float().std():.4f}")

    # Check if proj_out reverses proj_in
    proj_out_w = weights["proj_out.weight"]
    proj_out_b = weights["proj_out.bias"]
    h_out = torch.nn.functional.linear(h, proj_out_w.to(torch.bfloat16), proj_out_b.to(torch.bfloat16))
    print(f"proj_out(proj_in(noise)): {h_out.shape}, range=[{h_out.float().min():.4f}, {h_out.float().max():.4f}], std={h_out.float().std():.4f}")


if __name__ == "__main__":
    main()
