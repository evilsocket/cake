#!/usr/bin/env python3
"""Convert Wan2.1 VAE from .pth to safetensors with cake-expected key naming.

Original .pth format uses flat Sequential indexing:
  decoder.upsamples.{N}.residual.{0,2,3,6}.*  (ResBlock via Sequential)
  decoder.upsamples.{N}.resample.1.*           (spatial upsample Conv2d)
  decoder.upsamples.{N}.time_conv.*            (temporal upsample CausalConv3d)

Cake's vendored code expects stage-based naming:
  decoder.upsamples.{stage}.block.{block_idx}.norm1/conv1/norm2/conv2.*
  decoder.upsamples.{stage}.upsample.spatial_conv/time_conv.*

Also: residual.0 -> norm1, residual.2 -> conv1, residual.3 -> norm2, residual.6 -> conv2
And: gamma with shape [C,1,1,1] -> flattened to [C]
"""

import sys
import torch
from safetensors.torch import save_file

def remap_resblock_key(key):
    """Remap Sequential-indexed ResBlock key to named submodule key."""
    # residual.0 -> norm1 (RmsNorm gamma)
    # residual.2 -> conv1 (CausalConv3d weight/bias)
    # residual.3 -> norm2 (RmsNorm gamma)
    # residual.6 -> conv2 (CausalConv3d weight/bias)
    mapping = {
        "residual.0.": "norm1.",
        "residual.2.": "conv1.",
        "residual.3.": "norm2.",
        "residual.6.": "conv2.",
    }
    for old, new in mapping.items():
        if old in key:
            return key.replace(old, new)
    return key

def convert_wan_vae(input_path, output_path):
    print(f"Loading {input_path}...")
    state = torch.load(input_path, map_location="cpu", weights_only=True)

    # Wan VAE decoder has 4 stages: dim_mult=[1,2,4,4], temporal_downsample=[F,T,T]
    # The original .pth uses flat numbering for ALL blocks across all stages.
    # We need to figure out which flat indices correspond to which stages.

    # Analyze the structure by looking at the keys
    upsample_indices = set()
    for k in state.keys():
        if k.startswith("decoder.upsamples."):
            parts = k.split(".")
            idx = int(parts[2])
            upsample_indices.add(idx)

    print(f"Found upsample indices: {sorted(upsample_indices)}")
    # Expected: 0-14 (15 entries) for 4 stages
    # Stage 0 (ch=384): 3 resblocks [0,1,2] + upsample [3] (temporal+spatial)
    # Stage 1 (ch=384->192): 3 resblocks [4,5,6] + upsample [7] (temporal+spatial)
    # Stage 2 (ch=192): 3 resblocks [8,9,10] + upsample [11] (spatial only)
    # Stage 3 (ch=96): 3 resblocks [12,13,14] (no upsample - last stage)

    # Determine stage boundaries by channel size changes
    # We know the pattern: num_res_blocks=2 per stage, plus upsample
    # But looking at actual indices, it's more like:
    # Actually it's 3 resblocks each (based on index 0-2, 4-6, 8-10, 12-14)
    # With upsamples at 3, 7, 11

    new_state = {}

    for k, v in state.items():
        new_key = k

        if k.startswith("decoder.upsamples."):
            parts = k.split(".", 3)  # ['decoder', 'upsamples', 'idx', 'rest']
            flat_idx = int(parts[2])
            rest = parts[3]

            # Map flat index to (stage, type)
            # Stage 0: indices 0,1,2 = resblocks; 3 = upsample
            # Stage 1: indices 4,5,6 = resblocks; 7 = upsample
            # Stage 2: indices 8,9,10 = resblocks; 11 = upsample
            # Stage 3: indices 12,13,14 = resblocks (no upsample)
            if flat_idx <= 2:
                stage, block = 0, flat_idx
                rest = remap_resblock_key(rest)
                new_key = f"decoder.upsamples.{stage}.block.{block}.{rest}"
            elif flat_idx == 3:
                # Upsample for stage 0 (spatio-temporal)
                rest = rest.replace("resample.1.", "spatial_conv.")
                new_key = f"decoder.upsamples.0.upsample.{rest}"
            elif flat_idx <= 6:
                stage, block = 1, flat_idx - 4
                rest = remap_resblock_key(rest)
                new_key = f"decoder.upsamples.{stage}.block.{block}.{rest}"
            elif flat_idx == 7:
                rest = rest.replace("resample.1.", "spatial_conv.")
                new_key = f"decoder.upsamples.1.upsample.{rest}"
            elif flat_idx <= 10:
                stage, block = 2, flat_idx - 8
                rest = remap_resblock_key(rest)
                new_key = f"decoder.upsamples.{stage}.block.{block}.{rest}"
            elif flat_idx == 11:
                # Stage 2 is spatial-only upsample (no time_conv)
                # resample.1 -> conv (our Upsample2d uses "conv" prefix)
                rest = rest.replace("resample.1.", "conv.")
                new_key = f"decoder.upsamples.2.upsample.{rest}"
            elif flat_idx <= 14:
                stage, block = 3, flat_idx - 12
                rest = remap_resblock_key(rest)
                new_key = f"decoder.upsamples.{stage}.block.{block}.{rest}"

        elif k.startswith("decoder.middle."):
            # middle.0/2 are resblocks, middle.1 is attention
            parts = k.split(".", 3)
            idx = int(parts[2])
            rest = parts[3]
            if idx in (0, 2):
                rest = remap_resblock_key(rest)
                new_key = f"decoder.middle.{idx}.{rest}"
            # middle.1 (attention) keeps its naming

        # Flatten [C, 1, 1, 1] gamma to [C]
        if new_key.endswith(".gamma"):
            if v.dim() > 1:
                v = v.flatten()

        # Also rename gamma -> gamma (no change needed, our code expects gamma)
        # And handle norm naming

        new_state[new_key] = v.contiguous()
        if new_key != k:
            print(f"  {k} -> {new_key}  {list(v.shape)}")

    print(f"\nSaving {len(new_state)} tensors to {output_path}...")
    save_file(new_state, output_path)
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: convert_wan_vae.py <input.pth> <output.safetensors>")
        sys.exit(1)
    convert_wan_vae(sys.argv[1], sys.argv[2])
