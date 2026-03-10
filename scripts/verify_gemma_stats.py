#!/usr/bin/env python3
"""Verify Gemma-3 12B hidden state statistics for LTX-2.3 comparison.

Loads the Gemma-3 12B model, runs a forward pass, and prints per-layer
hidden state statistics to compare with the Rust implementation.

Usage:
    HF_TOKEN=... python scripts/verify_gemma_stats.py --prompt "a cat walking"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a cat walking on grass", help="Text prompt")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--model", default="google/gemma-3-12b-pt", help="Model name")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model {args.model} (float32 on CPU)...")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        output_hidden_states=True,
    )
    model.eval()

    # Tokenize with left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = int(attention_mask.sum().item())
    print(f"Prompt: '{args.prompt}' -> {seq_len} tokens (padded to {args.max_length})")
    print(f"Input IDs (last 10): {input_ids[0, -10:].tolist()}")

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    print(f"\nCollected {len(hidden_states)} hidden states (1 embedding + {len(hidden_states)-1} layers)")

    print("\n=== Per-layer hidden state statistics ===")
    for i, hs in enumerate(hidden_states):
        flat = hs.float().flatten()
        std = flat.std().item()
        min_val = flat.min().item()
        max_val = flat.max().item()
        mean = flat.mean().item()
        label = "embed" if i == 0 else f"layer {i-1}"
        print(f"  {label}: std={std:.2f}, mean={mean:.4f}, min={min_val:.2f}, max={max_val:.2f}")

    # Pack text embeds (same as Rust)
    print("\n=== Pack text embeds (Rust-equivalent) ===")
    SCALE_FACTOR = 8.0
    stacked = torch.stack(hidden_states, dim=-1)  # [B, L, D, num_layers]
    print(f"Stacked shape: {stacked.shape}")

    # Compute normalization stats per layer
    mask = attention_mask.float().unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
    masked = stacked * mask
    num_valid = (attention_mask.sum(dim=1).float() * stacked.shape[2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Mean per batch per layer
    sum_x = (masked).sum(dim=(1, 2), keepdim=True)
    mean = sum_x / (num_valid + 1e-6)

    # Min/max per batch per layer
    inv_mask = 1.0 - mask
    x_for_min = stacked + inv_mask * float('inf')
    x_for_max = stacked + inv_mask * float('-inf')
    x_min = x_for_min.flatten(1, 2).min(dim=1, keepdim=True).values.unsqueeze(1)
    x_max = x_for_max.flatten(1, 2).max(dim=1, keepdim=True).values.unsqueeze(1)

    range_val = x_max - x_min + 1e-6
    normalized = (stacked - mean) / range_val * SCALE_FACTOR
    packed = normalized.flatten(2, 3)  # [B, L, D * num_layers]
    packed = packed * attention_mask.float().unsqueeze(-1)

    packed_flat = packed.flatten()
    valid_packed = packed_flat[packed_flat.abs() > 1e-10]
    print(f"Packed shape: {packed.shape}")
    print(f"Packed (all): std={packed_flat.std():.6f}, mean={packed_flat.mean():.6f}")
    print(f"Packed (valid only): std={valid_packed.std():.6f}, mean={valid_packed.mean():.6f}")

    # Check a few layer ranges
    for layer_idx in [0, 24, 48]:
        if layer_idx < len(hidden_states):
            r = range_val[0, 0, 0, layer_idx].item()
            m = mean[0, 0, 0, layer_idx].item()
            print(f"  Layer {layer_idx}: mean={m:.4f}, range={r:.4f}")


if __name__ == "__main__":
    main()
