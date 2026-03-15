#!/usr/bin/env python3
"""Convert LTX-2.3 monolithic safetensors to diffusers directory format.

Usage:
    python scripts/convert_ltx23.py \
        --input /path/to/ltx-2.3-22b-dev.safetensors \
        --output /path/to/LTX-2.3-diffusers/

This creates the directory structure expected by Cake's Rust loader:
    output/
        transformer/
            config.json
            diffusion_pytorch_model.safetensors
        connectors/
            diffusion_pytorch_model.safetensors
        vae/
            diffusion_pytorch_model.safetensors
        vocoder/
            diffusion_pytorch_model.safetensors
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file


# Key rename mappings: monolithic key substring -> diffusers key substring
TRANSFORMER_RENAMES = [
    # Must be ordered: longer/more specific matches first
    ("patchify_proj.", "proj_in."),
    ("adaln_single.", "time_embed."),
    ("q_norm.", "norm_q."),
    ("k_norm.", "norm_k."),
]

CONNECTOR_RENAMES = [
    # In monolithic: model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.N.attn1.q_norm
    # In diffusers:  video_connector.transformer_blocks.N.attn1.norm_q
    ("transformer_1d_blocks.", "transformer_blocks."),
    ("q_norm.", "norm_q."),
    ("k_norm.", "norm_k."),
]

VAE_RENAMES = [
    ("res_blocks.", "resnets."),
    ("per_channel_statistics.mean-of-means", "latents_mean"),
    ("per_channel_statistics.std-of-means", "latents_std"),
]

# VAE block index remapping (monolithic -> diffusers)
# Monolithic: up_blocks.0 = mid_block, up_blocks.1 = up_blocks.0.upsamplers.0, etc.
VAE_DECODER_BLOCK_REMAP = [
    ("up_blocks.0.", "mid_block."),
    ("up_blocks.1.", "up_blocks.0.upsamplers.0."),
    ("up_blocks.2.", "up_blocks.0."),
    ("up_blocks.3.", "up_blocks.1.upsamplers.0."),
    ("up_blocks.4.", "up_blocks.1."),
    ("up_blocks.5.", "up_blocks.2.upsamplers.0."),
    ("up_blocks.6.", "up_blocks.2."),
    # LTX-2.3 has 4 up_blocks (vs 3 for LTX-2):
    ("up_blocks.7.", "up_blocks.3.upsamplers.0."),
    ("up_blocks.8.", "up_blocks.3."),
]

VAE_ENCODER_BLOCK_REMAP = [
    ("down_blocks.0.", "down_blocks.0."),
    ("down_blocks.1.", "down_blocks.0.downsamplers.0."),
    ("down_blocks.2.", "down_blocks.1."),
    ("down_blocks.3.", "down_blocks.1.downsamplers.0."),
    ("down_blocks.4.", "down_blocks.2."),
    ("down_blocks.5.", "down_blocks.2.downsamplers.0."),
    ("down_blocks.6.", "down_blocks.3."),
    ("down_blocks.7.", "down_blocks.3.downsamplers.0."),
    ("down_blocks.8.", "mid_block."),
]


def apply_renames(key: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        key = key.replace(old, new)
    return key


def apply_block_remap(key: str, remaps: list[tuple[str, str]]) -> str:
    """Apply block index remapping (must match longest prefix first)."""
    for old, new in sorted(remaps, key=lambda x: -len(x[0])):
        if old in key:
            return key.replace(old, new, 1)
    return key


def convert(input_path: str, output_dir: str, skip_audio: bool = True):
    print(f"Loading {input_path}...")
    checkpoint = load_file(input_path)
    print(f"Loaded {len(checkpoint)} tensors")

    # Categorize keys by component
    components = defaultdict(dict)
    feature_extractor = {}
    skipped = []

    for key, tensor in checkpoint.items():
        if key.startswith("model.diffusion_model."):
            stripped = key[len("model.diffusion_model."):]

            if stripped.startswith("video_embeddings_connector."):
                # Connector (video)
                conn_key = stripped[len("video_embeddings_connector."):]
                conn_key = apply_renames(conn_key, CONNECTOR_RENAMES)
                components["connectors"]["video_connector." + conn_key] = tensor

            elif stripped.startswith("audio_embeddings_connector."):
                if skip_audio:
                    skipped.append(key)
                    continue
                conn_key = stripped[len("audio_embeddings_connector."):]
                conn_key = apply_renames(conn_key, CONNECTOR_RENAMES)
                components["connectors"]["audio_connector." + conn_key] = tensor

            elif stripped.startswith("audio_"):
                if skip_audio:
                    skipped.append(key)
                    continue
                # Audio transformer components
                trans_key = apply_renames(stripped, TRANSFORMER_RENAMES)
                components["transformer"][trans_key] = tensor
            else:
                # Video transformer
                trans_key = apply_renames(stripped, TRANSFORMER_RENAMES)
                components["transformer"][trans_key] = tensor

        elif key.startswith("text_embedding_projection."):
            # Feature extractor — goes into connectors
            feat_key = key[len("text_embedding_projection."):]
            feature_extractor[feat_key] = tensor

        elif key.startswith("vae."):
            vae_key = key[len("vae."):]

            if vae_key.startswith("decoder."):
                inner = vae_key[len("decoder."):]
                inner = apply_block_remap(inner, VAE_DECODER_BLOCK_REMAP)
                inner = apply_renames(inner, VAE_RENAMES)
                components["vae"]["decoder." + inner] = tensor
            elif vae_key.startswith("encoder."):
                inner = vae_key[len("encoder."):]
                inner = apply_block_remap(inner, VAE_ENCODER_BLOCK_REMAP)
                inner = apply_renames(inner, VAE_RENAMES)
                components["vae"]["encoder." + inner] = tensor
            elif "per_channel_statistics" in vae_key:
                renamed = apply_renames(vae_key, VAE_RENAMES)
                components["vae"][renamed] = tensor
            else:
                components["vae"][vae_key] = tensor

        elif key.startswith("audio_vae."):
            if skip_audio:
                skipped.append(key)
                continue
            components["audio_vae"][key[len("audio_vae."):]] = tensor

        elif key.startswith("vocoder."):
            components["vocoder"][key[len("vocoder."):]] = tensor

        else:
            print(f"  WARNING: Unknown key prefix: {key}")
            skipped.append(key)

    # Add feature extractor to connectors
    if feature_extractor:
        # In LTX-2.3, text_embedding_projection replaces the connector's text_proj_in
        # Store as a separate component within connectors
        for feat_key, tensor in feature_extractor.items():
            components["connectors"]["feature_extractor." + feat_key] = tensor

    print(f"\nComponent summary:")
    for comp, tensors in sorted(components.items()):
        total_params = sum(t.numel() for t in tensors.values())
        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"  {comp}: {len(tensors)} tensors, {total_params:,} params, {total_bytes / 1e9:.2f} GB")
    if skipped:
        print(f"  skipped: {len(skipped)} tensors (audio)")

    # Save each component
    for comp_name, tensors in components.items():
        comp_dir = os.path.join(output_dir, comp_name)
        os.makedirs(comp_dir, exist_ok=True)
        out_path = os.path.join(comp_dir, "diffusion_pytorch_model.safetensors")
        print(f"\nSaving {comp_name} ({len(tensors)} tensors) -> {out_path}")
        save_file(tensors, out_path)

    # Write transformer config
    transformer_config = {
        "_class_name": "LTX2VideoTransformer3DModel",
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "cross_attention_dim": 4096,
        "num_layers": 48,
        "norm_eps": 1e-6,
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "timestep_scale_multiplier": 1000.0,
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "caption_channels": 3840,
        "cross_attention_adaln": True,
        # LTX-2.3 specific
        "gated_attention": True,
        "prompt_modulation": True,
    }
    config_path = os.path.join(output_dir, "transformer", "config.json")
    with open(config_path, "w") as f:
        json.dump(transformer_config, f, indent=2)
    print(f"Saved transformer config -> {config_path}")

    # Write connector config
    connector_config = {
        "caption_channels": 3840,
        "video_connector_num_layers": 8,
        "video_connector_num_attention_heads": 32,
        "video_connector_attention_head_dim": 128,
        "video_connector_num_learnable_registers": 128,
        "audio_connector_num_layers": 8,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 128,
        "audio_connector_num_learnable_registers": 128,
        "text_proj_in_factor": 49,
        "rope_theta": 10000.0,
        "connector_rope_base_seq_len": 4096,
        "has_feature_extractor": True,
        "feature_extractor_out_dim": 4096,
    }
    config_path = os.path.join(output_dir, "connectors", "config.json")
    with open(config_path, "w") as f:
        json.dump(connector_config, f, indent=2)
    print(f"Saved connector config -> {config_path}")

    # Write VAE config
    vae_config = {
        "latent_channels": 128,
        "block_out_channels": [256, 512, 1024, 2048],
        "decoder_block_out_channels": [128, 256, 512, 1024],
        "layers_per_block": [4, 6, 6, 2, 2],
        "decoder_layers_per_block": [4, 6, 4, 2, 2],
        "patch_size": 4,
        "patch_size_t": 1,
        "timestep_conditioning": False,
    }
    config_path = os.path.join(output_dir, "vae", "config.json")
    with open(config_path, "w") as f:
        json.dump(vae_config, f, indent=2)
    print(f"Saved VAE config -> {config_path}")

    print(f"\nDone! Output directory: {output_dir}")
    print(f"Use with: cake master --model {output_dir} --ltx-version 2.3 ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LTX-2.3 monolithic checkpoint to diffusers format")
    parser.add_argument("--input", "-i", required=True, help="Path to ltx-2.3-*.safetensors")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--include-audio", action="store_true", help="Include audio components")
    args = parser.parse_args()

    convert(args.input, args.output, skip_audio=not args.include_audio)
