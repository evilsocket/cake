#!/usr/bin/env python3
"""Convert LuxTTS PyTorch weights to safetensors format for Cake.

Usage:
    python scripts/convert_luxtts.py --model-dir /tmp/luxtts-weights --output-dir /tmp/luxtts-converted

Converts model.pt and vocoder/vocos.bin to safetensors format,
flattening the fm_decoder stack indexing to flat layer indices.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


# Stack sizes from config: fm_decoder_num_layers = [2, 2, 4, 4, 4]
STACK_SIZES = [2, 2, 4, 4, 4]


def flatten_fm_key(key: str) -> str:
    """Remap fm_decoder.encoders.{stack}.layers.{layer}.* to fm_decoder.layers.{flat}.*
    Also handle fm_decoder.encoders.{stack}.encoder.layers.{layer}.* for downsampled stacks.
    """
    import re

    # Pattern 1: fm_decoder.encoders.{stack}.layers.{layer}.{rest}
    m = re.match(r'fm_decoder\.encoders\.(\d+)\.layers\.(\d+)\.(.*)', key)
    if m:
        stack = int(m.group(1))
        layer = int(m.group(2))
        rest = m.group(3)
        flat = sum(STACK_SIZES[:stack]) + layer
        return f'fm_decoder.layers.{flat}.{rest}'

    # Pattern 2: fm_decoder.encoders.{stack}.encoder.layers.{layer}.{rest} (downsampled stacks)
    m = re.match(r'fm_decoder\.encoders\.(\d+)\.encoder\.layers\.(\d+)\.(.*)', key)
    if m:
        stack = int(m.group(1))
        layer = int(m.group(2))
        rest = m.group(3)
        flat = sum(STACK_SIZES[:stack]) + layer
        return f'fm_decoder.layers.{flat}.{rest}'

    # Pattern 3: fm_decoder.encoders.{stack}.encoder.time_emb.{rest} → fm_decoder.stack_time_emb.{stack}.{rest}
    m = re.match(r'fm_decoder\.encoders\.(\d+)\.encoder\.time_emb\.(.*)', key)
    if m:
        stack = int(m.group(1))
        rest = m.group(2)
        return f'fm_decoder.stack_time_emb.{stack}.{rest}'

    # Pattern 4: fm_decoder.encoders.{stack}.time_emb.{rest} → fm_decoder.stack_time_emb.{stack}.{rest}
    m = re.match(r'fm_decoder\.encoders\.(\d+)\.time_emb\.(.*)', key)
    if m:
        stack = int(m.group(1))
        rest = m.group(2)
        return f'fm_decoder.stack_time_emb.{stack}.{rest}'

    # Pattern 5: fm_decoder.encoders.{stack}.downsample.{rest} → fm_decoder.downsample.{stack}.{rest}
    m = re.match(r'fm_decoder\.encoders\.(\d+)\.downsample\.(.*)', key)
    if m:
        stack = int(m.group(1))
        rest = m.group(2)
        return f'fm_decoder.downsample.{stack}.{rest}'

    # Pattern 6: fm_decoder.encoders.{stack}.out_combiner.{rest} → fm_decoder.out_combiner.{stack}.{rest}
    m = re.match(r'fm_decoder\.encoders\.(\d+)\.out_combiner\.(.*)', key)
    if m:
        stack = int(m.group(1))
        rest = m.group(2)
        return f'fm_decoder.out_combiner.{stack}.{rest}'

    # Similarly for text_encoder
    m = re.match(r'text_encoder\.encoders\.0\.layers\.(\d+)\.(.*)', key)
    if m:
        layer = int(m.group(1))
        rest = m.group(2)
        return f'text_encoder.layers.{layer}.{rest}'

    return key


def convert_model(model_path: Path, output_path: Path):
    """Convert main model weights with key remapping."""
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'model' in state_dict:
        state_dict = state_dict['model']

    remapped = {}
    remap_count = 0
    for key, tensor in sorted(state_dict.items()):
        new_key = flatten_fm_key(key)

        # Scalar tensors need at least 1 dim for safetensors
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)

        # Convert to float16
        if tensor.dtype == torch.float32:
            tensor = tensor.half()

        remapped[new_key] = tensor
        if new_key != key:
            remap_count += 1
            print(f"  {key} -> {new_key} {list(tensor.shape)}")

    print(f"\nRemapped {remap_count}/{len(remapped)} keys")
    print(f"Saving {len(remapped)} tensors to {output_path}")
    save_file(remapped, str(output_path))
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")


def convert_vocoder(vocoder_path: Path, output_path: Path):
    """Convert vocoder weights."""
    print(f"\nLoading vocoder from {vocoder_path}")
    state_dict = torch.load(vocoder_path, map_location='cpu', weights_only=False)

    remapped = {}
    for key, tensor in sorted(state_dict.items()):
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype == torch.float32:
            tensor = tensor.half()
        remapped[key] = tensor

    print(f"Saving {len(remapped)} vocoder tensors to {output_path}")
    save_file(remapped, str(output_path))
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert LuxTTS weights to safetensors')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert main model
    model_pt = args.model_dir / 'model.pt'
    if model_pt.exists():
        convert_model(model_pt, args.output_dir / 'model.safetensors')

    # Convert vocoder
    vocos_bin = args.model_dir / 'vocoder' / 'vocos.bin'
    if vocos_bin.exists():
        convert_vocoder(vocos_bin, args.output_dir / 'vocos.safetensors')

    # Copy config, tokens, and vocoder config
    for src, dst_name in [
        (args.model_dir / 'config.json', 'config.json'),
        (args.model_dir / 'tokens.txt', 'tokens.txt'),
        (args.model_dir / 'vocoder' / 'config.yaml', 'vocoder_config.yaml'),
    ]:
        if src.exists():
            shutil.copy2(src, args.output_dir / dst_name)
            print(f"Copied {dst_name}")

    # Add architectures to config.json
    config_path = args.output_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if 'architectures' not in config:
            config['architectures'] = ['LuxTTSForTextToSpeech']
        # Add feature extraction params from vocoder config
        if 'n_fft' not in config.get('feature', {}):
            config['feature']['n_fft'] = 1024
            config['feature']['hop_length'] = 256
            config['feature']['n_mels'] = 100
            config['feature']['sample_rate'] = config['feature'].pop('sampling_rate', 24000)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Updated config.json with architectures and feature params")

    # Generate index file
    model_st = args.output_dir / 'model.safetensors'
    if model_st.exists():
        from safetensors import safe_open
        with safe_open(str(model_st), framework='pt') as f:
            keys = list(f.keys())
        index = {
            "metadata": {"total_size": model_st.stat().st_size},
            "weight_map": {k: "model.safetensors" for k in keys}
        }
        with open(args.output_dir / 'model.safetensors.index.json', 'w') as f:
            json.dump(index, f, indent=2)
        print(f"Generated index with {len(keys)} entries")

    print("\nConversion complete!")


if __name__ == '__main__':
    main()
