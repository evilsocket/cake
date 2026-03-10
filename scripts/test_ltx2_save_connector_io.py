"""
Save connector inputs/outputs for both cond and uncond from Python LTX-2 pipeline.
Uses monkey-patching within the pipeline call to avoid OOM.
"""
import torch
from safetensors.torch import save_file
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"
WIDTH = 512
HEIGHT = 384
NUM_FRAMES = 9

# Capture data via monkey-patches
captured = {}

# Patch encode_prompt to capture Gemma outputs
orig_get_gemma = pipe._get_gemma_prompt_embeds

def patched_get_gemma(prompt, **kwargs):
    result = orig_get_gemma(prompt=prompt, **kwargs)
    embeds, mask = result
    prompt_text = prompt[0] if isinstance(prompt, list) else prompt
    key = "prompt" if prompt_text.strip() else "neg"
    captured[f"{key}_packed_embeds"] = embeds.float().cpu().contiguous()
    captured[f"{key}_mask"] = mask.float().cpu().contiguous()
    print(f"  Gemma {key}: embeds={embeds.shape}, valid_tokens={mask.sum().item()}, "
          f"std={embeds.float().flatten().std():.6f}")
    return result

pipe._get_gemma_prompt_embeds = patched_get_gemma

# Patch connector to capture its I/O
orig_connector = pipe.connectors.forward

def patched_connector(text_hidden_states, attention_mask, additive_mask=False):
    result = orig_connector(text_hidden_states, attention_mask, additive_mask=additive_mask)
    video_emb = result[0]
    b = video_emb.shape[0]

    if b == 2:
        neg = video_emb[0:1]
        pos = video_emb[1:2]
        captured["neg_connector_out"] = neg.float().cpu().contiguous()
        captured["prompt_connector_out"] = pos.float().cpu().contiguous()

        diff = (pos - neg).float()
        print(f"\n  Connector output [batch=2]: {video_emb.shape}")
        print(f"    neg std={neg.float().flatten().std():.6f}")
        print(f"    pos std={pos.float().flatten().std():.6f}")
        print(f"    diff std={diff.flatten().std():.6f}")
        print(f"    first 30 diff std={diff[0, :30].flatten().std():.6f}")
        print(f"    last 30 diff std={diff[0, -30:].flatten().std():.6f}")
        per_tok = diff.squeeze(0).norm(dim=-1)
        nonzero = (per_tok > 0.01).sum().item()
        print(f"    tokens with diff > 0.01: {nonzero}/{video_emb.shape[1]}")
    elif b == 1:
        print(f"\n  Connector output [batch=1]: {video_emb.shape}, std={video_emb.float().flatten().std():.6f}")

    return result

pipe.connectors.forward = patched_connector

# Patch caption_projection to capture its output
if hasattr(pipe.transformer, 'caption_projection') and pipe.transformer.caption_projection is not None:
    orig_cap_proj = pipe.transformer.caption_projection.forward

    def patched_cap_proj(x):
        result = orig_cap_proj(x)
        b = result.shape[0]
        if b == 2:
            neg = result[0:1]
            pos = result[1:2]
            captured["neg_projected"] = neg.float().cpu().contiguous()
            captured["prompt_projected"] = pos.float().cpu().contiguous()
            diff = (pos - neg).float()
            print(f"\n  Caption projection [batch=2]: {result.shape}")
            print(f"    diff std={diff.flatten().std():.6f}")
            print(f"    first 30 diff std={diff[0, :30].flatten().std():.6f}")
            print(f"    last 30 diff std={diff[0, -30:].flatten().std():.6f}")
            per_tok = diff.squeeze(0).norm(dim=-1)
            nonzero = (per_tok > 0.01).sum().item()
            print(f"    tokens with diff > 0.01: {nonzero}/{result.shape[1]}")
        return result

    pipe.transformer.caption_projection.forward = patched_cap_proj

# Patch transformer to capture per-block stats (first call only)
block_call_count = [0]
orig_block_forward = pipe.transformer.transformer_blocks[0].__class__.forward

def patched_block_forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                           audio_encoder_hidden_states, temb, temb_audio,
                           temb_ca_scale_shift, temb_ca_audio_scale_shift,
                           temb_ca_gate, temb_ca_audio_gate,
                           video_rotary_emb=None, audio_rotary_emb=None,
                           ca_video_rotary_emb=None, ca_audio_rotary_emb=None,
                           encoder_attention_mask=None, audio_encoder_attention_mask=None,
                           a2v_cross_attention_mask=None, v2a_cross_attention_mask=None):
    result = orig_block_forward(self, hidden_states, audio_hidden_states, encoder_hidden_states,
                                 audio_encoder_hidden_states, temb, temb_audio,
                                 temb_ca_scale_shift, temb_ca_audio_scale_shift,
                                 temb_ca_gate, temb_ca_audio_gate,
                                 video_rotary_emb, audio_rotary_emb,
                                 ca_video_rotary_emb, ca_audio_rotary_emb,
                                 encoder_attention_mask, audio_encoder_attention_mask,
                                 a2v_cross_attention_mask, v2a_cross_attention_mask)

    block_call_count[0] += 1
    video_out = result[0] if isinstance(result, tuple) else result

    # Only log for first denoising step (step 0 has 2 transformer calls for CFG batch=2)
    if block_call_count[0] <= 48 and video_out.shape[0] == 2:
        neg = video_out[0:1].float()
        pos = video_out[1:2].float()
        diff = pos - neg
        block_idx = (block_call_count[0] - 1) % 48
        if block_idx < 5 or block_idx >= 45:
            print(f"    block {block_idx}: diff_std={diff.flatten().std():.6f}, "
                  f"pos_std={pos.flatten().std():.6f}")

    return result

for block in pipe.transformer.transformer_blocks:
    block.__class__.forward = patched_block_forward

print("Running pipeline...")
result = pipe(
    prompt=PROMPT,
    negative_prompt="",
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    num_inference_steps=2,
    guidance_scale=3.0,
    output_type="pt",
)

# Save captured tensors
print(f"\nSaving {len(captured)} captured tensors to /tmp/ltx2_connector_io.safetensors")
save_file(captured, "/tmp/ltx2_connector_io.safetensors")
for k, v in captured.items():
    print(f"  {k}: {v.shape}")

print("\nDone!")
