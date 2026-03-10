"""
Compare connector outputs for cond vs uncond within the pipeline.
Monkey-patches the connector to capture its inputs/outputs.
"""
import torch
from diffusers import LTX2Pipeline

pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

PROMPT = "A beautiful sunset over the ocean with waves crashing on rocks"

# Monkey-patch the connector forward to capture inputs/outputs
connector_calls = []
orig_connector_forward = pipe.connectors.forward

def patched_connector_forward(*args, **kwargs):
    result = orig_connector_forward(*args, **kwargs)
    # result is (video_embeds, video_attention_mask, audio_embeds, audio_attention_mask)
    video_emb = result[0]
    if video_emb is not None:
        b = video_emb.shape[0]
        if b == 2:
            neg = video_emb[0:1].float()
            pos = video_emb[1:2].float()
            diff = pos - neg
            print(f"\n  Connector output: {video_emb.shape}, dtype={video_emb.dtype}")
            print(f"    neg std={neg.flatten().std():.6f}")
            print(f"    pos std={pos.flatten().std():.6f}")
            print(f"    diff std={diff.flatten().std():.6f}")
            print(f"    diff abs max={diff.flatten().abs().max():.6f}")

            # Per-token analysis
            per_token_norm = diff.squeeze(0).norm(dim=-1)  # [L]
            nonzero = (per_token_norm > 0.01).sum().item()
            print(f"    Tokens with diff > 0.01: {nonzero} / {video_emb.shape[1]}")

            # Check first and last 30 tokens
            first_30_std = diff[0, :30].flatten().std().item()
            last_30_std = diff[0, -30:].flatten().std().item()
            print(f"    First 30 tokens diff std={first_30_std:.6f}")
            print(f"    Last 30 tokens diff std={last_30_std:.6f}")
        elif b == 1:
            print(f"\n  Connector output (single): {video_emb.shape}, std={video_emb.float().flatten().std():.6f}")

    return result

pipe.connectors.forward = patched_connector_forward

# Also patch caption_projection
if hasattr(pipe.transformer, 'caption_projection') and pipe.transformer.caption_projection is not None:
    orig_cap_proj = pipe.transformer.caption_projection.forward

    def patched_cap_proj(x):
        result = orig_cap_proj(x)
        b = result.shape[0]
        if b == 2:
            neg = result[0:1].float()
            pos = result[1:2].float()
            diff = pos - neg
            print(f"\n  Caption projection output: {result.shape}")
            print(f"    neg std={neg.flatten().std():.6f}")
            print(f"    pos std={pos.flatten().std():.6f}")
            print(f"    diff std={diff.flatten().std():.6f}")
            per_token_norm = diff.squeeze(0).norm(dim=-1)
            nonzero = (per_token_norm > 0.01).sum().item()
            print(f"    Tokens with diff > 0.01: {nonzero} / {result.shape[1]}")
        return result

    pipe.transformer.caption_projection.forward = patched_cap_proj

print("Running pipeline with connector diff instrumentation...")
result = pipe(
    prompt=PROMPT,
    negative_prompt="",
    width=512,
    height=384,
    num_frames=9,
    num_inference_steps=2,
    guidance_scale=3.0,
    output_type="pt",
)
print("\nDone!")
