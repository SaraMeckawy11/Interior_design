"""
LOCAL room redesign via FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0 -
free for COMMERCIAL use; the better 9B variant is gated non-commercial,
which is why this app uses the 4B).

Same job as Gen.py (photo + instruction -> redesigned room, geometry kept)
but fully offline on the RTX 4060: no API key, no per-image cost.
(Qwen-Image-Edit 20B cannot run here: ~12 GB even in 4-bit.)

First run downloads ~16 GB into HF_HOME (D:\huggingface). Resumable.

Memory strategy for this machine (16 GB RAM, little free / 8 GB VRAM):
the script runs in two stages, never holding both models on the GPU:
  stage 1: Qwen3 text encoder 4-bit on GPU -> encode the prompt -> free it
  stage 2: transformer 4-bit + VAE on GPU -> denoise with the cached embeds
Weights quantize shard-by-shard straight onto the GPU, so they never sit in
system RAM (the old fp32-in-RAM segfault path). Do NOT switch to
enable_model_cpu_offload(): it parks weights in RAM this machine lacks.
"""

import gc
import os

# Plain-HTTPS downloads (the default Xet backend stalled on this network).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from PIL import Image

from loader import room_image  # NumPy image, RGB, HWC, uint8

# ----------------------------- user preferences -----------------------------
ROOM_TYPE = "living room"        # e.g. "bedroom", "home office", "kitchen"
DESIGN_STYLE = "modern"          # e.g. "japandi", "minimalist", "industrial"
COLOR_TONE = "warm vanilla latte"  # e.g. "sage green", "charcoal and walnut"

# ----------------------------- config knobs -----------------------------
OUTPUT_DIR = "output"
# Each seed gives a different design (~10 s per render). Tried so far:
# seed 7 -> geometry 0.717 (best), 576906284 -> 0.620, 42 -> 0.616,
# 20260708 -> 0.602. The printed geometry score tells you how well a seed
# kept the room's architecture (higher = more faithful).
SEED = 7
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"  # Apache 2.0, commercial OK
# klein is step-distilled: 4 steps is the intended operating point and
# guidance_scale is ignored by distilled checkpoints (keep 1.0).
STEPS = 4
GUIDANCE = 1.0

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required (klein 4-bit needs the RTX 4060).")

# --- output size: keep the photo's orientation; dims must divide by 32 ---
orig_h, orig_w = room_image.shape[:2]
if orig_w > orig_h:
    WIDTH, HEIGHT = 1024, 768
else:
    WIDTH, HEIGHT = 768, 1024
print(f"Input {orig_w}x{orig_h} -> generating {WIDTH}x{HEIGHT}")

# ----------------------------- prompt (keep in sync with Gen.py) -----------
prompt = (
    f"Redesign this room as a {ROOM_TYPE} in {DESIGN_STYLE} style with a "
    f"{COLOR_TONE} color palette.\n\n"
    "HARD CONSTRAINTS - do not violate:\n"
    "- Keep the room's architectural GEOMETRY exactly as in the photo: every "
    "wall, window, door, ceiling, beam and column stays in its exact "
    "position, size and shape. Do not add, remove, move or resize any window "
    "or door. Surface finishes and materials MAY change; geometry may not.\n"
    "- Keep each window's exact size, proportions and SILL HEIGHT. Never "
    "enlarge a window, never convert a window into a door or floor-to-ceiling "
    "glazing, and keep every balcony door exactly where and as it is.\n"
    "- Keep the exact same camera position, angle and lens/perspective.\n"
    "- Keep daylight coming from the real windows; lighting must be "
    "physically plausible for this room.\n\n"
    "DESIGN BRIEF - work like a senior professional interior designer:\n"
    "- If the room is empty, furnish it completely. If it already has "
    "furniture or decor, replace all of it with the new design.\n"
    "- FULLY FINISH every surface (the photo may show an unfinished or "
    "under-construction room): lay a brand-new premium finished floor in a "
    "material that suits the style covering the ENTIRE floor, smooth painted "
    "walls, and a clean finished ceiling. No construction dust, debris, "
    "stains, bare concrete, exposed wires or unfinished surfaces anywhere.\n"
    f"- Full, cohesive furniture arrangement appropriate for a {ROOM_TYPE}: "
    "primary furniture pieces, correctly sized rug, curtains on the real "
    "windows, wall art, plants, and layered lighting.\n"
    "- Realistic scale and proportions; furniture sits properly on the floor "
    "with correct contact shadows; nothing floats or clips into walls.\n"
    "- Photorealistic output, high detail, natural soft shadows, styled like "
    "an Architectural Digest photo shoot."
)
print("PROMPT:\n", prompt, "\n" + "-" * 60)

# ============================================================================
# STAGE 0 - download everything to D: first (resumable). Loading from a LOCAL
# folder also sidesteps a transformers-on-Windows bug where subfolder shard
# paths are built with backslashes and fail against the Hub API.
# ============================================================================
from huggingface_hub import snapshot_download

print("Downloading model files (resumable, ~16 GB on first run)...")
LOCAL_DIR = snapshot_download(
    MODEL_ID,
    # everything except the duplicate single-file checkpoint (saves ~18 GB)
    allow_patterns=[
        "model_index.json", "scheduler/*", "tokenizer/*",
        "text_encoder/*", "transformer/*", "vae/*",
    ],
)
print("Download complete:", LOCAL_DIR)

# ============================================================================
# STAGE 1 - text encoder alone on the GPU: prompt -> embeddings, then free.
# ============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig as TransformersBnb4bit

print("Stage 1/2: loading text encoder (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(LOCAL_DIR, "tokenizer"))
text_encoder = AutoModelForCausalLM.from_pretrained(
    os.path.join(LOCAL_DIR, "text_encoder"),
    torch_dtype=torch.bfloat16,
    quantization_config=TransformersBnb4bit(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map={"": 0},
)

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnb4bit

with torch.no_grad():
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
        dtype=torch.bfloat16,
        device="cuda",
    ).cpu()  # tiny tensor; park it off-GPU while models swap

del text_encoder, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("Prompt encoded, text encoder freed.")

# ============================================================================
# STAGE 2 - transformer + VAE on the GPU: denoise with the cached embeddings.
# ============================================================================
print("Stage 2/2: loading transformer (4-bit) + VAE...")
transformer = Flux2Transformer2DModel.from_pretrained(
    LOCAL_DIR,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    quantization_config=DiffusersBnb4bit(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)

pipe = Flux2KleinPipeline.from_pretrained(
    LOCAL_DIR,
    transformer=transformer,
    text_encoder=None,   # already used and freed in stage 1
    tokenizer=None,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Activation-memory savers (weights are already handled by the 4-bit load).
try:
    pipe.enable_attention_slicing()
except Exception:
    pass
try:
    pipe.vae.enable_tiling()
except Exception:
    pass

gc.collect()
torch.cuda.empty_cache()

# ----------------------------- geometry scoring ----------------------------
# How much of the photo's structural edges (walls, window frames, doors,
# cornices) survive in a candidate. An empty room's edges ARE its
# architecture, so a candidate that moved/resized an opening scores lower.
# Furniture only ADDS edges, which this recall-style score doesn't punish.
import cv2
import numpy as np


def geometry_score(input_rgb, candidate_pil):
    w, h = candidate_pil.size
    inp = cv2.resize(input_rgb, (w, h))
    edges_in = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY), (5, 5), 0), 40, 120)
    edges_cand = cv2.Canny(cv2.cvtColor(np.array(candidate_pil), cv2.COLOR_RGB2GRAY), 40, 120)
    edges_cand = cv2.dilate(edges_cand, np.ones((7, 7), np.uint8))
    kept = ((edges_in > 0) & (edges_cand > 0)).sum()
    return kept / max((edges_in > 0).sum(), 1)


# ----------------------------- run -----------------------------------------
room_pil = Image.fromarray(room_image).convert("RGB")
os.makedirs(OUTPUT_DIR, exist_ok=True)
embeds_gpu = prompt_embeds.to("cuda")

result = pipe(
    prompt=None,
    prompt_embeds=embeds_gpu,
    image=[room_pil],
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    generator=torch.Generator(device="cuda").manual_seed(SEED),
).images[0]

print(f"seed {SEED}: geometry score {geometry_score(room_image, result):.3f}")

out_path = os.path.join(OUTPUT_DIR, "generated_interior_klein.png")
result.save(out_path)
room_pil.save(os.path.join(OUTPUT_DIR, "input.png"))
print(f"\nSaved: {out_path}")

try:
    result.show()
except Exception:
    pass
