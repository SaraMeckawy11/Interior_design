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
# Color: pick a designer-matched palette by number. Each has 3 colors that
# work together: dominant (60%), secondary (30%), accent (10%).
COLOR_PALETTES = {
    1: "warm vanilla latte, caramel and olive green",
    2: "sage green, cream and walnut brown",
    3: "cool grey, white and matte black",
    4: "sand beige, terracotta and rust",
    5: "navy blue, cream and brass gold",
    6: "charcoal, walnut and amber",
    7: "dusty rose, cream and muted gold",
    8: "ivory, light oak and forest green",
}
COLOR_CHOICE = 1
CUSTOM_COLORS = None  # e.g. "olive, cream and black" - overrides COLOR_CHOICE
COLOR_TONE = CUSTOM_COLORS or COLOR_PALETTES[COLOR_CHOICE]
# Split the palette into its three named roles so the prompt can assign
# concrete colors (models obey "sofa in warm vanilla latte" far better
# than "sofa in the dominant color").
_cparts = [c.strip() for c in COLOR_TONE.replace(" and ", ", ").split(",") if c.strip()]
DOM = _cparts[0]
SEC = _cparts[1] if len(_cparts) > 1 else DOM
ACC = _cparts[2] if len(_cparts) > 2 else SEC

# ----------------------------- config knobs -----------------------------
OUTPUT_DIR = "output"
# Each seed gives a different design. The printed geometry score says how
# well a seed kept the architecture (higher = more faithful); scores only
# compare within the SAME prompt version, steps and input photo. Try a few
# seeds per room and keep the best score.
SEED = 7
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"  # Apache 2.0, commercial OK
# klein is step-distilled: 4 steps is the intended operating point
# (~10 s/render). 8 steps measured higher geometry scores in one test but
# the user preferred the 4-step look. guidance_scale is ignored (keep 1.0).
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

# ----------------------------- prompt --------------------------------------
# The prompt is composed from the THREE user inputs (room type, design
# style, color tone). Models draw what you NAME, so each style defines its
# own shapes (round vs rectangular table!), materials, floor, curtains,
# art, plants, lamp and textures. Extend either table freely.
STYLE_SPECS = {
    "modern": dict(
        sofa="a sculptural curved sofa with a velvet back and boucle seat",
        table="a round travertine pedestal coffee table",
        floor="wide-plank warm honey oak laid straight",
        rug="a LARGE chunky-woven jute rug",
        curtains="cream double-layer drapery, glowing sheer plus linen panels",
        art="an oversized abstract artwork in the palette",
        plants="tall olive trees in matte travertine planters",
        lamp="a brass floor lamp with tapered fabric shade",
        ceiling="ONE wide disc pendant close to the ceiling",
        textures=("deeper tone-on-tone accents; boucle, velvet, travertine, "
                  "jute and warm oak, subtle brass; warm golden ambience"),
    ),
    "classic": dict(
        sofa="a tailored roll-arm sofa with carved wooden legs",
        table="a rectangular marble-top coffee table with carved legs",
        floor="herringbone oak parquet",
        rug="a LARGE bordered wool rug",
        curtains="heavy pleated drapery with elegant tiebacks",
        art="a large framed classical painting",
        plants="sculpted plants in ceramic urns",
        lamp="a column floor lamp with a pleated shade",
        ceiling="ONE crystal chandelier on a short chain, close to the ceiling",
        textures=("rich deeper accents; silk, velvet, marble and dark "
                  "polished wood, antique gold details; stately warm mood"),
    ),
    "scandinavian": dict(
        sofa="a clean-lined fabric sofa on tapered wooden legs",
        table="a round pale-wood coffee table",
        floor="pale matte oak boards",
        rug="a LARGE soft wool rug",
        curtains="airy white linen curtains",
        art="simple framed line-art prints",
        plants="a leafy plant in a simple white pot",
        lamp="a minimalist tripod floor lamp",
        ceiling="ONE small white dome pendant close to the ceiling",
        textures=("muted tone-on-tone accents; wool, linen, pale birch and "
                  "sheepskin, matte black details; bright airy calm"),
    ),
    "boho": dict(
        sofa="a relaxed low sofa with layered patterned cushions",
        table="a round carved-wood or rattan coffee table",
        floor="warm rustic wood boards",
        rug="LAYERED patterned rugs",
        curtains="light flowing natural-cotton curtains",
        art="an eclectic mix of woven and framed wall pieces",
        plants="abundant potted and trailing plants in terracotta and baskets",
        lamp="a woven rattan floor lamp",
        ceiling="ONE woven rattan pendant close to the ceiling",
        textures=("earthy playful accents; rattan, macrame, layered woven "
                  "textiles, jute and terracotta; relaxed sunlit warmth"),
    ),
    "japandi": dict(
        sofa="a low clean-lined sofa in natural linen",
        table="a low round dark-wood coffee table",
        floor="light matte wood boards",
        rug="a LARGE flat-woven neutral rug",
        curtains="plain linen panels",
        art="one minimal ink-brush artwork",
        plants="a single sculptural branch arrangement in a stone vessel",
        lamp="a paper-lantern floor lamp",
        ceiling="ONE round paper lantern close to the ceiling",
        textures=("quiet deeper accents; linen, pale and dark wood, stone "
                  "and paper, matte black; serene zen calm"),
    ),
    "industrial": dict(
        sofa="a cognac leather sofa",
        table="a rectangular reclaimed-wood and black steel coffee table",
        floor="wide dark wood boards",
        rug="a LARGE worn-look neutral rug",
        curtains="simple dark linen panels",
        art="large monochrome photography prints",
        plants="a tall plant in a black metal planter",
        lamp="a black tripod spotlight floor lamp",
        ceiling="ONE black metal ceiling light close to the ceiling",
        textures=("bold contrast accents; leather, black steel, reclaimed "
                  "wood and aged brass; moody warm light"),
    ),
    "minimalist": dict(
        sofa="a low straight-lined sofa in soft neutral fabric",
        table="a low rectangular seamless coffee table",
        floor="seamless pale oak boards",
        rug="a LARGE plain low-pile rug",
        curtains="plain full-height panels near the wall tone",
        art="one single large calm artwork",
        plants="one sculptural plant in a plain pot",
        lamp="a slim unobtrusive floor lamp",
        ceiling="ONE discreet flush ceiling light",
        textures=("subtle tone-on-tone accents; smooth plaster, pale wood "
                  "and soft matte fabric; serene uncluttered light"),
    ),
}
_st = STYLE_SPECS.get(DESIGN_STYLE.lower().strip()) or dict(
    sofa=f"an authentic {DESIGN_STYLE} statement sofa",
    table=f"a coffee table in authentic {DESIGN_STYLE} design",
    floor=f"premium flooring true to {DESIGN_STYLE} style",
    rug="a LARGE area rug true to the style",
    curtains="full-height drapery true to the style",
    art="one large artwork matching the style",
    plants="plants in style-matching pots",
    lamp="a floor lamp matching the style",
    ceiling="ONE style-matched ceiling light close to the ceiling",
    textures=(f"deeper tone-on-tone accents; materials and textures "
              f"authentic to {DESIGN_STYLE} style"),
)

FURNITURE_BY_ROOM = {
    "living room": (
        f"{_st['sofa']}, a neutral-tone armchair pair - EXACTLY 2 chairs "
        f"total in the room, NO third chair or extra seat - {_st['table']}, "
        "and a media console with a TV above it"
    ),
    "bedroom": (
        "an upholstered bed with layered premium bedding, two nightstands "
        "with warm lamps, and a bench at the foot of the bed"
    ),
    "dining room": (
        "a solid-wood dining table with sculptural chairs and a styled "
        "sideboard"
    ),
    "kitchen": (
        "fitted cabinetry with stone countertops, a breakfast counter with "
        "designer stools, and integrated appliances"
    ),
    "home office": (
        "a wide desk with a refined chair, full bookshelves, and a reading "
        "armchair"
    ),
    "kids room": (
        "a cozy bed with playful bedding, a study desk, a soft rug, and "
        "generous storage"
    ),
    "bathroom": (
        "a floating stone-top vanity with a backlit mirror, a glass shower, "
        "and premium tile"
    ),
}
_furniture = FURNITURE_BY_ROOM.get(
    ROOM_TYPE.lower().strip(),
    f"the essential furniture of a premium {ROOM_TYPE}, beautifully styled",
)

prompt = (
    f"Redesign this room as a {ROOM_TYPE} in {DESIGN_STYLE} style.\n\n"
    "HARD CONSTRAINTS - do not violate:\n"
    "- Keep the architectural GEOMETRY exactly as in the photo: every "
    "wall, window, door and ceiling keeps its exact position, size and "
    "shape; never add, remove, move or resize a window or door. Finishes "
    "MAY change; geometry may not.\n"
    "- Keep every window's exact size and SILL HEIGHT.\n"
    "- Keep the exact same camera position and angle.\n\n"
    "DESIGN BRIEF - senior interior designer:\n"
    "- Furnish if empty; else replace all.\n"
    f"- FULLY FINISH: {_st['floor']} floor, smooth walls, clean ceiling; "
    "no dust or bare concrete.\n"
    f"- Furnish with: {_furniture}; {_st['rug']} under all main "
    f"furniture, {_st['art']} on the main wall, {_st['plants']}, "
    f"{_st['lamp']}, {_st['ceiling']}, and layered cushions.\n"
    f"- Curtains: {_st['curtains']}, from a recessed ceiling slot, NO rod "
    "or gap, spanning the window wall, falling to the floor.\n"
    "- PLACEMENT: a LARGE 16:9 TV, TWICE as wide as tall, centered over "
    "the console, nothing behind it. Put the floor lamp and the plants "
    "on OPPOSITE sides of the room, never touching or adjacent; matching "
    "plants match in size; nothing blocks windows or doors; furniture "
    "square to walls; floor stays CLEAR except furniture, rug and pots.\n"
    f"- COLOR: {DOM} on walls and the sofa, {SEC} on large accents, "
    f"{ACC} ONLY in cushions and small decor - never {ACC} on the sofa; "
    f"{_st['textures']}.\n"
    "- Editorial photo, natural light, contact shadows.\n\n"
    "FINAL CHECK: the result overlays the input photo exactly - same "
    "walls, windows, doors, ceiling; only finishes and furnishings new."
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

# Guard against the encoder's hard 512-token limit: anything beyond it is
# SILENTLY truncated, cutting the FINAL CHECK constraints off the end.
_chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
_ntok = len(tokenizer(_chat)["input_ids"])
if _ntok > 512:
    print(f"*** WARNING: prompt = {_ntok} tokens > 512 — the end of the "
          "prompt WILL BE CUT OFF. Shorten it! ***")
else:
    print(f"Prompt tokens: {_ntok}/512")
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

# Never overwrite previous results; always continue after the highest
# existing number (so deleting old files can't reshuffle the order).
import glob
import re
_nums = [int(m.group(1)) for f in glob.glob(os.path.join(OUTPUT_DIR, "generated_interior_klein_*.png"))
         if (m := re.search(r"_(\d+)\.png$", f))]
out_path = os.path.join(OUTPUT_DIR, f"generated_interior_klein_{max(_nums, default=0) + 1:03d}.png")
result.save(out_path)
room_pil.save(os.path.join(OUTPUT_DIR, "input.png"))
print(f"\nSaved: {out_path}")

try:
    result.show()
except Exception:
    pass
