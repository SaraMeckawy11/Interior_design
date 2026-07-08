"""
LOCAL room redesign with GEOMETRY LOCK: FLUX.2 [klein] base 4B + RefControl
depth LoRA (both Apache 2.0, commercial OK).

Difference vs Gen_klein.py: the fast distilled klein follows the photo only
loosely (windows drift/enlarge on bad seeds). This variant conditions the
model on a DEPTH MAP of the real room via the RefControl LoRA
(thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora), the same idea
as the old ControlNet-depth pipeline but on a modern editing model:
  image = [depth map (structure), room photo (identity/content)]
Trade-off: the LoRA needs the BASE (non-distilled) klein -> 50 steps with
real guidance -> ~2-4 min per render on the RTX 4060 (vs ~10 s distilled).

First run downloads ~8 GB (base transformer+VAE; the text encoder is reused
from the distilled repo already on disk) + 92 MB LoRA. DPT depth model is
already cached. Same staged low-RAM loading as Gen_klein.py.
"""

import gc
import os

# Plain-HTTPS downloads (the default Xet backend stalled on this network).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from PIL import Image

from loader import room_image  # NumPy image, RGB, HWC, uint8

# ----------------------------- user preferences -----------------------------
ROOM_TYPE = "living room"        # e.g. "bedroom", "home office", "kitchen"
DESIGN_STYLE = "modern"          # e.g. "japandi", "minimalist", "industrial"
COLOR_TONE = "warm vanilla latte"  # e.g. "sage green", "charcoal and walnut"

# ----------------------------- config knobs -----------------------------
OUTPUT_DIR = "output"
SEED = 7
DISTILLED_ID = "black-forest-labs/FLUX.2-klein-4B"       # text encoder source
BASE_ID = "black-forest-labs/FLUX.2-klein-base-4B"       # LoRA's base model
LORA_ID = "thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora"
LORA_FILE = "flux2_klein_4b_refcontrol_depth.safetensors"
LORA_SCALE = 0.9   # 0.8-1.0 per the LoRA card; higher = stronger geometry lock
STEPS = 50         # base model operating point (not the distilled 4)
GUIDANCE = 4.0

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU required.")

# --- output size: keep the photo's orientation; dims must divide by 32 ---
orig_h, orig_w = room_image.shape[:2]
if orig_w > orig_h:
    WIDTH, HEIGHT = 1024, 768
else:
    WIDTH, HEIGHT = 768, 1024
print(f"Input {orig_w}x{orig_h} -> generating {WIDTH}x{HEIGHT}")

# ----------------------------- prompt --------------------------------------
# "refcontrol" is the LoRA trigger word.
prompt = (
    f"refcontrol. Redesign this room as a {ROOM_TYPE} in {DESIGN_STYLE} "
    f"style with a {COLOR_TONE} color palette.\n\n"
    "HARD CONSTRAINTS - do not violate:\n"
    "- Follow the depth map exactly: every wall, window, door, ceiling, beam "
    "and column stays in its exact position, size and shape. Do not add, "
    "remove, move or resize any window or door. Surface finishes and "
    "materials MAY change; geometry may not.\n"
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
# STAGE 0 - downloads (resumable; loads from LOCAL dirs to avoid the
# transformers-on-Windows backslash-subfolder bug).
# ============================================================================
from huggingface_hub import hf_hub_download, snapshot_download

print("Fetching model files (first run: ~8 GB base + 92 MB LoRA)...")
DISTILLED_DIR = snapshot_download(  # already on disk; instant
    DISTILLED_ID, allow_patterns=["tokenizer/*", "text_encoder/*"],
)
BASE_DIR = snapshot_download(
    BASE_ID,
    allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"],
)
LORA_PATH = hf_hub_download(LORA_ID, LORA_FILE)
print("Downloads complete.")

# ============================================================================
# STAGE 0.5 - depth map of the real room (DPT-large, cached), then free it.
# ============================================================================
from transformers import DPTForDepthEstimation, DPTImageProcessor

print("Computing depth map...")
dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to("cuda")
dpt_model.eval()

with torch.no_grad():
    inputs = dpt_processor(images=Image.fromarray(room_image), return_tensors="pt").to("cuda")
    depth = dpt_model(**inputs).predicted_depth

depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1), size=(HEIGHT, WIDTH), mode="bicubic", align_corners=False,
).squeeze().cpu().numpy()
depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
depth_pil = Image.fromarray(depth).convert("RGB")

del dpt_model, dpt_processor
gc.collect()
torch.cuda.empty_cache()

# ============================================================================
# STAGE 1 - text encoder alone on the GPU: prompt -> embeddings, then free.
# The base model uses real CFG (guidance 4.0), so the empty negative prompt
# must be encoded here too (the text encoder is gone at pipeline time).
# ============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig as TransformersBnb4bit

print("Stage 1/2: loading text encoder (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(DISTILLED_DIR, "tokenizer"))
text_encoder = AutoModelForCausalLM.from_pretrained(
    os.path.join(DISTILLED_DIR, "text_encoder"),
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
        text_encoder=text_encoder, tokenizer=tokenizer,
        prompt=prompt, dtype=torch.bfloat16, device="cuda",
    ).cpu()
    negative_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder, tokenizer=tokenizer,
        prompt="", dtype=torch.bfloat16, device="cuda",
    ).cpu()

del text_encoder, tokenizer
gc.collect()
torch.cuda.empty_cache()
print("Prompts encoded, text encoder freed.")

# ============================================================================
# STAGE 2 - base transformer (4-bit) + LoRA + VAE: denoise.
# ============================================================================
print("Stage 2/2: loading base transformer (4-bit) + RefControl LoRA + VAE...")
transformer = Flux2Transformer2DModel.from_pretrained(
    BASE_DIR,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    quantization_config=DiffusersBnb4bit(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)

pipe = Flux2KleinPipeline.from_pretrained(
    BASE_DIR,
    transformer=transformer,
    text_encoder=None,
    tokenizer=None,
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights(LORA_PATH)
try:
    pipe.set_adapters(pipe.get_active_adapters(), adapter_weights=[LORA_SCALE])
except Exception:
    pass  # default weight 1.0 is within the card's recommended range
pipe.to("cuda")

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
import cv2


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

# Per the LoRA card: first image = depth (structure), second = reference.
result = pipe(
    prompt=None,
    prompt_embeds=prompt_embeds.to("cuda"),
    negative_prompt_embeds=negative_embeds.to("cuda"),
    image=[depth_pil, room_pil],
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    generator=torch.Generator(device="cuda").manual_seed(SEED),
).images[0]

print(f"seed {SEED}: geometry score {geometry_score(room_image, result):.3f}")

# Never overwrite previous results: ..._001.png, _002.png, ...
n = 1
while os.path.exists(os.path.join(OUTPUT_DIR, f"generated_interior_klein_lora_{n:03d}.png")):
    n += 1
out_path = os.path.join(OUTPUT_DIR, f"generated_interior_klein_lora_{n:03d}.png")
result.save(out_path)
depth_pil.save(os.path.join(OUTPUT_DIR, "klein_depth_map.png"))
room_pil.save(os.path.join(OUTPUT_DIR, "input.png"))
print(f"\nSaved: {out_path}")

try:
    result.show()
except Exception:
    pass
