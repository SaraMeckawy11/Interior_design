"""
Geometry-faithful interior staging on DreamShaper-8 (img2img, depth + Canny).

For EXACT geometry this anchors three ways at once:
  1. img2img on your ACTUAL photo  -> real walls/window/doorway are the base
  2. depth ControlNet (DPT-large)  -> robust 3D structure even in low contrast
  3. Canny ControlNet              -> sharp architectural edges
The depth pass is what makes it hold the exact room (the receding wall, the
window plane, the doorway) regardless of restyling. It costs one extra model
load (DPT) -- slower than the pure-Canny version but still lighter than Gen.py
(no segmentation model), and DPT is freed before the SD pipeline loads.

Main knobs: STRENGTH (how far it may move from your room), DEPTH_SCALE and the
per-state Canny SCALE_* (how hard it sticks to the geometry).

Other details: clean structural edges, geometry prompt/negative, correct RGB,
device/dtype guard, attention/VAE slicing, fp16 folder fallback, optional hires
refine, saves result + depth + edge maps to output/.
"""

import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import DPTImageProcessor, DPTForDepthEstimation

# --- Load your custom image (user input) ---
from loader import room_image  # NumPy image, RGB, HWC, uint8

# ----------------------------- config knobs -----------------------------
OUTPUT_DIR = "output"
STEPS = 30
GUIDANCE = 7.5

# Optional refinement pass: a LIGHT img2img on the result to add sharpness/detail
# WITHOUT changing the layout. REFINE_SCALE = 1.0 keeps it at BASE resolution so
# it stays FAST -- your GPU thrashed to ~6.6s/step at 1.5x (768x1152). Bump to
# 1.25 only if you want more detail AND have the VRAM; REFINE = False to skip.
REFINE = True
REFINE_SCALE = 1.0
REFINE_STRENGTH = 0.35
REFINE_CONTROL = 0.40

# Canny carries the geometry now (KEEP IT > DEPTH_SCALE). An empty room's edges
# are only the walls/window/door -- the open floor has no edges, so high Canny
# locks the architecture WITHOUT blocking furniture on the floor.
SCALE_EMPTY = 0.80
SCALE_SEMI = 0.80
SCALE_FURNISHED = 0.80

# Depth is kept LOW on purpose. The depth map of an empty room is a flat empty
# floor, so a high DEPTH_SCALE tells the model "keep the floor clear" and the
# result comes out almost empty. Low depth gives a soft 3D hint without blocking
# furniture. Keep DEPTH_SCALE < the Canny SCALE_* above. Raise only if the walls/
# perspective start to drift; lower further (0.25) if it still won't furnish.
DEPTH_SCALE = 0.35

# img2img denoising strength = how much the room may change from your ACTUAL
# photo. The real pixels + depth + canny all anchor the geometry; strength only
# governs how much furniture/restyle is added. LOWER = more faithful but barer;
# HIGHER = more furniture but freer. Empty rooms need more to get furnished.
STRENGTH_EMPTY = 0.78
STRENGTH_SEMI = 0.64
STRENGTH_FURNISHED = 0.50

# Canny thresholds (on a denoised grayscale image). LOW values on purpose:
# construction photos are low-contrast (grey-on-grey zwalls, blown-out windows),
# so high thresholds miss the wall corners / window / doorway. 50/150 captures
# those faint architectural lines; the blur + dilate keep them clean.
CANNY_LOW = 50
CANNY_HIGH = 150

CANNY_CONTROLNET = "lllyasviel/control_v11p_sd15_canny"
DEPTH_CONTROLNET = "lllyasviel/sd-controlnet-depth"
DEPTH_MODEL = "Intel/dpt-large"
BASE_MODEL = "Lykon/dreamshaper-8"
FP16_DIR = os.environ.get("FP16_DIR", r"D:\interior_models_fp16")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def model_src(local_name, hub_id):
    local = os.path.join(FP16_DIR, local_name)
    return local if os.path.isdir(local) else hub_id


# --- Convert user image to a CLEAN structural Canny edge map ---
def get_canny_image(image, size):
    # `image` is RGB (loader already converted it). Build edges that describe
    # the architecture, not floor/wall texture noise:
    #   1. grayscale on luminance (proper edges, not per-channel noise)
    #   2. light blur so tiling / carpet speckle doesn't become fake edges
    #   3. Canny, then dilate 1px so wall/window/door lines are solid and
    #      continuous -- the ControlNet follows continuous lines far better.
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)            # raw, for classification
    solid = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)  # for control
    canny_rgb = cv2.cvtColor(solid, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb), edges


# --- DPT depth map: the robust 3D geometry signal (works in low contrast) ---
def get_depth_image(image, size, processor, model):
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)  # RGB already
    inputs = processor(images=Image.fromarray(resized), return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(size[1], size[0]), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = (depth * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(depth).convert("RGB")


# --- Detect if room is empty / semi / furnished (fast heuristic on edge density) ---
def classify_room(canny_edges):
    edge_ratio = float(np.count_nonzero(canny_edges)) / canny_edges.size
    print(f"Edge ratio: {edge_ratio:.4f}")

    if edge_ratio < 0.04:
        return "empty", SCALE_EMPTY, STRENGTH_EMPTY, 576906284
    elif edge_ratio < 0.07:
        return "semi", SCALE_SEMI, STRENGTH_SEMI, 576906284
    else:
        return "furnished", SCALE_FURNISHED, STRENGTH_FURNISHED, 42


# --- Resize with aspect ratio ---
# Cap the long side to 768. SD1.5 was trained at 512 and DUPLICATES the scene
# past ~768 on a side -- generating at 1024 tall is what produced two stacked
# rooms and destroyed the geometry. 512x768 / 768x512 is the safe range.
orig_h, orig_w = room_image.shape[:2]
target_size = (768, 512) if orig_w > orig_h else (512, 768)
canny_image, canny_edges = get_canny_image(room_image, target_size)

# img2img base = your ACTUAL room photo. This is what anchors the geometry:
# the real walls/window/doorway are the starting pixels, not noise.
base_image = Image.fromarray(cv2.resize(room_image, target_size, interpolation=cv2.INTER_AREA))

# --- Decide conditioning scale, strength and seed ---
room_type, conditioning_scale, strength, seed = classify_room(canny_edges)
print(f"Detected {room_type.upper()} room -> scale={conditioning_scale}, strength={strength}, seed={seed}")

# --- Depth map: load DPT, compute, then FREE it before the SD pipeline loads ---
print("Computing depth map (DPT-large) ...")
dpt_processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL)
dpt_model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
depth_image = get_depth_image(room_image, target_size, dpt_processor, dpt_model)
del dpt_model, dpt_processor
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# --- Prompt settings ---
# MUST stay under CLIP's 77-token limit or the tail is silently dropped. Geometry
# is held by depth + canny + the img2img base, so the prompt drives the DESIGN:
# a specific style, materials, palette and photographic lighting -- this is the
# real lever for how good it looks. Change "warm minimalist / vanilla latte" to
# any style you want (Scandinavian, Japandi, mid-century, boho...).
prompt = (
    "professional interior design photograph, cozy modern living room, warm minimalist style, "
    "plush beige linen sofa, walnut coffee table, soft wool rug, framed wall art, lush potted plants, "
    "brass floor lamp, throw blanket, warm cinematic lighting, golden hour daylight, "
    "vanilla latte and walnut palette, rich natural textures, architectural digest, "
    "photorealistic, ultra detailed, sharp focus"
)

negative_prompt = (
    "blurry, lowres, low quality, deformed, distorted furniture, bad proportions, "
    "ugly, messy, cluttered, oversaturated, cartoon, "
    "warped walls, curved walls, changed room shape, moved window, moved door, "
    "extra doors, extra windows, wrong perspective, "
    "fireplace, chimney, invented architecture"
)

# --- Load ControlNets: depth (3D geometry) + canny (edges) ---
depth_controlnet = ControlNetModel.from_pretrained(
    model_src("controlnet-depth", DEPTH_CONTROLNET), torch_dtype=dtype
)
canny_controlnet = ControlNetModel.from_pretrained(
    model_src("controlnet-canny", CANNY_CONTROLNET), torch_dtype=dtype
)

# --- Load pipeline (img2img so the real photo anchors the geometry) ---
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    model_src("dreamshaper-8", BASE_MODEL),
    controlnet=[depth_controlnet, canny_controlnet],
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
try:
    pipe.enable_vae_slicing()
except Exception:
    pass

# --- Run generation (img2img on the real photo + depth & Canny control) ---
output = pipe(
    prompt=prompt,
    image=base_image,                          # anchor on the real room
    control_image=[depth_image, canny_image],  # depth = 3D geometry, canny = edges
    strength=strength,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    controlnet_conditioning_scale=[DEPTH_SCALE, conditioning_scale],
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(seed),
)

output_image = output.images[0]

# --- Optional hi-res refinement: enhance detail on top of the good result ---
# Upscale + light img2img. Strength is low, so the layout/geometry is kept and
# only fine detail, sharpness and lighting are improved.
if REFINE:
    up_w = int(round(target_size[0] * REFINE_SCALE))
    up_h = int(round(target_size[1] * REFINE_SCALE))
    print(f"Refining at {up_w}x{up_h} (strength={REFINE_STRENGTH}) ...")
    up_image = output_image.resize((up_w, up_h), Image.LANCZOS)
    up_canny, _ = get_canny_image(room_image, (up_w, up_h))
    up_depth = depth_image.resize((up_w, up_h), Image.LANCZOS)
    refined = pipe(
        prompt=prompt,
        image=up_image,
        control_image=[up_depth, up_canny],
        strength=REFINE_STRENGTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        controlnet_conditioning_scale=[DEPTH_SCALE, REFINE_CONTROL],
        negative_prompt=negative_prompt,
        generator=torch.manual_seed(seed + 1),
    )
    output_image = refined.images[0]

# --- Save + show output ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_image.save(os.path.join(OUTPUT_DIR, "gan_generated_interior.png"))
canny_image.save(os.path.join(OUTPUT_DIR, "gan_canny_edges.png"))
depth_image.save(os.path.join(OUTPUT_DIR, "gan_depth_map.png"))
print(f"Saved result + depth + edge maps to: {OUTPUT_DIR}/")

try:
    output_image.show()
except Exception:
    pass
