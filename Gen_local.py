"""
Empty-room -> furnished interior, dual-ControlNet (depth + ADE20K seg) on DreamShaper-8.

This is the simple, working txt2img pipeline with targeted enhancements layered
on top of it (the core behaviour is unchanged):
  - correct RGB handling (loader already returns RGB; the old double BGR2RGB
    swap fed channel-flipped images to the depth/seg models)
  - official ADE20K palette for the seg ControlNet (was COLORMAP_JET, the wrong
    palette for control_v11p_sd15_seg)
  - window AND door detection feeding the prompt
  - depth/seg conditioning scales + seed/steps exposed as constants up top
  - intermediate maps + final result saved to output/ (not just .show())
  - memory-friendly: free depth/seg models before loading SD, attention/VAE
    slicing, and a local fp16 model-folder fallback (D:\\interior_models_fp16)
"""

import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import (
    DPTImageProcessor,
    DPTForDepthEstimation,
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
)

# --- Load your custom image ---
from loader import room_image  # NumPy image, RGB, HWC, uint8

# ----------------------------- config knobs -----------------------------
OUTPUT_DIR = "output"
SEED = 42
STEPS = 30 # a few more steps for cleaner detail
GUIDANCE = 7.5

# IMPORTANT lever for "how furnished": the seg map of an EMPTY room labels
# everything floor/wall, so a strong seg weight makes the model reproduce that
# emptiness. Keeping seg LOW (~0.15) lets the prompt actually fill the room with
# furniture, while the correct ADE20K palette means even this low weight still
# gently anchors the window/door. depth stays up to hold perspective (it does
# not suppress furniture). Raise SEG_SCALE toward 0.35 only if openings drift.
DEPTH_SCALE = 0.5
SEG_SCALE = 0.35

# Prefer locally-converted fp16 weights if present (faster + lighter on RAM),
# otherwise fall back to the Hub repos.
FP16_DIR = os.environ.get("FP16_DIR", r"D:\interior_models_fp16")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def model_src(local_name, hub_id):
    local = os.path.join(FP16_DIR, local_name)
    return local if os.path.isdir(local) else hub_id


# Official ADE20K palette (150 classes). The seg ControlNet was trained on these
# exact colors, so this is what makes the segmentation conditioning useful.
ADE20K_PALETTE = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
], dtype=np.uint8)


# --- Dynamically decide orientation (portrait or landscape) ---
orig_h, orig_w = room_image.shape[:2]
if orig_w > orig_h:
    TARGET_WIDTH, TARGET_HEIGHT = 1024, 768
else:
    TARGET_WIDTH, TARGET_HEIGHT = 768, 1024

print(f"Detected orientation: {'Landscape' if orig_w > orig_h else 'Portrait'}")
print(f"Using target size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
# Note: SD1.5 is happiest near 512-768. If you ever see duplicated furniture or
# run out of VRAM, drop the long side here from 1024 to 768.


def resize_image(image, width=TARGET_WIDTH, height=TARGET_HEIGHT, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(image, (width, height), interpolation=interpolation)


# --- Generate Depth map using DPT ---
dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
dpt_model.eval()


def get_depth_image(image):
    # `image` is already RGB (loader converted it once). Do NOT swap again.
    image_resized = resize_image(image)

    inputs = dpt_processor(images=Image.fromarray(image_resized), return_tensors="pt").to(device)

    with torch.no_grad():
        depth = dpt_model(**inputs).predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    depth_normalized = ((depth_resized - depth_resized.min()) /
                        (depth_resized.max() - depth_resized.min() + 1e-8) * 255).astype(np.uint8)
    return Image.fromarray(depth_normalized).convert("RGB")


# --- Generate Segmentation map using UPerNet ---
seg_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
seg_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(device)
seg_model.eval()
label_names = seg_model.config.id2label


def get_segmentation_map(image):
    # `image` is already RGB. No BGR2RGB swap here either.
    image_resized = resize_image(image)
    inputs = seg_processor(images=Image.fromarray(image_resized), return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return seg_map


def colorize_segmentation(seg_map):
    # Map class ids -> official ADE20K colors (RGB), then size to the target.
    seg_color = ADE20K_PALETTE[np.clip(seg_map, 0, len(ADE20K_PALETTE) - 1)].astype(np.uint8)
    seg_color = cv2.resize(seg_color, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(seg_color).convert("RGB")


# --------------------------------------------------------
# WINDOW + DOOR DETECTION
# --------------------------------------------------------

WINDOW_KEYWORDS = ["window", "windowpane"]
DOOR_KEYWORDS = ["door", "screen door"]

seg_map_raw = get_segmentation_map(room_image)
unique_classes = set(np.unique(seg_map_raw))

detected_window = False
detected_door = False
window_labels = []
door_labels = []

for class_id in unique_classes:
    name = label_names.get(int(class_id), "").lower()
    if any(word in name for word in WINDOW_KEYWORDS):
        detected_window = True
        window_labels.append(name)
    if any(word in name for word in DOOR_KEYWORDS):
        detected_door = True
        door_labels.append(name)

print("\nDetected windows:", window_labels or "none")
print("Detected doors:", door_labels or "none")


# --------------------------------------------------------
# PROMPT & NEGATIVE PROMPT LOGIC
# --------------------------------------------------------

# Main prompt stays CLEAN
prompt = (
    "modern living room, interior design, warm soft ambient lighting, "
    "vanilla latte palette, professional interior designer style, "
    "photorealistic 8k, high detail, natural shadows, "
    "includes sofa set, coffee table, area rug, wall art, curtain, "
    "TV cabinet, plants, bookshelf, accent lighting, "
    "cohesive furniture arrangement matching room layout"
)

# Base negative prompt
negative_prompt = (
    "blurry, lowres, distorted, floating furniture, bad lighting, wrong perspective"
)

# Keep the real openings, forbid invented ones.
if detected_window:
    prompt += ", window in its original place"
else:
    negative_prompt += ", no window, added window, fake window"

if detected_door:
    prompt += ", door in its original place"
else:
    negative_prompt += ", no door, added door, fake door"

print("\nFINAL PROMPT:\n", prompt)
print("\nFINAL NEGATIVE PROMPT:\n", negative_prompt)
print("--------------------------------------------------------")


# --------------------------------------------------------
# RUN GENERATION
# --------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

depth_image = get_depth_image(room_image)
seg_image = colorize_segmentation(seg_map_raw)

# Free the analysis models before loading the big SD pipeline (helps on
# low-RAM / low-VRAM machines where holding everything at once segfaults / OOMs).
del dpt_model, dpt_processor, seg_model, seg_processor
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

depth_controlnet = ControlNetModel.from_pretrained(
    model_src("controlnet-depth", "lllyasviel/sd-controlnet-depth"), torch_dtype=dtype
)
seg_controlnet = ControlNetModel.from_pretrained(
    model_src("controlnet-seg", "lllyasviel/control_v11p_sd15_seg"), torch_dtype=dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_src("dreamshaper-8", "Lykon/dreamshaper-8"),
    controlnet=[depth_controlnet, seg_controlnet],
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

output = pipe(
    prompt=prompt,
    image=[depth_image, seg_image],
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    controlnet_conditioning_scale=[DEPTH_SCALE, SEG_SCALE],
    negative_prompt=negative_prompt + ", empty room, no furniture",
    generator=torch.manual_seed(SEED),
)

output_image = output.images[0]

# Save everything (the old script only popped up a window and lost the result).
output_image.save(os.path.join(OUTPUT_DIR, "generated_interior.png"))
depth_image.save(os.path.join(OUTPUT_DIR, "depth_map.png"))
seg_image.save(os.path.join(OUTPUT_DIR, "seg_map.png"))
Image.fromarray(room_image).save(os.path.join(OUTPUT_DIR, "input.png"))
print(f"\nSaved result + maps to: {OUTPUT_DIR}/")

try:
    output_image.show()
except Exception:
    pass
