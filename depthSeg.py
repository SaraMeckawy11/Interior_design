import os
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

from transformers import (
    DPTImageProcessor,
    DPTForDepthEstimation,
    AutoImageProcessor,
    UperNetForSemanticSegmentation
)

# --------------------------------------------------------
# LOAD INPUT IMAGE
# --------------------------------------------------------
from loader import room_image  # numpy HWC BGR

device = "cuda" if torch.cuda.is_available() else "cpu"

orig_h, orig_w = room_image.shape[:2]
if orig_w > orig_h:
    TARGET_WIDTH, TARGET_HEIGHT = 768, 512
else:
    TARGET_WIDTH, TARGET_HEIGHT = 512, 768

def resize_image(image, width=TARGET_WIDTH, height=TARGET_HEIGHT):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

room_resized = resize_image(room_image)

# --------------------------------------------------------
# LOAD SEG + DEPTH MODELS ONCE
# --------------------------------------------------------
dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()

seg_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
seg_model = UperNetForSemanticSegmentation.from_pretrained(
    "openmmlab/upernet-convnext-small"
).to(device).eval()

# --------------------------------------------------------
# SEGMENTATION MAP
# --------------------------------------------------------
def get_segmentation_map(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = seg_processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    with torch.no_grad():
        out = seg_model(**inputs)
    return out.logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

seg_map = get_segmentation_map(room_resized)
label_names = seg_model.config.id2label

# --------------------------------------------------------
# AGGRESSIVE STRUCTURAL MASK
# --------------------------------------------------------
STRUCTURE = ["wall", "floor", "ceiling"]

mask = np.zeros_like(seg_map, dtype=np.uint8)

for cid, cname in label_names.items():
    cname = cname.lower()
    if cname not in STRUCTURE:
        mask[seg_map == int(cid)] = 255

# expand mask strongly
mask = cv2.dilate(mask, np.ones((35, 35), np.uint8), 1)

init_image = Image.fromarray(cv2.cvtColor(room_resized, cv2.COLOR_BGR2RGB))
mask_image = Image.fromarray(mask).convert("L")

# --------------------------------------------------------
# SHOW ORIGINAL + MASK BEFORE INPAINTING
# --------------------------------------------------------
print("Showing original resized image before inpainting...")
Image.fromarray(cv2.cvtColor(room_resized, cv2.COLOR_BGR2RGB)).show()

print("Showing mask before inpainting...")
mask_show_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
Image.fromarray(mask_show_colored).show()

# --------------------------------------------------------
# RUN INPAINTING
# --------------------------------------------------------
print("Running furniture removal ...")

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to(device)

empty_room_pil = inpaint_pipe(
    prompt="empty room interior, clean walls, no furniture, seamless background",
    image=init_image,
    mask_image=mask_image,
    negative_prompt="distortion, artifacts, blurry walls, messy texture, leftover furniture",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.manual_seed(42)
).images[0]

empty_room = cv2.cvtColor(np.array(empty_room_pil), cv2.COLOR_RGB2BGR)
empty_room = resize_image(empty_room)

# --------------------------------------------------------
# SHOW EMPTY ROOM BEFORE CONTINUING
# --------------------------------------------------------
print("Empty room generated. Showing image...")
Image.fromarray(cv2.cvtColor(empty_room, cv2.COLOR_BGR2RGB)).show()

# --------------------------------------------------------
# DEPTH & SEG FOR CONTROLNET (ON EMPTY ROOM)
# --------------------------------------------------------
def get_depth_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = dpt_processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)

    with torch.no_grad():
        depth = dpt_model(**inputs).predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    depth_norm = ((depth_resized - depth_resized.min()) /
                  (depth_resized.max() - depth_resized.min()) * 255).astype(np.uint8)

    return Image.fromarray(depth_norm).convert("RGB")


def get_segmentation_image(image):
    seg_map = get_segmentation_map(image)
    seg_colormap = cv2.applyColorMap((seg_map * 10).astype(np.uint8), cv2.COLORMAP_JET)
    return Image.fromarray(resize_image(seg_colormap))

# --------------------------------------------------------
# WINDOW DETECTION
# --------------------------------------------------------
WINDOW_KEYWORDS = ["window", "windowpane"]

seg_map_raw = get_segmentation_map(empty_room)
unique_classes = np.unique(seg_map_raw)

detected_window = False
window_labels = []

for cid in unique_classes:
    name = label_names.get(int(cid), "").lower()
    if any(w in name for w in WINDOW_KEYWORDS):
        detected_window = True
        window_labels.append(name)

print("Detected windows:", window_labels)

# --------------------------------------------------------
# PROMPTS
# --------------------------------------------------------
prompt = (
    "modern living room, interior design, soft ambient lighting, high detail, "
    "vanilla latte tones, realistic textures, highly detailed, "
    "photorealistic, 8k, designed by an interior architect"
)

negative_prompt = (
    "blurry, lowres, distorted, floating furniture, bad lighting, wrong perspective"
)

if detected_window:
    prompt += ", window in place"
else:
    negative_prompt += ", no window"

# --------------------------------------------------------
# CONTROLNET INPUTS
# --------------------------------------------------------
depth_image = get_depth_image(empty_room)
seg_image = get_segmentation_image(empty_room)

depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)
seg_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    controlnet=[depth_controlnet, seg_controlnet],
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

print("Running final generation ...")

output = pipe(
    prompt=prompt,
    image=[depth_image, seg_image],
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=[0.5, 0.1],
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(42)
)

final_image = output.images[0]
final_image.show()

print("FINAL IMAGE GENERATED.")
