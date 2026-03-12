from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import DPTImageProcessor, DPTForDepthEstimation, AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import torch
import cv2

# --- Load your custom image ---
from loader import room_image  # Returns NumPy image (HWC, BGR or RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dynamically decide orientation (portrait or landscape) ---
orig_h, orig_w = room_image.shape[:2]
if orig_w > orig_h:
    TARGET_WIDTH, TARGET_HEIGHT = 1024, 768
else:
    TARGET_WIDTH, TARGET_HEIGHT = 768, 1024

print(f"Detected orientation: {'Landscape' if orig_w > orig_h else 'Portrait'}")
print(f"Using target size: {TARGET_WIDTH}x{TARGET_HEIGHT}")

def resize_image(image, width=TARGET_WIDTH, height=TARGET_HEIGHT, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(image, (width, height), interpolation=interpolation)

# --- Generate Depth map using DPT ---
dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
dpt_model.eval()

def get_depth_image(image):
    image_resized = resize_image(image)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    inputs = dpt_processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)

    with torch.no_grad():
        depth = dpt_model(**inputs).predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(TARGET_HEIGHT, TARGET_WIDTH),
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    depth_normalized = ((depth_resized - depth_resized.min()) /
                        (depth_resized.max() - depth_resized.min()) * 255).astype(np.uint8)
    return Image.fromarray(depth_normalized).convert("RGB")

# --- Generate Segmentation map using UPerNet ---
seg_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
seg_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(device)
seg_model.eval()

def get_segmentation_map(image):
    image_resized = resize_image(image)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    inputs = seg_processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return seg_map

def get_segmentation_image(image):
    seg_map = get_segmentation_map(image)
    seg_colormap = cv2.applyColorMap((seg_map * 10).astype(np.uint8), cv2.COLORMAP_JET)
    seg_colormap_resized = cv2.resize(seg_colormap, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(seg_colormap_resized)


# --------------------------------------------------------
# 📌 WINDOW DETECTION ONLY
# --------------------------------------------------------

label_names = seg_model.config.id2label

WINDOW_KEYWORDS = ["window", "windowpane"]

seg_map_raw = get_segmentation_map(room_image)
unique_classes = set(np.unique(seg_map_raw))

detected_window = False
window_labels = []

for class_id in unique_classes:
    name = label_names.get(int(class_id), "").lower()

    if any(word in name for word in WINDOW_KEYWORDS):
        detected_window = True
        window_labels.append(name)

print("\nDetected windows:", window_labels)


# --------------------------------------------------------
# 📌 PROMPT & NEGATIVE PROMPT LOGIC
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

# Add "no window" ONLY to the negative prompt

if detected_window:
    prompt += ", window in place"
else:
    negative_prompt += ", no window"

print("\nFINAL PROMPT:\n", prompt)
print("\nFINAL NEGATIVE PROMPT:\n", negative_prompt)
print("--------------------------------------------------------")


# --------------------------------------------------------
# RUN GENERATION
# --------------------------------------------------------

depth_image = get_depth_image(room_image)
seg_image = get_segmentation_image(room_image)

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

output = pipe(
    prompt=prompt,
    image=[depth_image, seg_image],
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=[0.5, 0.1],
    negative_prompt=negative_prompt + ", empty room, no furniture",
    generator=torch.manual_seed(42)
)

output_image = output.images[0]
output_image.show()