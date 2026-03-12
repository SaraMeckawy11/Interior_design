import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from transformers import (
    DPTImageProcessor, DPTForDepthEstimation,
    AutoImageProcessor, UperNetForSemanticSegmentation
)

# ------------------------------
# Load your room image
# ------------------------------
from loader import room_image   # NumPy array

device = "cuda"

# ------------------------------
# Orientation (smaller for SDXL)
# ------------------------------
h, w = room_image.shape[:2]
if w > h:
    TARGET_W, TARGET_H = 768, 512
else:
    TARGET_W, TARGET_H = 512, 768

def resize_img(img):
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

print(f"Target: {TARGET_W}x{TARGET_H}")

# ------------------------------
# DPT Large Depth (best geometry)
# ------------------------------
dpt_proc = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
dpt_model.eval()

def get_depth(img_np):
    img = resize_img(img_np)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = dpt_proc(images=pil, return_tensors="pt").to(device)

    with torch.no_grad():
        depth = dpt_model(**inputs).predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(TARGET_H, TARGET_W),
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    d_norm = ((depth_resized - depth_resized.min()) /
              (depth_resized.max() - depth_resized.min()) * 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(d_norm, cv2.COLOR_GRAY2RGB))

# ------------------------------
# Segmentation for window detection
# ------------------------------
seg_proc = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
seg_model = UperNetForSemanticSegmentation.from_pretrained(
    "openmmlab/upernet-convnext-small"
).to(device)
seg_model.eval()

def get_seg_map(img_np):
    img = resize_img(img_np)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    inp = seg_proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = seg_model(**inp).logits

    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

# window detection
seg_map = get_seg_map(room_image)
label_names = seg_model.config.id2label
WINDOW_WORDS = ["window", "windowpane", "glass"]

detected_window = False
for cid in np.unique(seg_map):
    name = label_names.get(int(cid), "").lower()
    if any(w in name for w in WINDOW_WORDS):
        detected_window = True

print("Window detected:", detected_window)

# ------------------------------
# Prepare Control Images
# ------------------------------
depth_img = get_depth(room_image)

# ------------------------------
# Load SDXL + Depth ControlNet
# ------------------------------
depth_cn = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    controlnet=depth_cn,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()

# ------------------------------
# Prompt Logic
# ------------------------------
prompt = (
    "modern living room, interior design, soft ambient lighting, high detail, "
    "vanilla latte tones, realistic textures, highly detailed, "
    "photorealistic, 8k, designed by an interior architect, "
)

# if detected_window:
#     prompt += ", window in place"
# else:
#     prompt += ", coherent interior lighting"

negative = (
    "blurry, lowres, distorted, bad geometry, warped furniture, incorrect shadows"
)

# ------------------------------
# RUN SDXL
# ------------------------------
result = pipe(
    prompt=prompt,
    negative_prompt=negative,
    image=depth_img,
    num_inference_steps=28,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.7,
    generator=torch.manual_seed(42)
)

final = result.images[0]
final.show()
final.save("sdxl_output.png")
