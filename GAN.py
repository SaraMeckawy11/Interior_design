from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
import cv2

# --- Load your custom image (user input) ---
from loader import room_image  # This should return a NumPy image (HWC, BGR or RGB)

# --- Convert user image to Canny edge map ---
def get_canny_image(image, size=(768, 768)):
    image = cv2.resize(image, size)
    canny = cv2.Canny(image, 100, 200)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb), canny

# --- Detect if room is empty / semi / furnished ---
def classify_room(canny_edges):
    edge_pixels = np.sum(canny_edges > 0)
    total_pixels = canny_edges.size
    edge_ratio = edge_pixels / total_pixels
    print(f"Edge ratio: {edge_ratio:.4f}")

    if edge_ratio < 0.04:
        return "empty", 0.3, 576906284
    elif edge_ratio < 0.07:
        return "semi", 0.4, 576906284
    else:
        return "furnished", 0.5, 42

# --- Resize with aspect ratio ---
orig_h, orig_w = room_image.shape[:2]
target_size = (1024, 768) if orig_w > orig_h else (768, 1024)
canny_image, canny_edges = get_canny_image(room_image, target_size)

# --- Decide conditioning scale and seed ---
room_type, conditioning_scale, seed = classify_room(canny_edges)
print(f"Detected {room_type.upper()} room -> scale={conditioning_scale}, seed={seed}")

# --- Load ControlNet with Canny conditioning ---
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
)

# --- Base model ---
model4 = "Lykon/dreamshaper-8"

# --- Load pipeline ---
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model4,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# --- Prompt settings ---
prompt = (
    "modern living room, interior design, warm soft ambient lighting, "
    "vanilla latte palette, professional interior designer style, "
    "photorealistic 8k, high detail, natural shadows, "
    "includes sofa set, coffee table, area rug, wall art, curtain, "
    "TV cabinet, plants, bookshelf, accent lighting, "
    "cohesive furniture arrangement matching room layout, "
    "window in place"
)

negative_prompt = (
    "blurry, cartoon, low-resolution, floating furniture, distorted walls, extra doors or windows, "
)

# --- Run generation ---
output = pipe(
    prompt=prompt,
    image=canny_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=conditioning_scale,
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(seed)
)

# --- Show output ---
output_image = output.images[0]
output_image.show()
