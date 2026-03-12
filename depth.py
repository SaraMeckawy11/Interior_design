from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import numpy as np
import torch
import cv2

# --- Load your custom image ---
from loader import room_image  # Returns NumPy image (HWC, BGR or RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper to resize to nearest multiple of 64 ---
def resize_to_multiple_of_64(image, interpolation=cv2.INTER_CUBIC):
    h, w = image.shape[:2]
    new_h = (h // 64) * 64
    new_w = (w // 64) * 64
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)

# --- Generate Depth map using DPT ---
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
dpt_model.eval()

def get_depth_image(image):
    image_resized = resize_to_multiple_of_64(image)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)

    with torch.no_grad():
        depth = dpt_model(**inputs).predicted_depth

    # Resize depth to match multiples-of-64
    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=image_rgb.shape[:2], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize depth to 0-255
    depth_normalized = ((depth_resized - depth_resized.min()) / 
                        (depth_resized.max() - depth_resized.min()) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized).convert("RGB")
    return depth_image

# --- Prepare depth image ---
depth_image = get_depth_image(room_image)

# --- Load Depth ControlNet ---
depth_controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

# --- Base model ---
model_path = "Lykon/dreamshaper-8"

# --- Load pipeline with Depth ControlNet only ---
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_path,
    controlnet=[depth_controlnet],
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# --- Prompt with furniture explicitly ---
prompt = (
    "modern living room, interior design, soft ambient lighting, high detail, "
    "brown, white walls,"
    "realistic textures, highly detailed, photorealistic, 8k, designed by an interior architect, "
)

negative_prompt = (
    "blurry, cartoon, low-resolution, floating furniture, distorted walls, extra doors or windows"
)

# --- Run generation ---
output = pipe(
    prompt=prompt,
    image=[depth_image],  # only Depth
    num_inference_steps=30,  # increased steps for more detail
    guidance_scale=7.5,      # higher to enforce prompt
    controlnet_conditioning_scale=[0.8],  # strong depth guidance
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(576906284)  # reproducible
)

# --- Show output ---
output_image = output.images[0]
output_image.show()
