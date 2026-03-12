from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import (
    DPTImageProcessor,
    DPTForDepthEstimation,
)
from PIL import Image
import numpy as np
import torch
import cv2

# --- Load your custom image ---
from loader import room_image  # NumPy (HWC)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# --- Dynamically decide output resolution ---
orig_h, orig_w = room_image.shape[:2]
if orig_w > orig_h:
    TARGET_WIDTH, TARGET_HEIGHT = 1024, 768
else:
    TARGET_WIDTH, TARGET_HEIGHT = 768, 1024

print(f"Orientation: {'Landscape' if orig_w > orig_h else 'Portrait'}")
print(f"Using size: {TARGET_WIDTH}x{TARGET_HEIGHT}")


def resize_image(image, width=TARGET_WIDTH, height=TARGET_HEIGHT):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------
#  DEPTH MAP (DPT-Large)
# --------------------------------------------------------

dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
dpt_model.eval()


def get_depth_image(image):
    img_resized = resize_image(image)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    inputs = dpt_processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(device)

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


# --------------------------------------------------------
# PROMPT
# --------------------------------------------------------

prompt = (
    "modern living room, interior design, soft ambient lighting, high detail, "
    "vanilla latte, white walls,"
    "realistic textures, highly detailed, photorealistic, 8k, designed by an interior architect, "
)


negative_prompt = (
    "blurry, lowres, distorted, overexposed, bad lighting, wrong perspective, "
    "floating furniture, empty room, no furniture"
)

# --------------------------------------------------------
# LOAD CONTROLNET + PIPELINE
# --------------------------------------------------------

depth_image = get_depth_image(room_image)

depth_controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch_dtype
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-v2-turbo",
    controlnet=depth_controlnet,
    torch_dtype=torch_dtype,
    safety_checker=None
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


# --------------------------------------------------------
# GENERATE
# --------------------------------------------------------

output = pipe(
    prompt=prompt,
    image=depth_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.50,
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(42)
)

result = output.images[0]
result.show()
