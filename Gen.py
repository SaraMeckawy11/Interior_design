"""
Room redesign via Gemini image editing ("Nano Banana" family).

Same simple flat-script shape as before, but the model changed: instead of
SD1.5 + depth/seg ControlNet (which fights between "respect layout" and
"add furniture" through conditioning weights), we send the ACTUAL room photo
to an instruction-based image-editing model. It re-renders the photo itself,
so walls / windows / doors / perspective are preserved by construction —
no depth or seg weights to tune, and the room can never come out empty
because a control map suppressed the furniture.

Works for BOTH cases:
  - empty room      -> fully furnished per the preferences below
  - furnished room  -> existing furniture replaced with the new design

Setup (one time):
  1. Get a free API key at https://aistudio.google.com/apikey
  2. setx GEMINI_API_KEY "your-key-here"   (then open a NEW terminal)

Nothing runs on the GPU and no model weights are downloaded, so the
machine's RAM/VRAM/disk limits don't matter anymore.
"""

import base64
import os
import time
from io import BytesIO

import requests
from PIL import Image

from loader import room_image  # NumPy image, RGB, HWC, uint8

# ------------------------- user preferences -------------------------
ROOM_TYPE = "living room"          # e.g. "bedroom", "kitchen", "home office"
DESIGN_STYLE = "modern"            # e.g. "scandinavian", "japandi", "industrial", "classic"
COLOR_TONE = "warm vanilla latte"  # e.g. "earthy neutrals", "cool grey", "sage green"

# ------------------------------ config ------------------------------
OUTPUT_DIR = "output"
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Tried in order; first one that succeeds wins. NOTE: no Gemini image model
# has free API quota (verified: all 429 with limit 0) — billing required.
#   gemini-3.1-flash-image      -> default: best quality/cost (~$0.07 @1K)
#   gemini-3.1-flash-lite-image -> cheapest/fastest (~$0.034, looser)
#   gemini-2.5-flash-image      -> legacy fallback ($0.039)
# For maximum-quality final renders ($0.134):
#   set GEMINI_IMAGE_MODEL=gemini-3-pro-image
_first = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
MODELS = [_first] + [m for m in
                     ("gemini-3.1-flash-image", "gemini-3.1-flash-lite-image",
                      "gemini-2.5-flash-image")
                     if m != _first]
IMAGE_SIZE = "2K"  # "1K", "2K", or "4K" — only applies to 3.x models (2.5 is fixed 1024px)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ------------------------------ prompt ------------------------------
# One instruction covers both empty and furnished inputs. The editing model
# sees the real photo, so layout preservation is stated as a hard rule
# instead of being approximated by ControlNet weights.
prompt = (
    f"Redesign this room as a {ROOM_TYPE} in {DESIGN_STYLE} style with a "
    f"{COLOR_TONE} color palette.\n"
    "\n"
    "HARD CONSTRAINTS — do not violate:\n"
    "- Keep the room's architectural GEOMETRY exactly as in the photo: every "
    "wall, window, door, ceiling, beam and column stays in its exact "
    "position, size and shape. Do not add, remove, move or resize any window "
    "or door. Surface finishes and materials MAY change; geometry may not.\n"
    "- Keep the exact same camera position, angle and lens/perspective.\n"
    "- Keep daylight coming from the real windows; lighting must be "
    "physically plausible for this room.\n"
    "\n"
    "DESIGN BRIEF — work like a senior professional interior designer:\n"
    "- If the room is empty, furnish it completely. If it already has "
    "furniture or decor, replace all of it with the new design.\n"
    "- FULLY FINISH every surface (the photo may show an unfinished or "
    "under-construction room): lay a brand-new premium finished floor in a "
    "material that suits the style (e.g. wide-plank oak or large-format "
    "polished tile) covering the ENTIRE floor, smooth painted walls, and a "
    "clean finished ceiling. Absolutely no construction dust, debris, "
    "stains, bare concrete, exposed wires or unfinished surfaces may remain "
    "anywhere, including floor areas visible around rugs and furniture.\n"
    "- Full, cohesive furniture arrangement appropriate for a "
    f"{ROOM_TYPE}: primary seating/furniture pieces, rug sized correctly "
    "for the seating zone, curtains on the real windows, wall art, plants, "
    "and layered lighting (ambient + accent + task).\n"
    "- Realistic scale and proportions; furniture sits properly on the "
    "floor with correct contact shadows; nothing floats or clips into "
    "walls.\n"
    "- Cohesive materials and textures true to the style; tasteful, "
    "editorial-quality styling like an Architectural Digest photo shoot.\n"
    "- Photorealistic output, high detail, natural soft shadows.\n"
)


# ----------------------------- API call -----------------------------
def encode_input_image(np_image):
    buf = BytesIO()
    Image.fromarray(np_image).save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_image(response_json):
    # Image comes back as an inlineData part; a text part may accompany it.
    for candidate in response_json.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return Image.open(BytesIO(base64.b64decode(inline["data"])))
    return None


def generate(model, image_b64):
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ],
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }
    # Aspect ratio is intentionally omitted: for editing it defaults to the
    # input photo's. imageSize is a 3.x-only feature (2.5 rejects it).
    if not model.startswith("gemini-2.5"):
        body["generationConfig"]["imageConfig"] = {"imageSize": IMAGE_SIZE}
    for attempt in range(4):
        try:
            resp = requests.post(
                API_URL.format(model=model),
                headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
                json=body,
                timeout=300,
            )
        except requests.exceptions.RequestException as e:
            # Dropped connection etc. — transient, retry like a 5xx.
            if attempt < 3:
                wait = 15 * (attempt + 1)
                print(f"  {model}: network error ({type(e).__name__}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  {model}: network error: {e}")
            return None
        if resp.status_code == 200:
            return extract_image(resp.json())
        # 429/5xx are transient — back off and retry; anything else, move on.
        if resp.status_code in (429, 500, 502, 503) and attempt < 3:
            wait = 20 * (attempt + 1)
            print(f"  {model}: HTTP {resp.status_code}: {resp.text[:1500]}")
            print(f"  retrying in {wait}s...")
            time.sleep(wait)
            continue
        print(f"  {model}: HTTP {resp.status_code}: {resp.text[:1500]}")
        return None
    return None


# ------------------------------- run --------------------------------
if not API_KEY:
    raise SystemExit(
        "GEMINI_API_KEY is not set.\n"
        "Get a free key at https://aistudio.google.com/apikey then run:\n"
        '  setx GEMINI_API_KEY "your-key-here"\n'
        "and open a new terminal."
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)
image_b64 = encode_input_image(room_image)

print(f"Room type: {ROOM_TYPE} | Style: {DESIGN_STYLE} | Tone: {COLOR_TONE}")

output_image = None
for model in MODELS:
    print(f"Generating with {model}...")
    output_image = generate(model, image_b64)
    if output_image is not None:
        print(f"Success with {model}.")
        break

if output_image is None:
    raise SystemExit(
        "All models failed. Check the API key, quota/billing at "
        "https://aistudio.google.com, and the error messages above."
    )

output_image.save(os.path.join(OUTPUT_DIR, "generated_interior.png"))
Image.fromarray(room_image).save(os.path.join(OUTPUT_DIR, "input.png"))
print(f"\nSaved result to: {OUTPUT_DIR}/generated_interior.png")

try:
    output_image.show()
except Exception:
    pass
