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


def _load_api_key():
    # Env var first; fall back to gemini_key.txt next to this script
    # (gitignored), which works even when the editor was started before
    # setx and doesn't see fresh environment variables.
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_key.txt")
        if os.path.isfile(key_file):
            with open(key_file, encoding="utf-8") as f:
                key = f.read().strip()
    return key


API_KEY = _load_api_key()

# ------------------------- model choice -------------------------
# Pick the Nano Banana model by number (no fallback — exactly what you
# choose is what runs and what you pay for):
#   1 = Nano Banana 2       (gemini-3.1-flash-image)       ~$0.067/img — best quality/cost balance
#   2 = Nano Banana 2 Lite  (gemini-3.1-flash-lite-image)  ~$0.034/img — cheapest + fastest
#   3 = Nano Banana Pro     (gemini-3-pro-image)           ~$0.134/img — maximum quality, slowest
#   4 = Nano Banana legacy  (gemini-2.5-flash-image)       ~$0.039/img — old model, 1024px only
MODEL_CHOICE = 4

MODEL_OPTIONS = {
    1: "gemini-3.1-flash-image",
    2: "gemini-3.1-flash-lite-image",
    3: "gemini-3-pro-image",
    4: "gemini-2.5-flash-image",
}
MODEL = MODEL_OPTIONS[MODEL_CHOICE]
IMAGE_SIZE = "2K"  # "1K", "2K", or "4K" — only applies to 3.x models (2.5 is fixed 1024px)

# The plain Gemini API is prepay-only (the $300 Cloud credit does NOT apply),
# so calls go through Vertex AI instead: same models, billed to the Cloud
# credit. Requires a service-account-bound API key (July 2026 setup) with the
# "Vertex AI User" role. Set VERTEX_PROJECT = None to use the plain API.
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "gen-lang-client-0515245864")

if VERTEX_PROJECT:
    API_URL = (
        "https://aiplatform.googleapis.com/v1/projects/" + VERTEX_PROJECT +
        "/locations/global/publishers/google/models/{model}:generateContent"
    )
else:
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
            "role": "user",  # required by Vertex (plain API tolerates omitting it)
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
    # input photo's. imageSize: only 3.1-flash and pro support 1K/2K/4K —
    # Lite is fixed at 1K and legacy 2.5 at 1024px; both reject the field.
    if model in ("gemini-3.1-flash-image", "gemini-3-pro-image"):
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
print(f"Model {MODEL_CHOICE}: {MODEL}")

output_image = generate(MODEL, image_b64)

if output_image is None:
    raise SystemExit(
        f"{MODEL} failed. Check the error messages above and your "
        "billing/quota at https://console.cloud.google.com."
    )

# Never overwrite previous results; always continue after the highest
# existing number (so deleting old files can't reshuffle the order).
import glob
import re
_nums = [int(m.group(1)) for f in glob.glob(os.path.join(OUTPUT_DIR, "generated_interior_*.png"))
         if (m := re.search(r"generated_interior_(\d+)\.png$", f))]
out_path = os.path.join(OUTPUT_DIR, f"generated_interior_{max(_nums, default=0) + 1:03d}.png")
output_image.save(out_path)
Image.fromarray(room_image).save(os.path.join(OUTPUT_DIR, "input.png"))
print(f"\nSaved result to: {out_path}")

try:
    output_image.show()
except Exception:
    pass
