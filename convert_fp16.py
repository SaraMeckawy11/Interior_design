"""
One-time helper: build self-contained fp16 copies of the models Gen.py needs.

Why: this machine has very little free RAM. The stock weights are stored as
fp32 on disk (UNet 3.4 GB, each ControlNet 1.4 GB). Reading a file that large
into the ~3-4 GB of free RAM intermittently segfaults. We stream each tensor
with safetensors.safe_open (peak RAM = one tensor), cast to fp16, and write
half-size files. Gen.py then loads these local fp16 folders with small reads.

Run once:  python convert_fp16.py
Output:    ./models_fp16/{dreamshaper-8, controlnet-depth, controlnet-seg}
"""

import os
import glob
import json
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

# C: is full; write the fp16 copies to D: which has plenty of free space.
OUT_ROOT = os.environ.get("FP16_DIR", r"D:\interior_models_fp16")


def find_snapshot(repo_id):
    """Locate the local cache snapshot dir for a repo (download if missing)."""
    path = snapshot_download(repo_id, local_files_only=True)
    return path


def convert_safetensors_fp16(src, dst):
    """Stream src -> dst casting every tensor to fp16. Peak RAM ~ one tensor."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tensors = {}
    with safe_open(src, framework="pt") as f:
        meta = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key).half()
    meta = {**meta, "format": "pt"}
    save_file(tensors, dst, metadata=meta)
    print(f"  fp16 -> {dst} ({os.path.getsize(dst)//1024//1024} MB)")


def copy_tree_except_weights(src_dir, dst_dir, skip_dirs=()):
    """Copy configs/tokenizer/etc., skipping big weight files (handled separately)."""
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        if any(rel == s or rel.startswith(s + os.sep) for s in skip_dirs):
            continue
        for fn in files:
            if fn.endswith((".safetensors", ".bin")):
                continue  # weights are converted, not copied
            os.makedirs(os.path.join(dst_dir, rel), exist_ok=True)
            shutil.copy2(os.path.join(root, fn), os.path.join(dst_dir, rel, fn))


def build_pipeline_fp16(repo_id, out_name):
    print(f"\n[pipeline] {repo_id}")
    snap = find_snapshot(repo_id)
    out = os.path.join(OUT_ROOT, out_name)
    os.makedirs(out, exist_ok=True)

    # Skip the safety checker entirely (Gen.py disables it; saves ~1.2 GB).
    copy_tree_except_weights(snap, out, skip_dirs=("safety_checker",))

    # Convert the component weights we actually use.
    for comp in ("unet", "vae", "text_encoder"):
        src = os.path.join(snap, comp, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(src):
            src = os.path.join(snap, comp, "model.safetensors")
        if not os.path.exists(src):
            print(f"  (skip {comp}: no safetensors)")
            continue
        dst = os.path.join(out, comp, os.path.basename(src))
        convert_safetensors_fp16(src, dst)

    # Drop the safety_checker from model_index.json so the pipeline doesn't look for it.
    mi_path = os.path.join(out, "model_index.json")
    with open(mi_path) as f:
        mi = json.load(f)
    mi.pop("safety_checker", None)
    mi.pop("feature_extractor", None)
    with open(mi_path, "w") as f:
        json.dump(mi, f, indent=2)
    print(f"  done -> {out}")


def build_controlnet_fp16(repo_id, out_name):
    print(f"\n[controlnet] {repo_id}")
    snap = find_snapshot(repo_id)
    out = os.path.join(OUT_ROOT, out_name)
    os.makedirs(out, exist_ok=True)
    shutil.copy2(os.path.join(snap, "config.json"), os.path.join(out, "config.json"))
    src = os.path.join(snap, "diffusion_pytorch_model.safetensors")
    convert_safetensors_fp16(src, os.path.join(out, "diffusion_pytorch_model.safetensors"))
    print(f"  done -> {out}")


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    build_pipeline_fp16("Lykon/dreamshaper-8", "dreamshaper-8")
    build_controlnet_fp16("lllyasviel/sd-controlnet-depth", "controlnet-depth")
    build_controlnet_fp16("lllyasviel/control_v11p_sd15_seg", "controlnet-seg")
    print("\nAll fp16 models built under ./models_fp16/")
