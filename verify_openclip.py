#!/usr/bin/env python3
"""
verify_openclip.py - Robust OpenCLIP smoke test.

Recommended (uses your local BDD folder):
  python verify_openclip.py --dir data/bdd100k/images/100k/val --n 5

Or point at your subset folder:
  python verify_openclip.py --dir data/subsets/bdd100k_val_2000/images --n 10

Optional: use a single local image
  python verify_openclip.py --image /path/to/img.jpg

Optional: URL fallback (if your network allows)
  python verify_openclip.py --url "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=640&q=80"

Force CPU:
  python verify_openclip.py --cpu
"""

import argparse
import os
import sys
from pathlib import Path
from io import BytesIO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def pick_images_from_dir(img_dir: Path, n: int):
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    if not files:
        raise RuntimeError(f"No images found in: {img_dir.resolve()}")
    return files[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="Directory containing images to test (preferred).")
    ap.add_argument("--n", type=int, default=5, help="Number of images from --dir to test (default: 5).")
    ap.add_argument("--image", type=str, default=None, help="Path to a single local image.")
    ap.add_argument("--url", type=str, default=None, help="URL to a test image (fallback).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    ap.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name (default: ViT-B-32).")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="Pretrained tag.")
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:
        print("FAILED: torch import error:\n", e)
        return 1

    device = "cpu"
    if (not args.cpu) and torch.cuda.is_available():
        device = "cuda"

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("gpu:", torch.cuda.get_device_name(0))
        except Exception:
            pass
    print("device selected:", device)
    print("torch cuda runtime:", torch.version.cuda)

    try:
        import open_clip
    except Exception as e:
        print("FAILED: open_clip import error:\n", e)
        return 1

    try:
        from PIL import Image
    except Exception as e:
        print("FAILED: pillow import error (pip install pillow):\n", e)
        return 1

    # Load model
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        tokenizer = open_clip.get_tokenizer(args.model)
        model = model.to(device).eval()
    except Exception as e:
        print("FAILED: could not load OpenCLIP model:\n", e)
        return 1

    texts = [
        "a photo taken from a dashcam",
        "a photo of a road",
        "a photo of a car",
        "a photo of a pedestrian",
        "a photo of an intersection",
    ]

    # Precompute text features once
    try:
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            tf = model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)
    except Exception as e:
        print("FAILED: text encoding error:\n", e)
        return 1

    # Determine image sources
    img_paths = []
    if args.image:
        img_paths = [Path(args.image)]
    elif args.dir:
        img_dir = Path(args.dir)
        if not img_dir.exists():
            print(f"FAILED: --dir does not exist: {img_dir.resolve()}")
            return 1
        img_paths = pick_images_from_dir(img_dir, args.n)
    else:
        # URL fallback only if no local option provided
        if not args.url:
            args.url = "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=640&q=80"

        try:
            import requests
        except Exception as e:
            print("FAILED: requests not installed and no --dir/--image provided.")
            print("Install requests or use a local directory/image.")
            print(e)
            return 1

        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(args.url, headers=headers, timeout=25)
            ct = r.headers.get("Content-Type", "")
            print("HTTP:", r.status_code, "Content-Type:", ct, "bytes:", len(r.content))
            if r.status_code != 200 or "image" not in ct:
                print("Non-image response head:", r.content[:200])
                print("FAILED: URL did not return an image.")
                return 1
            # Use a sentinel path label for printing
            img_paths = ["<URL_IMAGE>"]
            img_from_url = Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            print("FAILED: could not download/parse URL image:\n", e)
            return 1

    # Run inference
    for idx, p in enumerate(img_paths):
        try:
            if p == "<URL_IMAGE>":
                img = img_from_url
                label = args.url
            else:
                p = Path(p)
                label = str(p)
                img = Image.open(p).convert("RGB")

            image = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                imf = model.encode_image(image)
                imf = imf / imf.norm(dim=-1, keepdim=True)
                sims = (imf @ tf.T).squeeze(0).detach().cpu().tolist()

            pairs = sorted(zip(texts, sims), key=lambda x: x[1], reverse=True)
            top_text, top_score = pairs[0]

            print("\n---")
            print(f"[{idx+1}/{len(img_paths)}] image: {label}")
            for t, s in pairs:
                print(f"{s:+.4f}  {t}")
            print(f"TOP: {top_score:+.4f}  {top_text}")

        except Exception as e:
            print(f"FAILED: inference error on {p}:\n", e)
            return 1

    print("\nSUCCESS: OpenCLIP works end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
