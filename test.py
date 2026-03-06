#!/usr/bin/env python3
"""
OpenCLIP smoke test.

Usage:
  python test.py
  python test.py --image /path/to/image.jpg
  python test.py --cpu   (force CPU)

Exit code:
  0 = success
  1 = failure
"""

import argparse
import sys
from io import BytesIO

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to a local image (jpg/png/webp).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name (default: ViT-B-32).")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="Pretrained weights tag.")
    args = parser.parse_args()

    # Import torch with a clear error message
    try:
        import torch
    except Exception as e:
        print("FAILED: could not import torch.")
        print(e)
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

    # Import open_clip with a clear error message
    try:
        import open_clip
    except Exception as e:
        print("FAILED: could not import open_clip.")
        print(e)
        return 1

    # PIL
    try:
        from PIL import Image
    except Exception as e:
        print("FAILED: could not import PIL.Image. Install pillow.")
        print(e)
        return 1

    # Load model
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        tokenizer = open_clip.get_tokenizer(args.model)
        model = model.to(device).eval()
    except Exception as e:
        print("FAILED: could not create OpenCLIP model/transforms.")
        print(e)
        return 1

    # Load an image (local preferred; URL fallback)
    img = None
    if args.image:
        try:
            img = Image.open(args.image).convert("RGB")
            print(f"loaded local image: {args.image}")
        except Exception as e:
            print("FAILED: could not open local image.")
            print(e)
            return 1
    else:
        # Robust URL fetch with header + content-type check
        try:
            import requests
        except Exception as e:
            print("FAILED: requests not installed. Either install requests or pass --image.")
            print(e)
            return 1

        url = "https://images.unsplash.com/photo-1502877338535-766e1452684a?auto=format&fit=crop&w=640&q=80"
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            r = requests.get(url, headers=headers, timeout=20)
            ct = r.headers.get("Content-Type", "")
            print("HTTP:", r.status_code, "Content-Type:", ct, "bytes:", len(r.content))
            if r.status_code != 200 or "image" not in ct:
                print("Non-image response head:", r.content[:200])
                print("FAILED: URL did not return an image. Pass --image to use a local file.")
                return 1
            img = Image.open(BytesIO(r.content)).convert("RGB")
            print("loaded URL image:", url)
        except Exception as e:
            print("FAILED: could not download/parse URL image. Pass --image to use a local file.")
            print(e)
            return 1

    # Inference
    texts = [
        "a photo of a person",
        "a photo of a car",
        "a photo of a road",
        "a photo taken from a dashcam",
    ]

    try:
        import torch
        image = preprocess(img).unsqueeze(0).to(device)
        text = tokenizer(texts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0).detach().cpu().tolist()

        pairs = list(zip(texts, sims))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        print("\nsimilarities:")
        for t, s in pairs_sorted:
            print(f"  {s:+.4f}  {t}")

        print("\ntop match:", pairs_sorted[0][1], "->", pairs_sorted[0][0])
    except Exception as e:
        print("FAILED: inference error.")
        print(e)
        return 1

    print("\nSUCCESS: OpenCLIP inference works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
