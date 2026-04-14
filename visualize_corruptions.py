#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import cv2

"""
python visualize_corruptions.py \
  --images_dir data/subsets/bdd100k_val_2000/images \
  --corruption gaussian_blur \
  --num_samples 4 \
  --severities 1,2,3,4,5 \
  --out_path results/visualized_noise/visual_gaussian_blur.png
"""

"""
for c in jpeg motion_blur fog gaussian_blur low_light; do
  python visualize_corruptions.py \
    --images_dir data/subsets/bdd100k_val_2000/images \
    --corruption "$c" \
    --num_samples 4 \
    --out_path "results/visualized_noise/visual_${c}.png"
done
"""


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


VALID_CORRUPTIONS = ["jpeg", "motion_blur", "fog", "gaussian_blur", "low_light"]


def clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_jpeg(arr_rgb: np.ndarray, quality: int) -> np.ndarray:
    # OpenCV expects BGR for imencode/decode
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", bgr, encode_params)
    if not ok:
        return arr_rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return rgb


def apply_motion_blur(arr_rgb: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return clip_uint8(cv2.filter2D(arr_rgb, -1, kernel))


def apply_fog(arr_rgb: np.ndarray, strength: float) -> np.ndarray:
    x = arr_rgb.astype(np.float32)
    veil = 255.0 * np.ones_like(x)
    out = (1.0 - strength) * x + strength * veil
    out = 128.0 + (out - 128.0) * (1.0 - 0.4 * strength)
    return clip_uint8(out)


def apply_low_light(arr_rgb: np.ndarray, gamma: float, noise_std: float) -> np.ndarray:
    x = arr_rgb.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    x = x * 255.0
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = x + noise
    return clip_uint8(x)


def get_corruption(name: str, severity: int):
    if not (1 <= severity <= 5):
        raise ValueError(f"Severity must be in [1, 5], got {severity}")

    if name == "fog":
        strength = [0.10, 0.18, 0.26, 0.34, 0.42][severity - 1]
        return lambda arr: apply_fog(arr, strength)

    if name == "low_light":
        gamma = [1.2, 1.4, 1.6, 1.8, 2.0][severity - 1]
        noise_std = [2, 4, 6, 8, 10][severity - 1]
        return lambda arr: apply_low_light(arr, gamma=gamma, noise_std=noise_std)

    if name == "motion_blur":
        k = [5, 9, 13, 17, 21][severity - 1]
        return lambda arr: apply_motion_blur(arr, k)

    if name == "gaussian_blur":
        #k = [3, 5, 7, 9, 11][severity - 1]
        k = [5,11,17,23,29][severity - 1]
        if k % 2 == 0:
            k += 1
        return lambda arr: cv2.GaussianBlur(arr, (k, k), 0)

    if name == "jpeg":
        q = [60, 45, 30, 20, 10][severity - 1]
        return lambda arr: apply_jpeg(arr, quality=q)

    raise ValueError(f"Unknown corruption: {name}")


def to_uint8_rgb(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def from_uint8_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGB")


def parse_severities(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("No severities were provided.")
    bad = [v for v in vals if v < 1 or v > 5]
    if bad:
        raise ValueError(f"Severities must be between 1 and 5, got: {bad}")
    return vals


def collect_images(images_dir: Path) -> list[Path]:
    paths = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(paths)


def fit_to_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    # Keep aspect ratio and center the image on a black canvas for consistent cell sizes.
    src = img.convert("RGB")
    w, h = src.size
    scale = min(cell_w / max(1, w), cell_h / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = src.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (cell_w, cell_h), color=(0, 0, 0))
    x = (cell_w - nw) // 2
    y = (cell_h - nh) // 2
    canvas.paste(resized, (x, y))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Visualize one corruption type across severity levels on random dataset samples."
        )
    )
    ap.add_argument("--images_dir", type=str, required=True, help="Root folder with dataset images.")
    ap.add_argument(
        "--corruption",
        type=str,
        required=True,
        choices=VALID_CORRUPTIONS,
        help="Corruption type to visualize.",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of random sample images (rows in output matrix).",
    )
    ap.add_argument(
        "--severities",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated severity levels in [1,5], e.g. 1,3,5.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for image sampling.")
    ap.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output image path. Default: results/visualized_noise/visual_<corruption>.png",
    )
    ap.add_argument(
        "--include_clean",
        action="store_true",
        help="Add a clean-reference column before corrupted columns.",
    )
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir.resolve()}")

    severities = parse_severities(args.severities)
    np.random.seed(args.seed)

    all_images = collect_images(images_dir)
    if not all_images:
        raise SystemExit(f"No images found in: {images_dir.resolve()}")

    n = min(args.num_samples, len(all_images))
    sampled_idx = np.random.choice(len(all_images), size=n, replace=False)
    sampled = [all_images[i] for i in sampled_idx]

    out_path = Path(args.out_path) if args.out_path else Path(
        f"results/visualized_noise/visual_{args.corruption}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    col_count = len(severities) + (1 if args.include_clean else 0)
    sample_img = Image.open(sampled[0]).convert("RGB")
    base_w, base_h = sample_img.size
    cell_w = min(420, max(140, base_w))
    cell_h = min(280, max(100, base_h))

    left_label_w = 220
    header_h = 42
    pad = 8
    canvas_w = left_label_w + pad + col_count * cell_w
    canvas_h = header_h + n * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    headers = []
    if args.include_clean:
        headers.append("clean")
    headers.extend([f"sev {s}" for s in severities])

    for c, htxt in enumerate(headers):
        x0 = left_label_w + pad + c * cell_w
        draw.text((x0 + 8, 12), htxt, fill=(0, 0, 0))

    for r, p in enumerate(sampled):
        img = Image.open(p).convert("RGB")
        arr = to_uint8_rgb(img)
        y0 = header_h + r * cell_h

        row_label = p.name
        draw.text((8, y0 + 8), row_label, fill=(0, 0, 0))

        c = 0
        if args.include_clean:
            tile = fit_to_cell(img, cell_w, cell_h)
            x0 = left_label_w + pad + c * cell_w
            canvas.paste(tile, (x0, y0))
            c += 1

        for sev in severities:
            T = get_corruption(args.corruption, sev)
            carr = T(arr)
            cimg = from_uint8_rgb(carr)
            tile = fit_to_cell(cimg, cell_w, cell_h)
            x0 = left_label_w + pad + c * cell_w
            canvas.paste(tile, (x0, y0))
            c += 1

    draw.text(
        (8, 10),
        f"Corruption: {args.corruption} | rows={n}",
        fill=(0, 0, 0),
    )
    canvas.save(out_path)

    print(f"Saved visualization to: {out_path.resolve()}")


if __name__ == "__main__":
    main()