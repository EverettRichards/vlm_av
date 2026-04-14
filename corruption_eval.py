#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import open_clip
from PIL import Image

#import albumentations as A
import cv2
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import numpy as np
import cv2

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
    # simple horizontal motion blur kernel (deterministic)
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return clip_uint8(cv2.filter2D(arr_rgb, -1, kernel))

def apply_fog(arr_rgb: np.ndarray, strength: float) -> np.ndarray:
    # strength in [0,1], blend with bright veil + slight contrast reduction
    x = arr_rgb.astype(np.float32)
    veil = 255.0 * np.ones_like(x)
    out = (1.0 - strength) * x + strength * veil
    # contrast reduction
    out = 128.0 + (out - 128.0) * (1.0 - 0.4 * strength)
    return clip_uint8(out)

def apply_low_light(arr_rgb: np.ndarray, gamma: float, noise_std: float) -> np.ndarray:
    # gamma > 1 darkens, noise_std in pixel units
    x = arr_rgb.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    x = x * 255.0
    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = x + noise
    return clip_uint8(x)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_corruption(name: str, severity: int):
    """
    Returns a callable f(arr_rgb_uint8) -> arr_rgb_uint8
    severity: 1..5
    """
    assert 1 <= severity <= 5

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
        k = [3, 5, 7, 9, 11][severity - 1]
        if k % 2 == 0:
            k += 1
        return lambda arr: cv2.GaussianBlur(arr, (k, k), 0)

    if name == "jpeg":
        # lower quality = stronger artifacts
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


def main():
    np.random.seed(0)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True,
                    help="Folder with images (your subset images dir).")
    ap.add_argument("--merged_csv", type=str, default="results/baseline_analysis/merged_lambda_labels.csv",
                    help="Output of analyze_baseline.py (merged labels).")
    ap.add_argument("--out_csv", type=str, default="results/corruption_eval.csv")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir.resolve()}")

    df = pd.read_csv(args.merged_csv)
    # Expect: filename, lambda, hazard_level, hazard_present
    needed = {"filename", "lambda", "hazard_level", "hazard_present"}
    if not needed.issubset(set(df.columns)):
        raise SystemExit(f"merged_csv must contain columns: {sorted(list(needed))}")

    device = "cuda" if (not args.cpu) and torch.cuda.is_available() else "cpu"
    print("device:", device)

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # Prompts (must match what you used for baseline lambda)
    hazard_prompts = [
        "a pedestrian crossing the road",
        "a cyclist in the lane",
        "a stopped vehicle blocking traffic",
        "road debris on the street",
        "a construction zone on the road",
        "an emergency vehicle with flashing lights",
    ]
    safe_prompts = [
        "a clear road with no obstacles",
        "a normal highway scene",
        "an empty intersection",
        "a clear lane ahead",
        "a normal driving scene",
    ]
    H = len(hazard_prompts)
    all_prompts = hazard_prompts + safe_prompts

    with torch.no_grad():
        text = tokenizer(all_prompts).to(device)
        tf = model.encode_text(text)
        tf = tf / tf.norm(dim=-1, keepdim=True)

    # Helper to encode a list of PIL images
    def encode_images(pil_list):
        batch = torch.stack([preprocess(im) for im in pil_list], dim=0).to(device)
        with torch.no_grad():
            imf = model.encode_image(batch)
            imf = imf / imf.norm(dim=-1, keepdim=True)
        return imf

    corruptions = ["fog", "low_light", "motion_blur", "gaussian_blur", "jpeg"]
    severities = [1, 2, 3, 4, 5]

    rows = []

    # We only evaluate the labeled subset for now (faster and directly tied to reliability)
    filenames = df["filename"].tolist()
    y = df["hazard_present"].astype(int).to_numpy()

    # Process in batches for speed
    for corr in corruptions:
        for sev in severities:
            T = get_corruption(corr, sev)
            print(f"\nRunning corruption={corr} severity={sev}")

            lam_list = []
            delta_l2_list = []
            dcos_list = []

            for start in tqdm(range(0, len(filenames), args.batch)):
                batch_files = filenames[start:start + args.batch]
                clean_imgs = []
                corr_imgs = []

                for fn in batch_files:
                    p = images_dir / fn
                    img = Image.open(p).convert("RGB")
                    clean_imgs.append(img)

                    arr = to_uint8_rgb(img)
                    carr = T(arr)
                    corr_imgs.append(from_uint8_rgb(carr))

                # Encode
                imf_clean = encode_images(clean_imgs)
                imf_corr = encode_images(corr_imgs)

                # Drift
                # Delta L2 between normalized embeddings
                delta = torch.norm(imf_clean - imf_corr, dim=-1).detach().cpu().numpy()
                # Cosine drop: 1 - cos
                cos = torch.sum(imf_clean * imf_corr, dim=-1).detach().cpu().numpy()
                dcos = 1.0 - cos

                # Lambda on corrupted image
                sims = (imf_corr @ tf.T).detach().cpu().numpy()  # [B, P]
                s_h = sims[:, :H].max(axis=1)
                s_n = sims[:, H:].max(axis=1)
                lam = s_h - s_n

                lam_list.append(lam)
                delta_l2_list.append(delta)
                dcos_list.append(dcos)

            lam = np.concatenate(lam_list, axis=0)
            delta = np.concatenate(delta_l2_list, axis=0)
            dcos = np.concatenate(dcos_list, axis=0)

            # Save per-image records
            for fn, y_i, lam_i, delta_i, dcos_i in zip(filenames, y, lam, delta, dcos):
                rows.append({
                    "filename": fn,
                    "hazard_present": int(y_i),
                    "corruption": corr,
                    "severity": sev,
                    "lambda_corrupt": float(lam_i),
                    "delta_l2": float(delta_i),
                    "dcos": float(dcos_i),
                })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv.resolve())

    # Quick summary analysis: AUC vs severity and drift correlation
    print("\nSummary (per corruption, severity):")
    for corr in corruptions:
        for sev in severities:
            sub = out_df[(out_df["corruption"] == corr) & (out_df["severity"] == sev)].copy()
            if len(sub["hazard_present"].unique()) < 2:
                continue
            auc = roc_auc_score(sub["hazard_present"].values, sub["lambda_corrupt"].values)

            # Drift vs lambda change proxy: use baseline lambda from merged_csv
            base = df.set_index("filename")["lambda"]
            sub["lambda_base"] = sub["filename"].map(base)
            sub["lambda_drop"] = sub["lambda_corrupt"] - sub["lambda_base"]

            rho, _ = spearmanr(sub["delta_l2"].values, sub["lambda_drop"].values)

            print(f"{corr:12s} sev={sev}  AUC={auc:.3f}  Spearman(delta_l2, lambda_drop)={rho:.3f}")


if __name__ == "__main__":
    main()
