#!/usr/bin/env python3
"""
Create a deterministic subset of BDD100K images from your local folder layout.

Your tree (confirmed):
  ./data/bdd100k/images/100k/{train,val,test}
  ./data/bdd100k/images/10k/{train,val,test}

Output:
  ./data/subsets/<name>/images/*.jpg
  ./data/subsets/<name>/manifest.csv

Examples:
  python make_subset.py --n 2000 --split val --size 100k --name bdd100k_val_2000
  python make_subset.py --n 1500 --split train --size 10k --name bdd10k_train_1500
  python make_subset.py --n 2000 --split val --size 100k --name bdd100k_val_2000 --method symlink
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import List


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(src_dir: Path) -> List[Path]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir.resolve()}")
    files = []
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()  # stable order before sampling
    if not files:
        raise RuntimeError(f"No images found in {src_dir.resolve()} (expected jpg/png/webp).")
    return files


def safe_link_or_copy(method: str, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if method == "copy":
        shutil.copy2(src, dst)
    elif method == "symlink":
        # Use relative symlinks so the subset folder remains movable as a unit
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    else:
        raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["10k", "100k"], default="100k",
                        help="Which BDD100K image folder to sample from (default: 100k).")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val",
                        help="Which split to sample from (default: val).")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of images to sample (default: 2000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic sampling (default: 42).")
    parser.add_argument("--name", type=str, default=None,
                        help="Subset name (default: auto-generated).")
    parser.add_argument("--method", choices=["copy", "symlink"], default="copy",
                        help="How to materialize subset images (default: copy). Symlink saves disk.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing subset directory if it exists.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "data" / "bdd100k" / "images" / args.size / args.split

    subset_name = args.name or f"bdd{args.size}_{args.split}_{args.n}_seed{args.seed}"
    out_dir = project_root / "data" / "subsets" / subset_name
    out_img_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.csv"

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    out_img_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(src_dir)

    if args.n > len(images):
        raise ValueError(f"Requested n={args.n} but only {len(images)} images exist in {src_dir}.")

    rng = random.Random(args.seed)
    # Deterministic sample without replacement
    subset = rng.sample(images, args.n)

    # Copy/symlink and write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset_name", subset_name])
        writer.writerow(["source_dir", str(src_dir.resolve())])
        writer.writerow(["size", args.size])
        writer.writerow(["split", args.split])
        writer.writerow(["n", args.n])
        writer.writerow(["seed", args.seed])
        writer.writerow([])  # blank line
        writer.writerow(["idx", "filename", "src_path", "dst_path"])

        for i, src in enumerate(subset):
            dst = out_img_dir / src.name
            safe_link_or_copy(args.method, src, dst)
            writer.writerow([i, src.name, str(src.resolve()), str(dst.resolve())])

    print(f"OK: created subset '{subset_name}'")
    print(f"Images:   {out_img_dir}  ({args.n} files)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
