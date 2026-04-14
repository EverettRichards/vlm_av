#!/usr/bin/env python3
"""
Ordinal labeling tool for driving images.

Labels:
  hazard_level in {0,1,2,3}
  hazard_present = 1 if hazard_level >= 2 else 0

Keys:
  0/1/2/3 : set hazard_level
  s       : skip image (no label written)
  b       : go back one image (removes last written label if it matches previous file)
  q or ESC: quit (safe, progress saved)

Usage:
  python label_tool.py --images_dir data/subsets/bdd100k_val_2000/images --out_csv data/labels/hazard_labels.csv
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

HELP_TEXT = [
    "Keys: 0/1/2/3=label   s=skip   b=back   q/ESC=quit",
    "hazard_level: 0 clear | 1 normal traffic | 2 elevated attention | 3 immediate hazard",
    "hazard_present = 1 if level >= 2 else 0",
]


def load_existing_labels(csv_path: Path) -> Dict[str, Tuple[int, int, float]]:
    """
    Returns map filename -> (hazard_level, hazard_present, timestamp)
    """
    labels = {}
    if not csv_path.exists():
        return labels
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["filename"]
            lvl = int(row["hazard_level"])
            pres = int(row["hazard_present"])
            ts = float(row.get("timestamp", "0") or 0)
            labels[fn] = (lvl, pres, ts)
    return labels


def write_all_labels(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["filename", "hazard_level", "hazard_present", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Directory containing images to label.")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path (will resume if exists).")
    ap.add_argument("--window", type=str, default="Hazard Labeling", help="OpenCV window title.")
    ap.add_argument("--max_width", type=int, default=1280, help="Resize for display if larger than this width.")
    ap.add_argument("--max_height", type=int, default=720, help="Resize for display if larger than this height.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_csv = Path(args.out_csv)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir.resolve()}")

    # Collect image list (deterministic)
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name)

    if not files:
        raise RuntimeError(f"No images found in {images_dir.resolve()}")

    # Load existing labels
    existing = load_existing_labels(out_csv)
    print(f"Found {len(files)} images.")
    print(f"Existing labels: {len(existing)} (resume enabled)")
    print("Output CSV:", out_csv.resolve())

    # Build ordered list of filenames to label (skip already labeled)
    remaining = [p for p in files if p.name not in existing]
    print(f"Remaining unlabeled: {len(remaining)}")

    # We'll maintain an ordered "rows" list for rewriting when using back.
    # Start from existing labels in filename order.
    rows = []
    for p in files:
        if p.name in existing:
            lvl, pres, ts = existing[p.name]
            rows.append({
                "filename": p.name,
                "hazard_level": str(lvl),
                "hazard_present": str(pres),
                "timestamp": str(ts),
            })

    # Helper: map filename -> index in rows (only for labeled)
    row_index = {r["filename"]: i for i, r in enumerate(rows)}

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(files):
        p = files[i]
        fn = p.name

        # If already labeled, auto-advance (but still allow viewing by stepping back)
        already = fn in existing

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: could not read {p}. Skipping.")
            i += 1
            continue

        # Resize for display (keep aspect)
        h, w = img.shape[:2]
        scale = min(args.max_width / w, args.max_height / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Overlay text
        overlay = img.copy()
        y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        status = f"[{i+1}/{len(files)}] {fn}"
        if already:
            lvl, pres, _ = existing[fn]
            status += f"   (labeled: level={lvl}, present={pres})"
        cv2.putText(overlay, status, (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 28
        for line in HELP_TEXT:
            cv2.putText(overlay, line, (10, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

        cv2.imshow(args.window, overlay)
        key = cv2.waitKey(0) & 0xFF

        # Quit
        if key in (27, ord("q")):  # ESC or q
            print("Quitting. Progress saved.")
            break

        # Skip (does not write label)
        if key == ord("s"):
            i += 1
            continue

        # Back (remove last label if possible, then step back one image)
        if key == ord("b"):
            # Move back one image index
            if i == 0:
                continue
            i -= 1
            prev_fn = files[i].name

            # If previous has a label in rows, remove it (only if it exists in our output file)
            if prev_fn in row_index:
                idx = row_index[prev_fn]
                removed = rows.pop(idx)
                print("Removed label:", removed)

                # Rebuild indices and existing map from rows
                existing = {r["filename"]: (int(r["hazard_level"]), int(r["hazard_present"]), float(r["timestamp"])) for r in rows}
                row_index = {r["filename"]: j for j, r in enumerate(rows)}

                # Rewrite the whole CSV (safe, still fast for a few hundred labels)
                write_all_labels(out_csv, rows)
                print("Rewrote CSV after back().")
            continue

        # Label keys 0-3
        if key in (ord("0"), ord("1"), ord("2"), ord("3")):
            lvl = int(chr(key))
            pres = 1 if lvl >= 2 else 0
            ts = time.time()

            # Update if already labeled; else append
            if fn in row_index:
                idx = row_index[fn]
                rows[idx] = {"filename": fn, "hazard_level": str(lvl), "hazard_present": str(pres), "timestamp": str(ts)}
            else:
                rows.append({"filename": fn, "hazard_level": str(lvl), "hazard_present": str(pres), "timestamp": str(ts)})

            # Update maps
            existing[fn] = (lvl, pres, ts)
            row_index = {r["filename"]: j for j, r in enumerate(rows)}

            # Append-only write is simpler, but back() needs consistent file state.
            # We'll just rewrite the CSV each time for correctness. It's still fast.
            write_all_labels(out_csv, rows)

            print(f"Labeled {fn}: level={lvl}, present={pres}")
            i += 1
            continue

        # Unrecognized key
        print("Unrecognized key. Use 0/1/2/3, s, b, q.")
        continue

    cv2.destroyAllWindows()
    print("Done.")
    print("Labels saved to:", out_csv.resolve())


if __name__ == "__main__":
    main()
