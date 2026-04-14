#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_csv", default="results/baseline_lambda.csv")
    parser.add_argument("--images_dir", default="data/subsets/bdd100k_val_2000/images")
    parser.add_argument("--out_file", default="results/demo_lambda_examples.png")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.lambda_csv)

    # Sort by lambda
    df_sorted = df.sort_values("lambda")

    safe = df_sorted.head(args.k)
    hazard = df_sorted.tail(args.k)

    safe = safe.assign(type="safe")
    hazard = hazard.assign(type="hazard")

    sample = pd.concat([safe, hazard])

    fig, axes = plt.subplots(2, args.k, figsize=(3*args.k, 6))

    for idx, (_, row) in enumerate(sample.iterrows()):

        img_path = Path(args.images_dir) / row["filename"]
        img = Image.open(img_path).convert("RGB")

        if idx < args.k:
            ax = axes[0, idx]
        else:
            ax = axes[1, idx - args.k]

        ax.imshow(img)
        ax.axis("off")

        label = f"{row['type']}  λ={row['lambda']:.3f}"
        ax.set_title(label)

    axes[0,0].set_ylabel("Lowest λ (safe scenes)", fontsize=12)
    axes[1,0].set_ylabel("Highest λ (hazard scenes)", fontsize=12)

    plt.tight_layout()

    Path(args.out_file).parent.mkdir(exist_ok=True)
    plt.savefig(args.out_file, dpi=200)

    print("Saved demo figure to:", args.out_file)

if __name__ == "__main__":
    main()
