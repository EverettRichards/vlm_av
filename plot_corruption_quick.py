#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, default="results/corruption_analysis/summary.csv")
    ap.add_argument("--out_dir", type=str, default="results/corruption_analysis/plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    corruptions = sorted(df["corruption"].unique())

    # AUC vs severity
    for corr in corruptions:
        sub = df[df["corruption"] == corr].sort_values("severity")
        plt.figure()
        plt.plot(sub["severity"], sub["auc"], marker="o")
        plt.xlabel("Severity")
        plt.ylabel("ROC AUC")
        plt.title(f"AUC vs Severity: {corr}")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"auc_vs_severity_{corr}.png", dpi=200)
        plt.close()

    # mean delta vs severity
    for corr in corruptions:
        sub = df[df["corruption"] == corr].sort_values("severity")
        plt.figure()
        plt.plot(sub["severity"], sub["mean_delta_l2"], marker="o")
        plt.xlabel("Severity")
        plt.ylabel("Mean Delta L2")
        plt.title(f"Embedding Drift vs Severity: {corr}")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"delta_vs_severity_{corr}.png", dpi=200)
        plt.close()

    # mean margin drift vs severity
    for corr in corruptions:
        sub = df[df["corruption"] == corr].sort_values("severity")
        plt.figure()
        plt.plot(sub["severity"], sub["mean_margin_drift"], marker="o")
        plt.xlabel("Severity")
        plt.ylabel("Mean |Lambda Drop| (Margin Drift)")
        plt.title(f"Margin Drift vs Severity: {corr}")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"margin_drift_vs_severity_{corr}.png", dpi=200)
        plt.close()

    print("Saved plots to:", out_dir.resolve())

if __name__ == "__main__":
    main()
