#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", type=str, default="results/corruption_eval.csv")
    ap.add_argument("--merged_csv", type=str, default="results/baseline_analysis/merged_lambda_labels.csv")
    ap.add_argument("--out_dir", type=str, default="results/corruption_plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(args.eval_csv)
    base_df = pd.read_csv(args.merged_csv).set_index("filename")

    # Add baseline lambda and lambda_drop
    eval_df["lambda_base"] = eval_df["filename"].map(base_df["lambda"])
    eval_df["lambda_drop"] = eval_df["lambda_corrupt"] - eval_df["lambda_base"]

    corruptions = sorted(eval_df["corruption"].unique())
    severities = sorted(eval_df["severity"].unique())

    # Plot AUC vs severity per corruption
    for corr in corruptions:
        aucs = []
        for sev in severities:
            sub = eval_df[(eval_df["corruption"] == corr) & (eval_df["severity"] == sev)]
            if sub["hazard_present"].nunique() < 2:
                aucs.append(np.nan)
            else:
                aucs.append(roc_auc_score(sub["hazard_present"], sub["lambda_corrupt"]))
        plt.figure()
        plt.plot(severities, aucs, marker="o")
        plt.xlabel("Severity")
        plt.ylabel("ROC AUC (hazard_present)")
        plt.title(f"AUC vs Severity: {corr}")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        p = out_dir / f"auc_vs_severity_{corr}.png"
        plt.savefig(p, dpi=200)
        plt.close()

    # Plot mean drift vs severity per corruption
    for corr in corruptions:
        means = []
        stds = []
        for sev in severities:
            sub = eval_df[(eval_df["corruption"] == corr) & (eval_df["severity"] == sev)]
            means.append(sub["delta_l2"].mean())
            stds.append(sub["delta_l2"].std())
        plt.figure()
        plt.errorbar(severities, means, yerr=stds, marker="o", capsize=3)
        plt.xlabel("Severity")
        plt.ylabel("Delta L2 (embedding drift)")
        plt.title(f"Embedding Drift vs Severity: {corr}")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        p = out_dir / f"drift_vs_severity_{corr}.png"
        plt.savefig(p, dpi=200)
        plt.close()

    # Plot relationship: drift vs lambda_drop (pooled), per corruption
    for corr in corruptions:
        sub = eval_df[eval_df["corruption"] == corr]
        plt.figure()
        plt.scatter(sub["delta_l2"], sub["lambda_drop"], s=8, alpha=0.4)
        plt.xlabel("Delta L2 (embedding drift)")
        plt.ylabel("Lambda drop (lambda_corrupt - lambda_base)")
        plt.title(f"Drift vs Lambda Drop: {corr}")
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        p = out_dir / f"drift_vs_lambda_drop_{corr}.png"
        plt.savefig(p, dpi=200)
        plt.close()

    print("Saved plots to:", out_dir.resolve())


if __name__ == "__main__":
    main()