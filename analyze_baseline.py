#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_csv", type=str, default="results/baseline_lambda.csv")
    ap.add_argument("--labels_csv", type=str, default="data/labels/hazard_labels.csv")
    ap.add_argument("--out_dir", type=str, default="results/baseline_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_df = pd.read_csv(args.lambda_csv)
    lab_df = pd.read_csv(args.labels_csv)

    # Basic sanity
    assert "filename" in lam_df.columns and "lambda" in lam_df.columns
    assert "filename" in lab_df.columns and "hazard_level" in lab_df.columns and "hazard_present" in lab_df.columns

    # Merge (inner join: only labeled examples)
    df = lam_df.merge(lab_df, on="filename", how="inner")
    df = df.dropna(subset=["lambda", "hazard_level", "hazard_present"]).copy()

    # Save merged CSV for future use
    merged_path = out_dir / "merged_lambda_labels.csv"
    df.to_csv(merged_path, index=False)

    print(f"Merged rows: {len(df)}")
    print(f"Saved merged CSV: {merged_path}")

    y = df["hazard_present"].astype(int).to_numpy()
    s = df["lambda"].astype(float).to_numpy()
    r = df["hazard_level"].astype(int).to_numpy()

    # Metrics
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both hazard_present classes in labels to compute AUC.")

    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)

    rho, rho_p = spearmanr(s, r)

    print("\nBaseline metrics on labeled subset")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR  AUC: {pr_auc:.4f}")
    print(f"Spearman(lambda, hazard_level): rho={rho:.4f}, p={rho_p:.3g}")

    # Summary by hazard_level
    summary = df.groupby("hazard_level")["lambda"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    summary_path = out_dir / "lambda_by_level.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(summary.to_string(index=False))

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y, s)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # PR curve plot
    prec, rec, _ = precision_recall_curve(y, s)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP={pr_auc:.3f})")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    pr_path = out_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # Boxplot of lambda by hazard level
    plt.figure()
    levels = sorted(df["hazard_level"].unique())
    data = [df[df["hazard_level"] == lv]["lambda"].values for lv in levels]
    plt.boxplot(data, labels=[str(lv) for lv in levels], showfliers=False)
    plt.xlabel("Hazard level")
    plt.ylabel("Lambda")
    plt.title("Lambda by Hazard Level")
    plt.grid(True, axis="y", linewidth=0.3)
    plt.tight_layout()
    box_path = out_dir / "lambda_by_level_boxplot.png"
    plt.savefig(box_path, dpi=200)
    plt.close()

    print("\nSaved plots:")
    print(f"- {roc_path}")
    print(f"- {pr_path}")
    print(f"- {box_path}")


if __name__ == "__main__":
    main()
