#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", type=str, default="results/corruption_eval.csv")
    ap.add_argument("--merged_csv", type=str, default="results/baseline_analysis/merged_lambda_labels.csv")
    ap.add_argument("--out_dir", type=str, default="results/corruption_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(args.eval_csv)
    base_df = pd.read_csv(args.merged_csv).set_index("filename")

    # Map baseline lambda + hazard_level into eval_df
    eval_df["lambda_base"] = eval_df["filename"].map(base_df["lambda"])
    eval_df["hazard_level"] = eval_df["filename"].map(base_df["hazard_level"])
    eval_df["hazard_present"] = eval_df["hazard_present"].astype(int)

    # Core task-aligned stability metrics
    eval_df["lambda_drop"] = eval_df["lambda_corrupt"] - eval_df["lambda_base"]
    eval_df["margin_drift"] = eval_df["lambda_drop"].abs()

    # Classification outcomes at a fixed threshold (choose on clean baseline)
    # Use median of baseline lambdas on labeled subset as a simple threshold; replace later with ROC-optimal threshold.
    tau = float(base_df["lambda"].median())
    eval_df["pred_hazard"] = (eval_df["lambda_corrupt"] > tau).astype(int)
    eval_df["error"] = (eval_df["pred_hazard"] != eval_df["hazard_present"]).astype(int)

    # Save augmented CSV
    augmented_path = out_dir / "corruption_eval_augmented.csv"
    eval_df.to_csv(augmented_path, index=False)
    print("Saved:", augmented_path.resolve())

    corruptions = sorted(eval_df["corruption"].unique())
    severities = sorted(eval_df["severity"].unique())

    # Summary table
    rows = []
    for corr in corruptions:
        for sev in severities:
            sub = eval_df[(eval_df["corruption"] == corr) & (eval_df["severity"] == sev)].copy()
            if len(sub) == 0:
                continue

            y = sub["hazard_present"].values
            s = sub["lambda_corrupt"].values

            # AUC/AP (if both classes exist)
            if sub["hazard_present"].nunique() >= 2:
                auc = roc_auc_score(y, s)
                apv = average_precision_score(y, s)
            else:
                auc = np.nan
                apv = np.nan

            # Mean drift / margin drift / error
            mean_delta = sub["delta_l2"].mean()
            mean_dcos = sub["dcos"].mean() if "dcos" in sub.columns else np.nan
            mean_margin = sub["margin_drift"].mean()
            err_rate = sub["error"].mean()

            # Correlations: which metric predicts lambda_drop / errors?
            # (Spearman is robust to nonlinearity)
            rho_delta_drop = spearmanr(sub["delta_l2"], sub["lambda_drop"]).correlation
            rho_margin_drop = spearmanr(sub["margin_drift"], sub["lambda_drop"]).correlation  # should be positive by construction
            rho_delta_err = spearmanr(sub["delta_l2"], sub["error"]).correlation
            rho_margin_err = spearmanr(sub["margin_drift"], sub["error"]).correlation

            rows.append({
                "corruption": corr,
                "severity": sev,
                "auc": auc,
                "ap": apv,
                "mean_delta_l2": mean_delta,
                "mean_dcos": mean_dcos,
                "mean_margin_drift": mean_margin,
                "error_rate_at_tau": err_rate,
                "spearman(delta_l2, lambda_drop)": rho_delta_drop,
                "spearman(margin_drift, lambda_drop)": rho_margin_drop,
                "spearman(delta_l2, error)": rho_delta_err,
                "spearman(margin_drift, error)": rho_margin_err,
            })

    summary = pd.DataFrame(rows).sort_values(["corruption", "severity"])
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path.resolve())

    # Print a compact view
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 140):
        print("\n=== SUMMARY (key columns) ===")
        print(summary[[
            "corruption","severity","auc","ap",
            "mean_delta_l2","mean_margin_drift","error_rate_at_tau",
            "spearman(delta_l2, error)","spearman(margin_drift, error)"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
