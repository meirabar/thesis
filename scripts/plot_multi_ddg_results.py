#!/usr/bin/env python3
"""
Generate figures for multi-DDG regression results.

Reads the JSON result files and produces:
  1. Coefficient bar plot: 20 AA coefficients, RMSF vs B-factor side by side
  2. Model comparison: R² bar chart across all models for both targets

Usage:
  python plot_multi_ddg_results.py \
      --results_dir /path/to/atlas_analysis/thermompnn \
      --output_dir /path/to/output
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def load_results(results_dir: str):
    """Load RMSF and B-factor multi-DDG results."""
    rmsf_path = Path(results_dir) / "multi_ddg_rmsf_results.json"
    bf_path = Path(results_dir) / "multi_ddg_bfactor_results.json"

    with open(rmsf_path) as f:
        rmsf = json.load(f)
    with open(bf_path) as f:
        bf = json.load(f)
    return rmsf, bf


def plot_coefficients(rmsf, bf, output_dir: str, scorer: str = "thermompnn"):
    """Bar plot of 20-AA Ridge coefficients for RMSF vs B-factor."""
    coefs_rmsf = rmsf["ridge_20ddg"]["feature_coefs_mean"]
    coefs_bf = bf["ridge_20ddg"]["feature_coefs_mean"]

    x = np.arange(len(AA_LIST))
    width = 0.38

    # Sort by RMSF coefficient magnitude for readability
    order = np.argsort(coefs_rmsf)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, [coefs_rmsf[i] for i in order], width,
           label="RMSF", color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, [coefs_bf[i] for i in order], width,
           label="B-factor", color="#DD8452", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([AA_LIST[i] for i in order], fontsize=11, fontweight="bold")
    ax.set_ylabel("Ridge coefficient (z-scored)", fontsize=12)
    ax.set_xlabel("Target amino acid", fontsize=12)
    ax.set_title("Per-amino-acid DDG coefficients for predicting dynamics\n"
                 "(Ridge regression, 20 DDG features, ThermoMPNN)",
                 fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = Path(output_dir) / f"{scorer}_multi_ddg_coefficients.pdf"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = Path(output_dir) / f"{scorer}_multi_ddg_coefficients.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)


def plot_model_comparison(rmsf, bf, output_dir: str, scorer: str = "thermompnn"):
    """Grouped bar chart comparing R² across models for both targets."""
    model_order = [
        "ols_mean_abs_ddg", "ols_sasa", "ols_plddt", "ols_mean_plddt",
        "ridge_nonlinear_only", "ridge_20ddg",
        "ridge_20ddg_nonlinear", "ridge_20ddg_plddt",
        "ridge_20ddg_nonlinear_plddt",
    ]
    labels = [
        "mean|DDG|", "SASA", "pLDDT", "mean|DDG|\n+pLDDT",
        "4 NL", "20 DDG",
        "20 DDG\n+4 NL", "20 DDG\n+pLDDT",
        "20 DDG\n+4 NL+pLDDT",
    ]

    r2_rmsf = [rmsf[m]["cv_r2_mean"] for m in model_order]
    r2_bf = [bf[m]["cv_r2_mean"] for m in model_order]
    std_rmsf = [rmsf[m]["cv_r2_std"] for m in model_order]
    std_bf = [bf[m]["cv_r2_std"] for m in model_order]

    x = np.arange(len(model_order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, r2_rmsf, width, yerr=std_rmsf,
           label="RMSF", color="#4C72B0", edgecolor="white",
           linewidth=0.5, capsize=3)
    ax.bar(x + width/2, r2_bf, width, yerr=std_bf,
           label="B-factor", color="#DD8452", edgecolor="white",
           linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Cross-validated $R^2$", fontsize=12)
    ax.set_title("Multi-DDG regression: model comparison\n"
                 "(ThermoMPNN, 5-fold protein-level CV, 1938 ATLAS proteins)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.30)

    # Add delta-R² annotation
    best_rmsf = r2_rmsf[-1]
    plddt_rmsf = r2_rmsf[2]
    best_bf = r2_bf[-1]
    plddt_bf = r2_bf[2]
    ax.annotate(f"$\\Delta R^2 = +{best_rmsf - plddt_rmsf:.3f}$",
                xy=(len(model_order)-1 - width/2, best_rmsf),
                xytext=(len(model_order)-2.5, best_rmsf + 0.025),
                fontsize=9, color="#4C72B0",
                arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2))
    ax.annotate(f"$\\Delta R^2 = +{best_bf - plddt_bf:.3f}$",
                xy=(len(model_order)-1 + width/2, best_bf),
                xytext=(len(model_order)-1.5, best_bf + 0.035),
                fontsize=9, color="#DD8452",
                arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))

    fig.tight_layout()
    out_path = Path(output_dir) / f"{scorer}_multi_ddg_model_comparison.pdf"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = Path(output_dir) / f"{scorer}_multi_ddg_model_comparison.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with multi_ddg_*_results.json files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output figures")
    parser.add_argument("--scorer", type=str, default="thermompnn",
                        help="Scorer name for figure filename prefix")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rmsf, bf = load_results(args.results_dir)
    plot_coefficients(rmsf, bf, args.output_dir, args.scorer)
    plot_model_comparison(rmsf, bf, args.output_dir, args.scorer)
    print("\nDone.")


if __name__ == "__main__":
    main()
