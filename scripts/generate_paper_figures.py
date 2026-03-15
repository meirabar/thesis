#!/usr/bin/env python3
"""
Generate all paper figures from analysis outputs.

Reads unified_results.json for summary statistics and per-protein TSV
files for distribution/scatter figures.

Usage:
  python generate_paper_figures.py \
      --results unified_results.json \
      --output-dir figures/
  python generate_paper_figures.py --results ... --figure fig1
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

from paper_config import (
    DATASETS,
    FIG1_PANELS, FIG2_PANELS,
)

# Global font settings for publication readability
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# ============================================================================
# HELPERS
# ============================================================================

def _load_per_protein_tsv(dataset, scorer, target) -> pd.DataFrame:
    """Load per-protein correlations TSV for a run."""
    from paper_config import AnalysisRun
    run = AnalysisRun(dataset=dataset, scorer=scorer, target=target)
    path = Path(run.per_protein_tsv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _load_pooled_data(dataset, scorer) -> pd.DataFrame:
    """Load merged per-residue data for pooled scatter/density plots.

    This loads the per-protein data files and concatenates them.
    Each protein's data is z-scored within-protein.
    Returns DataFrame with columns: robustness_z, target_z, protein_id.
    """
    ds = DATASETS[dataset]
    rob_dir = Path(ds.robustness_dir) / scorer
    data_dir = Path(ds.data_dir) / "proteins"

    if not data_dir.exists():
        return pd.DataFrame()

    rows = []
    for protein_dir in sorted(data_dir.iterdir()):
        if not protein_dir.is_dir():
            continue
        pid = protein_dir.name

        # Load robustness
        rob_path = rob_dir / f"{pid}_robustness.tsv"
        if not rob_path.exists():
            continue
        rob_df = pd.read_csv(rob_path, sep="\t")
        if "std_ddg" not in rob_df.columns:
            continue

        # Load target
        rmsf_path = list(protein_dir.glob("*_RMSF.tsv"))
        bfac_path = list(protein_dir.glob("*_Bfactor.tsv"))

        targets = {}
        if rmsf_path:
            rmsf_df = pd.read_csv(rmsf_path[0], sep="\t")
            rmsf_cols = [c for c in rmsf_df.columns
                         if c.lower().startswith("rmsf") or "r1" in c.lower()]
            if rmsf_cols:
                targets["rmsf"] = rmsf_df[rmsf_cols].mean(axis=1).values
        if bfac_path:
            bfac_df = pd.read_csv(bfac_path[0], sep="\t")
            bfac_cols = [c for c in bfac_df.columns
                         if "bfactor" in c.lower() or "b_factor" in c.lower()]
            if bfac_cols:
                targets["bfactor"] = bfac_df[bfac_cols[0]].values

        rob = rob_df["std_ddg"].values
        for tname, tvals in targets.items():
            n = min(len(rob), len(tvals))
            if n < 10:
                continue
            r, t = rob[:n], tvals[:n]
            # Z-score within protein
            r_z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-10)
            t_z = (t - np.nanmean(t)) / (np.nanstd(t) + 1e-10)
            for i in range(n):
                if np.isfinite(r_z[i]) and np.isfinite(t_z[i]):
                    rows.append({
                        "robustness_z": r_z[i],
                        "target_z": t_z[i],
                        "robustness_raw": float(r[i]),
                        "target_raw": float(t[i]),
                        "target_type": tname,
                        "protein_id": pid,
                    })

    return pd.DataFrame(rows)


# ============================================================================
# FIGURE 1: Per-protein correlation distributions (4 panels)
# ============================================================================

def generate_fig1(results: dict, output_dir: Path):
    """Per-protein rho histograms + scatter plots, one row per dataset-target."""
    n_panels = len(FIG1_PANELS)
    fig, axes = plt.subplots(n_panels, 2, figsize=(12, 4 * n_panels))
    if n_panels == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (ds_name, target) in enumerate(FIG1_PANELS):
        ds = DATASETS[ds_name]
        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"
        panel_label = f"{ds.display_name} {target_label}"

        # Histogram panel
        ax_hist = axes[row_idx, 0]
        ax_scat = axes[row_idx, 1]

        # Load per-protein data for each scorer
        for scorer, color, label in [
            ("thermompnn", "tab:blue", "ThMPNN"),
            ("esm1v", "tab:green", "ESM-1v"),
        ]:
            if scorer not in ds.available_scorers:
                continue
            pp = _load_per_protein_tsv(ds_name, scorer, target)
            if pp.empty:
                continue

            # Determine the rho column name
            rho_candidates = [
                f"rho_std_ddg_{target}",
                "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
            ]
            rho_col = next((c for c in rho_candidates if c in pp.columns), None)
            if rho_col is None:
                for c in pp.columns:
                    if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                        rho_col = c
                        break
            if rho_col is None or rho_col not in pp.columns:
                continue

            vals = pp[rho_col].dropna()
            ax_hist.hist(vals, bins=30, alpha=0.5, color=color, label=label)

        # pLDDT (from ThermoMPNN run)
        if ds.has_plddt:
            pp_th = _load_per_protein_tsv(ds_name, "thermompnn", target)
            if not pp_th.empty:
                plddt_col = f"rho_plddt_{target}"
                if plddt_col not in pp_th.columns:
                    plddt_col = "rho_plddt_rmsf" if target == "rmsf" else "rho_plddt_bfactor"
                if plddt_col in pp_th.columns:
                    vals = pp_th[plddt_col].dropna()
                    ax_hist.hist(vals, bins=30, alpha=0.5, color="tab:orange", label="pLDDT")

                    # Scatter: robustness rho vs pLDDT rho
                    rob_candidates = [
                        f"rho_std_ddg_{target}",
                        "rho_robustness_bfactor_target" if target == "bfactor" else "rho_std_ddg_rmsf",
                    ]
                    rob_col = next((c for c in rob_candidates if c in pp_th.columns), None)
                    if rob_col is None:
                        for c in pp_th.columns:
                            if "rho" in c and ("std_ddg" in c or "robustness_bfactor" in c):
                                rob_col = c
                                break
                    if rob_col:
                        both = pp_th[[rob_col, plddt_col]].dropna()
                        ax_scat.scatter(both[rob_col], both[plddt_col],
                                        alpha=0.3, s=10, c="tab:blue")
                        lim = [-1, 1]
                        ax_scat.plot(lim, lim, "k--", alpha=0.5)
                        ax_scat.set_xlim(lim)
                        ax_scat.set_ylim(lim)
                        ax_scat.set_xlabel(r"$\rho$(rob, target)")
                        ax_scat.set_ylabel(r"$\rho$(pLDDT, target)")
        else:
            ax_scat.text(0.5, 0.5, "No pLDDT\navailable",
                         transform=ax_scat.transAxes,
                         ha="center", va="center", fontsize=13, color="gray")
            ax_scat.set_xlim([-1, 1])
            ax_scat.set_ylim([-1, 1])
            ax_scat.set_xlabel(r"$\rho$(rob, target)")
            ax_scat.set_ylabel(r"$\rho$(pLDDT, target)")

        ax_hist.set_title(panel_label, fontweight="bold")
        ax_hist.set_xlabel(r"Per-protein Spearman $\rho$")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlim([-1, 1])

        # Legend only on first panel, no frame
        if row_idx == 0:
            ax_hist.legend(frameon=False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig1_per_protein_correlations.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated fig1_per_protein_correlations")


# ============================================================================
# FIGURE 2: 2D density scatter with marginals (4 panels)
# ============================================================================

def generate_fig2(results: dict, output_dir: Path):
    """2D density scatter plots with marginal distributions, one per dataset-target."""
    n_panels = len(FIG2_PANELS)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols  # ceil division

    fig = plt.figure(figsize=(20, 9 * n_rows))

    for panel_idx, (ds_name, target) in enumerate(FIG2_PANELS):
        ds = DATASETS[ds_name]
        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"

        # Load pooled z-scored data
        pooled = _load_pooled_data(ds_name, "thermompnn")
        if pooled.empty:
            continue

        target_data = pooled[pooled["target_type"] == target]
        if target_data.empty:
            continue

        # Subsample for plotting
        n_max = 50000
        if len(target_data) > n_max:
            target_data = target_data.sample(n_max, random_state=42)

        x = target_data["robustness_z"].values
        y = target_data["target_z"].values

        # Clip y-axis to remove heavy tail
        y_clip = np.percentile(y, 99)
        y_floor = np.percentile(y, 1)

        # Remap panel positions for n_rows x n_cols layout
        row = panel_idx // n_cols
        col = panel_idx % n_cols

        gs_inner = GridSpec(
            4, 4, figure=fig,
            left=0.07 + 0.48 * col, right=0.07 + 0.48 * col + 0.40,
            bottom=0.07 + (1.0 / n_rows) * (n_rows - 1 - row),
            top=0.07 + (1.0 / n_rows) * (n_rows - 1 - row) + (0.85 / n_rows),
            hspace=0.05, wspace=0.05,
        )

        ax_main = fig.add_subplot(gs_inner[1:, :-1])
        ax_top = fig.add_subplot(gs_inner[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs_inner[1:, -1], sharey=ax_main)

        # 2D density (hexbin)
        mask = (y >= y_floor) & (y <= y_clip)
        hb = ax_main.hexbin(x[mask], y[mask], gridsize=40, cmap="Blues",
                             mincnt=1, linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax_right, pad=0.1, shrink=0.8)
        cb.set_label("Count", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax_main.set_xlabel(r"$\operatorname{std}(\Delta\Delta G)$ (z-scored)")
        ax_main.set_ylabel(f"{target_label} (z-scored)")
        ax_main.set_ylim(y_floor, y_clip)

        # Marginal distributions
        ax_top.hist(x, bins=50, color="tab:blue", alpha=0.7, density=True)
        ax_top.set_ylabel("Density")
        plt.setp(ax_top.get_xticklabels(), visible=False)

        ax_right.hist(y[mask], bins=50, orientation="horizontal",
                       color="tab:orange", alpha=0.7, density=True)
        ax_right.set_xlabel("Density")
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # Title
        rho = scipy_stats.spearmanr(x, y)[0]
        ax_top.set_title(f"{ds.display_name} {target_label} "
                         f"($\\rho = {rho:.3f}$, $n = {len(x):,}$)",
                         fontsize=14, fontweight="bold")

    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig2_density_scatter.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated fig2_density_scatter")


# ============================================================================
# FIGURE 3 (merged): Model comparison (left) + Ridge coefficients (right)
#   3 rows: RMSF, B-factor, NMR.  2 columns: CV R², coefficients.
# ============================================================================

def generate_fig3(results: dict, output_dir: Path):
    """3x2 merged figure: model CV R² (left) and 24-feature Ridge
    coefficients with error bars (right), one row per target type."""

    model_display = {
        "ols_std_ddg": r"std($\Delta\Delta G$)",
        "ols_mean_abs_ddg": r"mean|$\Delta\Delta G$|",
        "ols_plddt": "pLDDT",
        "ols_sasa": "SASA",
        "ols_std_plddt": "std+pLDDT",
        "ridge_20ddg": r"20 $\Delta\Delta G$",
        "ridge_nonlinear_only": "4 NL",
        "ridge_20ddg_nonlinear": "20+NL",
        "ridge_20ddg_plddt": "20+pLDDT",
        "ridge_20ddg_nonlinear_plddt": "20+NL+pLDDT",
    }

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
    NONLINEAR_NAMES = ["std_ddg", "mean|DDG|", "max|DDG|", "min_ddg"]
    NONLINEAR_LABELS = [
        r"std($\Delta\Delta G$)",
        r"mean|$\Delta\Delta G$|",
        r"max|$\Delta\Delta G$|",
        r"min($\Delta\Delta G$)",
    ]
    ALL_FEATURES = AA_ORDER + NONLINEAR_NAMES
    ALL_LABELS = AA_ORDER + NONLINEAR_LABELS
    COEF_MODEL = "ridge_20ddg_nonlinear"

    # 3 rows: RMSF, B-factor, NMR
    row_configs = [
        ("RMSF", "rmsf", [
            ("atlas", "tab:blue", "ATLAS"),
            ("bbflow", "tab:orange", "BBFlow"),
        ]),
        ("B-factor", "bfactor", [
            ("atlas", "tab:blue", "ATLAS"),
            ("pdb_designs", "tab:green", "PDB designs"),
        ]),
        (r"NMR ($1 - S^2_\mathrm{RCI}$)", "bfactor", [
            ("rci_s2", "tab:purple", r"$S^2_\mathrm{RCI}$"),
        ]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(22, 18),
                             gridspec_kw={"width_ratios": [1, 1.8]})

    for row_idx, (title, target, dataset_list) in enumerate(row_configs):
        # ---- Left: CV R² model comparison ----
        ax_r2 = axes[row_idx, 0]

        all_model_names = []
        for mname in model_display:
            for ds_name, _, _ in dataset_list:
                run_key = f"{ds_name}_thermompnn_{target}"
                run = results.get("runs", {}).get(run_key, {})
                models = run.get("multi_ddg", {}).get("models", {})
                if mname in models:
                    if mname not in all_model_names:
                        all_model_names.append(mname)
                    break

        if all_model_names:
            n_models = len(all_model_names)
            n_datasets = len(dataset_list)
            bar_width = 0.8 / n_datasets
            x = np.arange(n_models)

            for ds_idx, (ds_name, color, label) in enumerate(dataset_list):
                run_key = f"{ds_name}_thermompnn_{target}"
                run = results.get("runs", {}).get(run_key, {})
                models = run.get("multi_ddg", {}).get("models", {})

                r2_vals = []
                r2_stds = []
                for mname in all_model_names:
                    m = models.get(mname, {})
                    r2_vals.append(m.get("cv_r2_mean", 0) or 0)
                    r2_stds.append(m.get("cv_r2_std", 0) or 0)

                offset = (ds_idx - (n_datasets - 1) / 2) * bar_width
                ax_r2.bar(x + offset, r2_vals, bar_width, yerr=r2_stds,
                          color=color, alpha=0.8, capsize=2, label=label)

            display_names = [model_display[m] for m in all_model_names]
            ax_r2.set_xticks(x)
            ax_r2.set_xticklabels(display_names, rotation=45, ha="right")

        ax_r2.set_ylabel("CV $R^2$")
        ax_r2.set_title(f"{title}: model comparison", fontweight="bold")
        # Legend on every left panel (datasets differ by row)
        ax_r2.legend(frameon=False)

        # ---- Right: 24-feature Ridge coefficients with error bars ----
        ax_coef = axes[row_idx, 1]
        n_series = len(dataset_list)
        width = 0.8 / n_series
        # Add gap between AA and nonlinear features
        x_coef = np.arange(len(ALL_FEATURES), dtype=float)
        x_coef[len(AA_ORDER):] += 1.0  # shift NL features right by 1 unit

        for series_idx, (ds_name, color, label) in enumerate(dataset_list):
            run_key = f"{ds_name}_thermompnn_{target}"
            run = results.get("runs", {}).get(run_key, {})
            models = run.get("multi_ddg", {}).get("models", {})
            ridge = models.get(COEF_MODEL, {})
            coefs = ridge.get("feature_coefs_mean")
            if not coefs:
                continue

            feat_names = ridge.get("feature_names", [])
            coef_dict = dict(zip(feat_names, coefs))
            vals = [coef_dict.get(f, 0) for f in ALL_FEATURES]

            # Error bars from std across CV folds
            coefs_std = ridge.get("feature_coefs_std")
            if coefs_std:
                std_dict = dict(zip(feat_names, coefs_std))
                errs = [std_dict.get(f, 0) for f in ALL_FEATURES]
            else:
                errs = None

            offset = (series_idx - (n_series - 1) / 2) * width
            ax_coef.bar(x_coef + offset, vals, width, yerr=errs,
                        color=color, alpha=0.8, capsize=2, label=label)

        ax_coef.set_xticks(x_coef)
        ax_coef.set_xticklabels(ALL_LABELS, rotation=45, ha="right")
        ax_coef.set_ylabel("Ridge coefficient")
        ax_coef.set_title(f"{title}: Ridge coefficients (20 AA + 4 NL)",
                          fontweight="bold")
        ax_coef.axhline(0, color="gray", linewidth=0.5)
        # Vertical separator: darker line in the gap between AA and NL
        sep_x = len(AA_ORDER) - 0.5 + 0.5  # midpoint of the gap
        ax_coef.axvline(sep_x, color="black", linewidth=1.2,
                        linestyle="-", alpha=0.7)
        # No legend on right panels (same colors as left)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig3_model_comparison.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated fig3_model_comparison")


def generate_fig4(results: dict, output_dir: Path):
    """Kept as no-op; merged into fig3."""
    print("  (fig4 merged into fig3, skipping)")


def generate_supp_fig1(results: dict, output_dir: Path):
    """Raw (un-normalized) scatter of robustness vs dynamics, motivating z-scoring."""
    TARGET_UNITS = {
        "rmsf": r"RMSF ($\AA$)",
        "bfactor": r"B-factor ($\AA^2$)",
    }

    n_panels = len(FIG2_PANELS)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    axes = np.atleast_2d(axes)

    for panel_idx, (ds_name, target) in enumerate(FIG2_PANELS):
        ds = DATASETS[ds_name]
        if target == "rmsf":
            target_label = "RMSF"
        elif ds_name == "rci_s2":
            target_label = r"$1{-}S^2_\mathrm{RCI}$"
        else:
            target_label = "B-factor"

        row, col = panel_idx // n_cols, panel_idx % n_cols
        ax = axes[row, col]

        pooled = _load_pooled_data(ds_name, "thermompnn")
        if pooled.empty:
            ax.set_title(f"{ds.display_name} {target_label}\n(no data)")
            continue

        target_data = pooled[pooled["target_type"] == target]
        if target_data.empty:
            ax.set_title(f"{ds.display_name} {target_label}\n(no data)")
            continue

        # Subsample for plotting
        n_max = 50000
        if len(target_data) > n_max:
            target_data = target_data.sample(n_max, random_state=42)

        x = target_data["robustness_raw"].values
        y = target_data["target_raw"].values

        # Color by protein to show between-protein dominance
        pids = target_data["protein_id"].values
        unique_pids = np.unique(pids)
        pid_colors = {p: i for i, p in enumerate(unique_pids)}
        c = np.array([pid_colors[p] for p in pids])

        ax.scatter(x, y, alpha=0.05, s=2, c=c, cmap="tab20", rasterized=True)

        rho = scipy_stats.spearmanr(x, y)[0]
        r_pearson = np.corrcoef(x, y)[0, 1]
        n_proteins = len(unique_pids)

        ax.set_title(f"{ds.display_name} {target_label}\n"
                     f"Spearman $\\rho = {rho:.3f}$, Pearson $r = {r_pearson:.3f}$, "
                     f"$n = {len(x):,}$ residues, {n_proteins} proteins",
                     fontsize=11)
        ax.set_xlabel(r"$\operatorname{std}(\Delta\Delta G)$ (kcal/mol)")
        if ds_name == "rci_s2":
            ax.set_ylabel(r"$1 - S^2_\mathrm{RCI}$ (unitless)")
        else:
            ax.set_ylabel(TARGET_UNITS.get(target, target_label))

    # Remove empty axes
    for idx in range(n_panels, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"supp_fig1_raw_scatter.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated supp_fig1_raw_scatter")


# ============================================================================
# MAIN
# ============================================================================

FIGURE_GENERATORS = {
    "fig1": ("Fig 1 (per-protein correlations)", generate_fig1),
    "fig2": ("Fig 2 (density scatter)", generate_fig2),
    "fig3": ("Fig 3 (model comparison)", generate_fig3),
    "fig4": ("Fig 4 (DDG coefficients)", generate_fig4),
    "supp_fig1": ("Supp Fig 1 (raw scatter)", generate_supp_fig1),
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to unified_results.json")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--figure", type=str, default=None,
                        help="Generate only this figure (e.g., 'fig1')")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs_to_gen = FIGURE_GENERATORS
    if args.figure:
        figs_to_gen = {args.figure: FIGURE_GENERATORS[args.figure]}

    for fig_id, (description, generator) in figs_to_gen.items():
        print(f"Generating {description}...")
        try:
            generator(results, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
