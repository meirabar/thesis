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
                        "target_type": tname,
                        "protein_id": pid,
                    })

    return pd.DataFrame(rows)


# ============================================================================
# FIGURE 1: Per-protein correlation distributions (4 panels)
# ============================================================================

def generate_fig1(results: dict, output_dir: Path):
    """4-panel per-protein rho histograms + scatter plots."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for col_idx, (ds_name, target) in enumerate(FIG1_PANELS):
        ds = DATASETS[ds_name]
        target_label = "RMSF" if target == "rmsf" else "B-factor"
        panel_label = f"{ds.display_name} {target_label}"

        # Histogram panel
        ax_hist = axes[0, col_idx]
        ax_scat = axes[1, col_idx]

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
        if col_idx == 0:
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
    """4-panel 2D density scatter plots with marginal distributions."""
    fig = plt.figure(figsize=(20, 18))

    for panel_idx, (ds_name, target) in enumerate(FIG2_PANELS):
        ds = DATASETS[ds_name]
        target_label = "RMSF" if target == "rmsf" else "B-factor"

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

        # Remap panel positions for 2x2 layout
        row = 0 if panel_idx < 2 else 1
        col = panel_idx % 2

        gs_inner = GridSpec(
            4, 4, figure=fig,
            left=0.07 + 0.48 * col, right=0.07 + 0.48 * col + 0.40,
            bottom=0.07 + 0.48 * (1 - row), top=0.07 + 0.48 * (1 - row) + 0.40,
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
# FIGURE 3: Multi-DDG model comparison (combined panels)
# ============================================================================

def generate_fig3(results: dict, output_dir: Path):
    """Model comparison: 2 panels (RMSF, B-factor) with grouped bars per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

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

    # Panel configs: (target, list of (dataset, color, label))
    panel_configs = [
        ("RMSF", "rmsf", [
            ("atlas", "tab:blue", "ATLAS"),
            ("bbflow", "tab:orange", "BBFlow"),
        ]),
        ("B-factor", "bfactor", [
            ("atlas", "tab:blue", "ATLAS"),
            ("pdb_designs", "tab:green", "PDB designs"),
        ]),
    ]

    for panel_idx, (title, target, dataset_list) in enumerate(panel_configs):
        ax = axes[panel_idx]

        # Collect model names from all datasets in this panel
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

        if not all_model_names:
            ax.set_title(f"{title}\n(no data)")
            continue

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
            ax.bar(x + offset, r2_vals, bar_width, yerr=r2_stds,
                   color=color, alpha=0.8, capsize=2, label=label)

        display_names = [model_display[m] for m in all_model_names]
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.set_ylabel("CV $R^2$")
        ax.set_title(title, fontweight="bold")

        ax.legend(frameon=False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig3_model_comparison.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated fig3_model_comparison")


# ============================================================================
# FIGURE 4: DDG coefficients (combined panels)
# ============================================================================

def generate_fig4(results: dict, output_dir: Path):
    """Per-amino-acid coefficient bar chart + scalar model R^2 comparison.

    Top row: 20 AA Ridge coefficients (from ridge_20ddg model).
    Bottom row: CV R^2 for scalar summary models (std_ddg, mean_ddg,
      mean_abs_ddg, pLDDT, SASA) as grouped bars.
    """
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

    # Scalar models to show in bottom panels (name in JSON, display label)
    SCALAR_MODELS = [
        ("ols_std_ddg", r"std($\Delta\Delta G$)"),
        ("ols_mean_ddg", r"mean($\Delta\Delta G$)"),
        ("ols_mean_abs_ddg", r"mean|$\Delta\Delta G$|"),
        ("ols_plddt", "pLDDT"),
        ("ols_sasa", "SASA"),
    ]

    # Panel columns: RMSF and B-factor
    panels = [
        ("RMSF", [
            ("atlas_thermompnn_rmsf", "tab:blue", "ATLAS"),
            ("bbflow_thermompnn_rmsf", "tab:orange", "BBFlow"),
        ]),
        ("B-factor", [
            ("atlas_thermompnn_bfactor", "tab:blue", "ATLAS"),
            ("pdb_designs_thermompnn_bfactor", "tab:green", "PDB designs"),
        ]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1]})

    for panel_idx, (title, series_list) in enumerate(panels):
        # --- Top: 20 AA Ridge coefficients ---
        ax_top = axes[0, panel_idx]
        for series_idx, (key, color, label) in enumerate(series_list):
            run = results.get("runs", {}).get(key, {})
            multi = run.get("multi_ddg", {})
            models = multi.get("models", {})
            ridge = models.get("ridge_20ddg", {})
            coefs = ridge.get("feature_coefs_mean")
            if not coefs:
                continue

            feat_names = ridge.get("feature_names")
            if isinstance(coefs, list) and feat_names:
                coef_dict = dict(zip(feat_names, coefs))
            elif isinstance(coefs, dict):
                coef_dict = coefs
            else:
                coef_dict = dict(zip(AA_ORDER, coefs)) if len(coefs) == 20 else {}
            vals = [coef_dict.get(aa, 0) for aa in AA_ORDER]
            x = np.arange(len(AA_ORDER))
            width = 0.35
            offset = -width/2 if series_idx == 0 else width/2
            ax_top.bar(x + offset, vals, width, color=color, alpha=0.8, label=label)

        ax_top.set_xticks(np.arange(len(AA_ORDER)))
        ax_top.set_xticklabels(AA_ORDER)
        ax_top.set_ylabel("Ridge coefficient")
        ax_top.set_title(title, fontweight="bold")
        ax_top.axhline(0, color="gray", linewidth=0.5)
        ax_top.legend(frameon=False, loc="upper left")

        # --- Bottom: scalar model CV R^2 ---
        ax_bot = axes[1, panel_idx]
        scalar_labels = [lbl for _, lbl in SCALAR_MODELS]
        x_sc = np.arange(len(SCALAR_MODELS))
        width_sc = 0.35

        for series_idx, (key, color, label) in enumerate(series_list):
            run = results.get("runs", {}).get(key, {})
            multi = run.get("multi_ddg", {})
            models = multi.get("models", {})
            r2_vals = []
            for model_name, _ in SCALAR_MODELS:
                m = models.get(model_name, {})
                r2 = m.get("cv_r2_mean")
                r2_vals.append(r2 if r2 is not None else 0)

            offset = -width_sc/2 if series_idx == 0 else width_sc/2
            ax_bot.bar(x_sc + offset, r2_vals, width_sc, color=color,
                       alpha=0.8, label=label)

        ax_bot.set_xticks(x_sc)
        ax_bot.set_xticklabels(scalar_labels, rotation=30, ha="right")
        ax_bot.set_ylabel("CV $R^2$")
        ax_bot.set_ylim(bottom=0)
        ax_bot.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"fig4_ddg_coefficients.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Generated fig4_ddg_coefficients")


# ============================================================================
# MAIN
# ============================================================================

FIGURE_GENERATORS = {
    "fig1": ("Fig 1 (per-protein correlations)", generate_fig1),
    "fig2": ("Fig 2 (density scatter)", generate_fig2),
    "fig3": ("Fig 3 (model comparison)", generate_fig3),
    "fig4": ("Fig 4 (DDG coefficients)", generate_fig4),
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
