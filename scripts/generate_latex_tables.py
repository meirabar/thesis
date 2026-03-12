#!/usr/bin/env python3
"""
Generate LaTeX tables for the robustness-dynamics paper from unified results.

Reads unified_results.json (produced by collect_results.py) and outputs
LaTeX table fragments that can be \\input{} into the paper or copy-pasted.

Usage:
  python generate_latex_tables.py --results unified_results.json --output-dir tables/
  python generate_latex_tables.py --results unified_results.json --table table1
"""

import json
import argparse
from pathlib import Path

from paper_config import (
    TABLE1_COLUMNS, TABLE1_PREDICTORS,
    TABLE2_SS_STRATA, TABLE2_BURIAL_STRATA,
    TABLE3_MODEL_ORDER, ALT_ROBUSTNESS_MEASURES,
    DATASETS,
)


def _fmt(val, decimals=3, sign=False):
    """Format a number for LaTeX, handling None/missing."""
    if val is None:
        return "---"
    if sign and val > 0:
        return f"$+${abs(val):.{decimals}f}"
    if sign and val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"{val:.{decimals}f}"


def _signed(val, decimals=3):
    """Format with explicit sign ($-$0.xxx or $+$0.xxx)."""
    if val is None:
        return "---"
    if val < 0:
        return f"$-${abs(val):.{decimals}f}"
    return f"$+${val:.{decimals}f}"


def _get_run(results, dataset, scorer, target):
    """Get a run from the unified results, or empty dict."""
    key = f"{dataset}_{scorer}_{target}"
    return results.get("runs", {}).get(key, {})


def _get_corr(run, field):
    """Safely get a correlation field."""
    return run.get("correlation", {}).get("pooled", {}).get(field)


def _get_pp(run, field):
    """Safely get a per-protein summary field."""
    return run.get("correlation", {}).get("per_protein_summary", {}).get(field)


def _get_strat(run, strat_type, category, field):
    """Safely get a stratified field."""
    return run.get("stratified", {}).get(strat_type, {}).get(category, {}).get(field)


# ============================================================================
# TABLE 1: Main bivariate results
# ============================================================================

def generate_table1(results: dict) -> str:
    """Generate Table 1: main bivariate, incremental, and partial results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Robustness vs.\ dynamics: bivariate correlations, incremental")
    lines.append(r"regression, and partial correlations across all four dataset--target")
    lines.append(r"combinations. Robustness: $\operatorname{std}(\Delta\Delta G)$.")
    lines.append(r"All pooled correlations significant at $p < 10^{-10}$.}")
    lines.append(r"\label{tab:pooled}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l cccc@{}}")
    lines.append(r"\toprule")

    # Column headers
    headers1 = [""]
    headers2 = [""]
    for ds_name, target in TABLE1_COLUMNS:
        ds = DATASETS[ds_name]
        headers1.append(r"\textbf{" + ds.display_name + "}")
        headers2.append(r"\textbf{" + ("RMSF" if target == "rmsf" else "B-factor") + "}")
    lines.append(" & ".join(headers1) + r" \\")
    lines.append(" & ".join(headers2) + r" \\")
    lines.append(r"\midrule")

    # Helper: get value for a (dataset, target) column
    def _val(dataset, target, scorer, getter, field):
        run = _get_run(results, dataset, scorer, target)
        if not run or run.get("status") == "missing":
            return None
        return getter(run, field)

    # n proteins / n residues
    row_np = ["$n$ proteins"]
    row_nr = ["$n$ residues"]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        np_val = run.get("n_proteins")
        nr_val = run.get("n_residues")
        row_np.append(f"{np_val:,}" if np_val else "---")
        row_nr.append(f"{nr_val:,}" if nr_val else "---")
    lines.append(" & ".join(row_np) + r" \\")
    lines.append(" & ".join(row_nr) + r" \\")
    lines.append(r"\midrule")

    # --- Median per-protein rho ---
    lines.append(r"\multicolumn{5}{l}{\textit{Median per-protein Spearman $\rho$ (predictor, target)}} \\")
    for pred in ["esm1v", "thermompnn", "plddt", "sasa"]:
        label = {"esm1v": "ESM-1v", "thermompnn": "ThermoMPNN",
                 "plddt": "pLDDT", "sasa": "SASA"}[pred]
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            if pred in ("plddt", "sasa"):
                # pLDDT and SASA are scorer-independent; use thermompnn run
                run = _get_run(results, ds_name, "thermompnn", target)
                field = f"median_rho_{pred}"
                val = _get_pp(run, field)
            else:
                run = _get_run(results, ds_name, pred, target)
                val = _get_pp(run, "median_rho_robustness")
            row.append(_signed(val))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Pooled rho ---
    lines.append(r"\multicolumn{5}{l}{\textit{Pooled Spearman $\rho$ (z-scored residues)}} \\")
    for pred in ["esm1v", "thermompnn", "plddt", "sasa"]:
        label = {"esm1v": "ESM-1v", "thermompnn": "ThermoMPNN",
                 "plddt": "pLDDT", "sasa": "SASA"}[pred]
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            if pred in ("plddt", "sasa"):
                run = _get_run(results, ds_name, "thermompnn", target)
                field = f"pooled_rho_{pred}"
                val = _get_corr(run, field)
            else:
                run = _get_run(results, ds_name, pred, target)
                val = _get_corr(run, "pooled_rho_robustness")
            row.append(_signed(val))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Pooled R² ---
    lines.append(r"\multicolumn{5}{l}{\textit{Pooled $R^2$ (OLS on z-scored residues)}} \\")
    for pred in ["esm1v", "thermompnn", "plddt"]:
        label = {"esm1v": "ESM-1v", "thermompnn": "ThermoMPNN", "plddt": "pLDDT"}[pred]
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            if pred == "plddt":
                run = _get_run(results, ds_name, "thermompnn", target)
                val = _get_corr(run, "pooled_r2_plddt")
            else:
                run = _get_run(results, ds_name, pred, target)
                val = _get_corr(run, "pooled_r2_robustness")
            row.append(_fmt(val))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Delta R² (ThermoMPNN over baselines) ---
    lines.append(r"\multicolumn{5}{l}{\textit{$\Delta R^2$ (adding ThermoMPNN to baseline)}} \\")
    for baseline, label in [("plddt", r"$+$ pLDDT"), ("sasa", r"$+$ SASA")]:
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            field = f"delta_r2_over_{baseline}"
            val = _get_corr(run, field)
            row.append(_signed(val) if val is not None else "---")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Partial rho ---
    lines.append(r"\multicolumn{5}{l}{\textit{Partial $\rho$ (ThermoMPNN $|$ confounder)}} \\")
    for conf, label in [("plddt", r"$|$\,pLDDT"), ("sasa", r"$|$\,SASA")]:
        row = [r"\quad " + label]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            field = f"pooled_partial_rho_{conf}"
            val = _get_corr(run, field)
            row.append(_signed(val))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Frac beats pLDDT ---
    row = [r"Frac $|\rho_\text{rob}| > |\rho_\text{pLDDT}|$"]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        val = _get_pp(run, "frac_robustness_beats_plddt")
        if val is not None:
            row.append(f"{val*100:.1f}\\%")
        else:
            row.append("---")
    lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# TABLE 2: Stratified correlations
# ============================================================================

def generate_table2(results: dict) -> str:
    """Generate Table 2: stratified pooled rho across all datasets."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stratified pooled Spearman $\rho$ (ThermoMPNN")
    lines.append(r"$\operatorname{std}(\Delta\Delta G)$ vs.\ target).")
    lines.append(r"pLDDT values shown in parentheses where available.}")
    lines.append(r"\label{tab:stratified}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l cccc@{}}")
    lines.append(r"\toprule")

    # Headers (same as Table 1)
    headers1 = [""]
    headers2 = [""]
    for ds_name, target in TABLE1_COLUMNS:
        ds = DATASETS[ds_name]
        headers1.append(r"\textbf{" + ds.display_name + "}")
        headers2.append(r"\textbf{" + ("RMSF" if target == "rmsf" else "B-factor") + "}")
    lines.append(" & ".join(headers1) + r" \\")
    lines.append(" & ".join(headers2) + r" \\")
    lines.append(r"\midrule")

    # Secondary structure
    lines.append(r"\multicolumn{5}{l}{\textit{Secondary structure}} \\")
    ss_labels = {"H": "Helix", "E": "Sheet", "C": "Coil"}
    for ss in TABLE2_SS_STRATA:
        row = [r"\quad " + ss_labels[ss]]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            rho_rob = _get_strat(run, "secondary_structure", ss, "rho_robustness")
            rho_plddt = _get_strat(run, "secondary_structure", ss, "rho_plddt")
            if rho_rob is not None:
                cell = _signed(rho_rob)
                if rho_plddt is not None:
                    cell += r"\,(" + _signed(rho_plddt) + ")"
            else:
                cell = "---"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # Burial
    lines.append(r"\multicolumn{5}{l}{\textit{Burial (SASA terciles)}} \\")
    burial_labels = {"core": "Core", "boundary": "Boundary", "surface": "Surface"}
    for burial in TABLE2_BURIAL_STRATA:
        row = [r"\quad " + burial_labels[burial]]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            rho_rob = _get_strat(run, "burial", burial, "rho_robustness")
            rho_plddt = _get_strat(run, "burial", burial, "rho_plddt")
            if rho_rob is not None:
                cell = _signed(rho_rob)
                if rho_plddt is not None:
                    cell += r"\,(" + _signed(rho_plddt) + ")"
            else:
                cell = "---"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# TABLE 3: Alternative measures + Multi-DDG regression (united)
# ============================================================================

def generate_table3(results: dict) -> str:
    """Generate Table 3: alternative robustness measures + multi-DDG regression."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Alternative robustness measures and multi-$\Delta\Delta G$")
    lines.append(r"regression (ThermoMPNN scorer).}")
    lines.append(r"\label{tab:alt_and_multi}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l r cccc@{}}")
    lines.append(r"\toprule")

    # Headers
    lines.append(r"& & \textbf{ATLAS} & \textbf{BBFlow} & \textbf{ATLAS} & \textbf{PDB des.} \\")
    lines.append(r"& Feats & \textbf{RMSF} & \textbf{RMSF} & \textbf{B-fac} & \textbf{B-fac} \\")
    lines.append(r"\midrule")

    # --- Top half: scalar robustness summaries (median per-protein |rho|) ---
    lines.append(r"\multicolumn{6}{l}{\textit{Scalar robustness summaries (med.\ per-protein $|\rho|$)}} \\")

    for measure_key, measure_label in ALT_ROBUSTNESS_MEASURES:
        is_primary = measure_key == "std_ddg"
        label = measure_label
        if is_primary:
            label += r" \textbf{(primary)}"
        row = [r"\quad " + label, "1"]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            alt = run.get("alt_robustness_medians", {})
            val = alt.get(measure_key)
            # Show absolute value (these are |rho|)
            if val is not None:
                row.append(f"{abs(val):.3f}")
            else:
                row.append("---")
        lines.append(" & ".join(row) + r" \\")

    # pLDDT baseline
    row = [r"\quad pLDDT (baseline)", "1"]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        val = _get_pp(run, "median_rho_plddt")
        row.append(f"{abs(val):.3f}" if val is not None else "---")
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # --- Bottom half: multi-DDG regression (CV R²) ---
    lines.append(r"\multicolumn{6}{l}{\textit{Regression models (5-fold protein-level CV $R^2$)}} \\")

    # Model display names and feature counts
    model_info = {
        "ols_std_ddg": (r"OLS $\operatorname{std}(\Delta\Delta G)$", 1),
        "ols_mean_abs_ddg": (r"OLS mean$|\Delta\Delta G|$", 1),
        "ols_sasa": ("OLS SASA", 1),
        "ols_plddt": ("OLS pLDDT", 1),
        "ols_mean_plddt": (r"OLS std $+$ pLDDT", 2),
        "ridge_20ddg": (r"Ridge: 20 $\Delta\Delta G$", 20),
        "ridge_nonlinear_only": ("Ridge: 4 NL only", 4),
        "ridge_20ddg_nonlinear": (r"Ridge: 20 $\Delta\Delta G$ + 4 NL", 24),
        "ridge_20ddg_plddt": (r"Ridge: 20 $\Delta\Delta G$ + pLDDT", 21),
        "ridge_20ddg_nonlinear_plddt": (r"Ridge: 20 $\Delta\Delta G$ + NL + pLDDT", 25),
    }

    for model_key in TABLE3_MODEL_ORDER:
        if model_key not in model_info:
            continue
        label, n_feats = model_info[model_key]
        row = [r"\quad " + label, str(n_feats)]
        for ds_name, target in TABLE1_COLUMNS:
            run = _get_run(results, ds_name, "thermompnn", target)
            multi = run.get("multi_ddg", {})
            models = multi.get("models", {})
            model = models.get(model_key, {})
            val = model.get("cv_r2_mean")
            row.append(_fmt(val) if val is not None else "---")
        lines.append(" & ".join(row) + r" \\")

        # Add midrule after baselines
        if model_key == "ols_mean_plddt":
            lines.append(r"\midrule")

    # Delta R² row
    lines.append(r"\midrule")
    row = [r"\quad $\Delta R^2$ (best $-$ pLDDT)", ""]
    for ds_name, target in TABLE1_COLUMNS:
        run = _get_run(results, ds_name, "thermompnn", target)
        multi = run.get("multi_ddg", {})
        val = multi.get("delta_r2_over_plddt")
        row.append(_signed(val) if val is not None else "---")
    lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

TABLE_GENERATORS = {
    "table1": ("Table 1 (bivariate results)", generate_table1),
    "table2": ("Table 2 (stratified)", generate_table2),
    "table3": ("Table 3 (alt measures + multi-DDG)", generate_table3),
}


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from unified results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to unified_results.json")
    parser.add_argument("--output-dir", type=str, default="tables",
                        help="Output directory for .tex files")
    parser.add_argument("--table", type=str, default=None,
                        help="Generate only this table (e.g., 'table1')")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_to_gen = TABLE_GENERATORS
    if args.table:
        tables_to_gen = {args.table: TABLE_GENERATORS[args.table]}

    for table_id, (description, generator) in tables_to_gen.items():
        print(f"Generating {description}...")
        latex = generator(results)
        out_path = out_dir / f"{table_id}.tex"
        with open(out_path, "w") as f:
            f.write(latex)
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
