#!/usr/bin/env python3
"""
Collect all analysis results into a unified JSON file.

Reads per-run outputs (pooled_results.json, stratified_results.json,
multi_ddg_*_results.json) and assembles a single unified_results.json
that the table/figure generators consume.

Usage:
  python collect_results.py --output unified_results.json
  python collect_results.py --output unified_results.json --verbose
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from paper_config import (
    DATASETS, CORRELATION_RUNS, MULTI_DDG_RUNS, CLUSTER
)


def _load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict if missing or invalid."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, Exception) as e:
        print(f"  WARNING: failed to load {p}: {e}")
        return {}


def _nan_safe(val):
    """Convert NaN to None for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def collect_correlation_run(run) -> dict:
    """Collect results for one correlation analysis run."""
    result = {
        "dataset": run.dataset,
        "scorer": run.scorer,
        "target": run.target,
        "dataset_type": run.ds.dataset_type,
        "display_name": run.ds.display_name,
    }

    # --- Load pooled results ---
    pooled_raw = _load_json(run.pooled_json_path)
    if not pooled_raw:
        result["status"] = "missing"
        return result

    result["status"] = "ok"

    # The correlation script always outputs RMSF-target fields.
    # For bfactor_only datasets, the RMSF fields contain bfactor data
    # (due to aliasing in the script). For ATLAS, there are separate
    # B-factor-as-target fields.
    is_bfactor_target = (run.target == "bfactor")
    is_bfactor_only = run.ds.bfactor_only

    # Helper: get a value from pooled_raw with NaN handling
    def _get(key, default=None):
        v = pooled_raw.get(key, default)
        return _nan_safe(v)

    # --- Bivariate correlations ---
    if is_bfactor_only:
        # PDB designs: bfactor aliased into RMSF fields
        corr = {
            "pooled_rho_robustness": _get("pooled_rho_robustness_rmsf"),
            "pooled_r2_robustness": _get("pooled_r2_robustness_rmsf"),
            "pooled_rho_plddt": _get("pooled_rho_plddt_rmsf"),
            "pooled_r2_plddt": _get("pooled_r2_plddt_rmsf"),
            "pooled_rho_sasa": _get("pooled_rho_sasa_rmsf"),
            "pooled_r2_sasa": _get("pooled_r2_sasa_rmsf"),
            "pooled_r2_joint_plddt": _get("pooled_r2_joint"),
            "pooled_r2_joint_sasa": _get("pooled_r2_joint_sasa"),
            "pooled_partial_rho_sasa": _get("pooled_rho_robustness_rmsf_partial_sasa"),
            "pooled_partial_rho_plddt": _get("pooled_rho_robustness_rmsf_partial_plddt"),
        }
        per_prot = {
            "median_rho_robustness": _get("median_rho_robustness_rmsf"),
            "median_rho_plddt": _get("median_rho_plddt_rmsf"),
            "median_rho_sasa": _get("median_rho_sasa_rmsf"),
            "median_rho_partial_sasa": _get("median_rho_partial_sasa"),
            "median_rho_partial_plddt": _get("median_rho_partial_plddt"),
            "frac_robustness_beats_plddt": _get("frac_robustness_beats_plddt"),
        }
    elif is_bfactor_target:
        # ATLAS B-factor target: use dedicated B-factor-as-target fields
        corr = {
            "pooled_rho_robustness": _get("pooled_rho_robustness_bfactor_target"),
            "pooled_r2_robustness": _get("pooled_r2_robustness_bfactor_target"),
            "pooled_rho_plddt": _get("pooled_rho_plddt_bfactor"),
            "pooled_r2_plddt": _get("pooled_r2_plddt_bfactor"),
            "pooled_rho_sasa": _get("pooled_rho_sasa_bfactor"),
            "pooled_r2_sasa": _get("pooled_r2_sasa_bfactor"),
            "pooled_r2_joint_plddt": _get("pooled_r2_bfactor_joint_plddt"),
            "pooled_r2_joint_sasa": _get("pooled_r2_bfactor_joint_sasa"),
            "pooled_partial_rho_sasa": _get("pooled_rho_robustness_bfactor_partial_sasa"),
            "pooled_partial_rho_plddt": _get("pooled_rho_robustness_bfactor_partial_plddt"),
        }
        per_prot = {
            "median_rho_robustness": _get("median_rho_robustness_bfactor_target"),
            "median_rho_plddt": _get("median_rho_plddt_bfactor"),
            "median_rho_sasa": _get("median_rho_sasa_bfactor"),
            "median_rho_partial_sasa": _get("median_rho_robustness_bfactor_partial_sasa"),
            "median_rho_partial_plddt": _get("median_rho_robustness_bfactor_partial_plddt"),
            "frac_robustness_beats_plddt": _get("frac_robustness_beats_plddt_bfactor"),
        }
    else:
        # RMSF target (default)
        corr = {
            "pooled_rho_robustness": _get("pooled_rho_robustness_rmsf"),
            "pooled_r2_robustness": _get("pooled_r2_robustness_rmsf"),
            "pooled_rho_plddt": _get("pooled_rho_plddt_rmsf"),
            "pooled_r2_plddt": _get("pooled_r2_plddt_rmsf"),
            "pooled_rho_sasa": _get("pooled_rho_sasa_rmsf"),
            "pooled_r2_sasa": _get("pooled_r2_sasa_rmsf"),
            "pooled_r2_joint_plddt": _get("pooled_r2_joint"),
            "pooled_r2_joint_sasa": _get("pooled_r2_joint_sasa"),
            "pooled_partial_rho_sasa": _get("pooled_rho_robustness_rmsf_partial_sasa"),
            "pooled_partial_rho_plddt": _get("pooled_rho_robustness_rmsf_partial_plddt"),
        }
        per_prot = {
            "median_rho_robustness": _get("median_rho_robustness_rmsf"),
            "median_rho_plddt": _get("median_rho_plddt_rmsf"),
            "median_rho_sasa": _get("median_rho_sasa_rmsf"),
            "median_rho_partial_sasa": _get("median_rho_partial_sasa"),
            "median_rho_partial_plddt": _get("median_rho_partial_plddt"),
            "frac_robustness_beats_plddt": _get("frac_robustness_beats_plddt"),
        }

    # Compute delta R² values
    r2_rob = corr.get("pooled_r2_robustness")
    r2_plddt = corr.get("pooled_r2_plddt")
    r2_joint_plddt = corr.get("pooled_r2_joint_plddt")
    r2_joint_sasa = corr.get("pooled_r2_joint_sasa")
    r2_sasa = corr.get("pooled_r2_sasa")

    corr["delta_r2_over_plddt"] = (
        round(r2_joint_plddt - r2_plddt, 4) if r2_joint_plddt and r2_plddt else None
    )
    corr["delta_r2_over_sasa"] = (
        round(r2_joint_sasa - r2_sasa, 4) if r2_joint_sasa and r2_sasa else None
    )

    result["correlation"] = {
        "pooled": corr,
        "per_protein_summary": per_prot,
    }

    # --- Load stratified results ---
    strat_raw = _load_json(run.stratified_json_path)
    if strat_raw:
        stratified = {}
        for strat_type in ["secondary_structure", "burial"]:
            if strat_type in strat_raw:
                strat_data = strat_raw[strat_type]
                # Normalize field names based on target
                normalized = {}
                for cat, vals in strat_data.items():
                    entry = {"n_residues": vals.get("n_residues")}
                    if is_bfactor_target or is_bfactor_only:
                        entry["rho_robustness"] = _nan_safe(vals.get(
                            "rho_robustness_bfactor",
                            vals.get("rho_robustness_rmsf")))
                        entry["rho_plddt"] = _nan_safe(vals.get(
                            "rho_plddt_bfactor",
                            vals.get("rho_plddt_rmsf")))
                    else:
                        entry["rho_robustness"] = _nan_safe(vals.get("rho_robustness_rmsf"))
                        entry["rho_plddt"] = _nan_safe(vals.get("rho_plddt_rmsf"))
                    normalized[cat] = entry
                stratified[strat_type] = normalized
        result["stratified"] = stratified

    # --- Alternative robustness measures (from per-protein data) ---
    # These are median per-protein rho for different robustness summaries.
    # Stored in per_protein_correlations.tsv columns.
    # We extract them from pooled_raw which has alt measure medians.
    alt = {}
    if is_bfactor_target and not is_bfactor_only:
        alt["std_ddg"] = _get("median_rho_robustness_bfactor_target")
        # The alt measure fields for B-factor target may not be in pooled JSON.
        # They'd need to be computed from per-protein TSV.
    elif is_bfactor_only:
        alt["std_ddg"] = _get("median_rho_robustness_rmsf")
    else:
        alt["std_ddg"] = _get("median_rho_robustness_rmsf")
    # Additional alt measures come from per-protein TSV (max_ddg, frac_destab etc.)
    # We read those from the TSV if it exists
    tsv_path = Path(run.per_protein_tsv_path)
    if tsv_path.exists():
        try:
            import pandas as pd
            pp_df = pd.read_csv(tsv_path, sep="\t")
            target_suffix = "bfactor" if (is_bfactor_target or is_bfactor_only) else "rmsf"
            for col_prefix, label in [
                ("rho_frac_destab", "frac_destab"),
                ("rho_frac_neutral", "frac_neutral"),
                ("rho_std_ddg", "std_ddg"),
                ("rho_max_ddg", "max_ddg"),
            ]:
                col = f"{col_prefix}_{target_suffix}" if f"{col_prefix}_{target_suffix}" in pp_df.columns else col_prefix
                if col in pp_df.columns:
                    vals = pp_df[col].dropna()
                    if len(vals) > 0:
                        alt[label] = _nan_safe(float(vals.median()))
        except Exception:
            pass
    result["alt_robustness_medians"] = alt

    # Protein/residue counts from pooled data
    result["n_proteins"] = _get("n_proteins", run.ds.n_proteins_approx)
    result["n_residues"] = _get("n_residues")

    return result


def collect_multi_ddg_run(run) -> dict:
    """Collect multi-DDG regression results for one run."""
    raw = _load_json(run.multi_ddg_json_path)
    if not raw:
        return {"status": "missing"}

    models = {}
    for model_name, model_data in raw.items():
        models[model_name] = {
            "n_features": model_data.get("n_features"),
            "cv_r2_mean": _nan_safe(model_data.get("cv_r2_mean")),
            "cv_r2_std": _nan_safe(model_data.get("cv_r2_std")),
            "cv_rho_mean": _nan_safe(model_data.get("cv_rho_mean")),
            "cv_rho_std": _nan_safe(model_data.get("cv_rho_std")),
            "per_protein_rho_median": _nan_safe(model_data.get("per_protein_rho_median")),
            "feature_coefs_mean": model_data.get("feature_coefs_mean"),
        }

    # Find best model (highest CV R²)
    best_name = max(models, key=lambda k: models[k].get("cv_r2_mean") or -1)
    plddt_r2 = models.get("ols_plddt", {}).get("cv_r2_mean")
    best_r2 = models[best_name].get("cv_r2_mean")

    return {
        "status": "ok",
        "models": models,
        "best_model": best_name,
        "best_r2": best_r2,
        "delta_r2_over_plddt": (
            round(best_r2 - plddt_r2, 4) if best_r2 and plddt_r2 else None
        ),
    }


def collect_all(verbose: bool = False) -> dict:
    """Collect all results into unified structure."""
    unified = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "collect_results.py",
        },
        "runs": {},
    }

    # Correlation runs
    print("Collecting correlation results...")
    for run in CORRELATION_RUNS:
        if verbose:
            print(f"  {run.key}: {run.pooled_json_path}")
        result = collect_correlation_run(run)
        unified["runs"][run.key] = result
        status = result.get("status", "unknown")
        if verbose:
            print(f"    -> {status}")

    # Multi-DDG runs
    print("Collecting multi-DDG regression results...")
    for run in MULTI_DDG_RUNS:
        key = run.key
        if verbose:
            print(f"  {key}: {run.multi_ddg_json_path}")
        multi_ddg = collect_multi_ddg_run(run)
        if key in unified["runs"]:
            unified["runs"][key]["multi_ddg"] = multi_ddg
        else:
            unified["runs"][key] = {"multi_ddg": multi_ddg}
        if verbose:
            print(f"    -> {multi_ddg.get('status', 'unknown')}")

    # Summary
    n_ok = sum(1 for r in unified["runs"].values()
               if r.get("status") == "ok" or r.get("multi_ddg", {}).get("status") == "ok")
    n_missing = sum(1 for r in unified["runs"].values()
                    if r.get("status") == "missing")
    print(f"\nCollected: {n_ok} runs with data, {n_missing} missing")

    return unified


def main():
    parser = argparse.ArgumentParser(description="Collect analysis results into unified JSON")
    parser.add_argument("--output", type=str,
                        default=f"{CLUSTER.paper_results_dir}/unified_results.json",
                        help="Output path for unified JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    unified = collect_all(verbose=args.verbose)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(unified, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
