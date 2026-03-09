#!/usr/bin/env python3
"""
Correlate per-residue mutational robustness with protein dynamics (RMSF)
using ATLAS database data.

This is the core analysis script for Direction 7:
"Does mutational robustness predict protein dynamics?"

Inputs:
  - ATLAS download directory (from download_atlas.py):
      proteins/{pdb_chain}/*_RMSF.tsv, *_pLDDT.tsv, *_Bfactor.tsv
  - Robustness output directory (from compute_robustness.py):
      {scorer}/*_robustness.tsv

Outputs:
  - Per-protein correlation results (TSV + JSON)
  - Pooled correlation analysis
  - Comparison: robustness vs. pLDDT vs. B-factor as dynamics predictors
  - Stratified analysis by secondary structure and burial
  - Publication-ready figures

Usage:
  python correlate_robustness_dynamics.py \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --robustness_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness \
      --scorer esm1v \
      --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis

  # Multiple scorers:
  python correlate_robustness_dynamics.py \
      --atlas_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas \
      --robustness_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_robustness \
      --scorer esm1v thermompnn \
      --output_dir /sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==========================================================================
# DATA LOADING
# ==========================================================================

def load_atlas_tsv(protein_dir: str, suffix: str) -> Optional[pd.DataFrame]:
    """Load a per-residue TSV file from an ATLAS protein directory."""
    protein_dir = Path(protein_dir)
    matches = list(protein_dir.glob(f"*{suffix}"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    return df


def load_atlas_rmsf(protein_dir: str) -> Optional[pd.DataFrame]:
    """Load RMSF data. Returns DataFrame with columns: position, rmsf_avg.

    ATLAS RMSF files have RMSF for 3 replicates. We average them.
    """
    df = load_atlas_tsv(protein_dir, "_RMSF.tsv")
    if df is None:
        return None
    # ATLAS RMSF TSV typically has columns like:
    # resid, resname, RMSF_R1, RMSF_R2, RMSF_R3 (or similar)
    # Handle different possible column formats
    rmsf_cols = [c for c in df.columns if "rmsf" in c.lower() or "r1" in c.lower()
                 or "r2" in c.lower() or "r3" in c.lower()]
    if not rmsf_cols:
        # Try numeric columns (skip residue id/name)
        rmsf_cols = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, float)]

    if not rmsf_cols:
        return None

    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["rmsf_avg"] = df[rmsf_cols].mean(axis=1).values
    # Also keep individual replicates if available
    for i, col in enumerate(rmsf_cols):
        result[f"rmsf_r{i+1}"] = df[col].values

    return result


def load_atlas_pldt(protein_dir: str) -> Optional[pd.DataFrame]:
    """Load pLDDT data. Returns DataFrame with columns: position, plddt."""
    df = load_atlas_tsv(protein_dir, "_pLDDT.tsv")
    if df is None:
        return None
    # Find the pLDDT column
    plddt_cols = [c for c in df.columns if "plddt" in c.lower() or "confidence" in c.lower()]
    if not plddt_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, float)]
        plddt_cols = numeric[-1:] if numeric else []
    if not plddt_cols:
        return None

    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["plddt"] = df[plddt_cols[0]].values
    return result


def load_atlas_bfactor(protein_dir: str) -> Optional[pd.DataFrame]:
    """Load B-factor data."""
    df = load_atlas_tsv(protein_dir, "_Bfactor.tsv")
    if df is None:
        return None
    bfac_cols = [c for c in df.columns if "bfactor" in c.lower() or "b_factor" in c.lower()]
    if not bfac_cols:
        numeric = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, float)]
        bfac_cols = numeric[-1:] if numeric else []
    if not bfac_cols:
        return None
    result = pd.DataFrame()
    result["position"] = range(1, len(df) + 1)
    result["bfactor"] = df[bfac_cols[0]].values
    return result


def load_robustness(robustness_dir: str, scorer: str, protein_id: str
                    ) -> Optional[pd.DataFrame]:
    """Load per-residue robustness TSV from compute_robustness.py output."""
    tsv_path = Path(robustness_dir) / scorer / f"{protein_id}_robustness.tsv"
    if not tsv_path.exists():
        return None
    return pd.read_csv(tsv_path, sep="\t")


def load_robustness_global(robustness_dir: str, scorer: str, protein_id: str
                           ) -> Optional[Dict]:
    """Load global robustness metrics from JSON."""
    json_path = Path(robustness_dir) / scorer / f"{protein_id}_robustness.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("global_metrics")


# ==========================================================================
# SECONDARY STRUCTURE FROM PDB
# ==========================================================================

def assign_secondary_structure(pdb_path: str) -> Optional[List[str]]:
    """Assign secondary structure using DSSP (BioPython) or mdtraj fallback.

    Returns list of 'H' (helix), 'E' (sheet), 'C' (coil) per residue.
    """
    # Try BioPython DSSP first (requires external mkdssp binary)
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
        ss_list = []
        for key in dssp.keys():
            ss = dssp[key][2]  # secondary structure code
            if ss in ("H", "G", "I"):
                ss_list.append("H")  # helix
            elif ss in ("E", "B"):
                ss_list.append("E")  # sheet
            else:
                ss_list.append("C")  # coil/loop
        return ss_list
    except Exception:
        pass

    # Fallback: mdtraj DSSP (no external binary needed)
    try:
        import mdtraj
        traj = mdtraj.load(pdb_path)
        dssp = mdtraj.compute_dssp(traj)[0]  # shape (n_residues,)
        ss_list = []
        for ss in dssp:
            if ss == 'H':
                ss_list.append('H')
            elif ss == 'E':
                ss_list.append('E')
            else:
                ss_list.append('C')
        return ss_list
    except Exception:
        return None


def compute_burial(pdb_path: str) -> Optional[List[float]]:
    """Compute per-residue relative solvent accessibility (RSA).

    Returns list of RSA values (0=buried, 1=exposed).
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
        rsa_list = [dssp[key][3] for key in dssp.keys()]
        return rsa_list
    except Exception:
        return None


def compute_sasa_from_pdb(pdb_path: str) -> Optional[pd.DataFrame]:
    """Compute per-residue SASA from a PDB file using mdtraj Shrake-Rupley.

    This is independent of DSSP (works even with --no_dssp) and provides
    a uniform burial baseline for all protein types (natural and designed).

    Returns DataFrame with columns: position, sasa (in nm^2).
    """
    try:
        import mdtraj
        traj = mdtraj.load(pdb_path)
        # Compute per-atom SASA, then sum by residue
        sasa_per_atom = mdtraj.shrake_rupley(traj, mode='atom')  # shape (1, n_atoms)
        sasa_per_residue = np.zeros(traj.topology.n_residues)
        for atom in traj.topology.atoms:
            sasa_per_residue[atom.residue.index] += sasa_per_atom[0, atom.index]
        result = pd.DataFrame({
            "position": range(1, len(sasa_per_residue) + 1),
            "sasa": sasa_per_residue,
        })
        return result
    except Exception:
        return None


# ==========================================================================
# CORRELATION ANALYSIS
# ==========================================================================

@dataclass
class PerProteinResult:
    """Correlation results for a single protein."""
    protein_id: str
    seq_length: int
    n_residues_used: int  # after filtering NaN

    # Robustness vs RMSF (primary measure: mean |DDG|)
    rho_robustness_rmsf: float = np.nan
    pval_robustness_rmsf: float = np.nan
    r2_robustness_rmsf: float = np.nan

    # Alternative robustness measures vs RMSF
    rho_frac_destab_rmsf: float = np.nan     # fraction of destabilizing mutations (ddG > 1)
    rho_frac_neutral_rmsf: float = np.nan    # fraction of neutral mutations (|ddG| < 0.5)
    rho_std_ddg_rmsf: float = np.nan         # std of ddG (landscape ruggedness)
    rho_max_ddg_rmsf: float = np.nan         # worst-case mutation effect

    # pLDDT vs RMSF (baseline)
    rho_plddt_rmsf: float = np.nan
    pval_plddt_rmsf: float = np.nan
    r2_plddt_rmsf: float = np.nan

    # B-factor vs RMSF (sanity check)
    rho_bfactor_rmsf: float = np.nan
    pval_bfactor_rmsf: float = np.nan

    # SASA vs RMSF (burial baseline)
    rho_sasa_rmsf: float = np.nan
    pval_sasa_rmsf: float = np.nan
    r2_sasa_rmsf: float = np.nan

    # Robustness vs pLDDT (are they correlated?)
    rho_robustness_plddt: float = np.nan

    # Robustness vs B-factor
    rho_robustness_bfactor: float = np.nan

    # Robustness vs SASA (are they correlated?)
    rho_robustness_sasa: float = np.nan

    # Partial correlation: robustness vs RMSF, controlling for SASA
    rho_robustness_rmsf_partial_sasa: float = np.nan
    pval_robustness_rmsf_partial_sasa: float = np.nan

    # Multiple regression: RMSF ~ robustness + pLDDT
    r2_joint: float = np.nan
    delta_r2_over_plddt: float = np.nan  # r2_joint - r2_plddt
    beta_robustness: float = np.nan      # regression coefficient
    beta_plddt: float = np.nan

    # Multiple regression: RMSF ~ robustness + SASA
    r2_joint_sasa: float = np.nan
    delta_r2_over_sasa: float = np.nan  # r2_joint_sasa - r2_sasa

    # Multiple regression with best robustness combo: RMSF ~ mean_abs_ddg + frac_destab + pLDDT
    r2_joint_multi_rob: float = np.nan
    delta_r2_multi_rob_over_plddt: float = np.nan

    # Global robustness metrics
    global_mean_abs_ddg: float = np.nan
    global_mean_ddg: float = np.nan

    scorer: str = ""


def correlate_single_protein(
    protein_id: str,
    robustness_df: pd.DataFrame,
    rmsf_df: pd.DataFrame,
    plddt_df: Optional[pd.DataFrame],
    bfactor_df: Optional[pd.DataFrame],
    global_metrics: Optional[Dict],
    scorer: str,
    sasa_df: Optional[pd.DataFrame] = None,
) -> Optional[PerProteinResult]:
    """Compute all correlations for a single protein."""

    # Merge on position — include all available robustness measures
    rob_cols = ["position", "mean_abs_ddg", "mean_ddg"]
    for extra in ["frac_destabilizing", "frac_neutral", "std_ddg", "max_ddg"]:
        if extra in robustness_df.columns:
            rob_cols.append(extra)
    merged = robustness_df[rob_cols].copy()
    merged = merged.merge(rmsf_df[["position", "rmsf_avg"]], on="position", how="inner")

    if plddt_df is not None:
        merged = merged.merge(plddt_df[["position", "plddt"]], on="position", how="left")
    else:
        merged["plddt"] = np.nan

    if bfactor_df is not None:
        merged = merged.merge(bfactor_df[["position", "bfactor"]], on="position", how="left")
    else:
        merged["bfactor"] = np.nan

    if sasa_df is not None:
        merged = merged.merge(sasa_df[["position", "sasa"]], on="position", how="left")
    else:
        merged["sasa"] = np.nan

    # Drop rows with NaN in core columns
    core = merged.dropna(subset=["mean_abs_ddg", "rmsf_avg"])
    if len(core) < 10:  # too few residues
        return None

    result = PerProteinResult(
        protein_id=protein_id,
        seq_length=len(robustness_df),
        n_residues_used=len(core),
        scorer=scorer,
    )

    # --- Robustness vs RMSF (primary: mean |DDG|) ---
    rho, pval = scipy_stats.spearmanr(core["mean_abs_ddg"], core["rmsf_avg"])
    result.rho_robustness_rmsf = rho
    result.pval_robustness_rmsf = pval
    # Pearson R^2
    r, _ = scipy_stats.pearsonr(core["mean_abs_ddg"], core["rmsf_avg"])
    result.r2_robustness_rmsf = r ** 2

    # --- Alternative robustness measures vs RMSF ---
    for col, attr in [("frac_destabilizing", "rho_frac_destab_rmsf"),
                       ("frac_neutral", "rho_frac_neutral_rmsf"),
                       ("std_ddg", "rho_std_ddg_rmsf"),
                       ("max_ddg", "rho_max_ddg_rmsf")]:
        if col in core.columns:
            valid_alt = core.dropna(subset=[col])
            if len(valid_alt) >= 10:
                rho_alt, _ = scipy_stats.spearmanr(valid_alt[col], valid_alt["rmsf_avg"])
                setattr(result, attr, rho_alt)

    # --- pLDDT vs RMSF ---
    plddt_valid = core.dropna(subset=["plddt"])
    if len(plddt_valid) >= 10:
        rho, pval = scipy_stats.spearmanr(plddt_valid["plddt"], plddt_valid["rmsf_avg"])
        result.rho_plddt_rmsf = rho
        result.pval_plddt_rmsf = pval
        r, _ = scipy_stats.pearsonr(plddt_valid["plddt"], plddt_valid["rmsf_avg"])
        result.r2_plddt_rmsf = r ** 2

    # --- B-factor vs RMSF ---
    bfac_valid = core.dropna(subset=["bfactor"])
    if len(bfac_valid) >= 10:
        rho, pval = scipy_stats.spearmanr(bfac_valid["bfactor"], bfac_valid["rmsf_avg"])
        result.rho_bfactor_rmsf = rho
        result.pval_bfactor_rmsf = pval

    # --- Robustness vs pLDDT ---
    if len(plddt_valid) >= 10:
        rho, _ = scipy_stats.spearmanr(plddt_valid["mean_abs_ddg"], plddt_valid["plddt"])
        result.rho_robustness_plddt = rho

    # --- Robustness vs B-factor ---
    if len(bfac_valid) >= 10:
        rho, _ = scipy_stats.spearmanr(bfac_valid["mean_abs_ddg"], bfac_valid["bfactor"])
        result.rho_robustness_bfactor = rho

    # --- SASA vs RMSF (burial baseline) ---
    sasa_valid = core.dropna(subset=["sasa"])
    if len(sasa_valid) >= 10:
        rho, pval = scipy_stats.spearmanr(sasa_valid["sasa"], sasa_valid["rmsf_avg"])
        result.rho_sasa_rmsf = rho
        result.pval_sasa_rmsf = pval
        r, _ = scipy_stats.pearsonr(sasa_valid["sasa"], sasa_valid["rmsf_avg"])
        result.r2_sasa_rmsf = r ** 2

        # Robustness vs SASA
        rho, _ = scipy_stats.spearmanr(sasa_valid["mean_abs_ddg"], sasa_valid["sasa"])
        result.rho_robustness_sasa = rho

        # Partial correlation: robustness vs RMSF, controlling for SASA
        # partial_rho(X,Y|Z) = (rho_XY - rho_XZ * rho_YZ) /
        #                      sqrt((1 - rho_XZ^2) * (1 - rho_YZ^2))
        rho_xy, _ = scipy_stats.spearmanr(sasa_valid["mean_abs_ddg"], sasa_valid["rmsf_avg"])
        rho_xz, _ = scipy_stats.spearmanr(sasa_valid["mean_abs_ddg"], sasa_valid["sasa"])
        rho_yz, _ = scipy_stats.spearmanr(sasa_valid["sasa"], sasa_valid["rmsf_avg"])
        denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
        if denom > 1e-10:
            partial_rho = (rho_xy - rho_xz * rho_yz) / denom
            result.rho_robustness_rmsf_partial_sasa = partial_rho
            # Approximate p-value for partial correlation (t-test)
            n = len(sasa_valid)
            if n > 3:
                t_stat = partial_rho * np.sqrt((n - 3) / (1 - partial_rho**2 + 1e-10))
                from scipy.stats import t as t_dist
                result.pval_robustness_rmsf_partial_sasa = 2 * t_dist.sf(abs(t_stat), n - 3)

    # --- Multiple regression: RMSF ~ robustness + SASA ---
    if len(sasa_valid) >= 10:
        from sklearn.linear_model import LinearRegression
        y_s = sasa_valid["rmsf_avg"].values
        X_sasa_only = sasa_valid[["sasa"]].values
        X_rob_sasa = sasa_valid[["mean_abs_ddg", "sasa"]].values
        r2_sasa_only = LinearRegression().fit(X_sasa_only, y_s).score(X_sasa_only, y_s)
        r2_rob_sasa = LinearRegression().fit(X_rob_sasa, y_s).score(X_rob_sasa, y_s)
        result.r2_joint_sasa = r2_rob_sasa
        result.delta_r2_over_sasa = r2_rob_sasa - r2_sasa_only

    # --- Multiple regression: RMSF ~ robustness + pLDDT ---
    joint_valid = core.dropna(subset=["plddt"])
    if len(joint_valid) >= 10:
        from sklearn.linear_model import LinearRegression

        y = joint_valid["rmsf_avg"].values
        X_plddt = joint_valid[["plddt"]].values
        X_joint = joint_valid[["mean_abs_ddg", "plddt"]].values

        # pLDDT alone
        reg_plddt = LinearRegression().fit(X_plddt, y)
        r2_plddt_only = reg_plddt.score(X_plddt, y)

        # Joint
        reg_joint = LinearRegression().fit(X_joint, y)
        r2_joint = reg_joint.score(X_joint, y)

        result.r2_joint = r2_joint
        result.delta_r2_over_plddt = r2_joint - r2_plddt_only
        result.beta_robustness = float(reg_joint.coef_[0])
        result.beta_plddt = float(reg_joint.coef_[1])

        # Multi-robustness regression: RMSF ~ mean_abs_ddg + frac_destabilizing + pLDDT
        if "frac_destabilizing" in joint_valid.columns:
            multi_valid = joint_valid.dropna(subset=["frac_destabilizing"])
            if len(multi_valid) >= 10:
                X_multi = multi_valid[["mean_abs_ddg", "frac_destabilizing", "plddt"]].values
                y_multi = multi_valid["rmsf_avg"].values
                r2_multi = LinearRegression().fit(X_multi, y_multi).score(X_multi, y_multi)
                result.r2_joint_multi_rob = r2_multi
                result.delta_r2_multi_rob_over_plddt = r2_multi - r2_plddt_only

    # --- Global metrics ---
    if global_metrics:
        result.global_mean_abs_ddg = global_metrics.get("global_mean_abs_ddg", np.nan)
        result.global_mean_ddg = global_metrics.get("global_mean_ddg", np.nan)

    return result


# ==========================================================================
# POOLED ANALYSIS
# ==========================================================================

@dataclass
class PooledResult:
    """Pooled correlation results across all proteins."""
    n_proteins: int = 0
    n_residues: int = 0
    scorer: str = ""

    # Pooled Spearman (z-scored per protein)
    pooled_rho_robustness_rmsf: float = np.nan
    pooled_pval_robustness_rmsf: float = np.nan
    pooled_rho_plddt_rmsf: float = np.nan
    pooled_pval_plddt_rmsf: float = np.nan

    # Pooled Pearson R^2
    pooled_r2_robustness_rmsf: float = np.nan
    pooled_r2_plddt_rmsf: float = np.nan

    # Pooled joint regression
    pooled_r2_joint: float = np.nan
    pooled_delta_r2: float = np.nan

    # Pooled SASA (burial baseline)
    pooled_rho_sasa_rmsf: float = np.nan
    pooled_r2_sasa_rmsf: float = np.nan
    pooled_rho_robustness_rmsf_partial_sasa: float = np.nan
    pooled_r2_joint_sasa: float = np.nan
    pooled_delta_r2_over_sasa: float = np.nan

    # Distribution of per-protein correlations
    median_rho_robustness_rmsf: float = np.nan
    mean_rho_robustness_rmsf: float = np.nan
    std_rho_robustness_rmsf: float = np.nan
    median_rho_plddt_rmsf: float = np.nan
    mean_rho_plddt_rmsf: float = np.nan
    median_rho_sasa_rmsf: float = np.nan
    median_rho_partial_sasa: float = np.nan

    # Fraction of proteins where robustness beats pLDDT
    frac_robustness_beats_plddt: float = np.nan


def run_pooled_analysis(
    per_protein_data: List[Tuple[pd.DataFrame, str]],
    per_protein_results: List[PerProteinResult],
    scorer: str,
) -> PooledResult:
    """Run pooled analysis across all proteins.

    per_protein_data: list of (merged_df, protein_id) for pooling residues.
    per_protein_results: list of PerProteinResult for summary stats.
    """
    result = PooledResult(scorer=scorer)

    # Summary of per-protein correlations
    rhos_rob = [r.rho_robustness_rmsf for r in per_protein_results
                if not np.isnan(r.rho_robustness_rmsf)]
    rhos_plddt = [r.rho_plddt_rmsf for r in per_protein_results
                  if not np.isnan(r.rho_plddt_rmsf)]
    rhos_sasa = [r.rho_sasa_rmsf for r in per_protein_results
                 if not np.isnan(r.rho_sasa_rmsf)]
    rhos_partial = [r.rho_robustness_rmsf_partial_sasa for r in per_protein_results
                    if not np.isnan(r.rho_robustness_rmsf_partial_sasa)]

    result.n_proteins = len(per_protein_results)

    if rhos_rob:
        result.median_rho_robustness_rmsf = float(np.median(rhos_rob))
        result.mean_rho_robustness_rmsf = float(np.mean(rhos_rob))
        result.std_rho_robustness_rmsf = float(np.std(rhos_rob))
    if rhos_plddt:
        result.median_rho_plddt_rmsf = float(np.median(rhos_plddt))
        result.mean_rho_plddt_rmsf = float(np.mean(rhos_plddt))
    if rhos_sasa:
        result.median_rho_sasa_rmsf = float(np.median(rhos_sasa))
    if rhos_partial:
        result.median_rho_partial_sasa = float(np.median(rhos_partial))

    # Fraction where |rho_robustness| > |rho_plddt|
    both = [(r.rho_robustness_rmsf, r.rho_plddt_rmsf) for r in per_protein_results
            if not np.isnan(r.rho_robustness_rmsf) and not np.isnan(r.rho_plddt_rmsf)]
    if both:
        beats = sum(1 for rr, rp in both if abs(rr) > abs(rp))
        result.frac_robustness_beats_plddt = beats / len(both)

    # Pool all residues (z-scored per protein)
    all_rows = []
    for merged_df, pid in per_protein_data:
        df = merged_df.dropna(subset=["mean_abs_ddg", "rmsf_avg"]).copy()
        if len(df) < 10:
            continue
        # Z-score within protein to remove protein-level differences
        for col in ["mean_abs_ddg", "rmsf_avg", "plddt", "sasa"]:
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                if sigma > 0:
                    df[f"{col}_z"] = (df[col] - mu) / sigma
                else:
                    df[f"{col}_z"] = 0.0
        df["protein_id"] = pid
        all_rows.append(df)

    if not all_rows:
        return result

    pooled = pd.concat(all_rows, ignore_index=True)
    result.n_residues = len(pooled)

    # Pooled correlations on z-scored values
    valid = pooled.dropna(subset=["mean_abs_ddg_z", "rmsf_avg_z"])
    if len(valid) >= 20:
        rho, pval = scipy_stats.spearmanr(valid["mean_abs_ddg_z"], valid["rmsf_avg_z"])
        result.pooled_rho_robustness_rmsf = rho
        result.pooled_pval_robustness_rmsf = pval
        r, _ = scipy_stats.pearsonr(valid["mean_abs_ddg_z"], valid["rmsf_avg_z"])
        result.pooled_r2_robustness_rmsf = r ** 2

    if "plddt_z" in pooled.columns:
        valid_plddt = pooled.dropna(subset=["plddt_z", "rmsf_avg_z"])
        if len(valid_plddt) >= 20:
            rho, pval = scipy_stats.spearmanr(valid_plddt["plddt_z"], valid_plddt["rmsf_avg_z"])
            result.pooled_rho_plddt_rmsf = rho
            result.pooled_pval_plddt_rmsf = pval
            r, _ = scipy_stats.pearsonr(valid_plddt["plddt_z"], valid_plddt["rmsf_avg_z"])
            result.pooled_r2_plddt_rmsf = r ** 2

        # Pooled joint regression
        joint_valid = pooled.dropna(subset=["mean_abs_ddg_z", "plddt_z", "rmsf_avg_z"])
        if len(joint_valid) >= 20:
            from sklearn.linear_model import LinearRegression
            y = joint_valid["rmsf_avg_z"].values
            X_plddt = joint_valid[["plddt_z"]].values
            X_joint = joint_valid[["mean_abs_ddg_z", "plddt_z"]].values

            r2_p = LinearRegression().fit(X_plddt, y).score(X_plddt, y)
            r2_j = LinearRegression().fit(X_joint, y).score(X_joint, y)
            result.pooled_r2_joint = r2_j
            result.pooled_delta_r2 = r2_j - r2_p

    # Pooled SASA analysis (burial baseline)
    if "sasa_z" in pooled.columns:
        valid_sasa = pooled.dropna(subset=["sasa_z", "rmsf_avg_z"])
        if len(valid_sasa) >= 20:
            rho, _ = scipy_stats.spearmanr(valid_sasa["sasa_z"], valid_sasa["rmsf_avg_z"])
            result.pooled_rho_sasa_rmsf = rho
            r, _ = scipy_stats.pearsonr(valid_sasa["sasa_z"], valid_sasa["rmsf_avg_z"])
            result.pooled_r2_sasa_rmsf = r ** 2

        # Pooled partial correlation: robustness vs RMSF | SASA
        valid_all = pooled.dropna(subset=["mean_abs_ddg_z", "sasa_z", "rmsf_avg_z"])
        if len(valid_all) >= 20:
            rho_xy, _ = scipy_stats.spearmanr(valid_all["mean_abs_ddg_z"], valid_all["rmsf_avg_z"])
            rho_xz, _ = scipy_stats.spearmanr(valid_all["mean_abs_ddg_z"], valid_all["sasa_z"])
            rho_yz, _ = scipy_stats.spearmanr(valid_all["sasa_z"], valid_all["rmsf_avg_z"])
            denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
            if denom > 1e-10:
                result.pooled_rho_robustness_rmsf_partial_sasa = (rho_xy - rho_xz * rho_yz) / denom

            # Pooled regression: RMSF ~ robustness + SASA
            from sklearn.linear_model import LinearRegression
            y = valid_all["rmsf_avg_z"].values
            X_sasa = valid_all[["sasa_z"]].values
            X_joint = valid_all[["mean_abs_ddg_z", "sasa_z"]].values
            r2_s = LinearRegression().fit(X_sasa, y).score(X_sasa, y)
            r2_j = LinearRegression().fit(X_joint, y).score(X_joint, y)
            result.pooled_r2_joint_sasa = r2_j
            result.pooled_delta_r2_over_sasa = r2_j - r2_s

    return result


# ==========================================================================
# STRATIFIED ANALYSIS
# ==========================================================================

def run_stratified_analysis(
    per_protein_data: List[Tuple[pd.DataFrame, str]],
    stratify_col: str,
) -> Dict[str, Dict[str, float]]:
    """Run correlation analysis stratified by a categorical column.

    Args:
        per_protein_data: list of (merged_df, protein_id)
        stratify_col: column name to stratify by (e.g., "ss", "burial_class")

    Returns:
        Dict mapping category -> {rho_robustness_rmsf, rho_plddt_rmsf, n_residues}
    """
    all_rows = []
    for merged_df, pid in per_protein_data:
        df = merged_df.dropna(subset=["mean_abs_ddg", "rmsf_avg"]).copy()
        if stratify_col not in df.columns or len(df) < 5:
            continue
        for col in ["mean_abs_ddg", "rmsf_avg", "plddt"]:
            if col in df.columns:
                mu, sigma = df[col].mean(), df[col].std()
                df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0
        all_rows.append(df)

    if not all_rows:
        return {}

    pooled = pd.concat(all_rows, ignore_index=True)
    results = {}

    for cat, group in pooled.groupby(stratify_col):
        if len(group) < 20:
            continue
        entry = {"n_residues": len(group)}

        valid = group.dropna(subset=["mean_abs_ddg_z", "rmsf_avg_z"])
        if len(valid) >= 20:
            rho, pval = scipy_stats.spearmanr(valid["mean_abs_ddg_z"], valid["rmsf_avg_z"])
            entry["rho_robustness_rmsf"] = rho
            entry["pval_robustness_rmsf"] = pval

        if "plddt_z" in group.columns:
            valid_p = group.dropna(subset=["plddt_z", "rmsf_avg_z"])
            if len(valid_p) >= 20:
                rho, _ = scipy_stats.spearmanr(valid_p["plddt_z"], valid_p["rmsf_avg_z"])
                entry["rho_plddt_rmsf"] = rho

        results[str(cat)] = entry

    return results


# ==========================================================================
# FIGURE GENERATION
# ==========================================================================

def generate_figures(
    per_protein_results: List[PerProteinResult],
    per_protein_data: List[Tuple[pd.DataFrame, str]],
    pooled_result: PooledResult,
    stratified_ss: Dict,
    stratified_burial: Dict,
    output_dir: str,
    scorer: str,
):
    """Generate publication figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping figures")
        return

    fig_dir = Path(output_dir) / "figures" / scorer
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig A: Distribution of per-protein rho (robustness vs RMSF) ---
    rhos_rob = [r.rho_robustness_rmsf for r in per_protein_results
                if not np.isnan(r.rho_robustness_rmsf)]
    rhos_plddt = [r.rho_plddt_rmsf for r in per_protein_results
                  if not np.isnan(r.rho_plddt_rmsf)]

    if rhos_rob:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of rho values
        ax = axes[0]
        ax.hist(rhos_rob, bins=40, alpha=0.7, label=f"Robustness ({scorer})",
                color="steelblue", edgecolor="black", linewidth=0.5)
        if rhos_plddt:
            ax.hist(rhos_plddt, bins=40, alpha=0.5, label="pLDDT",
                    color="coral", edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Spearman rho (predictor vs. RMSF)")
        ax.set_ylabel("Number of proteins")
        ax.set_title("Per-protein correlation with dynamics")
        ax.legend()

        # Scatter: rho_robustness vs rho_plddt
        ax = axes[1]
        if rhos_plddt:
            both = [(r.rho_robustness_rmsf, r.rho_plddt_rmsf)
                    for r in per_protein_results
                    if not np.isnan(r.rho_robustness_rmsf)
                    and not np.isnan(r.rho_plddt_rmsf)]
            if both:
                x, y = zip(*both)
                ax.scatter(x, y, alpha=0.3, s=15, color="steelblue")
                lim = max(abs(min(min(x), min(y))), abs(max(max(x), max(y)))) + 0.1
                ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="y=x")
                ax.set_xlabel(f"rho (robustness vs RMSF)")
                ax.set_ylabel("rho (pLDDT vs RMSF)")
                ax.set_title("Robustness vs pLDDT as dynamics predictors")
                ax.legend()

        plt.tight_layout()
        plt.savefig(fig_dir / "per_protein_correlations.png", dpi=150)
        plt.close()

    # --- Fig B: Pooled scatter (z-scored) ---
    all_rows = []
    for merged_df, pid in per_protein_data:
        df = merged_df.dropna(subset=["mean_abs_ddg", "rmsf_avg"]).copy()
        if len(df) < 10:
            continue
        for col in ["mean_abs_ddg", "rmsf_avg"]:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"{col}_z"] = (df[col] - mu) / sigma if sigma > 0 else 0.0
        all_rows.append(df)

    if all_rows:
        pooled_df = pd.concat(all_rows, ignore_index=True)
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        # Subsample for plotting if too many points
        if len(pooled_df) > 10000:
            plot_df = pooled_df.sample(10000, random_state=42)
        else:
            plot_df = pooled_df

        ax.scatter(plot_df["mean_abs_ddg_z"], plot_df["rmsf_avg_z"],
                   alpha=0.05, s=3, color="steelblue")
        ax.set_xlabel("Per-residue robustness (mean |DDG|, z-scored)")
        ax.set_ylabel("RMSF from MD (z-scored)")
        ax.set_title(f"Pooled: robustness vs dynamics (N={len(pooled_df):,} residues, "
                     f"{pooled_result.n_proteins} proteins)\n"
                     f"rho={pooled_result.pooled_rho_robustness_rmsf:.3f}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(fig_dir / "pooled_scatter.png", dpi=150)
        plt.close()

    # --- Fig C: Stratified bar chart ---
    if stratified_ss:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        categories = sorted(stratified_ss.keys())
        labels = {"H": "Helix", "E": "Sheet", "C": "Coil"}
        x_labels = [labels.get(c, c) for c in categories]

        rho_rob = [stratified_ss[c].get("rho_robustness_rmsf", 0) for c in categories]
        rho_pld = [stratified_ss[c].get("rho_plddt_rmsf", 0) for c in categories]
        x = np.arange(len(categories))
        w = 0.35

        ax.bar(x - w/2, rho_rob, w, label=f"Robustness ({scorer})",
               color="steelblue")
        ax.bar(x + w/2, rho_pld, w, label="pLDDT", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Spearman rho with RMSF")
        ax.set_title("Correlation with dynamics by secondary structure")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(fig_dir / "stratified_ss.png", dpi=150)
        plt.close()

    print(f"Figures saved to {fig_dir}")


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correlate mutational robustness with protein dynamics (RMSF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--atlas_dir", type=str, required=True,
                        help="ATLAS download directory (from download_atlas.py)")
    parser.add_argument("--robustness_dir", type=str, required=True,
                        help="Robustness output directory (from compute_robustness.py)")
    parser.add_argument("--scorer", type=str, nargs="+", default=["esm1v"],
                        help="Scorer name(s) to analyze (default: esm1v)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for analysis results")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--no_dssp", action="store_true",
                        help="Skip DSSP-based secondary structure assignment")
    parser.add_argument("--max_proteins", type=int, default=0,
                        help="Limit number of proteins (0=all, for testing)")
    parser.add_argument("--max_seq_length", type=int, default=0,
                        help="Skip proteins longer than this (0=no limit, "
                             "1024=match ESM-1v limit for fair comparison)")
    args = parser.parse_args()

    for scorer in args.scorer:
        print(f"\n{'='*60}")
        print(f"ANALYSIS: scorer = {scorer}")
        print(f"{'='*60}")
        run_analysis_for_scorer(
            atlas_dir=args.atlas_dir,
            robustness_dir=args.robustness_dir,
            scorer=scorer,
            output_dir=args.output_dir,
            make_figures=not args.no_figures,
            use_dssp=not args.no_dssp,
            max_proteins=args.max_proteins,
            max_seq_length=args.max_seq_length,
        )


def run_analysis_for_scorer(
    atlas_dir: str,
    robustness_dir: str,
    scorer: str,
    output_dir: str,
    make_figures: bool = True,
    use_dssp: bool = True,
    max_proteins: int = 0,
    max_seq_length: int = 0,
):
    """Run the full correlation analysis for one scorer."""
    out_dir = Path(output_dir) / scorer
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find proteins that have both data and robustness data
    atlas_proteins_dir = Path(atlas_dir) / "proteins"
    dataset_label = Path(atlas_dir).name  # e.g. "atlas", "bbflow_processed"
    if not atlas_proteins_dir.exists():
        print(f"ERROR: proteins dir not found: {atlas_proteins_dir}")
        return

    protein_ids = sorted([
        d.name for d in atlas_proteins_dir.iterdir()
        if d.is_dir() and (d / ".done").exists()
    ])

    if max_proteins > 0:
        protein_ids = protein_ids[:max_proteins]

    if max_seq_length > 0:
        print(f"Found {len(protein_ids)} proteins in {dataset_label} "
              f"(max_seq_length={max_seq_length})")
    else:
        print(f"Found {len(protein_ids)} proteins in {dataset_label}")

    # Process each protein
    per_protein_results = []
    per_protein_data = []  # for pooled analysis
    n_skip_no_robustness = 0
    n_skip_no_rmsf = 0
    n_skip_too_short = 0
    n_skip_too_long = 0

    for idx, pid in enumerate(protein_ids):
        protein_dir = str(atlas_proteins_dir / pid)

        # Load robustness
        rob_df = load_robustness(robustness_dir, scorer, pid)
        if rob_df is None:
            n_skip_no_robustness += 1
            continue

        # Skip proteins exceeding max sequence length (e.g. ESM-1v 1024 limit)
        if max_seq_length > 0 and len(rob_df) > max_seq_length:
            n_skip_too_long += 1
            continue

        # Load RMSF
        rmsf_df = load_atlas_rmsf(protein_dir)
        if rmsf_df is None:
            n_skip_no_rmsf += 1
            continue

        # Load pLDDT and B-factor (optional)
        plddt_df = load_atlas_pldt(protein_dir)
        bfactor_df = load_atlas_bfactor(protein_dir)
        global_metrics = load_robustness_global(robustness_dir, scorer, pid)

        # Compute SASA from PDB (uniform burial baseline for all proteins)
        sasa_df = None
        pdb_files = list(Path(protein_dir).glob("*.pdb"))
        if pdb_files:
            sasa_df = compute_sasa_from_pdb(str(pdb_files[0]))

        # Correlate
        result = correlate_single_protein(
            pid, rob_df, rmsf_df, plddt_df, bfactor_df, global_metrics, scorer,
            sasa_df=sasa_df,
        )
        if result is None:
            n_skip_too_short += 1
            continue

        per_protein_results.append(result)

        # Build merged DataFrame for pooled analysis
        rob_merge_cols = ["position", "mean_abs_ddg", "mean_ddg"]
        for extra in ["frac_destabilizing", "frac_neutral", "std_ddg", "max_ddg"]:
            if extra in rob_df.columns:
                rob_merge_cols.append(extra)
        merged = rob_df[rob_merge_cols].copy()
        merged = merged.merge(rmsf_df[["position", "rmsf_avg"]], on="position", how="inner")
        if plddt_df is not None:
            merged = merged.merge(plddt_df[["position", "plddt"]], on="position", how="left")
        if bfactor_df is not None:
            merged = merged.merge(bfactor_df[["position", "bfactor"]], on="position", how="left")
        if sasa_df is not None:
            merged = merged.merge(sasa_df[["position", "sasa"]], on="position", how="left")

        # Secondary structure and burial (if enabled)
        if use_dssp:
            pdb_files = list(Path(protein_dir).glob("*.pdb"))
            if pdb_files:
                ss = assign_secondary_structure(str(pdb_files[0]))
                if ss and len(ss) == len(merged):
                    merged["ss"] = ss
                burial = compute_burial(str(pdb_files[0]))
                if burial and len(burial) == len(merged):
                    merged["rsa"] = burial
                    merged["burial_class"] = pd.cut(
                        merged["rsa"], bins=[0, 0.2, 0.5, 1.0],
                        labels=["core", "boundary", "surface"]
                    )
                elif "sasa" in merged.columns and merged["sasa"].notna().sum() >= 10:
                    # Fallback: SASA-based burial (quantile terciles)
                    terciles = merged["sasa"].quantile([1/3, 2/3])
                    merged["burial_class"] = pd.cut(
                        merged["sasa"],
                        bins=[-np.inf, terciles.iloc[0], terciles.iloc[1], np.inf],
                        labels=["core", "boundary", "surface"]
                    )

        per_protein_data.append((merged, pid))

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(protein_ids)}] {len(per_protein_results)} processed")

    print(f"\nProcessed: {len(per_protein_results)} proteins")
    print(f"Skipped: {n_skip_no_robustness} no robustness, "
          f"{n_skip_no_rmsf} no RMSF, {n_skip_too_short} too short, "
          f"{n_skip_too_long} too long (>{max_seq_length})")

    if not per_protein_results:
        print("No proteins to analyze!")
        return

    # --- Save per-protein results ---
    results_df = pd.DataFrame([asdict(r) for r in per_protein_results])
    results_df.to_csv(out_dir / "per_protein_correlations.tsv", sep="\t", index=False)
    print(f"Per-protein results: {out_dir / 'per_protein_correlations.tsv'}")

    # --- Pooled analysis ---
    pooled = run_pooled_analysis(per_protein_data, per_protein_results, scorer)
    with open(out_dir / "pooled_results.json", "w") as f:
        json.dump(asdict(pooled), f, indent=2, default=_json_default)
    print(f"Pooled results: {out_dir / 'pooled_results.json'}")

    # --- Stratified analysis ---
    strat_ss = run_stratified_analysis(per_protein_data, "ss")
    strat_burial = run_stratified_analysis(per_protein_data, "burial_class")
    stratified = {"secondary_structure": strat_ss, "burial": strat_burial}
    with open(out_dir / "stratified_results.json", "w") as f:
        json.dump(stratified, f, indent=2, default=_json_default)
    print(f"Stratified results: {out_dir / 'stratified_results.json'}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY ({scorer})")
    print(f"{'='*60}")
    print(f"Proteins analyzed:  {pooled.n_proteins}")
    print(f"Residues pooled:    {pooled.n_residues:,}")
    print(f"")
    print(f"Per-protein median rho (robustness vs RMSF): "
          f"{pooled.median_rho_robustness_rmsf:.3f}")
    print(f"Per-protein median rho (pLDDT vs RMSF):      "
          f"{pooled.median_rho_plddt_rmsf:.3f}")
    print(f"Per-protein median rho (SASA vs RMSF):       "
          f"{pooled.median_rho_sasa_rmsf:.3f}")
    print(f"Per-protein median partial rho (rob|SASA):   "
          f"{pooled.median_rho_partial_sasa:.3f}")
    print(f"Frac where |rho_robustness| > |rho_pLDDT|:   "
          f"{pooled.frac_robustness_beats_plddt:.3f}")

    # Alternative robustness measures summary
    alt_measures = {
        "frac_destab": "rho_frac_destab_rmsf",
        "frac_neutral": "rho_frac_neutral_rmsf",
        "std_ddg": "rho_std_ddg_rmsf",
        "max_ddg": "rho_max_ddg_rmsf",
    }
    alt_medians = {}
    for label, attr in alt_measures.items():
        vals = [getattr(r, attr) for r in per_protein_results
                if not np.isnan(getattr(r, attr))]
        if vals:
            alt_medians[label] = np.median(vals)
    if alt_medians:
        print(f"\nAlternative robustness measures (median rho vs RMSF):")
        for label, med in sorted(alt_medians.items(), key=lambda x: -abs(x[1])):
            print(f"  {label:20s}: {med:.3f}")
    print(f"")
    print(f"Pooled rho (robustness vs RMSF):  {pooled.pooled_rho_robustness_rmsf:.3f} "
          f"(p={pooled.pooled_pval_robustness_rmsf:.2e})")
    print(f"Pooled rho (pLDDT vs RMSF):       {pooled.pooled_rho_plddt_rmsf:.3f}")
    print(f"Pooled rho (SASA vs RMSF):        {pooled.pooled_rho_sasa_rmsf:.3f}")
    print(f"Pooled partial rho (rob|SASA):    {pooled.pooled_rho_robustness_rmsf_partial_sasa:.3f}")
    print(f"Pooled R^2 (robustness):           {pooled.pooled_r2_robustness_rmsf:.3f}")
    print(f"Pooled R^2 (pLDDT):                {pooled.pooled_r2_plddt_rmsf:.3f}")
    print(f"Pooled R^2 (SASA):                 {pooled.pooled_r2_sasa_rmsf:.3f}")
    print(f"Pooled R^2 (rob+pLDDT):            {pooled.pooled_r2_joint:.3f}")
    print(f"Pooled R^2 (rob+SASA):             {pooled.pooled_r2_joint_sasa:.3f}")
    print(f"Delta R^2 (rob+pLDDT - pLDDT):     {pooled.pooled_delta_r2:.3f}")
    print(f"Delta R^2 (rob+SASA - SASA):       {pooled.pooled_delta_r2_over_sasa:.3f}")

    if strat_ss:
        print(f"\nBy secondary structure:")
        for cat in sorted(strat_ss.keys()):
            d = strat_ss[cat]
            label = {"H": "Helix", "E": "Sheet", "C": "Coil"}.get(cat, cat)
            print(f"  {label:8s}: rho_rob={d.get('rho_robustness_rmsf', float('nan')):.3f}  "
                  f"rho_plddt={d.get('rho_plddt_rmsf', float('nan')):.3f}  "
                  f"n={d.get('n_residues', 0):,}")

    # --- Figures ---
    if make_figures:
        generate_figures(
            per_protein_results, per_protein_data, pooled,
            strat_ss, strat_burial, output_dir, scorer
        )


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    main()
