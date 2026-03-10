#!/usr/bin/env python3
"""
Multi-dimensional DDG regression: predict dynamics (RMSF or B-factor)
using the full per-residue DDG profile (20 target-AA features) instead
of a single robustness summary statistic.

Each residue position i has a 20-dimensional feature vector:
  x_i = [DDG(S_i -> A), DDG(S_i -> C), ..., DDG(S_i -> Y)]
where the self-mutation entry DDG(S_i -> S_i) = 0.

This gives each column a consistent meaning across all positions
("cost of mutating TO alanine") regardless of the wild-type amino acid.

We compare:
  1. Ridge regression with full 20-DDG features
  2. Single-feature baselines (mean|DDG|, pLDDT, SASA)
  3. Combined: 20-DDG + pLDDT
All evaluated via protein-level k-fold cross-validation.

Usage:
  python multi_ddg_regression.py \
      --atlas_dir /path/to/atlas \
      --robustness_dir /path/to/robustness \
      --scorer thermompnn \
      --output_dir /path/to/output \
      --target rmsf          # or bfactor
      --n_folds 5
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Canonical amino acid ordering (matches compute_robustness.py)
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
N_AA = len(AA_LIST)


# ======================================================================
# DATA LOADING
# ======================================================================

def load_ddg_matrix_20col(robustness_dir: str, scorer: str,
                          protein_id: str) -> Optional[Tuple[np.ndarray, str]]:
    """Load L x 19 DDG matrix and expand to L x 20 (with 0 for self-mutation).

    Returns (ddg_20, sequence) where ddg_20 is shape (L, 20).
    """
    npy_path = Path(robustness_dir) / scorer / f"{protein_id}_ddg_matrix.npy"
    json_path = Path(robustness_dir) / scorer / f"{protein_id}_robustness.json"

    if not npy_path.exists() or not json_path.exists():
        return None

    ddg_19 = np.load(str(npy_path))  # (L, 19)
    with open(json_path) as f:
        meta = json.load(f)

    seq = meta.get("sequence", "")
    if not seq or len(seq) != ddg_19.shape[0]:
        return None

    L = len(seq)
    ddg_20 = np.zeros((L, N_AA), dtype=np.float32)

    for i in range(L):
        wt_aa = seq[i]
        wt_idx = AA_TO_IDX.get(wt_aa)
        if wt_idx is None:
            ddg_20[i, :] = np.nan
            continue
        # Reconstruct 20-column from 19-column by inserting 0 at wt position
        col = 0
        for j in range(N_AA):
            if j == wt_idx:
                ddg_20[i, j] = 0.0  # self-mutation
            else:
                ddg_20[i, j] = ddg_19[i, col]
                col += 1

    return ddg_20, seq


def load_atlas_column(protein_dir: str, suffix: str,
                      col_name: str) -> Optional[np.ndarray]:
    """Load a single column from an ATLAS TSV file."""
    protein_dir = Path(protein_dir)
    matches = list(protein_dir.glob(f"*{suffix}"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    # Find the right column
    candidates = [c for c in df.columns if col_name.lower() in c.lower()]
    if not candidates:
        numeric = [c for c in df.columns
                   if df[c].dtype in (np.float64, np.float32, float)]
        candidates = numeric[-1:] if numeric else []
    if not candidates:
        return None
    return df[candidates[0]].values


def load_rmsf(protein_dir: str) -> Optional[np.ndarray]:
    """Load RMSF, averaged across replicates."""
    protein_dir = Path(protein_dir)
    matches = list(protein_dir.glob("*_RMSF.tsv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep="\t")
    rmsf_cols = [c for c in df.columns if "rmsf" in c.lower()
                 or "r1" in c.lower() or "r2" in c.lower()
                 or "r3" in c.lower()]
    if not rmsf_cols:
        rmsf_cols = [c for c in df.columns
                     if df[c].dtype in (np.float64, np.float32, float)]
    if not rmsf_cols:
        return None
    return df[rmsf_cols].mean(axis=1).values


def compute_sasa(pdb_path: str) -> Optional[np.ndarray]:
    """Compute per-residue SASA using mdtraj."""
    try:
        import mdtraj
        traj = mdtraj.load(pdb_path)
        sasa_per_atom = mdtraj.shrake_rupley(traj, mode='atom')
        sasa_per_residue = np.zeros(traj.topology.n_residues)
        for atom in traj.topology.atoms:
            sasa_per_residue[atom.residue.index] += sasa_per_atom[0, atom.index]
        return sasa_per_residue
    except Exception:
        return None


# ======================================================================
# REGRESSION
# ======================================================================

@dataclass
class RegressionResult:
    """Results from cross-validated regression."""
    model_name: str
    n_features: int
    n_proteins_train: int
    n_proteins_test: int
    n_residues_train: int
    n_residues_test: int

    # Cross-validated metrics (mean +/- std across folds)
    cv_r2_mean: float = np.nan
    cv_r2_std: float = np.nan
    cv_rho_mean: float = np.nan
    cv_rho_std: float = np.nan

    # Per-protein CV metrics
    cv_per_protein_rho_median: float = np.nan
    cv_per_protein_rho_mean: float = np.nan

    # Feature importance (for multi-DDG model)
    feature_names: List[str] = None
    feature_coefs_mean: List[float] = None


def build_dataset(
    atlas_dir: str,
    robustness_dir: str,
    scorer: str,
    target: str,
    max_seq_length: int = 0,
    max_proteins: int = 0,
) -> Tuple[List[dict], List[str]]:
    """Build list of per-protein data dicts for regression.

    Each dict has: ddg_20 (L,20), target (L,), plddt (L,), sasa (L,),
                   mean_abs_ddg (L,), protein_id, seq_length.
    """
    atlas_proteins_dir = Path(atlas_dir) / "proteins"
    protein_ids = sorted([
        d.name for d in atlas_proteins_dir.iterdir()
        if d.is_dir() and (d / ".done").exists()
    ])

    if max_proteins > 0:
        protein_ids = protein_ids[:max_proteins]

    dataset = []
    skipped = {"no_ddg": 0, "no_target": 0, "too_long": 0, "too_short": 0,
               "length_mismatch": 0}

    for pid in protein_ids:
        protein_dir = str(atlas_proteins_dir / pid)

        # Load DDG matrix
        result = load_ddg_matrix_20col(robustness_dir, scorer, pid)
        if result is None:
            skipped["no_ddg"] += 1
            continue

        ddg_20, seq = result
        L = len(seq)

        if max_seq_length > 0 and L >= max_seq_length:
            skipped["too_long"] += 1
            continue

        # Load target
        if target == "rmsf":
            y = load_rmsf(protein_dir)
        elif target == "bfactor":
            y = load_atlas_column(protein_dir, "_Bfactor.tsv", "bfactor")
        else:
            raise ValueError(f"Unknown target: {target}")

        if y is None:
            skipped["no_target"] += 1
            continue

        if len(y) != L:
            skipped["length_mismatch"] += 1
            continue

        # Load baselines
        plddt = load_atlas_column(protein_dir, "_pLDDT.tsv", "plddt")
        if plddt is not None and len(plddt) != L:
            plddt = None

        pdb_files = list(Path(protein_dir).glob("*.pdb"))
        sasa = compute_sasa(str(pdb_files[0])) if pdb_files else None
        if sasa is not None and len(sasa) != L:
            sasa = None

        # mean_abs_ddg from robustness TSV
        rob_tsv = Path(robustness_dir) / scorer / f"{pid}_robustness.tsv"
        mean_abs_ddg = None
        if rob_tsv.exists():
            rob_df = pd.read_csv(rob_tsv, sep="\t")
            if "mean_abs_ddg" in rob_df.columns and len(rob_df) == L:
                mean_abs_ddg = rob_df["mean_abs_ddg"].values

        # Filter valid rows (no NaN in DDG or target)
        valid = ~(np.isnan(ddg_20).any(axis=1) | np.isnan(y))
        n_valid = valid.sum()
        if n_valid < 10:
            skipped["too_short"] += 1
            continue

        entry = {
            "protein_id": pid,
            "seq_length": L,
            "ddg_20": ddg_20[valid],
            "target": y[valid],
            "plddt": plddt[valid] if plddt is not None else None,
            "sasa": sasa[valid] if sasa is not None else None,
            "mean_abs_ddg": mean_abs_ddg[valid] if mean_abs_ddg is not None else None,
            "n_residues": int(n_valid),
        }
        dataset.append(entry)

    print(f"Loaded {len(dataset)} proteins for {scorer}/{target}")
    print(f"Skipped: {skipped}")

    return dataset


def run_cv_regression(
    dataset: List[dict],
    n_folds: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
) -> Dict[str, RegressionResult]:
    """Run protein-level k-fold CV for multiple models.

    Models compared:
      1. ridge_20ddg: Ridge on 20 DDG features
      2. ridge_20ddg_plddt: Ridge on 20 DDG + pLDDT
      3. ols_mean_abs_ddg: OLS on mean|DDG| (1 feature)
      4. ols_plddt: OLS on pLDDT (1 feature)
      5. ols_mean_abs_ddg_plddt: OLS on mean|DDG| + pLDDT (2 features)
      6. ols_sasa: OLS on SASA (1 feature)
    """
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import KFold
    from scipy import stats as scipy_stats

    np.random.seed(seed)
    n = len(dataset)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Protein indices
    indices = np.arange(n)

    # --- Helper: compute nonlinear summary features from DDG matrix ---
    def ddg_nonlinear_features(ddg_20):
        """Compute nonlinear summary statistics from L x 20 DDG matrix.

        Returns L x 4 array: [std_ddg, mean_abs_ddg, max_abs_ddg, min_ddg].
        These capture landscape shape properties (variance, extremes) that
        a linear combination of the 20 raw DDG values cannot represent.
        """
        # Mask out the self-mutation (0 values) for correct statistics
        ddg_masked = ddg_20.copy()
        ddg_masked[ddg_masked == 0] = np.nan
        std_ddg = np.nanstd(ddg_masked, axis=1)
        mean_abs = np.nanmean(np.abs(ddg_masked), axis=1)
        max_abs = np.nanmax(np.abs(ddg_masked), axis=1)
        min_ddg = np.nanmin(ddg_masked, axis=1)
        return np.column_stack([std_ddg, mean_abs, max_abs, min_ddg])

    nonlinear_names = ["std_ddg", "mean|DDG|", "max|DDG|", "min_ddg"]

    # --- Model definitions: name -> (feature_extractor, regularized) ---
    def extract_20ddg(entry):
        return entry["ddg_20"]

    def extract_20ddg_nonlinear(entry):
        """20 raw DDG + 4 nonlinear summary features = 24 features.
        This hybrid model can capture both per-AA-specific effects AND
        landscape shape (variance, extremes) that pure linear models miss.
        """
        nl = ddg_nonlinear_features(entry["ddg_20"])
        return np.column_stack([entry["ddg_20"], nl])

    def extract_20ddg_nonlinear_plddt(entry):
        """24 DDG features + pLDDT = 25 features."""
        if entry["plddt"] is None:
            return None
        nl = ddg_nonlinear_features(entry["ddg_20"])
        return np.column_stack([entry["ddg_20"], nl, entry["plddt"]])

    def extract_nonlinear_only(entry):
        """4 nonlinear DDG summary features only (no per-AA detail)."""
        return ddg_nonlinear_features(entry["ddg_20"])

    def extract_20ddg_plddt(entry):
        if entry["plddt"] is None:
            return None
        return np.column_stack([entry["ddg_20"], entry["plddt"]])

    def extract_mean_abs_ddg(entry):
        if entry["mean_abs_ddg"] is None:
            return None
        return entry["mean_abs_ddg"].reshape(-1, 1)

    def extract_plddt(entry):
        if entry["plddt"] is None:
            return None
        return entry["plddt"].reshape(-1, 1)

    def extract_mean_plddt(entry):
        if entry["mean_abs_ddg"] is None or entry["plddt"] is None:
            return None
        return np.column_stack([entry["mean_abs_ddg"], entry["plddt"]])

    def extract_sasa(entry):
        if entry["sasa"] is None:
            return None
        return entry["sasa"].reshape(-1, 1)

    models = {
        "ridge_20ddg": {
            "extractor": extract_20ddg,
            "use_ridge": True,
            "feature_names": list(AA_LIST),
            "n_features": 20,
        },
        "ridge_20ddg_nonlinear": {
            "extractor": extract_20ddg_nonlinear,
            "use_ridge": True,
            "feature_names": list(AA_LIST) + nonlinear_names,
            "n_features": 24,
        },
        "ridge_20ddg_nonlinear_plddt": {
            "extractor": extract_20ddg_nonlinear_plddt,
            "use_ridge": True,
            "feature_names": list(AA_LIST) + nonlinear_names + ["pLDDT"],
            "n_features": 25,
        },
        "ridge_nonlinear_only": {
            "extractor": extract_nonlinear_only,
            "use_ridge": True,
            "feature_names": nonlinear_names,
            "n_features": 4,
        },
        "ridge_20ddg_plddt": {
            "extractor": extract_20ddg_plddt,
            "use_ridge": True,
            "feature_names": list(AA_LIST) + ["pLDDT"],
            "n_features": 21,
        },
        "ols_mean_abs_ddg": {
            "extractor": extract_mean_abs_ddg,
            "use_ridge": False,
            "feature_names": ["mean|DDG|"],
            "n_features": 1,
        },
        "ols_plddt": {
            "extractor": extract_plddt,
            "use_ridge": False,
            "feature_names": ["pLDDT"],
            "n_features": 1,
        },
        "ols_mean_plddt": {
            "extractor": extract_mean_plddt,
            "use_ridge": False,
            "feature_names": ["mean|DDG|", "pLDDT"],
            "n_features": 2,
        },
        "ols_sasa": {
            "extractor": extract_sasa,
            "use_ridge": False,
            "feature_names": ["SASA"],
            "n_features": 1,
        },
    }

    results = {}

    for model_name, model_def in models.items():
        print(f"\n  Model: {model_name}")
        extractor = model_def["extractor"]
        use_ridge = model_def["use_ridge"]

        fold_r2s = []
        fold_rhos = []
        fold_per_protein_rhos = []
        fold_coefs = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
            # Collect train/test residues
            X_train_parts, y_train_parts = [], []
            X_test_parts, y_test_parts = [], []
            test_protein_boundaries = []  # for per-protein rho

            for idx in train_idx:
                entry = dataset[idx]
                X = extractor(entry)
                if X is None:
                    continue
                X_train_parts.append(X)
                y_train_parts.append(entry["target"])

            offset = 0
            for idx in test_idx:
                entry = dataset[idx]
                X = extractor(entry)
                if X is None:
                    continue
                X_test_parts.append(X)
                y_test_parts.append(entry["target"])
                n_res = len(entry["target"])
                test_protein_boundaries.append((offset, offset + n_res,
                                                entry["protein_id"]))
                offset += n_res

            if not X_train_parts or not X_test_parts:
                continue

            X_train = np.vstack(X_train_parts)
            y_train = np.concatenate(y_train_parts)
            X_test = np.vstack(X_test_parts)
            y_test = np.concatenate(y_test_parts)

            # Z-score features and target using TRAIN statistics
            X_mean = np.nanmean(X_train, axis=0)
            X_std = np.nanstd(X_train, axis=0)
            X_std[X_std < 1e-10] = 1.0
            y_mean = np.mean(y_train)
            y_std = np.std(y_train)
            if y_std < 1e-10:
                continue

            X_train_z = (X_train - X_mean) / X_std
            X_test_z = (X_test - X_mean) / X_std
            y_train_z = (y_train - y_mean) / y_std
            y_test_z = (y_test - y_mean) / y_std

            # Replace any remaining NaN with 0
            X_train_z = np.nan_to_num(X_train_z, nan=0.0)
            X_test_z = np.nan_to_num(X_test_z, nan=0.0)

            # Fit
            if use_ridge:
                reg = Ridge(alpha=alpha).fit(X_train_z, y_train_z)
            else:
                reg = LinearRegression().fit(X_train_z, y_train_z)

            # Evaluate
            y_pred = reg.predict(X_test_z)
            r2 = 1 - np.sum((y_test_z - y_pred)**2) / np.sum((y_test_z - np.mean(y_test_z))**2)
            rho, _ = scipy_stats.spearmanr(y_test_z, y_pred)

            fold_r2s.append(r2)
            fold_rhos.append(rho)
            fold_coefs.append(reg.coef_.copy())

            # Per-protein rho on test set
            pp_rhos = []
            for start, end, pid in test_protein_boundaries:
                if end - start < 10:
                    continue
                rho_pp, _ = scipy_stats.spearmanr(y_test_z[start:end],
                                                   y_pred[start:end])
                if not np.isnan(rho_pp):
                    pp_rhos.append(rho_pp)
            fold_per_protein_rhos.extend(pp_rhos)

        if not fold_r2s:
            continue

        # Aggregate
        n_train_total = sum(len(dataset[i]["target"])
                            for i in range(n) if extractor(dataset[i]) is not None)
        res = RegressionResult(
            model_name=model_name,
            n_features=model_def["n_features"],
            n_proteins_train=int(n * (n_folds - 1) / n_folds),
            n_proteins_test=int(n / n_folds),
            n_residues_train=0,  # approximate
            n_residues_test=0,
            cv_r2_mean=float(np.mean(fold_r2s)),
            cv_r2_std=float(np.std(fold_r2s)),
            cv_rho_mean=float(np.mean(fold_rhos)),
            cv_rho_std=float(np.std(fold_rhos)),
            feature_names=model_def["feature_names"],
        )

        if fold_per_protein_rhos:
            res.cv_per_protein_rho_median = float(np.median(fold_per_protein_rhos))
            res.cv_per_protein_rho_mean = float(np.mean(fold_per_protein_rhos))

        if fold_coefs:
            res.feature_coefs_mean = [float(x) for x in np.mean(fold_coefs, axis=0)]

        print(f"    CV R²: {res.cv_r2_mean:.4f} ± {res.cv_r2_std:.4f}")
        print(f"    CV rho: {res.cv_rho_mean:.4f} ± {res.cv_rho_std:.4f}")
        print(f"    Per-protein median rho: {res.cv_per_protein_rho_median:.4f}")

        if res.feature_coefs_mean and len(res.feature_coefs_mean) <= 25:
            print(f"    Coefficients:")
            for name, coef in zip(res.feature_names, res.feature_coefs_mean):
                print(f"      {name:6s}: {coef:+.4f}")

        results[model_name] = res

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-DDG regression for dynamics prediction",
    )
    parser.add_argument("--atlas_dir", type=str, required=True)
    parser.add_argument("--robustness_dir", type=str, required=True)
    parser.add_argument("--scorer", type=str, default="thermompnn")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target", type=str, default="rmsf",
                        choices=["rmsf", "bfactor"],
                        help="Target variable to predict")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength")
    parser.add_argument("--max_seq_length", type=int, default=0)
    parser.add_argument("--max_proteins", type=int, default=0)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Multi-DDG Regression: {args.scorer} -> {args.target}")
    print(f"{'='*60}")

    dataset = build_dataset(
        atlas_dir=args.atlas_dir,
        robustness_dir=args.robustness_dir,
        scorer=args.scorer,
        target=args.target,
        max_seq_length=args.max_seq_length,
        max_proteins=args.max_proteins,
    )

    if not dataset:
        print("No data loaded!")
        return

    print(f"\nRunning {args.n_folds}-fold protein-level CV "
          f"(alpha={args.alpha})...")
    results = run_cv_regression(
        dataset,
        n_folds=args.n_folds,
        alpha=args.alpha,
    )

    # Save results
    out_dir = Path(args.output_dir) / args.scorer
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"multi_ddg_{args.target}_results.json"
    serializable = {k: asdict(v) for k, v in results.items()}
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and np.isnan(x) else x)
    print(f"\nResults saved to {out_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"COMPARISON TABLE ({args.scorer} -> {args.target})")
    print(f"{'='*60}")
    print(f"{'Model':<25s} {'n_feat':>6s} {'CV R²':>12s} "
          f"{'CV rho':>12s} {'PP med rho':>10s}")
    print("-" * 70)
    for name in ["ols_plddt", "ols_mean_abs_ddg", "ols_sasa",
                  "ols_mean_plddt", "ridge_nonlinear_only",
                  "ridge_20ddg", "ridge_20ddg_plddt",
                  "ridge_20ddg_nonlinear", "ridge_20ddg_nonlinear_plddt"]:
        if name not in results:
            continue
        r = results[name]
        print(f"{r.model_name:<25s} {r.n_features:>6d} "
              f"{r.cv_r2_mean:>6.4f}±{r.cv_r2_std:.4f} "
              f"{r.cv_rho_mean:>6.4f}±{r.cv_rho_std:.4f} "
              f"{r.cv_per_protein_rho_median:>10.4f}")


if __name__ == "__main__":
    main()
