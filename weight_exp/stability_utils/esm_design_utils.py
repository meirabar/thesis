#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/bin/bash

# In[1]:

"""
esm_design_utils.py
Utilities to run the existing design + evaluation scripts (keeps them unchanged),
then extract the numeric fields you requested.

Exports:
    process_pdb(pdb_id, output_root, python_bin="python", lambda_dg=1.0, lambda_ddg=1.0, gamma_hessian=0.0, force=True)
        -> returns a dict with keys:
           ['pdb id','DG esm WT','DDG esm WT','DG esm Design','DDG esm Design','WT loss','Design loss','summary_path']
"""
import os
import json
import subprocess
from pathlib import Path
import time
import pandas as pd
import ast


import sys
sys.path.append("/sci/labs/orzuk/meirab/stability_algo/weight0/stability_utils")
from esm_utils import *

# default paths 
EVAL_ROOT = Path("/sci/labs/orzuk/meirab/stability_algo/evaluation")
FULL_OUTPUT_ROOT = Path("/sci/labs/orzuk/meirab/stability_algo/full_pipeline_outputs")
PER_PDB_OUT = FULL_OUTPUT_ROOT / "per_pdb"

# ensure dirs
PER_PDB_OUT.mkdir(parents=True, exist_ok=True)

def _try_load_summary(eval_dir: Path):
    """
    Load evaluation summary from a text file (e.g., evaluation_summary.txt or *summary*.txt).
    Returns (parsed_dict, Path_to_file) or (None, None) if nothing found/parsable.
    Parsing rules:
      - Lines like "key: value" are parsed.
      - Numeric values coerced to int/float when possible.
      - Python-like lists/tuples/dicts parsed with ast.literal_eval.
      - Comma-separated integer lists like "1,2,3" are parsed to [1,2,3].
    """
    # look for summary text files only
    for p in sorted(Path(eval_dir).glob("*summary*.txt")):
        try:
            txt = p.read_text()
        except Exception:
            continue

        parsed = {}
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ":" not in ln:
                continue
            key, val = ln.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val == "" or val.lower() in ("none", "null", "nan"):
                parsed_val = None
            else:
                # try literal eval for list/tuple/dict/numeric-like strings
                parsed_val = None
                try:
                    # if it looks like a python literal: try literal_eval
                    if val.startswith(("(", "[", "{")):
                        parsed_val = ast.literal_eval(val)
                    else:
                        # try int
                        try:
                            parsed_val = int(val)
                        except Exception:
                            # try float
                            try:
                                parsed_val = float(val)
                            except Exception:
                                # comma-separated numeric list: "1,2,3" -> [1,2,3]
                                if "," in val and all(s.strip().lstrip("-").replace(".","",1).isdigit() for s in val.split(",")):
                                    pieces = [s.strip() for s in val.split(",")]
                                    # decide ints vs floats
                                    if all(p.isdigit() or (p.startswith("-") and p[1:].isdigit()) for p in pieces):
                                        parsed_val = [int(p) for p in pieces]
                                    else:
                                        parsed_val = [float(p) for p in pieces]
                                else:
                                    parsed_val = val
                except Exception:
                    parsed_val = val
            parsed[key] = parsed_val

        # heuristics: only return if we found at least one useful key
        if any(k in parsed for k in ("pdb_id", "wt_dg", "design_dg", "wt_overall_ddg", "design_overall_ddg","wt_loss", "design_loss")):
            return parsed, p

    # nothing found
    return None, None

def _derive_from_ddg_csv(eval_dir: Path, pdb_id: str):
    """
    If summary json not available, try to compute summary from ddg csv and/or recompute DGs.
    Returns dict (may have partial data).
    """
    out = {}
    csv_path = eval_dir / f"{pdb_id}_ddg_landscape_weight0.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Try common column names
            if "ddg" in df.columns:
                out["wt_overall_ddg"] = float(df["ddg"].mean())
            else:
                # take mean of all numeric columns except first (index)
                num = df.select_dtypes("number")
                if not num.empty:
                    out["wt_overall_ddg"] = float(num.values.mean())
        except Exception:
            pass
    return out

def process_pdb(
    pdb_id: str,
    weight_exp: str = "weight_exp",
    output_root: str = str(FULL_OUTPUT_ROOT),
    python_bin: str = "python",
    dg_mean: float = 0,
    dg_std: float = 0,
    ddg_mean: float = 0,
    ddg_std: float = 0,
    lambda_dg: float = 1.0,
    lambda_ddg: float = 1.0,
    gamma_hessian: float = 0.0,
    force: bool = True,
    run_pipeline_cmd: str = (
        "{python_bin} run_pipeline_and_design.py "
        "--pdb_id {pdb} "
        "--lambda_dg {lambda_dg} "
        "--lambda_ddg {lambda_ddg}"
        "--weight_exp {weight_exp}"

    ),
    run_eval_cmd: str = (
        "{python_bin} run_evaluation_pipeline.py "
        "--pdb_id {pdb} "
        "--lambda_dg {lambda_dg} "
        "--lambda_ddg {lambda_ddg}"
        "--weight_exp {weight_exp}"
    ),
):   
    """
    Runs design pipeline and evaluation for a single pdb_id and returns a dict with columns requested.

    - run_pipeline_cmd and run_eval_cmd are format strings with {pdb} substitution.
    - This function writes the per-pdb JSON into FULL_OUTPUT_ROOT/per_pdb/{pdb}.json.
    - If the evaluation already produced a summary JSON in the evaluation dir, we use it.
    """
    pdb = pdb_id.strip().upper()
    out_root = Path(output_root)
    per_pdb_dir = out_root / "per_pdb"
    per_pdb_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = Path("/sci/labs/orzuk/meirab/stability_algo/evaluation/")/ f"{pdb}_{weight_exp}" 
    # 1) run design pipeline (unless we decide to skip when per-pdb exists and not force)
    per_pdb_json = per_pdb_dir / f"{pdb}_{weight_exp}_dg{lambda_dg}_ddg{lambda_ddg}.json"

    # If forcing, remove existing per-pdb json so we always rerun
    if per_pdb_json.exists() and force:
        try:
            per_pdb_json.unlink()
            print(f"[INFO] Removed existing per-pdb JSON to force rerun: {per_pdb_json}")
        except Exception as e:
            print(f"[WARN] Could not remove {per_pdb_json}: {e}. Continuing and will overwrite if possible.")

       # Run design pipeline
    try:
        cmd = run_pipeline_cmd.format(
            python_bin=python_bin,
            pdb=pdb,
            lambda_dg=lambda_dg,
            lambda_ddg=lambda_ddg,
            weight_exp=weight_exp,
        ).split()
        print(f"[run_pipeline] CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Design pipeline failed for {pdb}: {e}")
    # Run evaluation pipeline
        # Run evaluation pipeline
    try:
        cmd = run_eval_cmd.format(
            python_bin=python_bin,
            pdb=pdb,
            lambda_dg=lambda_dg,
            lambda_ddg=lambda_ddg,
            weight_exp=weight_exp,
        ).split()
        print(f"[run_eval] CMD: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation pipeline failed for {pdb}: {e}")


    time.sleep(1)

    # Attempt to load summary from evaluation folder
    summary_data, summary_path = _try_load_summary(eval_dir)
    derived = {}
    if not summary_data:
        # Try to derive minimal info from ddg CSV
        derived = _derive_from_ddg_csv(eval_dir, pdb)
    else:
        derived = summary_data

    # canonical keys and defaults
    wt_dg = derived.get("wt_dg") or derived.get("wt_dg_val") or derived.get("wt_dg_value")
    design_dg = derived.get("design_dg") or derived.get("design_dg_val") or derived.get("design_dg_value")
    wt_ddg = derived.get("wt_overall_ddg") or derived.get("overall_ddg_wt") or derived.get("wt_mean_ddg")
    design_ddg = derived.get("design_overall_ddg") or derived.get("overall_ddg_design") or derived.get("design_mean_ddg")
    wt_loss = derived.get("wt_loss") 
    design_loss = derived.get("design_loss")

    # If any of the DGs are missing, try to compute quickly using esm_utils if available
    if (wt_dg is None or design_dg is None) and compute_dg_esm1v is not None:
        try:
            # fetch sequences from saved files used by your scripts
            # Attempt to read greedy output file path
            beam_path = Path(f"/sci/labs/orzuk/meirab/stability_algo/beam_outputs/{pdb_id}_beam_best_{exp_weights}.txt")
            if greedy_path.exists():
                design_seq = beam_path.read_text().strip()
            else:
                design_seq = None

            # Use evaluate_utils.fetch_and_extract_sequence if present; fallback: None
            try:
                sys.path.append("/sci/labs/orzuk/meirab/stability_algo/stability_utils")
                from evaluate_utils import fetch_and_extract_sequence
                wt_seq = fetch_and_extract_sequence(pdb, "A")
            except Exception:
                wt_seq = None

            if wt_dg is None and wt_seq and compute_dg_esm1v is not None:
                wt_dg = float(compute_dg_esm1v(wt_seq))
            if design_dg is None and design_seq and compute_dg_esm1v is not None:
                design_dg = float(compute_dg_esm1v(design_seq))
        except Exception:
            pass

    # If ddgs are missing and full_compute_ddg_esm1v exists and we have sequences, compute them (expensive)
    if (wt_ddg is None or design_ddg is None) and full_compute_ddg_esm1v is not None:
        try:
            if 'wt_seq' not in locals():
                try:
                    from evaluate_utils import fetch_and_extract_sequence
                    wt_seq = fetch_and_extract_sequence(pdb, "A")
                except Exception:
                    wt_seq = None
            if 'design_seq' not in locals():
                beam_path = Path(f"/sci/labs/orzuk/meirab/stability_algo/beam_outputs/{pdb_id}_beam_best_{exp_weights}.txt")

                design_seq = beam_path.read_text().strip() if greedy_path.exists() else None

            if wt_ddg is None and wt_seq:
                wt_ddg, _ = full_compute_ddg_esm1v(wt_seq, None, None, verbose=False)
            if design_ddg is None and design_seq:
                design_ddg, _ = full_compute_ddg_esm1v(design_seq, None, None, verbose=False)
        except Exception:
            pass

    # Compute losses with compute_loss if available, otherwise set None
    if (wt_loss is None or design_loss is None) and compute_loss is not None:
        try:
            # use ddg (mean overall) and hess=0 as default
            wt_loss = float(compute_loss(dg= wt_dg, ddg=overall_ddg_wt, hess=0, dg_mean=dg_mean, dg_std=dg_std, ddg_mean=ddg_mean,
                                         ddg_std=ddg_std, lambda_dg = lambda_dg, lambda_ddg = lambda_ddg ))
            design_loss = float(compute_loss(dg= design_dg, ddg=design_ddg, hess=0, dg_mean=dg_mean, dg_std=dg_std, ddg_mean=ddg_mean,
                                         ddg_std=ddg_std, lambda_dg = lambda_dg, lambda_ddg = lambda_ddg ))
        except Exception:
            wt_loss = None
            design_loss = None
    # Define pdb_path (fix for NameError)
    pdb_path = f"/sci/labs/orzuk/meirab/stability_algo/data/full_pdbs/{pdb.lower()}.pdb"
    seq_len = get_sequence_length(pdb_path, "A")
    
    out = {
            "pdb id": pdb,
            "DG esm WT": round(wt_dg, 3),
            "DDG esm WT": round(wt_ddg, 3),
            "DG esm Design": round(design_dg, 3),
            "DDG esm Design": round(design_ddg, 3),
            "WT loss": round(wt_loss, 3),
            "Design loss": round(design_loss, 3),
            "Sequence length": seq_len,
            "summary_path": str(summary_path) if summary_path else None,
           }


    # save per-pdb json
    per_pdb_json.write_text(json.dumps(out, indent=2))
    return out
# 1) run design pipeline (unless we decide to skip when per-pdb exists and not force)
