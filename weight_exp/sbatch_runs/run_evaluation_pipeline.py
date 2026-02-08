#!/usr/bin/env python3
"""
run_evaluation_pipeline.py

Full evaluation pipeline:
1) DG WT vs DG design (PLL mean and DG proxy)
2) DDG WT vs DDG design (full landscape + overall mean)
3) Combined loss WT vs design (compute_loss from esm_utils)
4) Heatmaps saved in evaluation folder (plot_wt_vs_design_heatmaps)
5) PyMOL overlay of WT vs design predicted structure using overlay.pml

Usage:
    python run_evaluation_pipeline.py --pdb_id 1PSE
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import numpy as np
import pandas as pd

# Ensure local utils are importable
sys.path.append("/sci/labs/orzuk/meirab/stability_algo/weight_exp/stability_utils")
from evaluate_utils import *
from esm_utils import *
# from pymol_structure_plotting import run_align_and_circle





def run_evaluation(
    pdb_id: str,
    chain: str = "A",
    weight_exp: str = "weight_exp",
    lambda_dg: float = 1.0,
    lambda_ddg: float = 1.0,
    design_pdb_path: str = None,
    design_predicted_pdb: str = None,
    overlay_pml: str = "overlay.pml",
    force_use_pdb: bool = False,
    device: str = None,
    verbose: bool = True
):
    start_time = time.time()
    pdb_id = pdb_id.strip()
    eval_dir = Path(f"/sci/labs/orzuk/meirab/stability_algo/evaluation/{pdb_id}_{weight_exp}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "evaluation_summary.txt"

    print(f"[INFO] Started evaluationof weight0:")
    # 0) get WT sequence
    print(f"[INFO] Fetching WT sequence for {pdb_id} (chain {chain}) ...")
    wt_seq = fetch_and_extract_sequence(pdb_id, chain)

    dg_mean, dg_std, ddg_mean, ddg_std = load_norm_stats(pdb_id)
    print(f" {pdb_id}: DG μ={dg_mean:.3f}, σ={dg_std:.3f}, DDG μ={ddg_mean:.3f}, σ={ddg_std:.3f}")


    # 1) Get design sequence: prefer *_esm1v_sequence.txt -> optional design_pdb_path -> fallback to mutated_fixed.pdb
    if not force_use_pdb:
        # first try beam search output
        beam_path = Path(f"/sci/labs/orzuk/meirab/stability_algo/beam_outputs/{pdb_id}_beam_best_{weight_exp}.txt")
        print(beam_path)
        if beam_path.exists():
            design_seq = beam_path.read_text().strip()
            print(f"[INFO] Loaded design sequence from beam_outputs/{pdb_id}_beam_best_{weight_exp}.txt")

    # 2) length check and truncation
    if len(wt_seq) != len(design_seq):
        L = min(len(wt_seq), len(design_seq))
        print(f"[WARN] WT and design seq lengths differ; truncating to first {L} residues.")
        wt_seq, design_seq = wt_seq[:L], design_seq[:L]

    print(f"[INFO] WT length={len(wt_seq)}, design length={len(design_seq)}")

    # 3) Load ESM-1v model + alphabet (device-aware)
    device = device or ("cuda" if (hasattr(torch := __import__("torch"), "cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Loading ESM-1v model on device: {device}")
    model, alphabet = load_esm1v_model(device=device if "device" in load_esm1v_model.__code__.co_varnames else None)
    # if load_esm1v_model doesn't accept device arg, tolerate above

    # 4) DG: compute PLL loss and DG proxy for both sequences
    print("[STEP 1] Computing DG  (lower = better)...")
    wt_dg = compute_dg_esm1v(wt_seq, model, alphabet)
    ds_dg = compute_dg_esm1v(design_seq, model, alphabet)
    print("\n=== DG results (lower = better) ===")
    print(f"WT DG: {wt_dg:.3f}")
    print(f"DES DG: {ds_dg:.3f}")

    # 5) DDG landscapes (full matrices) for WT and design (can be slow)
    print("\n[STEP 2] Computing full DDG landscapes (this may take time)...")
    overall_ddg_wt,ddg_wt_matrix = full_compute_ddg_esm1v(wt_seq, model, alphabet,  verbose=False)
    overall_ddg_design,ddg_design_matrix = full_compute_ddg_esm1v(design_seq, model, alphabet, verbose=False)
    
    print("\n=== DDG summary (mut - wt; negative = stabilizing) ===")
    print(f"WT overall mean DDG   : {overall_ddg_wt:.6f}")
    print(f"DES overall mean DDG  : {overall_ddg_design:.6f}")
    print(f"Design - WT overall DDG: {overall_ddg_design - overall_ddg_wt:.6f}")
    
    
    # 6) Also produce the CSV listing all single-mutation ddg values for plotting/inspection
    csv_out = eval_dir / f"{pdb_id}_ddg_landscape_{weight_exp}.csv"
    print(f"[INFO] Producing mutation CSV at: {csv_out}")
    df_ddg = validate_design_ddg(wt_seq, design_seq, model, alphabet, str(csv_out))

    # 7) Plot heatmaps and save to eval dir
    print("\n[STEP 4] Plotting DDG heatmaps ...")
    heatmap_file = eval_dir / f"{pdb_id}_ddg_heatmaps_{weight_exp}.png"
    mutated_positions = [
        f"{wt_seq[i]}{i+1}{design_seq[i]}"
        for i in range(len(wt_seq))
        if wt_seq[i] != design_seq[i]
    ]

    plot_wt_vs_design_heatmaps(
        csv_path=str(csv_out),
        outfile=str(heatmap_file),
        title=f"{pdb_id}: WT vs Design DDG heatmaps",
        mutated_positions=mutated_positions,
        wt_seq=wt_seq
    )

    wt_loss =  compute_loss(
        dg= wt_dg,
        ddg=overall_ddg_wt,
        dg_mean=dg_mean,
        dg_std=dg_std,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        lambda_dg = lambda_dg,
        lambda_ddg = lambda_ddg
    )
 
    design_loss = compute_loss(
        dg= ds_dg,
        ddg=overall_ddg_design,
        dg_mean=dg_mean,
        dg_std=dg_std,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        lambda_dg = lambda_dg,
        lambda_ddg = lambda_ddg
    )


    print(f"[DONE] Saved heatmaps to {heatmap_file}")
        # 10) Save numeric summary
    summary = {
        "pdb_id": pdb_id,
        "wt_dg": wt_dg,
        "design_dg": ds_dg,
        "wt_overall_ddg": overall_ddg_wt,
        "design_overall_ddg": overall_ddg_design,
        "overall_ddg_diff": overall_ddg_design - overall_ddg_wt,
	"wt_loss" : wt_loss,
	"design_loss" : design_loss,
        "mutated_positions": mutated_positions,
        "heatmap_file": str(heatmap_file),
        "ddg_csv": str(csv_out),
        "runtime_sec": round(time.time() - start_time, 1),
    }
    save_summary(eval_dir, summary)
    print(f"[ALL DONE] Evaluation folder: {eval_dir}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pdb_id", required=True, help="PDB ID (e.g., 1PSE)")
    p.add_argument("--chain", default="A", help="Chain ID")
    p.add_argument("--design_pdb", default=None, help="Optional path to design PDB (if not using *_esm1v_sequence.txt)")
    p.add_argument("--design_predicted_pdb", default=None, help="Optional path to AlphaFold predicted design PDB")
    p.add_argument("--overlay_pml", default="overlay.pml", help="PyMOL overlay script (overlay.pml)")
    p.add_argument("--force_use_pdb", action="store_true",
                   help="Force loading design seq from provided PDB instead of esm1v_sequence.txt")

    # === Added experiment arguments ===
    p.add_argument("--lambda_dg", type=float, default=1.0, help="Weight for DG term")
    p.add_argument("--lambda_ddg", type=float, default=1.0, help="Weight for DDG term")
    p.add_argument("--weight_exp", default="default", help="Experiment name for output naming")

    args = p.parse_args()

    print(f"[INFO] Starting evaluation for {args.pdb_id} "
          f"(λDG={args.lambda_dg}, λDDG={args.lambda_ddg}, exp={args.weight_exp})")

    run_evaluation(
        pdb_id=args.pdb_id,
        chain=args.chain,
        design_pdb_path=args.design_pdb,
        design_predicted_pdb=args.design_predicted_pdb,
        overlay_pml=args.overlay_pml,
        force_use_pdb=args.force_use_pdb,
        lambda_dg=args.lambda_dg,
        lambda_ddg=args.lambda_ddg,
        weight_exp=args.weight_exp,
    )
