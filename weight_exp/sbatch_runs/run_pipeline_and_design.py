
import subprocess
import pandas as pd
from esm import pretrained
from tqdm.notebook import tqdm
import os
import pyfoldx
import itertools
import numpy as np
from pyfoldx.structure.structure import Structure
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import time
import plotly.graph_objects as go
import sklearn
import sys
import contextlib
import io
from Bio.PDB import PDBList

import argparse
import multiprocessing as mp
import traceback
import shutil
import glob

import pyrosetta
from pyrosetta import pose_from_pdb

_PYROSETTA_INITIALIZED = False

def init_pyrosetta_once():
    global _PYROSETTA_INITIALIZED
    if not _PYROSETTA_INITIALIZED:
        pyrosetta.init("-mute all -ignore_unrecognized_res")
        _PYROSETTA_INITIALIZED = True


sys.path.append('/sci/labs/orzuk/meirab/stability_algo/weight_exp/stability_utils')
from run_greedy_design_exp import *
from beam_search_design_exp import *
from greedy_utils import * 
from spatial_filter_utils import * 
from fasta_file_create import *
from MSA_utils import *
from PSSM_calc import *
from filter_utils import *
from esm_utils import *


def run_full_pipeline(pdb_id: str,
                      chain_id: str = "A",
                      wt_pdb_path: str = None,   # <-- NEW
                      lambda_dg: float = 1.0,
                      lambda_ddg: float = 1.0,
                      weight_exp: str = "weight_exp",
                      msa_dir="/sci/labs/orzuk/meirab/stability_algo/mmseqs_output/",
                      fasta_dir="/sci/labs/orzuk/meirab/stability_algo/data/fasta_inputs/",
                      pssm_out_dir="/sci/labs/orzuk/meirab/stability_algo/data/pssm_outputs/",
                      ddg_csv_path=None,
                      output_seq_path="greedy_esm_output.txt"):

    pdb_id_lower = pdb_id.lower()
    init_pyrosetta_once()
    pose = pose_from_pdb(wt_pdb_path)

    if pose.num_chains() != 1:
        raise RuntimeError(
            f"WT PDB {wt_pdb_path} has {pose.num_chains()} chains; "
            "expected exactly 1 (chain A only)"
        )

    print(f"\n[Step 1] Fetching FASTA for {pdb_id} chain {chain_id}")
    fasta_path, resmap_path = extract_chain_fasta_and_residue_map_from_pdb(
        pdb_path=wt_pdb_path,
        chain_id=chain_id,
        save_dir=fasta_dir,
        pdb_id=pdb_id
    )

    with open(fasta_path) as f:
        seq = "".join(line.strip() for line in f if not line.startswith(">"))

    assert len(seq) == pose.total_residue(), (
        f"Pipeline invariant broken: FASTA length {len(seq)} "
        f"!= Rosetta pose length {pose.total_residue()}"
    )

    #fetch_fasta_from_pdb(pdb_id, chain_id=chain_id, save_dir=fasta_dir)
	
    print(f"[Step 2] Submitting MMseqs2 search job...")
    # Submit job and get SLURM Job ID
    res = subprocess.run(["sbatch", "mmseqs_search.sh", fasta_path, pdb_id],capture_output=True, text=True)
    job_id = res.stdout.strip().split()[-1]
    print(f"Submitted MMseqs job with Job ID: {job_id}")

    # Wait for job to complete
    while True:
        result = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in result.stdout:
            print("MMseqs job completed.")
            break
        print("MMseqs job still running... waiting 30s")
        time.sleep(30)

    # === Wait for aligned MSA output ===
    msa_path = os.path.join(msa_dir, f"{pdb_id_lower}_result_aligned.fasta")
    print(f"[Step 3] Waiting for MSA result at {msa_path}...")
    while not os.path.exists(msa_path):
        print(" Waiting 30s...")
        time.sleep(30)

    print(f"[Step 4] Computing PSSM...")
    freqs, pssm = compute_pssm(msa_path)
    pssm_out_csv = os.path.join(pssm_out_dir, f"{pdb_id_lower}_pssm.csv")
    pssm.to_csv(pssm_out_csv)

    json_out = pssm_out_csv.replace(".csv", ".json")
    create_pssm_jsonl_from_df(pssm_df=pssm, pdb_id=pdb_id, output_jsonl_path=json_out, threshold=0)

    print(f"[Step 5] Loading PSSM and DDG matrices...")
    pssm_df = pd.read_csv(pssm_out_csv, index_col=0)
    ddg_df = pd.read_csv(ddg_csv_path, index_col=0) if ddg_csv_path else None

    print(f"[Step 6] Filtering allowed mutations...")
    allowed = allowed_mutations(pssm_df, ddg_df, threshold=-0.45)


    print(f"[Step 6b] Calculating spatial proximity map...")

    # === Define paths ===
    pdb_dir = "/sci/labs/orzuk/meirab/stability_algo/data/full_pdbs"
    proximity_out_dir = "/sci/labs/orzuk/meirab/stability_algo/data/proximity_maps"
    os.makedirs(proximity_out_dir, exist_ok=True)

    pdb_path = os.path.join(pdb_dir, f"{pdb_id.lower()}.pdb")

    if not os.path.exists(pdb_path):
        print(f"[WARN] Full PDB not found locally for {pdb_id}.")
        print("Run the downloader script first or fallback to PyFoldX.")
        structure = Structure(code=pdb_id)
        structure.downloadPDB()
        pdb_path = f"{pdb_id}.pdb"

    # === Define proximity map path ===
    proximity_path = os.path.join(
        proximity_out_dir,
        f"{pdb_id.lower()}_residue_proximity_r6.json"  # if radius=6.0
    )

    # === Check if proximity map already exists ===
    if os.path.exists(proximity_path):
        print(f"[INFO] Proximity map already exists for {pdb_id}, skipping computation.")
    else:
        print(f"[INFO] Computing new proximity map for {pdb_id}...")
        proximity_path = save_residue_proximity_map(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            chain_id=chain_id,
            radius=6.0,
            out_dir=proximity_out_dir
        )
        print(f"[INFO] Proximity map saved to: {proximity_path}")

    print(f"[Step 7] Fetching wildtype sequence...")
    from pyfoldx.structure.structure import Structure 
    sequence = Structure(code=pdb_id).getSequence(chain=chain_id)

    print(f"[Step 8] Running ESM-1v greedy design...")
    dg_mean, dg_std, ddg_mean, ddg_std = load_norm_stats(pdb_id)
    print(f" {pdb_id}: DG μ={dg_mean:.3f}, σ={dg_std:.3f}, DDG μ={ddg_mean:.3f}, σ={ddg_std:.3f}")

    final_seq, mutation_log, logs = run_greedy_design_exp(
        pdb_id=pdb_id,
        sequence=sequence,
        allowed_mutations=allowed,
        weight_exp=weight_exp,
        ddg_df=ddg_df,
        n_rounds=1,
        dg_mean=dg_mean,
        dg_std=dg_std,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        lambda_dg= lambda_dg,
        lambda_ddg= lambda_ddg,
        top_n=15,  # greedy: apply top single-choice per round
        output_dir="/sci/labs/orzuk/meirab/stability_algo/greedy_outputs/"
    )
    print("[Step 9] Running beam search refinement...")
    mutation_df = pd.DataFrame(mutation_log)   # full cumulative log
    top_df = extract_top_mutations(mutation_df, top_n=15)
    top_mutations = [(int(r["position"]), str(r["mutated_aa"]), float(r["loss"]))
                     for _, r in top_df.iterrows()]
    print(top_mutations)
    # load proximity map (already saved earlier in pipeline)
    with open(proximity_path) as f:
        proximity_dict = json.load(f)


    # run beam search directly
    best_seq, best_log, all_results = beam_search_design_exp(
        pdb_id=pdb_id,
        sequence=sequence,
        top_mutations=top_mutations,
        proximity_dict=proximity_dict,
        ddg_df=ddg_df,
        n_rounds=10,
        dg_mean=dg_mean,
        dg_std=dg_std,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        lambda_dg= lambda_dg,
        lambda_ddg= lambda_ddg,
        top_k=5,
        output_dir="/sci/labs/orzuk/meirab/stability_algo/beam_outputs/"
    )


    print("Beam search best seq:", best_seq)
    print("Beam search log:", best_log[-5:] if best_log else "No mutations applied")

    # save to file
    beam_out_path = f"/sci/labs/orzuk/meirab/stability_algo/beam_outputs/{pdb_id}_beam_best_{weight_exp}.txt"
    with open(beam_out_path, "w") as f:
        f.write(best_seq)
    print(f"Beam search sequence saved to {beam_out_path}")


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--pdb_id", required=True, help="PDB ID, e.g. 2JVD")
    p.add_argument("--lambda_dg", type=float, default=1.0, help="Weight for DG term")
    p.add_argument("--lambda_ddg", type=float, default=1.0, help="Weight for DDG term")
    p.add_argument("--weight_exp", default="default", help="Experiment name for output naming")
    args = p.parse_args()

    pdb_id = args.pdb_id.upper()
    weight_exp = args.weight_exp

    print(f"[INFO] Running pipeline for {pdb_id} (λDG={args.lambda_dg}, λDDG={args.lambda_ddg}, exp={weight_exp})")

    run_full_pipeline(
        pdb_id=pdb_id,
        wt_pdb_path=f"/sci/labs/orzuk/meirab/stability_algo/data/pross_wt_pdbs/{pdb_id}_PROSS_WT.pdb",
        ddg_csv_path=f"/sci/labs/orzuk/meirab/pyfoldx_ddg_csvs/ddGs_df_{pdb_id}.csv",
        output_seq_path=f"/sci/labs/orzuk/meirab/stability_algo/greedy_outputs/{pdb_id}_esm1v_sequence_{weight_exp}.txt",
        lambda_dg=args.lambda_dg,
        lambda_ddg=args.lambda_ddg,
        weight_exp=weight_exp,
    )
