#!/usr/bin/env python
# coding: utf-8
#!/bin/bash

# In[1]:


from tqdm.notebook import tqdm
import os
import pyfoldx
import itertools
import numpy as np
from pyfoldx.structure.structure import Structure
import pandas as pd
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
from pathlib import Path
import argparse
import multiprocessing as mp
import traceback
import shutil
import glob

import logging
import json
from datetime import datetime


from esm import pretrained, FastaBatchedDataset, ProteinBertModel
# importing util functions 
sys.path.append('/sci/labs/orzuk/meirab/stability_algo/weight_exp/stability_utils')
from pyfoldx_utils import *
from fasta_file_create import *
from MSA_utils import *
from PSSM_calc import *
from esm_utils import *


# === Helper: compute full DG with ESM-1v ===
def compute_dg_full(seq: str, model, alphabet):
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.cuda()
    with torch.no_grad():
        logits = model(tokens)["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        tgt = tokens[:, 1:]
        nll = -log_probs[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        dg_val = nll.mean().item()
    return dg_val


# === Optional: skip spatially close pairs ===
def is_structurally_conflicting(new_mut, existing, proximity_dict, max_dist=6.0):
    """Return True if new_mut too close (<max_dist Å) to any existing mutation."""
    pos_new = new_mut[0]
    for pos_old, *_ in existing:
        if pos_new in proximity_dict.get(pos_old, {}) and \
           proximity_dict[pos_old][pos_new] < max_dist:
            return True
    return False



def beam_search_design_exp(
    pdb_id: str,
    sequence: str,
    top_mutations: list,
    proximity_dict: dict,
    ddg_df: dict = None,
    n_rounds: int = 5,   
    dg_mean: float = 0,
    dg_std: float = 0,
    ddg_mean: float = 0,
    ddg_std: float = 0,
    lambda_dg: float = 1.0,
    lambda_ddg: float = 1.0,
    top_k: int = 3,
    output_dir: str = "/sci/labs/orzuk/meirab/stability_algo/beam_outputs/"
):
    """
    Beam search mutation design with ESM-1v scoring.
    - Starts from WT sequence
    - Iteratively adds mutations
    - Keeps top_k sequences each round (beam width)
    """
    # === Setup logging ===
    log_dir = "/sci/labs/orzuk/meirab/stability_algo/weight_exp/traces"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, f"{pdb_id}_beam_trace.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"=== Starting beam search trace for {pdb_id} ===")

    # JSON trace for structured tracking
    trace_dir = "/sci/labs/orzuk/meirab/stability_algo/traces/"
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    trace_file = os.path.join(trace_dir, f"{pdb_id}_beam_trace.jsonl")

    def trace_json(round_idx, record):
        record = {"round": round_idx, **record}
        with open(trace_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load ESM-1v
    model, alphabet = pretrained.load_model_and_alphabet("esm1v_t33_650M_UR90S_1")

    # Make sequence mutable
    wt_seq = list(sequence)

    # Compute baseline WT loss
    wt_dg = compute_dg_esm1v("".join(sequence), model, alphabet)
    overall_ddg_wt, ddg_wt_matrix = full_compute_ddg_esm1v("".join(sequence), model, alphabet,  verbose=False)

    baseline_loss = compute_loss(
        dg= wt_dg,
        ddg=overall_ddg_wt,
        dg_mean=dg_mean,
        dg_std=dg_std,
        ddg_mean=ddg_mean,
        ddg_std=ddg_std,
        lambda_dg = lambda_dg,
        lambda_ddg = lambda_ddg
    )

    print(f"baseline loss = {baseline_loss}")
    # Initialize beam with WT
    candidates = [wt_seq]
    mutation_logs = [[]]

    for round_idx in range(n_rounds):
        logging.info(f"\n--- Round {round_idx + 1} ---")
        new_candidates, new_logs, all_losses = [], [], []
        round_records = []

        for seq, log in zip(candidates, mutation_logs):
            parent_seq_str = "".join(seq)
            print(f"\n Starting from sequence: {parent_seq_str}")
            if log:
                print(f"   Previous mutations: {', '.join([f'{p}{a}' for p, a, _ in log])}")
            else:
                print("   (wild-type starting point)")

        mutation_candidates = []
        for pos, aa, _ in top_mutations:
            # Skip if already mutated
            if seq[pos - 1] == aa:
               continue
            new_seq = list(seq)
            new_seq[pos - 1] = aa
            mut_str = f"{pos}{aa}"
            mutation_candidates.append((pos, aa, mut_str, new_seq))

            print(f"   → Testing {len(mutation_candidates)} new mutations this round...")

        for pos, aa, mut_str, new_seq in mutation_candidates:
            try:
                new_seq_str = "".join(new_seq)
    
                # === Compute DG ===
                dg = compute_dg_esm1v(new_seq_str, model, alphabet)
    
                # === Compute DDG using neighbors ===
                ddg = compute_ddg_from_neighbors(new_seq_str, pos, model, alphabet, proximity_dict)
    
                # === Compute total loss ===
                loss = compute_loss(
                    dg=dg,
                    ddg=ddg,
                    dg_mean=dg_mean,
                    dg_std=dg_std,
                    ddg_mean=ddg_mean,
                    ddg_std=ddg_std,
                    lambda_dg=lambda_dg,
                    lambda_ddg=lambda_ddg,
                )

                print(
                    f"     Mutation {mut_str:>6}:  "
                    f"DG={dg:8.3f}  |  DDG={ddg:8.3f}  |  Loss={loss:8.3f}"
                )

                round_records.append({
                    "mutation": mut_str,
                    "pos": pos,
                    "aa": aa,
                    "DG": float(dg),
                    "DDG": float(ddg),
                    "Loss": float(loss),
                    "seq_before": parent_seq_str
                })

            except Exception as e:
                logging.warning(f"     [ERROR] {mut_str}: {repr(e)}")

            new_candidates.append(new_seq)
            new_logs.append(log + [(pos, aa, loss)])
            all_losses.append(loss)
            
        # === After scoring all candidates for this round ===
        if round_records:
            sorted_round = sorted(round_records, key=lambda x: x["Loss"])
            topk = sorted_round[:top_k]
            logging.info(f"Top {n_rounds} mutations this round:")
            for r in topk:
                logging.info(f"     {r['mutation']:<6} | Loss={r['Loss']:.3f} | DG={r['DG']:.2f} | DDG={r['DDG']:.2f}")
            trace_json(round_idx + 1, {"tested": round_records, "top_mutations": topk})
        else:
            logging.warning("   No valid mutations found this round.")

        print("\n Round summary:")
        sorted_round = sorted(round_records, key=lambda x: x["Loss"])
        for r in sorted_round[:5]:
            print(f"   {r['mutation']:>6}  |  Loss={r['Loss']:.3f}  |  DG={r['DG']:.3f}  |  DDG={r['DDG']:.3f}")

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>> CHANGE #1: NEW DOMINANCE RULE (DG < WT & DDG < WT)
        # Identify mutations that improve BOTH DG AND DDG versus WT
        dominant = [
            r for r in sorted_round
            if (r["DG"] < wt_dg and r["DDG"] < overall_ddg_wt)
        ]

        if dominant:
            best_dom = min(dominant, key=lambda x: x["Loss"])
            print("\n🌟 Dominant mutation chosen (DG and DDG both improved vs WT):")
            print(f"   → {best_dom['mutation']} | DG={best_dom['DG']:.3f} | DDG={best_dom['DDG']:.3f}")

            # Apply the mutation
            new_seq = list(candidates[0])
            new_seq[best_dom["pos"] - 1] = best_dom["aa"]

            candidates = [new_seq]
            mutation_logs = [[(best_dom["pos"], best_dom["aa"], best_dom["Loss"])]]

            continue
        # <<< CHANGE END
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # === Strict DG & DDG early stopping rule ===
        if not sorted_round:
            logging.warning("No valid mutations this round, stopping early.")
            break

        best_entry = sorted_round[0]
        best_loss_round = best_entry["Loss"]
        DG_new = best_entry["DG"]
        DDG_new = best_entry["DDG"]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # >>> CHANGE #2: REMOVE EARLY STOP FOR DOMINANT MUTATIONS
        if not (DG_new < wt_dg and DDG_new < overall_ddg_wt):
            print("\⚠️  Early stopping: DG or DDG did not improve vs WT.")
            print(f"   WT DG   = {wt_dg:.4f}   |   New DG   = {DG_new:.4f}")
            print(f"   WT DDG  = {overall_ddg_wt:.4f} |   New DDG  = {DDG_new:.4f}")

            logging.info(
                f"Early stop at round {round_idx+1}: "
                f"DG_new={DG_new:.4f} (wt={wt_dg:.4f}), "
                f"DDG_new={DDG_new:.4f} (wt={overall_ddg_wt:.4f})"
            )

            best_idx, best_loss = min(
                enumerate([log[-1][2] if log else baseline_loss for log in mutation_logs]),
                key=lambda x: x[1]
            )
            best_seq = candidates[best_idx]
            best_log = mutation_logs[best_idx]

            trace_json("early_stop_DG_DDG", {
                "round": round_idx + 1,
                "DG_new": DG_new,
                "DDG_new": DDG_new,
                "DG_wt": wt_dg,
                "DDG_wt": overall_ddg_wt,
                "best_seq": "".join(best_seq)
            })

            return "".join(best_seq), best_log, list(zip(candidates, mutation_logs))
        # <<< CHANGE END
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Deduplicate by sequence (keep lowest loss version)
        unique = {}
        for seq, log, loss in zip(new_candidates, new_logs, all_losses):
            seq_str = "".join(seq)
            if seq_str not in unique or unique[seq_str][1] > loss:
                unique[seq_str] = (log, loss)

        # Keep top_k sequences by loss
        top = sorted(
            [(list(seq), log, loss) for seq, (log, loss) in unique.items()],
            key=lambda x: x[2]
        )[:top_k]

        if not top:
            print(" No surviving candidates; stopping beam search early.")
            break

        # === FULL ESM RESCORING for top_k ===
        rescored_entries = []
        for seq, log, loss in top:
            seq_str = "".join(seq)
            true_dg = compute_dg_full(seq_str, model, alphabet)
            true_ddg, _ = full_compute_ddg_esm1v(seq_str, model, alphabet, verbose=False)
            true_loss = lambda_dg * true_dg + lambda_ddg * true_ddg

            rescored_entries.append({
                "seq": seq,
                "log": log,
                "pred_loss": loss,
                "true_dg": round(true_dg, 3),
                "true_ddg": round(true_ddg, 3),
                "true_loss": round(true_loss, 3)
            })

        rescored_entries = sorted(rescored_entries, key=lambda x: x["true_loss"])
        candidates = [r["seq"] for r in rescored_entries[:top_k]]
        mutation_logs = [r["log"] for r in rescored_entries[:top_k]]
        losses = [r["true_loss"] for r in rescored_entries[:top_k]]

        # === LOG predicted vs true ===
        for e in rescored_entries:
            trace_json(round_idx + 1, {
                "mutation_log": e["log"],
                "pred_loss": e["pred_loss"],
                "true_loss": e["true_loss"],
                "true_dg": e["true_dg"],
                "true_ddg": e["true_ddg"]
            })


    # Pick best overall
    all_final = list(zip(candidates, mutation_logs))
    best_idx, best_loss = min(
        enumerate([log[-1][2] if log else baseline_loss for log in mutation_logs]),
        key=lambda x: x[1]
    )
    best_seq = candidates[best_idx]
    best_log = mutation_logs[best_idx]

    # Save results
    for i, (seq, log) in enumerate(all_final):
        prefix = os.path.join(output_dir, f"{pdb_id}_beam_seq{i}")
        with open(prefix + ".txt", "w") as f:
            f.write("".join(seq))
        if log:
            pd.DataFrame(log, columns=["position", "aa", "loss"]).to_csv(
                prefix + "_log.csv", index=False
            )
    logging.info(f"\nBeam search finished for {pdb_id}")
    logging.info(f"Best loss = {best_loss:.3f}")
    logging.info(f"Best sequence saved to {output_dir}")
    trace_json("end", {"best_loss": best_loss, "best_seq": "".join(best_seq)})

    print(f"\nBeam search finished. Best loss: {best_loss:.4f}")
    return "".join(best_seq), best_log, all_final
