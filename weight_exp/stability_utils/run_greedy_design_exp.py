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
from esm import pretrained, FastaBatchedDataset, ProteinBertModel
# importing util functions 
sys.path.append('/sci/labs/orzuk/meirab/stability_algo/weight_exp/stability_utils')
from fasta_file_create import *
from MSA_utils import *
from PSSM_calc import *
from esm_utils import *





def run_greedy_design_exp(
    pdb_id: str,
    sequence: str,
    allowed_mutations: dict,
    weight_exp: str = "weight_exp",
    ddg_df: dict = None,
    n_rounds: int = 3,
    dg_mean: float = 0,
    dg_std: float = 0,
    ddg_mean: float = 0,
    ddg_std: float = 0,
    lambda_dg: float = 1.0,
    lambda_ddg: float = 1.0,
    top_n: int = 1,
    output_dir: str = "/sci/labs/orzuk/meirab/stability_algo/greedy_outputs/"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    proximity_map_path = f"/sci/labs/orzuk/meirab/stability_algo/data/proximity_maps/{pdb_id.lower()}_residue_proximity_r6.json"
    with open(proximity_map_path) as f:
        proximity_dict = json.load(f)

    model, alphabet = pretrained.load_model_and_alphabet("esm1v_t33_650M_UR90S_1")
    sequence = list(sequence)            # mutable
    mutation_log = []                    # cumulative list of applied mutations (detailed)
    round_best = []                      # one entry per round: the best mutation (flattened logs for part 9)

    output_seq_path = os.path.join(output_dir, f"{pdb_id}_esm1v_sequence_{weight_exp}.txt")
    mutation_log_path = os.path.join(output_dir, f"{pdb_id}_esm1v_log_{weight_exp}.csv")

    wt_dg = compute_dg_esm1v("".join(sequence), model, alphabet)
    overall_ddg_wt,ddg_wt_matrix = full_compute_ddg_esm1v("".join(sequence), model, alphabet,  verbose=False)
    baseline_loss = compute_loss(dg= wt_dg, ddg=overall_ddg_wt, dg_mean=dg_mean, dg_std=dg_std, ddg_mean=ddg_mean,
        ddg_std=ddg_std, lambda_dg = lambda_dg, lambda_ddg = lambda_ddg )
    
    print(f"Baseline WT DG: {wt_dg:.3f},Baseline WT DDG: {overall_ddg_wt:.3f}")
    print(f"Baseline WT Loss: {baseline_loss:.3f}")
    print("\n Starting Greedy Design with ESM-1v")

    for round_idx in range(n_rounds):
        print(f"\n--- Round {round_idx + 1} ---")
        candidates = generate_mutation_candidates(sequence, allowed_mutations)
        losses = []
        dgs = []
        ddgs = []
        seqs_after = []

        # Score each candidate and record metrics in parallel lists
        for pos, aa, mut_str, new_seq in candidates:
            try:
                seq_before = "".join(sequence)
                dg = compute_dg_esm1v("".join(new_seq), model, alphabet)
                ddg = compute_ddg_from_neighbors("".join(new_seq), pos, model, alphabet, proximity_dict)
                loss = compute_loss(dg= dg,ddg=ddg, dg_mean=dg_mean, dg_std=dg_std, ddg_mean=ddg_mean,ddg_std=ddg_std, lambda_dg = lambda_dg, lambda_ddg = lambda_ddg)

                dgs.append(dg)
                ddgs.append(ddg)
                losses.append(loss)
                seqs_after.append("".join(new_seq))

                print(f" Mut {pos}{sequence[pos-1]}->{aa} | DG={dg:.2f}, DDG={ddg:.2f}, Loss={loss:.2f}")
            except Exception as e:
                # keep parallel list lengths consistent; mark as unusable
                dgs.append(np.nan)
                ddgs.append(np.nan)
                losses.append(np.inf)
                seqs_after.append(None)
                print(f" Skipping {pos}{aa} due to error: {e}")
        
        # Choose top_n candidates by lowest loss

        losses_arr = np.array(losses)
        if np.all(np.isinf(losses_arr)):
            print("No valid mutation candidates this round (all failed). Breaking.")
            break

        # ----- select top_n unique-position candidates (prefer loss < baseline_loss) -----
        sorted_idxs = list(np.argsort(losses_arr))
        selected_idxs = []

        # positions already mutated in previous rounds (cumulative)
        prev_positions = set([m['position'] for m in mutation_log])

        # 1) Prefer new positions with loss < baseline_loss
        for idx in sorted_idxs:
            if len(selected_idxs) >= top_n:
                break
            if not np.isfinite(losses_arr[idx]):
                continue
            pos = candidates[idx][0]
            if pos in prev_positions:
                continue
            if losses_arr[idx] < baseline_loss:
                selected_idxs.append(idx)
                prev_positions.add(pos)

        
        # === NEW SELECTION LOGIC ===
        losses_arr = np.array(losses)

        # Step 1: Group candidates by position and pick the best mutation per position
        best_per_position = {}
        for i, (pos, aa, mut_str, new_seq) in enumerate(candidates):
            L = losses_arr[i]
            if not np.isfinite(L):
                continue
            # keep only the best mutation at each position
            if pos not in best_per_position or L < losses_arr[best_per_position[pos]]:
                best_per_position[pos] = i

        # Now we have: one best mutation per position
        pos_best_indices = list(best_per_position.values())

        # Step 2: Keep only mutations better than baseline loss
        improving_idxs = [i for i in pos_best_indices if losses_arr[i] < baseline_loss]

        # Step 3: Sort improvements by loss (best first)
        improving_idxs = sorted(improving_idxs, key=lambda i: losses_arr[i])

        # Step 4: Select up to top_n
        top_idxs = improving_idxs[:top_n]

        # Debug print
        print(f"Found {len(improving_idxs)} improving unique-position mutations.")
        print(f"Selecting {len(top_idxs)} mutations this round.")


        # Apply selected mutations to the current sequence (update in place)
        applied_in_round = []
        seq_before_round = "".join(sequence)
        for idx in top_idxs:
            pos, aa, mut_str, _ = candidates[idx]
            loss = float(losses[idx]) if losses[idx] != np.inf else np.inf
            dg = dgs[idx]
            ddg = ddgs[idx]
            seq_after = seqs_after[idx] if seqs_after[idx] is not None else "".join(sequence)

            # apply mutation (careful about indexing convention in your code; this assumes pos is 1-based)
            sequence[pos - 1] = aa

            mut_entry = {
                "round": round_idx + 1,
                "position": pos,
                "mutated_aa": aa,
                "loss": loss,
                "dg": dg,
                "ddg": ddg,
                "seq_before": seq_before_round,
                "seq_after": "".join(sequence)
            }
            applied_in_round.append(mut_entry)
            mutation_log.append(mut_entry)   # cumulative

        # record the best mutation for this round (if top_n>1, this will contain multiple; choose the best single one)
        best_idx = top_idxs[0]
        round_best_entry = {
            "round": round_idx + 1,
            "position": candidates[best_idx][0],
            "mutated_aa": candidates[best_idx][1],
            "loss": float(losses[best_idx]),
            "dg": dgs[best_idx],
            "ddg": ddgs[best_idx],
            "seq_before": seq_before_round,
            "seq_after": seqs_after[best_idx] if seqs_after[best_idx] is not None else "".join(sequence)
        }
        round_best.append(round_best_entry)

        # build applied strings safely and print
        applied_strs = [f"{m['position']}{m['mutated_aa']}" for m in applied_in_round]
        print(f" Applied (round {round_idx+1}): {applied_strs}")

    # Final evaluation
    final_seq = "".join(sequence)
    final_dg = compute_dg_esm1v(final_seq, model, alphabet)

    # total_ddg: sum ddg of applied mutations recorded in mutation_log
    total_ddg, _ = full_compute_ddg_esm1v(final_seq, model, alphabet,  verbose=False)


    # Save cumulative mutation log (flattened)
    df_log = pd.DataFrame(mutation_log)
    if not df_log.empty:
        df_log.to_csv(mutation_log_path, index=False)

    # Save final sequence
    with open(output_seq_path, "w") as f:
        f.write(final_seq)

    print(f"\n Final sequence saved to: {output_seq_path}")
    print(f" Mutation log saved to: {mutation_log_path}")
    print(f" Final DG (log-likelihood): {final_dg:.3f}")
    print(f" Total DDG across mutations: {total_ddg:.3f}")

    # returns: final sequence, full cumulative mutation_log, and round-best logs (flattened per round)
    return final_seq, mutation_log, round_best
