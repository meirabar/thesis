#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/bin/bash

# In[1]:

from tqdm.notebook import tqdm
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
import os
import copy
import json
import torch
from pathlib import Path
from esm import pretrained, FastaBatchedDataset, ProteinBertModel
from Bio.PDB import PDBParser

from itertools import combinations
from collections import defaultdict
from typing import Union,Tuple,Iterable, List, Dict, Any

sys.path.insert(0, '/sci/labs/orzuk/meirab/stability_algo/weight_exp/stability_utils')
from filter_utils import *



def load_norm_stats(pdb_id, eval_dir="/sci/labs/orzuk/meirab/stability_algo/evaluation"):
    stats_path = Path(eval_dir) / f"{pdb_id}_sequence" / f"{pdb_id}_sequence_norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"No normalization stats found for {pdb_id}: {stats_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats["dg_mean"], stats["dg_std"], stats["ddg_mean"], stats["ddg_std"]


def run_foldx(mutation_string, pdb_id, chain="A", foldx_dir="FoldX"):
    """
    Run FoldX BuildModel on a given PDB with specified mutations.
    Returns the DDG and a boolean indicating success.
    """
    structure = Structure(pdb_id)
    mutation_list = mutation_string if mutation_string.endswith(";") else mutation_string + ";"
    try:
        ddGs, mutModels, wtModels = structure.mutate(mutations=mutation_list, verbose=False)
        mutated_model = mutModels.frames[0]
        output_path = f"/sci/labs/orzuk/meirab/stability_algo/greedy_outputs/{pdb_id}_mutated.pdb"
        with open(output_path, "w") as f:
            f.write("".join(mutated_model.toPdb()))

        avg_ddg = float(ddGs['total'].values[0])  # single mutation set assumed
        return avg_ddg, True
    except Exception as e:
        print(f"FoldX mutation failed for {mutation_string}: {e}")
        return None, False




def compute_loss(dg: float, ddg: float, 
                 dg_mean: float, dg_std: float,
                 ddg_mean: float, ddg_std: float,
                 lambda_dg: float = 1.0,
                 lambda_ddg: float = 1.0) -> float:

    dg_norm  = (dg - dg_mean) / dg_std if dg_std != 0 else dg
    ddg_norm = (ddg - ddg_mean) / ddg_std if ddg_std != 0 else ddg

    return lambda_dg * dg_norm + lambda_ddg * ddg_norm 



def full_compute_ddg_esm1v(seq: str, model: Any, alphabet: Any, verbose: bool = False) -> float:
    """
    Compute overall mean DDG using ESM-1v-style scoring.
    ddg = mut_score - wt_score  (so negative => mutant predicted more stable).

    NOTE: returns TWO values:
        1) ddg_matrix (np.ndarray of shape L x 20 with np.nan at WT columns)
        2) overall_mean (float) mean across all non-nan ddg entries

    Args:
        seq: WT sequence (string).
        model, alphabet: forwarded to compute_dg_esm1v (must exist in the module).
        verbose: print progress.
    """
    import numpy as np
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.upper()
    L = len(seq)
    A = len(aa_list)

    # prepare output matrix filled with NaN
    ddg_matrix = np.full((L, A), np.nan, dtype=float)

    # compute WT score once (compute_dg_esm1v must be defined elsewhere)
    wt_score = float(compute_dg_esm1v(seq, model, alphabet))
    if verbose:
        print(f"[compute_ddg_esm1v] WT score = {wt_score:.6f}  (L={L})")

    aa_to_col = {aa: idx for idx, aa in enumerate(aa_list)}

    for i in range(L):
        wt_aa = seq[i]
        for aa in aa_list:
            col = aa_to_col[aa]
            if aa == wt_aa:
                ddg_matrix[i, col] = np.nan
                continue

            mut_seq = seq[:i] + aa + seq[i+1:]
            try:
                mut_score = float(compute_dg_esm1v(mut_seq, model, alphabet))
                ddg_matrix[i, col] = mut_score - wt_score
            except Exception as e:
                # keep NaN on failure and optionally log
                if verbose:
                    print(f"  [pos {i+1}] failed to score mutant {wt_aa}{i+1}{aa}: {e}")
                ddg_matrix[i, col] = np.nan

        if verbose and ((i + 1) % 50 == 0 or i == L - 1):
            print(f"  processed pos {i+1}/{L}")

    # overall mean across non-nan values (safety: return 0.0 if all NaN)
    if np.all(np.isnan(ddg_matrix)):
        overall_mean = 0.0
    else:
        overall_mean = float(np.nanmean(ddg_matrix))

    return overall_mean , ddg_matrix

def apply_top_mutations(sequence, candidates, losses, top_n=1):
    """
    Apply top-N lowest-loss mutations.
    """
    sorted_candidates = sorted(zip(losses, candidates), key=lambda x: x[0])
    applied = []

    for i in range(min(top_n, len(sorted_candidates))):
        loss, (pos, aa, _, _) = sorted_candidates[i]
        sequence[pos - 1] = aa
        applied.append((pos, aa, loss))

    return sequence, applied

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


def generate_mutation_candidates(sequence, allowed_mutations):
    """
    Generate all valid single-point mutation candidates.
    Returns list of (position, new_aa, mut_str, new_sequence)
    """
    candidates = []
    for pos, aa_list in allowed_mutations.items():  # pos is 0-based
        if pos >= len(sequence):
            print(f"Skipping position {pos}: out of bounds.")
            continue
        wt = sequence[pos]
        for aa in aa_list:
            if wt == aa:
                continue
            mut_str = f"{wt}{pos + 1}{aa}"  # pos+1 for human-readable mutation name
            new_seq = sequence.copy()
            new_seq[pos] = aa
            candidates.append((pos, aa, mut_str, new_seq))
    return candidates

def run_esmfold_on_sequence(seq: str):
    """
    Run ESMFold to get predicted structure from sequence.
    Returns structure data or confidence score.
    """
    

    model = esmfold_v1()
    model.eval().cuda()

    with torch.no_grad():
        output = model.infer_pdb(seq)
        return output  # PDB string


def compute_dg_esm1v(seq: str, model, alphabet):
    """
    Use ESM-1v to compute negative log-likelihood (proxy for DG) for the whole sequence.

    Interpretation:
    - Lower DG model assigns higher probability to the sequence  predicted more stable.
    - Higher DG model assigns lower probability  predicted less stable.
    """
    batch_converter = alphabet.get_batch_converter()
    model.eval().cuda()

    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        logits = model(batch_tokens)["logits"]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    true_probs = log_probs[0, 1:len(seq)+1, :]  # exclude start/end tokens

    aa_indices = [alphabet.get_idx(aa) for aa in seq]
    ll_per_pos = true_probs[range(len(seq)), aa_indices]
    return -ll_per_pos.sum().item()  # total negative log-likelihood (proxy for DG)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def compute_ddg_esm1v(seq: str, pos: int, model, alphabet):
    """
    Compute the *mean* DDG over all 19 non-WT amino acids at a given position.
    """
    wt_seq = list(seq)
    wt_aa = wt_seq[pos - 1]
    wt_score = compute_dg_esm1v("".join(wt_seq), model, alphabet)

    ddg_vals = []
    for aa in AMINO_ACIDS:
        if aa == wt_aa:
            continue
        seq_mut = wt_seq.copy()
        seq_mut[pos - 1] = aa
        mut_score = compute_dg_esm1v("".join(seq_mut), model, alphabet)
        ddg_vals.append(mut_score - wt_score)

    if len(ddg_vals) == 0:
        return 0.0
    return np.mean(ddg_vals)


def compute_ddg_from_neighbors(wt_seq, center_pos, model, alphabet, proximity_dict):
    """
    Compute DDG as mean of mean-19-DDGs over the neighbor residues of center_pos.
    """
    wt_seq = list(wt_seq)
    neighbors = proximity_dict.get(str(center_pos), [])
    ddg_means = []

    if not neighbors:
        print(f"[WARN] No neighbors for residue {center_pos}, using only self.")
        neighbors = [center_pos]

    for neighbor in neighbors:
        neighbor_pos = int(neighbor)
        if neighbor_pos < 1 or neighbor_pos > len(wt_seq):
            continue
        try:
            ddg_avg19 = compute_ddg_esm1v("".join(wt_seq), neighbor_pos, model, alphabet)
           
            ddg_means.append(ddg_avg19)
        except Exception as e:
            print(f" Error computing mean19 DDG for neighbor {neighbor_pos}: {e}")
            continue
	
    if len(ddg_means) == 0:
        return 0.0
    return np.mean(ddg_means)



def extract_top_mutations(logs: Any, top_n: int = 10) -> pd.DataFrame:
    """
    Normalize `logs` to a DataFrame and return the top_n entries by loss (ascending).
    Accepts:
      - pd.DataFrame
      - list[dict]
      - list[tuple] or list[list] (expects at least 3 elements: pos, aa, loss)
      - list[scalar] (interpreted as loss)
    Returns:
      - pd.DataFrame with at least columns ['position','mutated_aa','loss'] where possible.
    """
    # 1. Normalize to DataFrame
    if isinstance(logs, pd.DataFrame):
        df = logs.copy()
    elif isinstance(logs, list):
        if len(logs) == 0:
            return pd.DataFrame(columns=["position", "mutated_aa", "loss"])

        first = logs[0]
        if isinstance(first, dict):
            df = pd.DataFrame(logs)
        elif isinstance(first, (list, tuple)):
            # handle list of tuples/lists: try to map to (position, mutated_aa, loss, ...)
            # if tuples longer than 3, keep extra columns as well
            maxlen = max(len(x) for x in logs)
            cols = []
            if maxlen >= 3:
                cols = ["position", "mutated_aa", "loss"] + [f"c{i}" for i in range(4, maxlen+1)]
            else:
                # shorter tuples: name generically
                cols = [f"c{i}" for i in range(1, maxlen+1)]
            df = pd.DataFrame([list(x) + [None]*(maxlen - len(x)) for x in logs], columns=cols)
        else:
            # list of scalars -> treat as losses only
            df = pd.DataFrame({"loss": list(logs)})
    else:
        # try to coerce (e.g., numpy arrays)
        try:
            df = pd.DataFrame(logs)
        except Exception:
            raise TypeError("Unsupported type for `logs`. Provide DataFrame, list[dict], list[tuple], or list[scalars].")

    # 2. Find loss column (common aliases)
    loss_col = None
    lower_cols = [c.lower() for c in df.columns]
    for candidate in ("loss", "score", "value", "objective"):
        if candidate in lower_cols:
            loss_col = df.columns[lower_cols.index(candidate)]
            break
    if loss_col is None:
        # fallback: pick a numeric column with 'loss' substring or the first numeric col
        for i, c in enumerate(df.columns):
            if "loss" in c.lower():
                loss_col = c
                break
        if loss_col is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) == 1:
                loss_col = numeric_cols[0]
            elif len(numeric_cols) > 1:
                # pick the most-likely: prefer 'loss' containing, else first numeric
                loss_col = numeric_cols[0]
            else:
                # last resort: if dataframe has exactly one column, treat it as loss
                if df.shape[1] == 1:
                    loss_col = df.columns[0]
                else:
                    raise ValueError(
                        "Could not detect a loss column in logs. Provide a column named 'loss' or pass "
                        "logs as list-of-dicts with a 'loss' key or list-of-tuples (pos, aa, loss)."
                    )

    # 3. Ensure position/aa columns exist (create if missing)
    # position column candidates
    pos_col = None
    for cand in ("position", "pos", "p", "residue"):
        if cand in df.columns:
            pos_col = cand
            break
    # aa column candidates
    aa_col = None
    for cand in ("mutated_aa", "aa", "mutation", "mut"):
        if cand in df.columns:
            aa_col = cand
            break

    # If they are missing but df has appropriate columns (e.g., from tuple), try to map them
    if pos_col is None or aa_col is None:
        # heuristic: if there are at least 3 columns and they are named generically (c1,c2,c3...), map them
        if "position" not in df.columns and "mutated_aa" not in df.columns:
            if "c1" in df.columns and "c2" in df.columns and "c3" in df.columns:
                if pos_col is None:
                    pos_col = "c1"
                if aa_col is None:
                    aa_col = "c2"
                # only set loss_col if it's c3 and loss_col not yet set
                if loss_col is None and "c3" in df.columns:
                    loss_col = "c3"

    # if still missing, leave as None - downstream code can still use returned DataFrame
    # 4. Convert loss column to numeric and drop rows where loss is NaN or infinite
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df_valid = df.dropna(subset=[loss_col]).copy()

    if df_valid.empty:
        return pd.DataFrame(columns=df.columns)

    # 5. Return top_n by smallest loss
    top_df = df_valid.nsmallest(top_n, columns=loss_col).reset_index(drop=True)

    # nice-to-have: ensure standard column names exist if possible
    if pos_col and pos_col in top_df.columns:
        top_df = top_df.rename(columns={pos_col: "position"}, errors="ignore")
    if aa_col and aa_col in top_df.columns:
        top_df = top_df.rename(columns={aa_col: "mutated_aa"}, errors="ignore")
    top_df = top_df.rename(columns={loss_col: "loss"}, errors="ignore")

    return top_df


Mutation = Union[Tuple[int, str], Tuple[int, str, object]]  # allow extra fields (e.g. loss)

def generate_valid_combinations(mutations: Iterable[Mutation],
                                min_len: int = 2,
                                pos_base: int = 1) -> List[Tuple]:
    """
    Generate all non-conflicting combinations of given mutations.

    - `mutations` may be an iterable of (pos, aa) or (pos, aa, ...).
    - `pos_base` = 1 if pos values are 1-based (default, matches your greedy code).
      Set pos_base=0 if you use 0-based positions.
    Returns a list of combinations where each combination is a tuple of (pos, aa)
    (extra elements in original tuples are dropped).
    """
    # Normalize to (pos, aa) pairs
    norm = []
    for m in mutations:
        if len(m) < 2:
            raise ValueError(f"Mutation tuple too short: {m}")
        pos = int(m[0])
        aa = str(m[1])
        # store normalized pos as the original pos (still 1- or 0-based depending on pos_base)
        norm.append((pos, aa))

    def is_valid(comb):
        positions = [pos for pos, _ in comb]
        return len(positions) == len(set(positions))

    combos = []
    n = len(norm)
    for r in range(min_len, n + 1):
        for comb in combinations(norm, r):
            if is_valid(comb):
                combos.append(comb)
    return combos


def apply_mutations_to_sequence(base_seq: Union[str, List[str]],
                                mutations: Iterable[Mutation],
                                pos_base: int = 1) -> Union[str, List[str]]:
    """
    Apply mutations to a base sequence.
    - base_seq: either a list of single-letter AAs OR a string.
    - mutations: iterable of (pos, aa) or (pos, aa, ...). pos is interpreted using pos_base.
    - pos_base: 1 if pos is 1-based (default), 0 if pos is 0-based.
    Returns same type as base_seq (string -> string, list -> list).
    """
    # create mutable copy
    was_string = isinstance(base_seq, str)
    if was_string:
        seq_list = list(base_seq)
    else:
        seq_list = base_seq.copy()

    n = len(seq_list)

    for m in mutations:
        if len(m) < 2:
            raise ValueError(f"Mutation tuple too short: {m}")
        pos = int(m[0])
        aa = str(m[1])

        idx = pos - pos_base  # convert to 0-based index
        if idx < 0 or idx >= n:
            raise IndexError(f"Mutation position {pos} (index {idx}) out of bounds for sequence length {n}")
        seq_list[idx] = aa

    return "".join(seq_list) if was_string else seq_list


def score_combinations(combos,original_seq,model,alphabet,
    lambda_dg=1.0,lambda_ddg=1.0,save_path=None):

    """
    Compute loss for each mutated combination using ESM.
    Optionally saves results to a JSON file.
    """
    scored = []
    for comb in combos:
        mut_seq = apply_mutations_to_sequence(list(original_seq), comb)
        dg = compute_dg_esm1v("".join(mut_seq), model, alphabet)
        ddgs = [compute_ddg_esm1v("".join(original_seq), pos, aa, model, alphabet) for pos, aa in comb]
        ddg = np.nanmean(ddgs)
        loss = compute_loss(lambda_dg, dg, ddg)
        scored.append({
            "mutations": comb,
            "mutated_sequence": "".join(mut_seq),
            "dg": dg,
            "ddg": ddg,
            "loss": loss
        })

    # Save results if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(scored, f, indent=2)

    return scored



def get_sequence_length(pdb_path, chain_id="A"):
    """
    Returns the number of residues (with CA atoms) in a given chain of a PDB file.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        chain = structure[0][chain_id]
        residues = [res for res in chain if 'CA' in res]
        return len(residues)
    except Exception as e:
        print(f"[WARN] Could not compute sequence length for {pdb_path}: {e}")
        return None
        return False, None, None, None, None
