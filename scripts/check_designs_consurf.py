#!/usr/bin/env python3
"""Check which PDB design proteins have ConSurf-DB entries.

Reports MSA depth from ConSurf JSON to help distinguish:
- Natural proteins (deep MSA, hundreds of homologs)
- Designed proteins on natural scaffolds (moderate MSA from parent)
- Truly novel designs (shallow/no MSA)

Usage:
    python scripts/check_designs_consurf.py
"""

import json
import gzip
from pathlib import Path

PROJECT = "/sci/labs/orzuk/orzuk/projects/ProteinStability"
CONSURF_FILES = Path(PROJECT) / "data" / "ConSurf" / "files"
DESIGNS_DIR = Path(PROJECT) / "data" / "pdb_designs" / "proteins"
MAP_FILE = CONSURF_FILES.parent / "identical_to_unique_dict.txt"
METADATA = Path(PROJECT) / "data" / "pdb_designs" / "metadata.tsv"


def load_mapping():
    mapping = {}
    if MAP_FILE.exists():
        with open(MAP_FILE) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    mapping[k.strip().lower()] = v.strip()
    return mapping


def get_consurf_info(json_path):
    """Extract MSA depth and score stats from ConSurf JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        scores = data.get("SCORE", [])
        msa = data.get("MSA_DATA", {})
        n_seqs = msa.get("n_sequences", msa.get("num_sequences",
                  msa.get("NUMBER_OF_SEQS", "?")))
        n_residues = len(scores)
        non_none = [s for s in scores if s is not None]
        n_scored = len(non_none)
        mean_score = sum(non_none) / len(non_none) if non_none else None
        return {
            "n_seqs": n_seqs,
            "n_residues": n_residues,
            "n_scored": n_scored,
            "mean_score": mean_score,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    mapping = load_mapping()

    design_ids = sorted([d.name for d in DESIGNS_DIR.iterdir() if d.is_dir()])
    print(f"Total PDB design proteins: {len(design_ids)}")

    # Load metadata if available
    meta = {}
    if METADATA.exists():
        with open(METADATA) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) >= 2:
                    row = dict(zip(header, fields))
                    pid = row.get("protein_id", row.get("pdb_chain", fields[0]))
                    meta[pid] = row

    results = []
    for pid in design_ids:
        parts = pid.split("_")
        if len(parts) != 2:
            continue
        pdb, chain = parts

        json_path = None
        # Direct match
        cand = CONSURF_FILES / f"{pdb.upper()}_{chain.upper()}_consurf_info.json"
        if cand.exists():
            json_path = cand
        else:
            # Via mapping
            pid_lower = pid.lower()
            if pid_lower in mapping:
                mapped = mapping[pid_lower]
                mp, mc = mapped.split("_")
                mc_file = CONSURF_FILES / f"{mp.upper()}_{mc.upper()}_consurf_info.json"
                if mc_file.exists():
                    json_path = mc_file

        if json_path is not None:
            info = get_consurf_info(json_path)
            results.append((pid, info))

    print(f"Have ConSurf scores: {len(results)} / {len(design_ids)}")
    print(f"\n{'PDB_ID':12s} {'MSA_seqs':>10s} {'residues':>10s} {'scored':>8s} {'mean_score':>12s}")
    print("-" * 60)
    for pid, info in sorted(results, key=lambda x: -(x[1].get("n_seqs", 0)
                            if isinstance(x[1].get("n_seqs"), (int, float)) else 0)):
        n_seqs = info.get("n_seqs", "?")
        n_res = info.get("n_residues", "?")
        n_scored = info.get("n_scored", "?")
        mean_s = info.get("mean_score")
        mean_str = f"{mean_s:.3f}" if mean_s is not None else "?"
        print(f"{pid:12s} {str(n_seqs):>10s} {str(n_res):>10s} {str(n_scored):>8s} {mean_str:>12s}")

    # Also print the MSA_DATA keys from first file to understand structure
    if results:
        first_path = None
        pid0 = results[0][0]
        parts = pid0.split("_")
        pdb, chain = parts
        cand = CONSURF_FILES / f"{pdb.upper()}_{chain.upper()}_consurf_info.json"
        if cand.exists():
            first_path = cand
        if first_path:
            with open(first_path) as f:
                data = json.load(f)
            print(f"\nExample ConSurf JSON keys for {pid0}:")
            print(f"  Top-level: {list(data.keys())}")
            msa = data.get("MSA_DATA", {})
            if msa:
                print(f"  MSA_DATA keys: {list(msa.keys())}")
                for k, v in msa.items():
                    if isinstance(v, (int, float, str)):
                        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
