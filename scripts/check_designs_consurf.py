#!/usr/bin/env python3
"""Check which PDB design proteins have ConSurf-DB entries.

Proteins with ConSurf scores are likely natural proteins that slipped
through the keyword-based PDB design filter, or have close natural
homologs that ConSurf-DB processed.

Usage:
    python scripts/check_designs_consurf.py
"""

from pathlib import Path

PROJECT = "/sci/labs/orzuk/orzuk/projects/ProteinStability"
CONSURF_FILES = Path(PROJECT) / "data" / "ConSurf" / "files"
DESIGNS_DIR = Path(PROJECT) / "data" / "pdb_designs"
MAP_FILE = CONSURF_FILES.parent / "identical_to_unique_dict.txt"


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


def main():
    mapping = load_mapping()
    print(f"ConSurf mapping entries: {len(mapping)}")

    design_ids = sorted([d.name for d in DESIGNS_DIR.iterdir() if d.is_dir()])
    print(f"Total PDB design proteins: {len(design_ids)}")

    has_consurf = []
    has_via_mapping = []

    for pid in design_ids:
        parts = pid.split("_")
        if len(parts) != 2:
            continue
        pdb, chain = parts

        # Direct match
        cand = CONSURF_FILES / f"{pdb.upper()}_{chain.upper()}_consurf_info.json"
        if cand.exists():
            has_consurf.append(pid)
            continue

        # Via identical_to_unique mapping
        pid_lower = pid.lower()
        if pid_lower in mapping:
            mapped = mapping[pid_lower]
            mp, mc = mapped.split("_")
            mc_file = CONSURF_FILES / f"{mp.upper()}_{mc.upper()}_consurf_info.json"
            if mc_file.exists():
                has_consurf.append(pid)
                has_via_mapping.append(f"{pid} -> {mapped}")
                continue

    print(f"\nHave ConSurf scores: {len(has_consurf)} / {len(design_ids)}")
    print(f"  Direct match: {len(has_consurf) - len(has_via_mapping)}")
    print(f"  Via mapping:  {len(has_via_mapping)}")
    print(f"\nProteins with ConSurf (likely natural or have natural homologs):")
    for pid in has_consurf:
        print(f"  {pid}")
    if has_via_mapping:
        print(f"\nMapped entries:")
        for entry in has_via_mapping:
            print(f"  {entry}")


if __name__ == "__main__":
    main()
