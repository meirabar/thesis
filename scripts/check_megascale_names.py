#!/usr/bin/env python3
"""Quick diagnostic: inspect MegaScale Dataset1 name column."""
import csv
import re
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "/sci/labs/orzuk/meirab/protien_mega_exp_heatmaps/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"

names = set()
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        names.add(row["name"])

pdb_pat = re.compile(r"^[0-9][A-Za-z0-9]{3}\.pdb$")
pdb_names = sorted(n for n in names if pdb_pat.match(n))
other = sorted(n for n in names if not pdb_pat.match(n))

print(f"Total unique names: {len(names)}")
print(f"4-char PDB.pdb names: {len(pdb_names)}")
print(f"Sample PDB names: {pdb_names[:10]}")
print(f"Other names (sample): {other[:10]}")

has_pdb_prefix = [n for n in other[:100] if re.match(r"^[0-9][A-Za-z0-9]{3}", n)]
print(f"Other names starting with PDB-like prefix (of first 100): {len(has_pdb_prefix)}")
print(f"Examples: {has_pdb_prefix[:5]}")
