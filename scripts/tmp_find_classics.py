#!/usr/bin/env python3
"""Find well-known proteins in ATLAS dataset by searching protein names/titles."""
import pandas as pd

df = pd.read_csv('/sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis/thermompnn/per_protein_correlations.tsv', sep='\t')

# First: just print all PDB IDs so we can grep for known ones
pdb_ids = df['protein_id'].str.split('_').str[0].tolist()

# Search for common protein keywords in PDB IDs
# Many classic proteins have multiple PDB entries; let's try broader patterns
import os, glob

# Approach 1: Look for proteins with good rho AND reasonable size, print PDB IDs
# so we can manually identify them
good = df[(df['seq_length'] >= 80) & (df['seq_length'] <= 300) &
          (df['rho_std_ddg_rmsf'].abs() > 0.6) &
          (df['rho_robustness_bfactor_target'].abs() > 0.4)]
good = good.sort_values('rho_std_ddg_rmsf')

print(f"Found {len(good)} proteins with |rho_rmsf|>0.6, |rho_bfac|>0.4, L=80-300")
print(f"\n{'protein_id':12s} {'L':>4s}  {'rho_rmsf':>9s}  {'rho_bfac':>9s}  {'plddt_rmsf':>11s}  {'plddt_bfac':>11s}")
print("-" * 70)

for _, row in good.head(50).iterrows():
    print(f"{row['protein_id']:12s} {row['seq_length']:4d}  {row['rho_std_ddg_rmsf']:9.3f}  {row['rho_robustness_bfactor_target']:9.3f}  {row['rho_plddt_rmsf']:11.3f}  {row['rho_plddt_bfactor']:11.3f}")

# Approach 2: Try to read PDB titles from ATLAS directory
atlas_dir = '/sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas'
print("\n\n=== Checking PDB titles for top candidates ===")
for _, row in good.head(20).iterrows():
    pid = row['protein_id']
    pdb_code = pid.split('_')[0]
    # Try to find and read the PDB file header
    pdb_patterns = [
        os.path.join(atlas_dir, pdb_code, f"{pdb_code}*.pdb"),
        os.path.join(atlas_dir, pid, f"*.pdb"),
        os.path.join(atlas_dir, pdb_code, "*.pdb"),
    ]
    title = "?"
    for pat in pdb_patterns:
        files = glob.glob(pat)
        if files:
            try:
                with open(files[0]) as f:
                    for line in f:
                        if line.startswith("TITLE"):
                            title = line[10:].strip()
                            break
            except:
                pass
            break
    print(f"{pid:12s}  {title}")
