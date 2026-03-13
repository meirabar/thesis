#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('/sci/labs/orzuk/orzuk/projects/ProteinStability/data/atlas_analysis/thermompnn/per_protein_correlations.tsv', sep='\t')

classics = [
    '1ubq', '1ubi', '2ci2', '1lz1', '1lyz', '3lzm', '2lzm', '1l58', '4lzt',
    '1ake', '4ake', '2eck',
    '1rx2', '1dfj', '3dfr', '1dyr',
    '1cll', '1cdl', '1osa',
    '2rn2', '1a2p', '1bni',
    '1hiv', '1htg', '1aid',
    '1a3n', '1hho', '2hhb',
    '1mbo', '1myg', '1wla',
    '1tim', '7tim', '1tph',
    '3pga', '1pgk',
    '1cyx', '1cwp', '2cpb',
    '1stn', '1snc',
    '1shf', '1ten', '3hhr',
    '1pgb', '1pga', '2ptl',
    '1crn', '1l2y',
    '1e4e', '2e4e',
]

pdb_ids = df['protein_id'].str.split('_').str[0]

print(f"{'protein_id':12s} {'L':>4s}  {'rho_rmsf':>9s}  {'rho_bfac':>9s}  {'plddt_rmsf':>11s}  {'plddt_bfac':>11s}")
print("-" * 70)

for c in classics:
    matches = df[pdb_ids == c]
    if len(matches) > 0:
        for _, row in matches.iterrows():
            print(f"{row['protein_id']:12s} {row['seq_length']:4d}  {row['rho_std_ddg_rmsf']:9.3f}  {row['rho_robustness_bfactor_target']:9.3f}  {row['rho_plddt_rmsf']:11.3f}  {row['rho_plddt_bfactor']:11.3f}")
