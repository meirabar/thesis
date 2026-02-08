# Stable and Robust Protein Design

This repository contains the full pipeline used in my MSc thesis for **protein stability and mutational robustness design** using pretrained protein language models (ESM-1v), combined with evolutionary filtering, physics-based pruning, and combinatorial optimization.

You can find the **final summary results table** here:

- **Final Results Table (WT vs Design, clickable structure overlays):**  
  [Results Table](https://meirabar.github.io/thesis/results/weight05_05_summary_table.html)

Each protein entry links to **structural validation results**, including WT vs designed AlphaFold overlays.

---

## Pipeline Steps

Each step below corresponds to a concrete stage in the design pipeline implemented in  
`run_pipeline_and_design.py`.

- **`extract_chain_fasta_and_residue_map_from_pdb`**  
  Extract chain-A FASTA sequence and residue index mapping from the WT structure.

- **`mmseqs_search`**  
  Generate a deep multiple sequence alignment (MSA) using MMseqs2.

- **`compute_pssm`**  
  Compute per-position amino-acid frequencies and the PSSM from the MSA.

- **`allowed_mutations`**  
  Filter candidate substitutions using PSSM thresholds and FoldX ΔΔG constraints.

- **`save_residue_proximity_map`**  
  Compute residue–residue spatial proximity graph (Cα distance cutoff).

- **`load_norm_stats`**  
  Load global ΔG and ΔΔG normalization statistics for loss scaling.

- **`run_greedy_design_exp`**  
  Perform greedy ESM-1v single-mutation optimization under a weighted ΔG/ΔΔG loss.

- **`beam_search_design_exp`**  
  Refine designs via combinatorial beam search with spatial constraints.

---

## Typical Workflows

### Run full pipeline for one protein

```bash
python run_pipeline_and_design.py 6FVC 0.5 0.5 weight05
python run_evaluation_pipeline.py 6FVC 0.5 0.5 weight05


## Other Scripts

The repository also includes supporting utilities and scripts for evaluation and reporting.

### Utils Folder

- esm_utils.py – ESM-1v scoring and sequence likelihood utilities

- esm_design_utils.py – shared helpers for ESM-based design

- evaluate_utils.py – evaluation helpers (metrics, parsing, aggregation)

- beam_search_design_exp.py – beam search implementation

- run_greedy_design_exp.py – greedy design implementation