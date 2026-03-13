#!/bin/bash
# Re-run ATLAS correlations (with new bfactor alt measures) + step 6 + postprocessing.
# Usage: nohup bash scripts/slurm/tmp_rerun_atlas_and_postprocess.sh > logs/rerun_atlas.log 2>&1 &
set -e

cd /sci/labs/orzuk/orzuk/github/meira_thesis
source /sci/labs/orzuk/orzuk/projects/ProteinStability/envs/robustness/bin/activate
P=/sci/labs/orzuk/orzuk/projects/ProteinStability

echo "Started at $(date)"

echo "=== 1/3: Re-running ATLAS correlations (new bfactor alt measures) ==="
python scripts/correlate_robustness_dynamics.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn esm1v --output_dir $P/data/atlas_analysis

echo "=== 2/3: Multi-DDG - ATLAS B-factor ==="
python scripts/multi_ddg_regression.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn --target bfactor --output_dir $P/data/atlas_analysis

echo "=== 3/3: Postprocessing (collect + tables + figures) ==="
python scripts/run_all_analyses.py --postprocess-only

echo "=== ALL DONE at $(date) ==="
