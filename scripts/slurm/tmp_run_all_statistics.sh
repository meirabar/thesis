#!/bin/bash
# Run all correlation + multi-DDG + postprocessing jobs sequentially.
# Usage: nohup bash scripts/slurm/tmp_run_all_statistics.sh > logs/run_all_statistics.log 2>&1 &
set -e

cd /sci/labs/orzuk/orzuk/github/meira_thesis
source /sci/labs/orzuk/orzuk/projects/ProteinStability/envs/robustness/bin/activate
P=/sci/labs/orzuk/orzuk/projects/ProteinStability

echo "Started at $(date)"

echo "=== 1/9: Correlations - PDB designs ThermoMPNN ==="
python scripts/correlate_robustness_dynamics.py --atlas_dir $P/data/pdb_designs --robustness_dir $P/data/pdb_designs_robustness --scorer thermompnn --output_dir $P/data/pdb_designs_analysis --target bfactor

echo "=== 2/9: Correlations - PDB designs ESM-1v ==="
python scripts/correlate_robustness_dynamics.py --atlas_dir $P/data/pdb_designs --robustness_dir $P/data/pdb_designs_robustness --scorer esm1v --output_dir $P/data/pdb_designs_analysis --target bfactor

echo "=== 3/9: Correlations - ATLAS (ThermoMPNN + ESM-1v) ==="
python scripts/correlate_robustness_dynamics.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn esm1v --output_dir $P/data/atlas_analysis

echo "=== 4/9: Correlations - BBFlow (ThermoMPNN + ESM-1v) ==="
python scripts/correlate_robustness_dynamics.py --atlas_dir $P/data/bbflow_processed --robustness_dir $P/data/bbflow_robustness --scorer thermompnn esm1v --output_dir $P/data/bbflow_analysis

echo "=== 5/9: Multi-DDG - ATLAS RMSF ==="
python scripts/multi_ddg_regression.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn --target rmsf --output_dir $P/data/atlas_analysis

echo "=== 6/9: Multi-DDG - ATLAS B-factor ==="
python scripts/multi_ddg_regression.py --atlas_dir $P/data/atlas --robustness_dir $P/data/atlas_robustness --scorer thermompnn --target bfactor --output_dir $P/data/atlas_analysis

echo "=== 7/9: Multi-DDG - BBFlow RMSF ==="
python scripts/multi_ddg_regression.py --atlas_dir $P/data/bbflow_processed --robustness_dir $P/data/bbflow_robustness --scorer thermompnn --target rmsf --output_dir $P/data/bbflow_analysis

echo "=== 8/9: Multi-DDG - PDB designs B-factor ==="
python scripts/multi_ddg_regression.py --atlas_dir $P/data/pdb_designs --robustness_dir $P/data/pdb_designs_robustness --scorer thermompnn --target bfactor --output_dir $P/data/pdb_designs_analysis

echo "=== 9/9: Postprocessing (collect + tables + figures) ==="
python scripts/run_all_analyses.py --postprocess-only

echo "=== ALL DONE at $(date) ==="
