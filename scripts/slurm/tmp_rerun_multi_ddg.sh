#!/bin/bash
# Rerun multi-DDG regression for all datasets (to pick up ols_mean_ddg model)
# Then collect results and regenerate figures.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p $LOG_DIR

cat > $LOG_DIR/rerun_multi_ddg.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rerun_mddg
#SBATCH --output=$LOG_DIR/rerun_mddg_%j.out
#SBATCH --error=$LOG_DIR/rerun_mddg_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=$CPU_PARTITION

source $VENV_DIR/bin/activate
cd $REPO_DIR

echo "=== Rerun multi-DDG for all datasets ==="
echo "Started: \$(date)"

# ATLAS RMSF
echo "--- ATLAS RMSF ---"
python scripts/multi_ddg_regression.py \\
    --atlas_dir $ATLAS_DIR \\
    --robustness_dir $ROBUSTNESS_DIR \\
    --scorer thermompnn \\
    --output_dir $ANALYSIS_DIR \\
    --target rmsf

# ATLAS B-factor
echo "--- ATLAS B-factor ---"
python scripts/multi_ddg_regression.py \\
    --atlas_dir $ATLAS_DIR \\
    --robustness_dir $ROBUSTNESS_DIR \\
    --scorer thermompnn \\
    --output_dir $ANALYSIS_DIR \\
    --target bfactor

# BBFlow RMSF
echo "--- BBFlow RMSF ---"
python scripts/multi_ddg_regression.py \\
    --atlas_dir $BBFLOW_PROCESSED \\
    --robustness_dir $BBFLOW_ROBUSTNESS \\
    --scorer thermompnn \\
    --output_dir $BBFLOW_ANALYSIS \\
    --target rmsf

# PDB designs B-factor
echo "--- PDB designs B-factor ---"
python scripts/multi_ddg_regression.py \\
    --atlas_dir $PDB_DESIGNS_DIR \\
    --robustness_dir $PDB_DESIGNS_ROBUSTNESS \\
    --scorer thermompnn \\
    --output_dir $PDB_DESIGNS_ANALYSIS \\
    --target bfactor

echo "=== All multi-DDG done ==="
echo "Finished: \$(date)"

# Collect results
echo "--- Collecting results ---"
python scripts/collect_results.py \\
    --output $PROJECT_DIR/data/paper_results/unified_results.json \\
    --verbose

# Generate figures
echo "--- Generating figures ---"
python scripts/generate_paper_figures.py \\
    --results $PROJECT_DIR/data/paper_results/unified_results.json \\
    --output-dir $PROJECT_DIR/data/paper_results/figures

echo "=== All done ==="
echo "Finished: \$(date)"
EOF

sbatch $LOG_DIR/rerun_multi_ddg.slurm
echo "Submitted. Monitor: tail -f $LOG_DIR/rerun_mddg_*.out"
