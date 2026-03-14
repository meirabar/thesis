#!/bin/bash
#
# Pipeline for RCI-S2 dataset: preprocess, compute ThermoMPNN, run correlations.
# Run each step sequentially on the cluster.
#
# Usage:
#   bash run_rci_pipeline.sh [step]
#   Steps: preprocess, thermompnn, correlate, multi_ddg, all
#

set -e

# Paths
PROJECT=/sci/labs/orzuk/orzuk/projects/ProteinStability
REPO=/sci/labs/orzuk/orzuk/github/meira_thesis
VENV=$PROJECT/envs/robustness
LOGDIR=$PROJECT/logs
SCRIPTS=$REPO/scripts

# RCI data paths
RCI_CSV=$PROJECT/data/gradation_nmr/zenodo_submission_v2/rci/rci_final.csv
PDB_DIR=$PROJECT/data/gradation_nmr/zenodo_submission_v2/rci/pdb_files
OUTPUT_DIR=$PROJECT/data/rci_s2_processed
ROBUSTNESS_DIR=$PROJECT/data/rci_s2_robustness
ANALYSIS_DIR=$PROJECT/data/rci_s2_analysis

STEP=${1:-all}

# ============================================================
# Step 1: Preprocess RCI dataset
# ============================================================
if [[ "$STEP" == "preprocess" || "$STEP" == "all" ]]; then
    echo "=== Step 1: Preprocess RCI dataset ==="
    source $VENV/bin/activate
    python $SCRIPTS/preprocess_rci_dataset.py \
        --rci_csv $RCI_CSV \
        --pdb_dir $PDB_DIR \
        --output_dir $OUTPUT_DIR
    echo "Done. Check $OUTPUT_DIR/proteins/"
fi

# ============================================================
# Step 2: Compute ThermoMPNN robustness (SLURM array job)
# ============================================================
if [[ "$STEP" == "thermompnn" || "$STEP" == "all" ]]; then
    echo "=== Step 2: Submit ThermoMPNN SLURM job ==="

    # Create PDB list file for --pdb_list mode
    PDB_LIST=$OUTPUT_DIR/pdb_list.txt
    find $OUTPUT_DIR/proteins -name "*.pdb" -type l | sort > $PDB_LIST
    N_PDBS=$(wc -l < $PDB_LIST)
    echo "  Found $N_PDBS PDB files"

    # Submit array job in batches of ~100
    BATCH_SIZE=100
    N_BATCHES=$(( (N_PDBS + BATCH_SIZE - 1) / BATCH_SIZE ))

    cat > $LOGDIR/rci_thermompnn.slurm << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=rci_thermo
#SBATCH --output=LOGDIR_PLACEHOLDER/rci_thermo_%a_%j.out
#SBATCH --error=LOGDIR_PLACEHOLDER/rci_thermo_%a_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=galileo
#SBATCH --array=0-NBATCHES_PLACEHOLDER

source VENV_PLACEHOLDER/bin/activate
cd REPO_PLACEHOLDER

BATCH_SIZE=100
START=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))

# Extract batch of PDB paths
PDB_LIST=OUTPUT_PLACEHOLDER/pdb_list.txt
BATCH_FILE=${SLURM_TMPDIR}/batch_pdbs.txt
sed -n "$((START+1)),$((END))p" $PDB_LIST > $BATCH_FILE

N=$(wc -l < $BATCH_FILE)
if [ "$N" -eq 0 ]; then
    echo "No PDBs in batch $SLURM_ARRAY_TASK_ID, exiting."
    exit 0
fi

echo "=== Batch $SLURM_ARRAY_TASK_ID: $N PDBs (indices $START to $END) ==="
echo "Started: $(date)"

python scripts/compute_robustness.py \
    --scorer thermompnn \
    --pdb_list $BATCH_FILE \
    --output_dir ROBUSTNESS_PLACEHOLDER \
    --skip_existing

echo "Finished: $(date)"
SLURM_EOF

    # Replace placeholders
    sed -i "s|LOGDIR_PLACEHOLDER|$LOGDIR|g" $LOGDIR/rci_thermompnn.slurm
    sed -i "s|NBATCHES_PLACEHOLDER|$((N_BATCHES - 1))|g" $LOGDIR/rci_thermompnn.slurm
    sed -i "s|VENV_PLACEHOLDER|$VENV|g" $LOGDIR/rci_thermompnn.slurm
    sed -i "s|REPO_PLACEHOLDER|$REPO|g" $LOGDIR/rci_thermompnn.slurm
    sed -i "s|OUTPUT_PLACEHOLDER|$OUTPUT_DIR|g" $LOGDIR/rci_thermompnn.slurm
    sed -i "s|ROBUSTNESS_PLACEHOLDER|$ROBUSTNESS_DIR|g" $LOGDIR/rci_thermompnn.slurm

    echo "  Submitting $N_BATCHES array jobs..."
    sbatch $LOGDIR/rci_thermompnn.slurm
    echo "  Monitor with: squeue -u \$USER | grep rci_thermo"
fi

# ============================================================
# Step 3: Run correlation analysis
# ============================================================
if [[ "$STEP" == "correlate" || "$STEP" == "all_analysis" ]]; then
    echo "=== Step 3: Correlation analysis ==="

    cat > $LOGDIR/rci_correlate.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_corr
#SBATCH --output=$LOGDIR/rci_corr_%j.out
#SBATCH --error=$LOGDIR/rci_corr_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=galileo

source $VENV/bin/activate
cd $REPO

echo "=== RCI-S2 Correlation Analysis ==="
echo "Started: \$(date)"

python scripts/correlate_robustness_dynamics.py \\
    --atlas_dir $OUTPUT_DIR \\
    --robustness_dir $ROBUSTNESS_DIR \\
    --scorer thermompnn \\
    --output_dir $ANALYSIS_DIR \\
    --target bfactor

echo "Finished: \$(date)"
EOF

    sbatch $LOGDIR/rci_correlate.slurm
    echo "  Submitted correlation job"
fi

# ============================================================
# Step 4: Multi-DDG regression
# ============================================================
if [[ "$STEP" == "multi_ddg" || "$STEP" == "all_analysis" ]]; then
    echo "=== Step 4: Multi-DDG regression ==="

    cat > $LOGDIR/rci_multi_ddg.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_mddg
#SBATCH --output=$LOGDIR/rci_mddg_%j.out
#SBATCH --error=$LOGDIR/rci_mddg_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=galileo

source $VENV/bin/activate
cd $REPO

echo "=== RCI-S2 Multi-DDG Regression ==="
echo "Started: \$(date)"

python scripts/multi_ddg_regression.py \\
    --atlas_dir $OUTPUT_DIR \\
    --robustness_dir $ROBUSTNESS_DIR \\
    --scorer thermompnn \\
    --output_dir $ANALYSIS_DIR \\
    --target bfactor

echo "Finished: \$(date)"
EOF

    sbatch $LOGDIR/rci_multi_ddg.slurm
    echo "  Submitted multi-DDG regression job"
fi

echo ""
echo "Pipeline steps:"
echo "  1. bash run_rci_pipeline.sh preprocess      # ~1 min, run interactively"
echo "  2. bash run_rci_pipeline.sh thermompnn       # ~hours, SLURM array job"
echo "  3. bash run_rci_pipeline.sh all_analysis     # after ThermoMPNN done"
