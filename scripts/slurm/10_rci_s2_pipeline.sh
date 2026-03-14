#!/bin/bash
#
# Pipeline for RCI-S2 dataset: preprocess, compute ThermoMPNN, run correlations.
# All steps submitted via sbatch -- nothing blocks your terminal.
#
# Usage:
#   bash run_rci_pipeline.sh preprocess     # Step 1: ~1 min
#   bash run_rci_pipeline.sh thermompnn     # Step 2: ~hours (array job)
#   bash run_rci_pipeline.sh all_analysis   # Step 3+4: after ThermoMPNN done
#   bash run_rci_pipeline.sh collect        # Step 5: after analysis done
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

STEP=${1:?Usage: bash run_rci_pipeline.sh [preprocess|thermompnn|all_analysis|collect]}

mkdir -p $LOGDIR

# ============================================================
# Step 1: Preprocess RCI dataset (sbatch, ~1 min)
# ============================================================
if [[ "$STEP" == "preprocess" ]]; then
    cat > $LOGDIR/rci_preprocess.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_prep
#SBATCH --output=$LOGDIR/rci_prep_%j.out
#SBATCH --error=$LOGDIR/rci_prep_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=galileo

source $VENV/bin/activate
cd $REPO

echo "=== Preprocess RCI dataset ==="
echo "Started: \$(date)"

python scripts/preprocess_rci_dataset.py \\
    --rci_csv $RCI_CSV \\
    --pdb_dir $PDB_DIR \\
    --output_dir $OUTPUT_DIR

echo "Finished: \$(date)"
echo "Protein dirs:"
ls $OUTPUT_DIR/proteins/ | wc -l
EOF

    sbatch $LOGDIR/rci_preprocess.slurm
    echo "Submitted preprocessing job. Check: tail -n 20 $LOGDIR/rci_prep_*.out"
fi

# ============================================================
# Step 2: Compute ThermoMPNN robustness (SLURM array job)
# ============================================================
if [[ "$STEP" == "thermompnn" ]]; then
    echo "=== Step 2: Submit ThermoMPNN SLURM job ==="

    # Create PDB list file for --pdb_list mode
    PDB_LIST=$OUTPUT_DIR/pdb_list.txt
    find $OUTPUT_DIR/proteins -name "*.pdb" -type l | sort > $PDB_LIST
    N_PDBS=$(wc -l < $PDB_LIST)
    echo "  Found $N_PDBS PDB files"

    # Submit array job in batches of ~100
    BATCH_SIZE=100
    N_BATCHES=$(( (N_PDBS + BATCH_SIZE - 1) / BATCH_SIZE ))
    MAX_IDX=$(( N_BATCHES - 1 ))

    cat > $LOGDIR/rci_thermompnn.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_thermo
#SBATCH --output=$LOGDIR/rci_thermo_%a_%j.out
#SBATCH --error=$LOGDIR/rci_thermo_%a_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=galileo
#SBATCH --array=0-${MAX_IDX}

source $VENV/bin/activate
cd $REPO

BATCH_SIZE=100
START=\$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1 ))
END=\$(( START + BATCH_SIZE - 1 ))

# Extract batch of PDB paths
BATCH_FILE=\${SLURM_TMPDIR}/batch_pdbs.txt
sed -n "\${START},\${END}p" $PDB_LIST > \$BATCH_FILE

N=\$(wc -l < \$BATCH_FILE)
if [ "\$N" -eq 0 ]; then
    echo "No PDBs in batch \$SLURM_ARRAY_TASK_ID, exiting."
    exit 0
fi

echo "=== Batch \$SLURM_ARRAY_TASK_ID: \$N PDBs (lines \$START to \$END) ==="
echo "Started: \$(date)"

python scripts/compute_robustness.py \\
    --scorer thermompnn \\
    --pdb_list \$BATCH_FILE \\
    --output_dir $ROBUSTNESS_DIR \\
    --skip_existing

echo "Finished: \$(date)"
EOF

    echo "  Submitting $N_BATCHES array jobs..."
    sbatch $LOGDIR/rci_thermompnn.slurm
    echo "  Monitor: squeue -u \$USER | grep rci_thermo"
    echo "  Progress: ls $ROBUSTNESS_DIR/thermompnn/ | wc -l"
fi

# ============================================================
# Step 3+4: Correlation + Multi-DDG (sbatch)
# ============================================================
if [[ "$STEP" == "all_analysis" ]]; then
    # Correlation
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
    echo "Submitted correlation job"

    # Multi-DDG
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
    echo "Submitted multi-DDG regression job"
fi

# ============================================================
# Step 5: Collect results (sbatch)
# ============================================================
if [[ "$STEP" == "collect" ]]; then
    cat > $LOGDIR/rci_collect.slurm << EOF
#!/bin/bash
#SBATCH --job-name=rci_coll
#SBATCH --output=$LOGDIR/rci_collect_%j.out
#SBATCH --error=$LOGDIR/rci_collect_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=galileo

source $VENV/bin/activate
cd $REPO

echo "=== Collect all results ==="
echo "Started: \$(date)"

python scripts/collect_results.py \\
    --output $PROJECT/data/paper_results/unified_results.json \\
    --verbose

echo "Finished: \$(date)"
EOF

    sbatch $LOGDIR/rci_collect.slurm
    echo "Submitted collect job"
fi
