#!/bin/bash
#SBATCH --job-name=correlate
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#
# Run correlation analysis: robustness vs. RMSF/dynamics.
# CPU-only job (no GPU needed). Run after robustness computation is done.
#
# Usage:
#   sbatch scripts/slurm/3_correlate.sh              # ESM-1v only
#   sbatch scripts/slurm/3_correlate.sh --both        # both scorers
#   sbatch scripts/slurm/3_correlate.sh --no-dssp     # skip DSSP
#   sbatch scripts/slurm/3_correlate.sh --both --max-seq-length 1024
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Parse args
SCORERS="esm1v"
EXTRA_FLAGS=""
MAX_SEQ_LEN=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --both)             SCORERS="esm1v thermompnn" ;;
        --no-dssp)          EXTRA_FLAGS="${EXTRA_FLAGS} --no_dssp" ;;
        --max-seq-length)   MAX_SEQ_LEN="${ARGS[i+1]}"; ((i++)) ;;
    esac
done
[[ -n "${MAX_SEQ_LEN}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --max_seq_length ${MAX_SEQ_LEN}"

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "Correlation Analysis"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Scorers:    ${SCORERS}"
echo "Output:     ${ANALYSIS_DIR}"
echo "============================================"

python "${REPO_DIR}/scripts/correlate_robustness_dynamics.py" \
    --atlas_dir "${ATLAS_DIR}" \
    --robustness_dir "${ROBUSTNESS_DIR}" \
    --scorer ${SCORERS} \
    --output_dir "${ANALYSIS_DIR}" \
    ${EXTRA_FLAGS}

echo ""
echo "Analysis finished at $(date)"
echo "Results in: ${ANALYSIS_DIR}"
