#!/bin/bash
#SBATCH --job-name=bbflow_corr
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#
# Run correlation analysis for BBFlow designed proteins.
# CPU-only job. Run after robustness + pLDDT are computed.
#
# Usage:
#   sbatch scripts/slurm/6_bbflow_correlate.sh
#   sbatch scripts/slurm/6_bbflow_correlate.sh --robustness-col std_ddg
#
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Parse args
EXTRA_FLAGS=""
ROB_COL=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --robustness-col)   ROB_COL="${ARGS[i+1]}"; ((i++)) ;;
    esac
done
[[ -n "${ROB_COL}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --robustness_col ${ROB_COL}"

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "BBFlow Correlation Analysis"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Scorers:    esm1v thermompnn"
echo "Output:     ${BBFLOW_ANALYSIS}"
echo "============================================"
echo ""

python "${REPO_DIR}/scripts/correlate_robustness_dynamics.py" \
    --atlas_dir "${BBFLOW_PROCESSED}" \
    --robustness_dir "${BBFLOW_ROBUSTNESS}" \
    --scorer esm1v thermompnn \
    --output_dir "${BBFLOW_ANALYSIS}" \
    ${EXTRA_FLAGS}

echo ""
echo "BBFlow analysis finished at $(date)"
echo "Results in: ${BBFLOW_ANALYSIS}"
