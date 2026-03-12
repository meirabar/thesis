#!/bin/bash
#SBATCH --job-name=pdb_des_corr
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#
# Run correlation analysis for PDB de novo designed proteins (B-factor only).
# CPU-only job. Run after robustness computation is done.
#
# Usage:
#   sbatch scripts/slurm/10_pdb_designs_correlate.sh
#   sbatch scripts/slurm/10_pdb_designs_correlate.sh --robustness-col std_ddg
#   sbatch scripts/slurm/10_pdb_designs_correlate.sh --both   # ESM-1v + ThermoMPNN
#
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Parse args
SCORERS="thermompnn"
EXTRA_FLAGS=""
ROB_COL=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --both)             SCORERS="esm1v thermompnn" ;;
        --robustness-col)   ROB_COL="${ARGS[i+1]}"; ((++i)) ;;
    esac
done
[[ -n "${ROB_COL}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --robustness_col ${ROB_COL}"

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "PDB De Novo Designs — Correlation Analysis"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Scorers:    ${SCORERS}"
echo "Output:     ${PDB_DESIGNS_ANALYSIS}"
echo "============================================"
echo ""

python "${REPO_DIR}/scripts/correlate_robustness_dynamics.py" \
    --atlas_dir "${PDB_DESIGNS_DIR}" \
    --robustness_dir "${PDB_DESIGNS_ROBUSTNESS}" \
    --scorer ${SCORERS} \
    --output_dir "${PDB_DESIGNS_ANALYSIS}" \
    --target bfactor \
    ${EXTRA_FLAGS}

echo ""
echo "PDB designs analysis finished at $(date)"
echo "Results in: ${PDB_DESIGNS_ANALYSIS}"
