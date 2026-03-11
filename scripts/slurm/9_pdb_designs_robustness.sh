#!/bin/bash
#SBATCH --job-name=pdb_des_rob
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#
# Compute ThermoMPNN robustness for PDB de novo designed proteins.
# GPU job. Run after 8_pdb_designs_download.sh completes.
#
# Usage:
#   sbatch scripts/slurm/9_pdb_designs_robustness.sh
#   sbatch scripts/slurm/9_pdb_designs_robustness.sh --both   # ESM-1v + ThermoMPNN
#
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Parse args
RUN_ESM=false
for arg in "$@"; do
    case "$arg" in
        --both) RUN_ESM=true ;;
    esac
done

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "PDB De Novo Designs — Robustness Computation"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Input:      ${PDB_DESIGNS_DIR}"
echo "Output:     ${PDB_DESIGNS_ROBUSTNESS}"
echo "============================================"
echo ""

# ---- ThermoMPNN (primary scorer) ----
echo "==== Computing ThermoMPNN robustness ===="
python "${REPO_DIR}/scripts/compute_robustness.py" \
    --scorer thermompnn \
    --atlas_dir "${PDB_DESIGNS_DIR}" \
    --output_dir "${PDB_DESIGNS_ROBUSTNESS}" \
    --batch \
    --batch_start 0 \
    --batch_end 9999 \
    --device cuda \
    --skip_existing \
    --thermompnn_dir "${THERMOMPNN_DIR}"
echo ""

# ---- ESM-1v (optional, for comparison) ----
if [ "$RUN_ESM" = true ]; then
    echo "==== Computing ESM-1v robustness ===="
    python "${REPO_DIR}/scripts/compute_robustness.py" \
        --scorer esm1v \
        --atlas_dir "${PDB_DESIGNS_DIR}" \
        --output_dir "${PDB_DESIGNS_ROBUSTNESS}" \
        --batch \
        --batch_start 0 \
        --batch_end 9999 \
        --device cuda \
        --skip_existing
    echo ""
fi

echo "============================================"
echo "Robustness computation finished at $(date)"
echo "Results in: ${PDB_DESIGNS_ROBUSTNESS}"
echo "============================================"
