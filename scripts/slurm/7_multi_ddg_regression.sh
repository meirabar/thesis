#!/bin/bash
#SBATCH --job-name=multi_ddg
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Multi-dimensional DDG regression: predict dynamics using full 20-AA
# DDG profile per residue, with protein-level k-fold cross-validation.
# CPU-only job (no GPU needed). Run after robustness computation is done.
#
# Usage:
#   sbatch scripts/slurm/7_multi_ddg_regression.sh                          # ATLAS, RMSF
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --target bfactor         # ATLAS, B-factor
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --bbflow                 # BBFlow, RMSF
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --both-datasets          # ATLAS + BBFlow
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --both-targets           # RMSF + B-factor
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --all                    # everything
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --max-seq-length 1024    # length filter
#   sbatch scripts/slurm/7_multi_ddg_regression.sh --alpha 0.1              # regularization
# ============================================================================

set -euo pipefail
if [[ -z "${REPO_DIR:-}" ]]; then
    _cfg="$(dirname "${BASH_SOURCE[0]}")/config.sh"
    [[ ! -f "$_cfg" ]] && _cfg="${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/slurm/config.sh"
    source "$_cfg"
fi

# Defaults
SCORER="thermompnn"
TARGETS="rmsf"
RUN_ATLAS=true
RUN_BBFLOW=false
MAX_SEQ_LEN=""
ALPHA="1.0"
N_FOLDS="5"

# Parse args
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --target)            TARGETS="${ARGS[i+1]}"; ((i++)) ;;
        --both-targets)      TARGETS="rmsf bfactor" ;;
        --bbflow)            RUN_ATLAS=false; RUN_BBFLOW=true ;;
        --both-datasets)     RUN_ATLAS=true; RUN_BBFLOW=true ;;
        --all)               RUN_ATLAS=true; RUN_BBFLOW=true; TARGETS="rmsf bfactor" ;;
        --scorer)            SCORER="${ARGS[i+1]}"; ((i++)) ;;
        --max-seq-length)    MAX_SEQ_LEN="${ARGS[i+1]}"; ((i++)) ;;
        --alpha)             ALPHA="${ARGS[i+1]}"; ((i++)) ;;
        --n-folds)           N_FOLDS="${ARGS[i+1]}"; ((i++)) ;;
    esac
done

EXTRA_FLAGS=""
[[ -n "${MAX_SEQ_LEN}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --max_seq_length ${MAX_SEQ_LEN}"

source "${VENV_DIR}/bin/activate"

echo "============================================"
echo "Multi-DDG Regression"
echo "Date:       $(date)"
echo "Node:       $(hostname)"
echo "Scorer:     ${SCORER}"
echo "Targets:    ${TARGETS}"
echo "Alpha:      ${ALPHA}"
echo "Folds:      ${N_FOLDS}"
echo "ATLAS:      ${RUN_ATLAS}"
echo "BBFlow:     ${RUN_BBFLOW}"
echo "============================================"

# --- ATLAS ---
if [[ "${RUN_ATLAS}" == "true" ]]; then
    for target in ${TARGETS}; do
        echo ""
        echo ">>> ATLAS / ${SCORER} / ${target}"
        echo "--------------------------------------------"
        python "${REPO_DIR}/scripts/multi_ddg_regression.py" \
            --atlas_dir "${ATLAS_DIR}" \
            --robustness_dir "${ROBUSTNESS_DIR}" \
            --scorer "${SCORER}" \
            --output_dir "${ANALYSIS_DIR}" \
            --target "${target}" \
            --alpha "${ALPHA}" \
            --n_folds "${N_FOLDS}" \
            ${EXTRA_FLAGS}
    done
fi

# --- BBFlow ---
if [[ "${RUN_BBFLOW}" == "true" ]]; then
    for target in ${TARGETS}; do
        # BBFlow has no B-factors
        if [[ "${target}" == "bfactor" ]]; then
            echo ""
            echo ">>> Skipping BBFlow / bfactor (no crystal structures)"
            continue
        fi
        echo ""
        echo ">>> BBFlow / ${SCORER} / ${target}"
        echo "--------------------------------------------"
        python "${REPO_DIR}/scripts/multi_ddg_regression.py" \
            --atlas_dir "${BBFLOW_PROCESSED}" \
            --robustness_dir "${BBFLOW_ROBUSTNESS}" \
            --scorer "${SCORER}" \
            --output_dir "${BBFLOW_ANALYSIS}" \
            --target "${target}" \
            --alpha "${ALPHA}" \
            --n_folds "${N_FOLDS}" \
            ${EXTRA_FLAGS}
    done
fi

echo ""
echo "Multi-DDG regression finished at $(date)"
