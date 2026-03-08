#!/bin/bash
# ============================================================================
# Shared configuration for all SLURM pipeline scripts.
#
# THIS IS THE ONLY FILE YOU NEED TO EDIT when changing paths.
# All other scripts source this file.
#
# To use on a different machine or as a collaborator:
#   1. Clone the repo
#   2. Edit the 3 paths below
#   3. Run 0_setup_env.sh to create the venv
#   4. Run submit_pipeline.sh
# ============================================================================

# --- Code repositories ---
export REPO_DIR="/sci/labs/orzuk/orzuk/github/meira_thesis"
export THERMOMPNN_DIR="/sci/labs/orzuk/orzuk/github/ThermoMPNN"

# --- Project directory (data, results, logs — outside the git repo) ---
export PROJECT_DIR="/sci/labs/orzuk/orzuk/projects/ProteinStability"

# --- Derived paths (no need to edit unless you want a custom layout) ---
export VENV_DIR="${PROJECT_DIR}/envs/robustness"
export ATLAS_DIR="${PROJECT_DIR}/data/atlas"
export ROBUSTNESS_DIR="${PROJECT_DIR}/data/atlas_robustness"
export ANALYSIS_DIR="${PROJECT_DIR}/data/atlas_analysis"
export LOG_DIR="${PROJECT_DIR}/logs"

# --- SLURM settings ---
export GPU_PARTITION="catfish"
export CPU_PARTITION="glacier"
export CHUNK_SIZE=50
# GPU request syntax for this cluster: --gres=gpu:l4:1 (catfish L4 24GB)
# Alternatives: gpu:l40s:1 (salmon 48GB), gpu:h200:1 (goldfish 141GB)
export GPU_GRES="gpu:l4:1"

# --- Create directories ---
mkdir -p "${LOG_DIR}"
