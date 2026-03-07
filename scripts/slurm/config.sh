#!/bin/bash
# ============================================================================
# Shared configuration for all SLURM pipeline scripts.
#
# THIS IS THE ONLY FILE YOU NEED TO EDIT when changing paths.
# All other scripts source this file.
#
# To use on a different machine or as a collaborator:
#   1. Clone the repo
#   2. Edit the paths below
#   3. Run 0_setup_env.sh to create the venv
#   4. Run submit_pipeline.sh
# ============================================================================

# --- Code repositories ---
REPO_DIR="/sci/labs/orzuk/orzuk/github/meira_thesis"
THERMOMPNN_DIR="/sci/labs/orzuk/orzuk/github/ThermoMPNN"

# --- Project directory (data, results, logs — outside the git repo) ---
PROJECT_DIR="/sci/labs/orzuk/orzuk/projects/ProteinStability"

# --- Derived paths (no need to edit unless you want a custom layout) ---
VENV_DIR="${PROJECT_DIR}/envs/robustness"
ATLAS_DIR="${PROJECT_DIR}/data/atlas"
ROBUSTNESS_DIR="${PROJECT_DIR}/data/atlas_robustness"
ANALYSIS_DIR="${PROJECT_DIR}/data/atlas_analysis"
LOG_DIR="${PROJECT_DIR}/logs"

# --- SLURM settings ---
GPU_PARTITION="gpu"          # partition with GPUs (sinfo to check)
CPU_PARTITION="long"         # partition for CPU-only / long jobs
CHUNK_SIZE=50                # proteins per array task

# --- Create directories ---
mkdir -p "${LOG_DIR}"
