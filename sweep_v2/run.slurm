#!/bin/bash
# Slurm HPC hyperparameter sweep launcher for EuroEval with W&B Sweep integration.
#
# Submit to cluster with:
#   sbatch --export=ALL,MODELS="model1|model2",...  sweep/run_sweep.slurm
#
# Note on sbatch --export:
#   sbatch interprets commas in the --export list as variable separators.
#   Use '|' or ';' for separating multiple values within a single variable:
#   MODELS="model1|model2" (NOT "model1,model2")

#SBATCH --job-name=euroeval-sweep
#SBATCH --partition=small-g
#SBATCH --account=project_465002893
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002893/nb-embed/huggingface_cache/
export HIP_VISIBLE_DEVICES=0
export TORCH_COMPILE_DISABLE=1
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export VENV=/scratch/project_465002893/nb-embed/EuroEval/venv.sqsh

# === Configuration (override with sbatch --export) ===
# Swept hyperparameters
MODEL="${MODEL:-ltg/norbert4-xsmall}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEP="${MAX_STEP:-1280}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"

NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
PRIORITIZE_MASK="${PRIORITIZE_MASK:-1}"

singularity exec -B $VENV:/user-software:image-src=/ $SIF sweep_v2/run.sh

