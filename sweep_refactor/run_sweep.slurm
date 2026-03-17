#!/bin/bash
#SBATCH --job-name=euroeval-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

# Set/override these with: sbatch --export=ALL,MODELS=org/model-a,org/model-b ...
export MODELS="${MODELS:-ltg/norbert4-xsmall}"
export LEARNING_RATES="${LEARNING_RATES:-5e-6,1e-5,2e-5,5e-5,8e-5,1e-4,2e-4,5e-4}"
export WARMUP_RATIOS="${WARMUP_RATIOS:-0.0,0.01,0.05,0.1}"
export BATCH_SIZES="${BATCH_SIZES:-32}"
export MAX_STEPS="${MAX_STEPS:-160,320,640,1280}"
export LANGUAGE="${LANGUAGE:-no}"
export TASKS="${TASKS:-}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
export CACHE_DIR="${CACHE_DIR:-.euroeval_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export PRIORITIZE_MASK="${PRIORITIZE_MASK:-1}"
export WANDB="${WANDB:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-nb-embed-euroeval-sweeps}"
export WANDB_ENTITY="${WANDB_ENTITY:-nbailab}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_GROUP="${WANDB_GROUP:-}"
export WANDB_TAGS="${WANDB_TAGS:-euroeval,sweep}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-hyperparameter-sweep}"
export WANDB_NOTES="${WANDB_NOTES:-}"

# Optional cluster-specific setup goes here.
# module purge
# module load cuda/12.1

bash sweep_refactor/run_sweep.sh
