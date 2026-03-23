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
#SBATCH --account=project_465002270
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002270/nb-embed/huggingface_cache/
export HIP_VISIBLE_DEVICES=0
export TORCH_COMPILE_DISABLE=1
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

# === Configuration (override with sbatch --export) ===
# Swept hyperparameters
MODELS="${MODELS:-ltg/norbert4-xsmall}"
LEARNING_RATES="${LEARNING_RATES:-5e-6|1e-5|2e-5|5e-5|8e-5|1e-4|2e-4|5e-4}"
WARMUP_RATIOS="${WARMUP_RATIOS:-0.0|0.01|0.05|0.1}"
BATCH_SIZES="${BATCH_SIZES:-32}"
MAX_STEPS="${MAX_STEPS:-160|320|640|1280}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.0}"
LR_SCHEDULER_TYPES="${LR_SCHEDULER_TYPES:-linear}"

# Fixed hyperparameters
EVAL_STEPS="${EVAL_STEPS:-30}"
LOGGING_STEPS="${LOGGING_STEPS:-30}"
SAVE_STEPS="${SAVE_STEPS:-30}"
EVAL_ACCUMULATION_STEPS="${EVAL_ACCUMULATION_STEPS:-32}"
GRADIENT_ACCUMULATION_BASE="${GRADIENT_ACCUMULATION_BASE:-32}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw_torch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"

# Benchmark configuration
LANGUAGE="${LANGUAGE:-no}"
TASKS="${TASKS:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CACHE_DIR="${CACHE_DIR:-.euroeval_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
PRIORITIZE_MASK="${PRIORITIZE_MASK:-1}"

# W&B configuration
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-euroeval-sweeps}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-euroeval,sweep,slurm}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-hyperparameter-sweep}"
WANDB_NOTES="${WANDB_NOTES:-}"

# W&B Sweep configuration
USE_WANDB_SWEEP="${USE_WANDB_SWEEP:-0}"
MANUAL_ONLY="${MANUAL_ONLY:-1}"

# === Container setup (if using Singularity) ===
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SIF}"
CONTAINER_BIND="${CONTAINER_BIND:-}"

# === Setup ===
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Starting EuroEval sweep on $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Models: $MODELS"
echo "Learning rates: $LEARNING_RATES"

# === Build command ===
CMD=(
    python -m sweep_v2.sweep_hyperparams
    --models "$MODELS"
    --learning-rates "$LEARNING_RATES"
    --warmup-ratios "$WARMUP_RATIOS"
    --batch-sizes "$BATCH_SIZES"
    --max-steps "$MAX_STEPS"
    --eval-steps "$EVAL_STEPS"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --eval-accumulation-steps "$EVAL_ACCUMULATION_STEPS"
    --gradient-accumulation-base "$GRADIENT_ACCUMULATION_BASE"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --optimizer-name "$OPTIMIZER_NAME"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --weight-decays "$WEIGHT_DECAYS"
    --lr-scheduler-types "$LR_SCHEDULER_TYPES"
    --language "$LANGUAGE"
    --tasks "$TASKS"
    --num-iterations "$NUM_ITERATIONS"
    --cache-dir "$CACHE_DIR"
)

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ -n "$PER_DEVICE_EVAL_BATCH_SIZE" ]]; then
    CMD+=(--per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    CMD+=(--trust-remote-code)
fi

if [[ "$PRIORITIZE_MASK" == "1" ]]; then
    CMD+=(--prioritize-mask)
fi

if [[ "$WANDB" == "1" ]]; then
    CMD+=(
        --wandb
        --wandb-project "$WANDB_PROJECT"
        --wandb-tags "$WANDB_TAGS"
        --wandb-mode "$WANDB_MODE"
        --wandb-job-type "$WANDB_JOB_TYPE"
    )

    if [[ -n "$WANDB_ENTITY" ]]; then
        CMD+=(--wandb-entity "$WANDB_ENTITY")
    fi

    if [[ -n "$WANDB_RUN_NAME" ]]; then
        CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
    fi

    if [[ -n "$WANDB_GROUP" ]]; then
        CMD+=(--wandb-group "$WANDB_GROUP")
    fi

    if [[ -n "$WANDB_NOTES" ]]; then
        CMD+=(--wandb-notes "$WANDB_NOTES")
    fi

    if [[ "$USE_WANDB_SWEEP" == "0" || "$MANUAL_ONLY" == "1" ]]; then
        CMD+=(--manual-only)
    else
        CMD+=(--use-wandb-sweep)
    fi
fi

# === Execute with or without container ===
if [[ -n "$CONTAINER_IMAGE" ]]; then
    SINGULARITY_BIND="${CONTAINER_BIND:-.}"
    echo "Running in Singularity container: $CONTAINER_IMAGE"
    singularity exec --bind "$SINGULARITY_BIND" "$CONTAINER_IMAGE" /scratch/project_465002270/nb-embed/EuroEval/sweep_v2/run_sweep.sh
else
    echo "Running on host system"
    "${CMD[@]}"
fi

echo "Sweep completed with exit code: $?"
