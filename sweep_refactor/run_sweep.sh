#!/bin/bash

set -euo pipefail

source /scratch/project_465002270/nb-embed/EuroEval/.venv/bin/activate

MODELS="${MODELS:-ltg/norbert4-xsmall}"
LEARNING_RATES="${LEARNING_RATES:-5e-6,1e-5,2e-5,5e-5,8e-5,1e-4,2e-4,5e-4}"
WARMUP_RATIOS="${WARMUP_RATIOS:-0.0,0.01,0.05,0.1}"
BATCH_SIZES="${BATCH_SIZES:-32}"
MAX_STEPS="${MAX_STEPS:-160,320,640,1280}"
EVAL_STEPS="${EVAL_STEPS:-30}"
LOGGING_STEPS="${LOGGING_STEPS:-30}"
SAVE_STEPS="${SAVE_STEPS:-30}"
EVAL_ACCUMULATION_STEPS="${EVAL_ACCUMULATION_STEPS:-32}"
GRADIENT_ACCUMULATION_BASE="${GRADIENT_ACCUMULATION_BASE:-32}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw_torch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.0}"
LR_SCHEDULER_TYPES="${LR_SCHEDULER_TYPES:-linear}"
LANGUAGE="${LANGUAGE:-no}"
TASKS="${TASKS:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CACHE_DIR="${CACHE_DIR:-.euroeval_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
PRIORITIZE_MASK="${PRIORITIZE_MASK:-1}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-euroeval-sweeps}"
WANDB_ENTITY="${WANDB_ENTITY:-nbailab}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-euroeval,sweep}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-hyperparameter-sweep}"
WANDB_NOTES="${WANDB_NOTES:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CMD=(
    python sweep_refactor/sweep_hyperparams.py
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
        --wandb-notes "$WANDB_NOTES"
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
fi

echo ${CMD[@]}

"${CMD[@]}"
