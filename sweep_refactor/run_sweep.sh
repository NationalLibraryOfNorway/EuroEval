#!/bin/bash

set -euo pipefail

MODELS="${MODELS:-ltg/norbert4-xsmall}"
LEARNING_RATES="${LEARNING_RATES:-5e-6,1e-5,2e-5,5e-5,8e-5,1e-4,2e-4,5e-4}"
WARMUP_RATIOS="${WARMUP_RATIOS:-0.0,0.01,0.05,0.1}"
BATCH_SIZES="${BATCH_SIZES:-32}"
MAX_STEPS="${MAX_STEPS:-160,320,640,1280}"
LANGUAGE="${LANGUAGE:-no}"
TASKS="${TASKS:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CACHE_DIR="${CACHE_DIR:-.euroeval_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
PRIORITIZE_MASK="${PRIORITIZE_MASK:-1}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-euroeval-sweeps}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
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
    --language "$LANGUAGE"
    --tasks "$TASKS"
    --num-iterations "$NUM_ITERATIONS"
    --cache-dir "$CACHE_DIR"
)

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
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

"${CMD[@]}"
