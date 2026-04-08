#!/bin/bash
# Local hyperparameter sweep launcher for EuroEval with W&B Sweep integration.
#
# Usage:
#   bash run_sweep.sh
#   MODELS="model1|model2" LEARNING_RATES="1e-5|2e-5" bash run_sweep.sh
#   WANDB=1 WANDB_PROJECT=my-project bash run_sweep.sh

set -euo pipefail

# === Configuration ===
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

# === Build command ===
CMD=(
    python  sweep_v2/run.py
    --model "$MODEL"
    --learning-rate "$LEARNING_RATE"
    --warmup-ratio "$WARMUP_RATIO"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEP"
    --weight-decay "$WEIGHT_DECAY"
    --lr-scheduler-type "$LR_SCHEDULER_TYPE"
    --num-iterations "$NUM_ITERATIONS"
    --trust-remote-code
    --prioritize-mask
    --no-progress-bar
)

# === Execute ===
"${CMD[@]}"
