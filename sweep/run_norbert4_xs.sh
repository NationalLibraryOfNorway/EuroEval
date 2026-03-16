#!/bin/bash

MODEL="ltg/norbert4-xsmall"
LEARNING_RATES="5e-6,1e-5,2e-5,5e-5,8e-5,1e-4,2e-4,5e-4"
WARMUP_RATIOS="0.0,0.01,0.05,0.1"
BATCH_SIZES="32"
MAX_STEPS="160,320,640,1280"
LANGUAGE="no"
TASKS=""
NUM_ITERATIONS=3
CACHE_DIR=".euroeval_cache"
OUTPUT_DIR="sweep_runs/norbert4_xsmall"

source scratch/project_465002270/nb-embed/.venv/bin/activate

python sweep/sweep_norbert4_xsmall.py \
        --model "$MODEL" \
        --learning-rates "$LEARNING_RATES" \
        --warmup-ratios "$WARMUP_RATIOS" \
        --batch-sizes "$BATCH_SIZES" \
        --max-steps "$MAX_STEPS" \
        --language "$LANGUAGE" \
        --tasks "$TASKS" \
        --num-iterations "$NUM_ITERATIONS" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --trust-remote-code \
        --prioritize-mask