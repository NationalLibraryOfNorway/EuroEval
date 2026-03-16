#!/bin/bash
#SBATCH --job-name=euroeval-sweep-norbert4-xsmall
#SBATCH --account=project_465002270
#SBATCH --partition=small-g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

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

module use /appl/local/laifs/modules 
module load lumi-aif-singularity-bindings
export HF_HOME=/scratch/project_465002270/nb-embed/huggingface_cache/
export HIP_VISIBLE_DEVICES=0
export TORCH_COMPILE_DISABLE=1
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

singularity exec ./run_norbert4_xs.sh
