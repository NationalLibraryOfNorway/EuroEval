# Sweep Refactor

Model-agnostic hyperparameter sweep utilities for EuroEval finetuning benchmarks.

## Files

- `sweep_hyperparams.py`: Main sweep runner.
- `run_sweep.sh`: Local launcher using environment variables.
- `run_sweep.slurm`: Slurm launcher (generic template).

## What It Does

The runner sweeps combinations of:

- `models` (one or more)
- `learning_rate`
- `warmup_ratio`
- `finetuning_batch_size`
- `max_steps`

and evaluates each trial with `euroeval.Benchmarker`.

Outputs are compact and unified in a single output directory:

- `sweep_detailed_results.jsonl`: full benchmark records for all trials
- `sweep_summary.json`: ranked trial summary
- `sweep_summary.csv`: ranked trial summary (CSV)

## Weights and Biases Logging

Install W&B in your environment:

```bash
pip install wandb
```

Authenticate once (online mode):

```bash
wandb login
```

Enable W&B for local runs:

```bash
WANDB=1 \
WANDB_PROJECT="euroeval-sweeps" \
WANDB_ENTITY="your-team-or-user" \
WANDB_GROUP="norwegian-models" \
WANDB_TAGS="euroeval,sweep,no" \
bash sweep_refactor/run_sweep.sh
```

Enable W&B for Slurm runs:

```bash
sbatch --export=ALL,WANDB=1,WANDB_PROJECT=euroeval-sweeps,WANDB_ENTITY=your-team-or-user sweep_refactor/run_sweep.slurm
```

Offline logging (sync later):

```bash
WANDB=1 WANDB_MODE=offline bash sweep_refactor/run_sweep.sh
```

When enabled, the script logs:

- Full sweep configuration
- Per-trial status, runtime, and objective score
- Per-benchmark record metadata and numeric total metrics
- Ranked trial summary as a W&B table
- Detailed benchmark records as a W&B table
- Output files (`sweep_summary.json`, `sweep_summary.csv`,
  `sweep_detailed_results.jsonl`) as a W&B artifact

## Run Locally

From repository root:

```bash
bash sweep_refactor/run_sweep.sh
```

Override parameters via environment variables:

```bash
MODELS="ltg/norbert4-xsmall,google/mt5-small" \
LEARNING_RATES="1e-5,2e-5" \
WARMUP_RATIOS="0.0,0.1" \
BATCH_SIZES="16,32" \
MAX_STEPS="320,640" \
LANGUAGE="no" \
TASKS="sentiment-classification,linguistic-acceptability" \
OUTPUT_DIR="sweep_runs/multi_model_experiment" \
bash sweep_refactor/run_sweep.sh
```

## Run With Slurm

Submit with defaults:

```bash
sbatch sweep_refactor/run_sweep.slurm
```

Submit with overrides:

```bash
sbatch --export=ALL,MODELS=ltg/norbert4-xsmall,google/mt5-small,OUTPUT_DIR=sweep_runs/multi_model_experiment sweep_refactor/run_sweep.slurm
```

## Direct Python Usage

```bash
python sweep_refactor/sweep_hyperparams.py \
  --models "ltg/norbert4-xsmall,google/mt5-small" \
  --learning-rates "1e-5,2e-5" \
  --warmup-ratios "0.0,0.1" \
  --batch-sizes "16,32" \
  --max-steps "320,640" \
  --language "no" \
  --tasks "" \
  --num-iterations 3 \
  --cache-dir ".euroeval_cache" \
  --output-dir "sweep_runs/multi_model_experiment" \
  --trust-remote-code \
  --prioritize-mask
```

## Notes

- `--models` is comma-separated and supports one or many model IDs.
- If `--output-dir` is omitted, the script defaults to `sweep_runs/<first-model-slug>`.
- Use `--stop-on-error` if you want the sweep to terminate on the first failed trial.
