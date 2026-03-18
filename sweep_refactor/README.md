# Sweep Refactor

Model-agnostic hyperparameter sweep utilities for EuroEval finetuning benchmarks.

## Overview

This sweep tool systematically evaluates model performance across different hyperparameter
configurations. It supports sweeping over learning rate, warmup ratio, batch size, and
max steps for one or more models simultaneously.

### How It Works

1. **Configuration Cartesian Product**: All combinations of specified hyperparameters are
generated (Cartesian product)
2. **Trial Execution**: Each trial configuration:
   - Instantiates a `Benchmarker` with the trial's hyperparameters
   - Runs evaluation across specified tasks and languages
   - Collects results from all datasets
3. **Result Aggregation**: For each trial:
   - Per-dataset metrics are collected
   - Results are grouped and analyzed by task
   - Mean metrics and standard errors are computed
   - A composite total benchmark score is calculated (average of task-level scores)
4. **Output Generation**:
   - Detailed JSONL records for all benchmarks
   - Ranked CSV/JSON summary of trials
   - W&B run tracking (optional) with comprehensive metrics and artifacts

## Files

- `sweep_hyperparams.py`: Main sweep runner (750+ lines)
- `run_sweep.sh`: Local launcher using environment variables
- `run_sweep.slurm`: HPC job template for Slurm clusters

## What It Sweeps

Cartesian product combinations of:

- `models`: One or more model IDs (e.g., `ltg/norbert4-xsmall,google/mt5-small`)
- `learning_rate`: Learning rates for finetuning (e.g., `1e-5,2e-5,5e-5`)
- `warmup_ratio`: Warmup schedule ratios (e.g., `0.0,0.1`)
- `finetuning_batch_size`: Training batch sizes (e.g., `16,32`)
- `max_steps`: Maximum finetuning steps (e.g., `320,640,1280`)

Each configuration is evaluated via `euroeval.Benchmarker` across specified tasks and languages.

## Output Files

All outputs are saved to a single directory (configurable via `--output-dir`):

- **`sweep_detailed_results.jsonl`**: Full benchmark records (one JSON per line)
  - Includes trial config, dataset, task, all metrics
  - Machine-readable for post-processing and analysis
  
- **`sweep_summary.json`**: Ranked trial summary with all hyperparameters and objective
scores
  
- **`sweep_summary.csv`**: Same as JSON but in CSV format (human-readable spreadsheet)

## Weighting & Biases (W&B) Integration

### Setup

Install W&B:

```bash
pip install wandb
```

Authenticate once:

```bash
wandb login
```

### What Gets Logged

When W&B is enabled (`--wandb` or `WANDB=1`), the sweep logs comprehensive data organized
hierarchically:

#### Per-Trial Run

- **Config**: All hyperparameters (model, learning_rate, warmup_ratio, batch_size,
max_steps, language, tasks, etc.)
- **Status & Runtime**: execution status (success/failed), elapsed time in seconds
- **Dataset-Level Metrics**: Individual metric values for each dataset evaluated
  - Example: `dataset_name/accuracy`, `dataset_name/f1`, etc.
  - Includes model parameter count per dataset
- **Task-Level Metrics**: Aggregated statistics per task
  - `task/{task_name}/{metric_name}_mean`: Average metric across datasets in that task
  - `task/{task_name}/{metric_name}_se`: Standard error of the metric
  - `task/{task_name}/primary_mean`: Task's composite score (mean of test_* metrics)
  - `task/{task_name}/primary_se`: Standard error of the composite score
- **Total Benchmark Score**: `total_benchmark_score`
  - Single metric averaging the primary score across all tasks
  - Useful for comparing trial performance across different task sets

#### Summary Run

A separate W&B run (`sweep-summary`) aggregates the entire sweep with:

- Sweep group identifier and configuration
- Number of trials, successful trials, failed trials
- Best trial information (name, hyperparameters, objective score)
- W&B tables: ranked trials and benchmark records
- Artifacts: uploaded output files (JSONL, CSV, JSON)

### W&B Graph Types

The logging structure enables multiple useful visualizations:

#### 1. **Task-Specific Performance Curves**

- Graphs like `task/sentiment-classification/accuracy_mean` with error bars (`accuracy_se`)
- Shows how each task's metrics improve across trials
- Useful for identifying task-specific hyperparameter sensitivity

#### 2. **Total Benchmark Score Tracking**

- Single line showing `total_benchmark_score` across all trials
- Holistic view of overall performance progression
- Simple visual comparison of trial quality

#### 3. **Trial Status Dashboard**

- Execution time (`elapsed_seconds`) across trials
- Success/failure rates
- Status tracking for sweep completeness

#### 4. **Comparison Tables**

- **Trials Table**: Ranked trials with all hyperparameters and scores
- **Benchmarks Table**: Per-dataset results with task assignments

#### 5. **Group-Level Filtering**

- All trials are grouped under the same W&B group name
- Enables parallel sweeps on different models/configurations
- Easy filtering and comparison within W&B dashboard

### Example W&B Dashboard Features

- **Parallel coordinates plot**: Visualize hyperparameter space vs. total_benchmark_score
- **Scatter plots**: Learning rate vs. warmup_ratio colored by objective_score
- **Line charts**: Task-specific metrics with error bands
- **Tables**: Download ranked results for further analysis

## Running the Sweep

### Locally

From repository root:

```bash
bash sweep_refactor/run_sweep.sh
```

Override parameters via environment variables (exported):

```bash
export MODELS="ltg/norbert4-xsmall,google/mt5-small"
export LEARNING_RATES="1e-5,2e-5"
export WARMUP_RATIOS="0.0,0.1"
export BATCH_SIZES="16,32"
export MAX_STEPS="320,640"
export LANGUAGE="no"
export TASKS="sentiment-classification,linguistic-acceptability"
export OUTPUT_DIR="sweep_runs/multi_model_experiment"
bash sweep_refactor/run_sweep.sh
```

Or set them inline (one-liner):

```bash
MODELS="ltg/norbert4-xsmall,google/mt5-small" LEARNING_RATES="1e-5,2e-5" WARMUP_RATIOS="0.0,0.1" BATCH_SIZES="16,32" MAX_STEPS="320,640" LANGUAGE="no" TASKS="sentiment-classification,linguistic-acceptability" OUTPUT_DIR="sweep_runs/multi_model_experiment" bash sweep_refactor/run_sweep.sh
```

### With W&B Logging

Enable W&B for local runs:

```bash
export WANDB=1
export WANDB_PROJECT="euroeval-sweeps"
export WANDB_ENTITY="your-team-or-user"
export WANDB_GROUP="norwegian-models"
export WANDB_TAGS="euroeval,sweep,no"
bash sweep_refactor/run_sweep.sh
```

Offline mode (sync results later):

```bash
export WANDB=1
export WANDB_MODE=offline
bash sweep_refactor/run_sweep.sh
```

### On Slurm Clusters

Submit with defaults:

```bash
sbatch sweep_refactor/run_sweep.slurm
```

Submit with custom parameters and W&B:

```bash
sbatch --export=ALL,\
MODELS=ltg/norbert4-xsmall,\
WANDB=1,\
WANDB_PROJECT=euroeval-sweeps,\
WANDB_ENTITY=your-team-or-user,\
OUTPUT_DIR=sweep_runs/cluster_run \
sweep_refactor/run_sweep.slurm
```

### Direct Python Invocation

```bash
python sweep_refactor/sweep_hyperparams.py \
  --models "ltg/norbert4-xsmall,google/mt5-small" \
  --learning-rates "1e-5,2e-5" \
  --warmup-ratios "0.0,0.1" \
  --batch-sizes "16,32" \
  --max-steps "320,640" \
  --language "no" \
  --tasks "sentiment-classification,linguistic-acceptability" \
  --num-iterations 3 \
  --cache-dir ".euroeval_cache" \
  --output-dir "sweep_runs/multi_model_experiment" \
  --trust-remote-code \
  --prioritize-mask \
  --wandb \
  --wandb-project "euroeval-sweeps" \
  --wandb-entity "your-team-or-user" \
  --wandb-group "experiment-001" \
  --wandb-tags "euroeval,sweep,baseline"
```

## Configuration Reference

### Hyperparameter Arguments

| Argument | Default | Description |
| ---------- | --------- | ------------- |
| `--models` | `ltg/norbert4-xsmall` | Comma-separated model IDs |
| `--learning-rates` | `5e-6,1e-5,...,5e-4` | Comma-separated learning rates |
| `--warmup-ratios` | `0.0,0.01,0.05,0.1` | Comma-separated warmup ratios |
| `--batch-sizes` | `32` | Comma-separated batch sizes |
| `--max-steps` | `160,320,640,1280` | Comma-separated max steps |
| `--language` | `no` | Language filter for Benchmarker |
| `--tasks` | (empty) | Optional comma-separated tasks to restrict sweep |
| `--num-iterations` | `3` | Benchmark iterations per trial |

### Output & Control

| Argument | Default | Description |
| ---------- | --------- | ------------- |
| `--output-dir` | `sweep_runs/<model-slug>` | Output directory |
| `--cache-dir` | `.euroeval_cache` | EuroEval cache directory |
| `--stop-on-error` | (off) | Stop sweep on first trial failure |
| `--force` | (off) | Force evaluation despite existing records |
| `--no-progress-bar` | (off) | Disable progress bars |
| `--trust-remote-code` | (off) | Trust remote model code |
| `--prioritize-mask` | (off) | Use priority mask in benchmarking |

### W&B Arguments

| Argument | Default | Description |
| ---------- | --------- | ------------- |
| `--wandb` | (off) | Enable W&B logging |
| `--wandb-project` | `euroeval-sweeps` | W&B project name |
| `--wandb-entity` | (empty) | W&B entity/team (optional) |
| `--wandb-group` | (auto) | W&B run group (auto-generated if omitted) |
| `--wandb-tags` | `euroeval,sweep` | Comma-separated W&B tags |
| `--wandb-mode` | `online` | W&B mode: `online`, `offline`, or `disabled` |
| `--wandb-job-type` | `hyperparameter-sweep` | W&B job type |
| `--wandb-notes` | (empty) | Optional notes for the run |

## Example Workflows

### Quick Local Test

```bash
export MODELS="ltg/norbert4-xsmall"
export LEARNING_RATES="1e-5"
export WARMUP_RATIOS="0.1"
export BATCH_SIZES="32"
export MAX_STEPS="160"
export LANGUAGE="no"
export TASKS="sentiment-classification"
bash sweep_refactor/run_sweep.sh
```

### Multi-Model Production Sweep

```bash
export WANDB=1
export WANDB_PROJECT="euroeval-sweeps"
export WANDB_ENTITY="ml-team"
export WANDB_GROUP="v2-models-benchmark"
export MODELS="ltg/norbert4-xsmall,google/mt5-small,xlm-roberta-base"
export LEARNING_RATES="1e-5,2e-5,5e-5"
export WARMUP_RATIOS="0.0,0.1"
export BATCH_SIZES="16,32"
export MAX_STEPS="320,640"
export LANGUAGE="en"
export OUTPUT_DIR="sweep_runs/v2_models_multilingual"
bash sweep_refactor/run_sweep.sh
```

### Cluster Submission with Custom Config

```bash
sbatch \
  --export=ALL,\
MODELS="ltg/norbert4-small",\
LEARNING_RATES="1e-5,5e-5,1e-4",\
WANDB=1,\
WANDB_PROJECT="euroeval-sweeps",\
WANDB_GROUP="large-batch-experiment",\
OUTPUT_DIR="sweep_runs/large_batch_trial" \
  sweep_refactor/run_sweep.slurm
```

## Notes

- Each trial is independent; failures do not affect other trials unless `--stop-on-error`
is set
- The Cartesian product can grow large; e.g., 2 models × 5 LRs × 3 WRs × 2 BSs × 3
max_steps = **180 trials**
- W&B enables real-time monitoring of sweep progress via dashboard
- Output files are cumulative; re-running with the same output dir appends to JSONL
- Task-level metrics with standard errors enable error-aware hyperparameter selection
