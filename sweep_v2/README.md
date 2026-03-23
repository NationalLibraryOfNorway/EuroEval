# Refactored EuroEval Hyperparameter Sweep

A modern, W&B-integrated hyperparameter sweep toolkit for EuroEval finetuning benchmarks
with support for local and distributed (HPC) execution.

## Features

- **W&B Sweep Integration**: Native support for W&B Sweep API for distributed sweeping
and comprehensive metric tracking
- **Manual Mode**: Traditional manual iteration through hyperparameter grid for backward
compatibility
- **Modular Architecture**: Clean separation of concerns with standalone modules for
metrics, output, config, and utilities
- **Comprehensive Logging**: Track all metrics, configurations, and results in JSON,
CSV, and JSONL formats
- **Slurm Support**: First-class support for HPC job submission with singularity
container integration
- **Flexible List Parsing**: Support for `,`, `;`, and `|` separators in hyperparameter
lists for maximum compatibility

## Quick Start

### Local Execution

```bash
# Run a simple sweep with manual iteration
bash sweep/run_sweep.sh

# With W&B logging and W&B Sweep API
WANDB=1 USE_WANDB_SWEEP=1 bash sweep/run_sweep.sh

# Customize hyperparameters
MODELS="ltg/norbert4-small|google/mt5-small" \
LEARNING_RATES="1e-5|2e-5|5e-5" \
BATCH_SIZES="16|32" \
bash sweep/run_sweep.sh
```

### HPC (Slurm)

```bash
# Submit to cluster with custom hyperparameters
sbatch --export=ALL,\
MODELS="ltg/norbert4-small|google/mt5-small",\
LEARNING_RATES="1e-5|2e-5|5e-5",\
BATCH_SIZES="16|32",\
NUM_ITERATIONS="5" \
sweep/run_sweep.slurm

# With W&B logging
sbatch --export=ALL,\
MODELS="model1|model2",\
WANDB=1,\
WANDB_PROJECT="my-project",\
... \
sweep/run_sweep.slurm
```

## Architecture

The refactored sweep system is organized into clean, testable modules:

```terminal
sweep/
├── __init__.py              # Package definition
├── cli.py                   # Argument parsing
├── config.py                # TrialConfig dataclasses
├── metrics.py               # Metric computation and aggregation
├── output.py                # Output file generation
├── runner.py                # Main sweep orchestration with W&B integration
├── utils.py                 # Utility functions (parsing, naming, etc.)
├── sweep_hyperparams.py     # Entry point script
├── run_sweep.sh             # Local launcher
├── run_sweep.slurm          # Slurm HPC launcher
└── README.md                # This file
```

### Module Overview

#### `config.py`

Defines core data structures:

- `TrialConfig`: Stores swept hyperparameters (model, lr, warmup_ratio, batch_size,
max_steps, weight_decay, lr_scheduler_type)
- `TrialResult`: Stores trial outcome (config, score, num_records, error)

#### `utils.py`

Utility functions:

- `parse_float_list()`, `parse_int_list()`, `parse_str_list()`:
Parse comma/semicolon/pipe-separated lists
- `build_trial_name()`: Generate human-readable trial identifiers
- `score_from_metrics()`, `sanitize_model_name()`, `generate_sweep_group()`

#### `metrics.py`

Metric aggregation:

- `aggregate_by_task()`: Group benchmark results by task
- `compute_task_statistics()`: Mean and standard error per task
- `compute_total_benchmark_score()`: Overall benchmark score
- `aggregate_objective()`: Summary objective score per trial

#### `output.py`

File I/O:

- `append_trial_results()`: Write detailed JSONL records
- `write_summary_json()`, `write_summary_csv()`: Write ranked summary files

#### `runner.py`

Main orchestration:

- `run_sweep()`: Top-level sweep coordinator
- `create_sweep_config()`: Generate W&B sweep definition
- `run_trial()`: Execute a single trial
- `_run_with_wandb_sweep()`: W&B Sweep API integration
- `_run_manual_sweep()`: Traditional manual iteration

#### `cli.py`

Argument parsing with comprehensive options for:

- Swept hyperparameters (models, learning_rates, batch_sizes, max_steps, etc.)
- Fixed hyperparameters (eval_steps, logging_steps, etc.)
- W&B configuration
- Sweep mode selection (W&B Sweep API vs. manual)

## Environment Variables Reference

### Swept Hyperparameters (Cartesian Product)

```bash
MODELS="model1|model2|model3"           # Model IDs
LEARNING_RATES="1e-5|2e-5|5e-5"         # Learning rates
WARMUP_RATIOS="0.0|0.1"                 # Warmup schedule ratios
BATCH_SIZES="16|32|64"                  # Training batch sizes
MAX_STEPS="320|640|1280"                # Maximum training steps
WEIGHT_DECAYS="0.0|0.01|0.1"            # L2 regularization
LR_SCHEDULER_TYPES="linear|cosine"      # Learning rate scheduler types
```

### Fixed Hyperparameters (Applied to All Trials)

```bash
EVAL_STEPS="30"                         # Evaluation frequency
LOGGING_STEPS="30"                      # Logging frequency
SAVE_STEPS="30"                         # Checkpoint frequency
EVAL_ACCUMULATION_STEPS="32"
GRADIENT_ACCUMULATION_BASE="32"
EARLY_STOPPING_PATIENCE="5"
OPTIMIZER_NAME="adamw_torch"
SAVE_TOTAL_LIMIT="1"
PER_DEVICE_EVAL_BATCH_SIZE=""           # (optional, defaults to train batch size)
```

### Benchmark Configuration

```bash
LANGUAGE="no"                           # Language filter
TASKS=""                                # Optional task filter (comma-separated)
NUM_ITERATIONS="3"                      # Iterations per trial
CACHE_DIR=".euroeval_cache"             # Model/data cache
OUTPUT_DIR=""                           # Results output directory (auto if empty)
TRUST_REMOTE_CODE="1"                   # Allow remote code execution
PRIORITIZE_MASK="1"                     # Use prioritization mask
```

### W&B Configuration

```bash
WANDB="1"                               # Enable W&B logging
WANDB_PROJECT="euroeval-sweeps"         # W&B project name
WANDB_ENTITY=""                         # W&B team/account
WANDB_GROUP=""                          # Group name for runs
WANDB_TAGS="euroeval,sweep"             # Tags for filtering
WANDB_MODE="online"                     # online|offline|disabled
WANDB_JOB_TYPE="hyperparameter-sweep"   # Job type
WANDB_NOTES=""                          # Run notes
```

### Sweep Mode Configuration

```bash
USE_WANDB_SWEEP="1"                     # Use W&B Sweep API (requires WANDB=1)
MANUAL_ONLY="0"                         # Force manual iteration mode
```

## Sweep Modes

### Mode 1: Manual Iteration (Default)

Traditional sequential execution through the hyperparameter grid.

```bash
python -m sweep.sweep_hyperparams \
    --models ltg/norbert4-xsmall \
    --learning-rates 1e-5,2e-5,5e-5 \
    --batch-sizes 16,32 \
    --wandb \
    --manual-only  # Forces manual mode even with --wandb
```

**Advantages:**

- Simple, predictable execution
- Works without W&B account
- Easy local development

**Output:**

- `sweep_detailed_results.jsonl`: All benchmark records
- `sweep_summary.json`: Ranked trial summary
- `sweep_summary.csv`: Same as JSON, spreadsheet format
- Individual W&B runs (if `--wandb` enabled)

### Mode 2: W&B Sweep API (Distributed)

Uses W&B's sweep infrastructure for distributed hyperparameter optimization.

```bash
python -m sweep.sweep_hyperparams \
    --models ltg/norbert4-xsmall \
    --learning-rates 1e-5,2e-5,5e-5 \
    --batch-sizes 16,32 \
    --wandb \
    --use-wandb-sweep \
    --wandb-project my-project
```

**Advantages:**

- Distributed execution across multiple machines
- Automatic agent coordination
- Centralized metric collection
- W&B dashboard integration

**Requirements:**

- W&B account and API key
- `wandb` Python package

**Workflow:**

1. Define sweep configuration → W&B Sweep API
2. Create agents on local/remote machines
3. Each agent pulls trials and reports metrics
4. Central W&B dashboard aggregates results

## Output Files

All outputs are saved to `--output-dir` (default: `sweep_runs/<model-slug>/`):

### `sweep_detailed_results.jsonl`

Machine-readable detailed records (one JSON per line):

```json
{
  "trial_name": "model_ltg__norbert4_xsmall_lr_1e-05_wr_0p01_bs_32_ms_320_wd_0p0_sched_linear",
  "trial_config": {
    "model": "ltg/norbert4-xsmall",
    "learning_rate": 1e-05,
    "warmup_ratio": 0.01,
    ...
  },
  "benchmark_result": {
    "dataset": "...",
    "task": "...",
    "results": {...}
  }
}
```

### `sweep_summary.json`

Human-readable ranked summary of all trials:

```json
[
  {
    "trial_name": "...",
    "model": "ltg/norbert4-xsmall",
    "learning_rate": 1e-05,
    "objective_score": 0.85,
    "num_records": 10,
    "error": null
  },
  ...
]
```

### `sweep_summary.csv`

Same as JSON but in spreadsheet-friendly CSV format.

## List Separator Support

The sweep tools support multiple separators for flexibility, especially with HPC job submission:

- **Comma**: `1e-5,2e-5,5e-5` (standard)
- **Semicolon**: `1e-5;2e-5;5e-5` (alternative)
- **Pipe**: `1e-5|2e-5|5e-5` (HPC-friendly, avoids sbatch issues)
- **Escaped comma**: `1e-5,2e-5\,5`, comma literals in values

**HPC Tip**: When using `sbatch --export`, sbatch treats commas as variable separators.
Use pipes instead:

```bash
# ❌ DON'T: First value only is parsed
sbatch --export=ALL,LEARNING_RATES=1e-5,2e-5,5e-5

# ✅ DO: All values are parsed correctly
sbatch --export=ALL,LEARNING_RATES="1e-5|2e-5|5e-5"
```

## Examples

### Example 1: Multi-Model Comparison (Local)

```bash
cd /path/to/EuroEval

MODELS="ltg/norbert4-xsmall|ltg/norbert4-small|google/mt5-small" \
LEARNING_RATES="1e-5|2e-5|5e-5" \
BATCH_SIZES="16|32" \
NUM_ITERATIONS="3" \
bash sweep/run_sweep.sh
```

### Example 2: Learning Rate Grid with W&B (Local)

```bash
WANDB=1 \
WANDB_PROJECT="euroeval-lr-tuning" \
WANDB_TAGS="learning-rate,grid-search" \
LEARNING_RATES="5e-6|1e-5|2e-5|5e-5|1e-4|2e-4|5e-4" \
bash sweep/run_sweep.sh
```

### Example 3: HPC Submission with Distributed W&B Sweep

```bash
sbatch \
    --export=ALL,\
MODELS="ltg/norbert4-small|google/mt5-small",\
LEARNING_RATES="1e-5|2e-5|5e-5|1e-4",\
BATCH_SIZES="16|32",\
MAX_STEPS="320|640|1280",\
NUM_ITERATIONS="5",\
WANDB=1,\
WANDB_PROJECT="euroeval-sweeps",\
WANDB_ENTITY="my-team",\
USE_WANDB_SWEEP=1,\
MANUAL_ONLY=0 \
    sweep/run_sweep.slurm
```

### Example 4: W&B Sweep with Custom Scheduler

```bash
WANDB=1 \
WANDB_PROJECT="euroeval-schedules" \
USE_WANDB_SWEEP=1 \
LR_SCHEDULER_TYPES="linear|cosine|constant_with_warmup" \
bash sweep/run_sweep.sh
```

## Troubleshooting

### W&B Login Issues

```bash
# Initialize W&B
wandb login

# Check authentication
wandb verify

# Use offline mode if network is restricted
WANDB_MODE=offline bash sweep/run_sweep.sh
```

### HPC Sbatch Variable Parsing

If only the first value from a comma-separated list is used:

```bash
# This happens because sbatch --export uses commas as separators
sbatch --export=ALL,LEARNING_RATES="1e-5,2e-5"  # ❌ Only 1e-5 is used

# Fix: Use alternative separator
sbatch --export=ALL,LEARNING_RATES="1e-5|2e-5"  # ✅ Works correctly
```

### Memory Issues on HPC

Increase memory allocation:

```bash
sbatch --mem=64G sweep/run_sweep.slurm
```

### Monitoring Slurm Job

```bash
# Check job status
squeue -j <job-id>

# Monitor in real-time
tail -f slurm_<job-id>.log
```

## Migration from Old Sweep Format

The new sweep maintains backward compatibility with command-line arguments but improves
internal structure:

**Old format:**

```bash
python sweep_refactor/sweep_hyperparams.py --models ... --learning-rates ...
```

**New format:**

```bash
python -m sweep.sweep_hyperparams --models ... --learning-rates ...
```

All argument names and defaults remain the same for easy migration.

## Advanced Usage

### Using W&B Sweep API Programmatically

```python
from sweep.runner import run_sweep
from sweep.utils import parse_float_list, parse_str_list

# Programmatic sweep without CLI parsing
run_sweep(
    models=["ltg/norbert4-xsmall", "google/mt5-small"],
    learning_rates=[1e-5, 2e-5, 5e-5],
    warmup_ratios=[0.0, 0.1],
    batch_sizes=[32],
    max_steps_values=[320, 640],
    weight_decays=[0.0],
    lr_scheduler_types=["linear"],
    args=args,
    use_wandb_sweep=True,
)
```

### Extending Metrics

Edit `metrics.py` to add custom metric aggregation:

```python
def compute_custom_metric(results_by_task):
    # Custom aggregation logic
    ...
```

## Performance Tips

1. **Use W&B Sweep API** for distributed execution across multiple machines
2. **Increase NUM_ITERATIONS** on stable hardware (can reduce variance)
3. **Use pipe `|` separator** for lists to avoid shell expansion issues
4. **Enable HPC caching** with `--cache-dir` to avoid redundant downloads

## Contributing

To extend the sweep system:

1. Add new utility functions to `utils.py`
2. Add new metrics to `metrics.py`
3. Add new CLI options to `cli.py`
4. Update the launcher scripts and documentation

## See Also

- Main README: [../../README.md](../../README.md)
- EuroEval Documentation: [../../docs/](../../docs/)
- W&B Documentation: [https://wandb.ai/docs/](https://wandb.ai/docs/)
