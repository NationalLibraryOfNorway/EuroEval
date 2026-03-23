"""Command-line argument parsing for sweep."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sweep script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a W&B-integrated EuroEval hyperparameter sweep \
            for one or more models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ltg/norbert4-xsmall",
        help="Model IDs to benchmark, separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="5e-6,1e-5,2e-5,5e-5,8e-5,1e-4,2e-4,5e-4",
        help="Learning rates separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--warmup-ratios",
        type=str,
        default="0.0,0.01,0.05,0.1",
        help="Warmup ratios separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="32",
        help="Finetuning batch sizes separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--max-steps",
        type=str,
        default="160,320,640,1280",
        help="Maximum finetuning steps separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=30,
        help="How often to evaluate the model during training.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=30,
        help="How often to log training metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=30,
        help="How often to save model checkpoints.",
    )
    parser.add_argument(
        "--eval-accumulation-steps",
        type=int,
        default=32,
        help="Number of steps to accumulate gradients for evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-base",
        type=int,
        default=32,
        help="Base value for computing gradient accumulation \
            (actual: base / batch_size).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of evaluation steps with no improvement before stopping.",
    )
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="adamw_torch",
        help="Optimizer to use (e.g., adamw_torch, adamw_8bit, sgd).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="Maximum number of model checkpoints to keep.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (if None, uses training batch size).",
    )
    parser.add_argument(
        "--weight-decays",
        type=str,
        default="0.0",
        help="Weight decay values to sweep, separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--lr-scheduler-types",
        type=str,
        default="linear",
        help="LR scheduler types to sweep, separated by ',', ';' or '|'.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="no",
        help="Language filter passed to Benchmarker.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help=(
            "Optional comma-separated tasks to restrict the sweep, "
            "for example: sentiment-classification,linguistic-acceptability"
        ),
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations for each trial.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".euroeval_cache",
        help="Cache directory used by EuroEval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where sweep outputs are stored. "
            "If omitted, defaults to sweep_runs/<first-model-slug>."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Benchmarker.",
    )
    parser.add_argument(
        "--prioritize-mask",
        action="store_true",
        help="Pass prioritize_mask=True when benchmarking.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep immediately if one trial fails (manual mode only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force evaluation even if existing records are present.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable progress bars in benchmark runs.",
    )
    parser.add_argument(
        "--use-wandb-sweep",
        action="store_true",
        default=True,
        help="Use W&B Sweep API for distributed sweep (requires W&B).",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Run sweep manually without W&B Sweep API.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--wandb-project", type=str, default="euroeval-sweeps", help="W&B project name."
    )
    parser.add_argument(
        "--wandb-entity", type=str, default="", help="Optional W&B entity (team/user)."
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default="", help="Optional W&B run name."
    )
    parser.add_argument(
        "--wandb-group", type=str, default="", help="Optional W&B group name."
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="euroeval,sweep",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="W&B mode. Use 'offline' if internet access is restricted.",
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default="hyperparameter-sweep",
        help="W&B job type.",
    )
    parser.add_argument(
        "--wandb-notes", type=str, default="", help="Optional notes for the W&B run."
    )
    return parser.parse_args()
