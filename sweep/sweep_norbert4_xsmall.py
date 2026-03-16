"""Run a hyperparameter sweep for EuroEval finetuning benchmarks.

This script sweeps learning rate, warmup ratio, finetuning batch size, and max
steps for a single model (default: ltg/norbert4-xsmall), using EuroEval's
Benchmarker API. Each trial writes its own results JSONL in an isolated folder,
and the script builds a ranked summary file with the best configuration.
"""

# ruff: noqa: I001

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from euroeval import Benchmarker  # type: ignore[reportMissingImports]


IGNORED_METRIC_SUFFIXES = ("_se",)
IGNORED_METRIC_KEYS = {
    "num_failed_instances",
    "test_loss",
    "test_runtime",
    "test_samples_per_second",
    "test_steps_per_second",
}


@dataclass(frozen=True)
class TrialConfig:
    """Sweep trial configuration."""

    learning_rate: float
    warmup_ratio: float
    finetuning_batch_size: int
    max_steps: int


@dataclass
class TrialResult:
    """Sweep trial result."""

    config: TrialConfig
    objective_score: float | None
    num_records: int
    trial_dir: Path
    error: str | None = None


def _parse_float_list(raw: str) -> list[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(value) for value in values]


def _parse_int_list(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(value) for value in values]


def _score_from_total_metrics(total_metrics: dict[str, Any]) -> float | None:
    values: list[float] = []
    for key, value in total_metrics.items():
        if not key.startswith("test_"):
            continue
        if key in IGNORED_METRIC_KEYS:
            continue
        if any(key.endswith(suffix) for suffix in IGNORED_METRIC_SUFFIXES):
            continue
        if isinstance(value, int | float):
            values.append(float(value))

    if not values:
        return None

    return sum(values) / len(values)


def _aggregate_objective(results: list[Any]) -> tuple[float | None, int]:
    row_scores: list[float] = []

    for result in results:
        result_dict = (
            result.model_dump() if hasattr(result, "model_dump") else dict(result)
        )
        score_dict = result_dict.get("results", {})
        total_metrics = score_dict.get("total", {})
        if not isinstance(total_metrics, dict):
            continue

        row_score = _score_from_total_metrics(total_metrics=total_metrics)
        if row_score is not None:
            row_scores.append(row_score)

    if not row_scores:
        return None, 0

    return sum(row_scores) / len(row_scores), len(row_scores)


def _build_trial_name(config: TrialConfig) -> str:
    lr = f"{config.learning_rate:.1e}".replace("+", "")
    wr = str(config.warmup_ratio).replace(".", "p")
    bs = str(config.finetuning_batch_size)
    ms = str(config.max_steps)
    return f"lr_{lr}_wr_{wr}_bs_{bs}_ms_{ms}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sweep script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a EuroEval hyperparameter sweep for one model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ltg/norbert4-xsmall",
        help="Model ID to benchmark.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="5e-6,1e-5,2e-5,5e-5,8e-5,1e-4,2e-4,5e-4",
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--warmup-ratios",
        type=str,
        default="0.0,0.01,0.05,0.1",
        help="Comma-separated warmup ratios.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="32",
        help="Comma-separated finetuning batch sizes.",
    )
    parser.add_argument(
        "--max-steps",
        type=str,
        default="160,320,640,1280",
        help="Comma-separated maximum finetuning steps.",
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
        default=Path("sweep_runs/norbert4_xsmall"),
        help="Directory where sweep trial outputs and summary are stored.",
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
        help="Stop the sweep immediately if one trial fails.",
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
    return parser.parse_args()


def main() -> None:
    """Run sweep trials and save ranked summary files."""
    args = parse_args()

    learning_rates = _parse_float_list(args.learning_rates)
    warmup_ratios = _parse_float_list(args.warmup_ratios)
    batch_sizes = _parse_int_list(args.batch_sizes)
    max_steps_values = _parse_int_list(args.max_steps)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()] or None

    configs = [
        TrialConfig(
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            finetuning_batch_size=batch_size,
            max_steps=max_steps,
        )
        for learning_rate, warmup_ratio, batch_size, max_steps in itertools.product(
            learning_rates, warmup_ratios, batch_sizes, max_steps_values
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[TrialResult] = []

    print(f"Running {len(configs)} sweep trials for model: {args.model}")

    for index, config in enumerate(configs, start=1):
        trial_name = _build_trial_name(config)
        trial_dir = args.output_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(configs)}] Trial {trial_name}")

        try:
            benchmarker = Benchmarker(
                task=tasks,
                language=args.language,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                finetuning_batch_size=config.finetuning_batch_size,
                max_steps=config.max_steps,
                num_iterations=args.num_iterations,
                progress_bar=not args.no_progress_bar,
                save_results=True,
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir,
                force=args.force,
                raise_errors=True,
            )

            benchmarker.results_path = trial_dir / "euroeval_benchmark_results.jsonl"
            benchmark_results = list(
                benchmarker.benchmark(
                    model=[args.model],
                    prioritize_mask=args.prioritize_mask,
                )
            )

            objective_score, num_records = _aggregate_objective(benchmark_results)
            summary_rows.append(
                TrialResult(
                    config=config,
                    objective_score=objective_score,
                    num_records=num_records,
                    trial_dir=trial_dir,
                )
            )

        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            summary_rows.append(
                TrialResult(
                    config=config,
                    objective_score=None,
                    num_records=0,
                    trial_dir=trial_dir,
                    error=error_message,
                )
            )
            print(f"  Failed: {error_message}")
            if args.stop_on_error:
                break

    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            float("-inf") if row.objective_score is None else row.objective_score
        ),
        reverse=True,
    )

    summary_json = args.output_dir / "sweep_summary.json"
    summary_csv = args.output_dir / "sweep_summary.csv"

    json_payload = [
        {
            "learning_rate": row.config.learning_rate,
            "warmup_ratio": row.config.warmup_ratio,
            "finetuning_batch_size": row.config.finetuning_batch_size,
            "max_steps": row.config.max_steps,
            "objective_score": row.objective_score,
            "num_records": row.num_records,
            "trial_dir": str(row.trial_dir),
            "error": row.error,
        }
        for row in ranked_rows
    ]
    summary_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "learning_rate",
                "warmup_ratio",
                "finetuning_batch_size",
                "max_steps",
                "objective_score",
                "num_records",
                "trial_dir",
                "error",
            ],
        )
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    "learning_rate": row.config.learning_rate,
                    "warmup_ratio": row.config.warmup_ratio,
                    "finetuning_batch_size": row.config.finetuning_batch_size,
                    "max_steps": row.config.max_steps,
                    "objective_score": row.objective_score,
                    "num_records": row.num_records,
                    "trial_dir": str(row.trial_dir),
                    "error": row.error or "",
                }
            )

    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")

    best = next((row for row in ranked_rows if row.objective_score is not None), None)
    if best is None:
        print("No successful trial produced an objective score.")
        return

    print("Best trial:")
    print(f"  learning_rate={best.config.learning_rate}")
    print(f"  warmup_ratio={best.config.warmup_ratio}")
    print(f"  finetuning_batch_size={best.config.finetuning_batch_size}")
    print(f"  max_steps={best.config.max_steps}")
    print(f"  objective_score={best.objective_score:.4f}")
    print(f"  trial_dir={best.trial_dir}")


if __name__ == "__main__":
    main()
