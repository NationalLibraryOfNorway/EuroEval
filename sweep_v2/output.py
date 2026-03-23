"""Output generation and file writing utilities."""

import csv
import json
from pathlib import Path
from typing import Any

from .config import TrialResult


def append_trial_results(
    results_path: Path,
    trial_name: str,
    config_dict: dict[str, Any],
    benchmark_results: list[Any],
) -> None:
    """Append trial results to JSONL file.

    Args:
        results_path: Path to JSONL file.
        trial_name: Readable name for the trial.
        config_dict: Trial configuration as dictionary.
        benchmark_results: List of benchmark result objects.
    """
    with results_path.open("a", encoding="utf-8") as handle:
        for benchmark_result in benchmark_results:
            result_dict = (
                benchmark_result.model_dump()
                if hasattr(benchmark_result, "model_dump")
                else dict(benchmark_result)
            )
            payload = {
                "trial_name": trial_name,
                "trial_config": config_dict,
                "benchmark_result": result_dict,
            }
            handle.write(json.dumps(payload) + "\n")


def write_summary_json(summary_path: Path, ranked_rows: list[TrialResult]) -> None:
    """Write trial summary to JSON file.

    Args:
        summary_path: Path to output JSON file.
        ranked_rows: List of TrialResult objects, ranked by objective score.
    """
    json_payload = [
        {
            "trial_name": row.trial_name,
            "model": row.config.model,
            "learning_rate": row.config.learning_rate,
            "warmup_ratio": row.config.warmup_ratio,
            "finetuning_batch_size": row.config.finetuning_batch_size,
            "max_steps": row.config.max_steps,
            "weight_decay": row.config.weight_decay,
            "lr_scheduler_type": row.config.lr_scheduler_type,
            "objective_score": row.objective_score,
            "num_records": row.num_records,
            "error": row.error,
        }
        for row in ranked_rows
    ]
    summary_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def write_summary_csv(summary_path: Path, ranked_rows: list[TrialResult]) -> None:
    """Write trial summary to CSV file.

    Args:
        summary_path: Path to output CSV file.
        ranked_rows: List of TrialResult objects, ranked by objective score.
    """
    fieldnames = [
        "trial_name",
        "model",
        "learning_rate",
        "warmup_ratio",
        "finetuning_batch_size",
        "max_steps",
        "weight_decay",
        "lr_scheduler_type",
        "objective_score",
        "num_records",
        "error",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    "trial_name": row.trial_name,
                    "model": row.config.model,
                    "learning_rate": row.config.learning_rate,
                    "warmup_ratio": row.config.warmup_ratio,
                    "finetuning_batch_size": row.config.finetuning_batch_size,
                    "max_steps": row.config.max_steps,
                    "weight_decay": row.config.weight_decay,
                    "lr_scheduler_type": row.config.lr_scheduler_type,
                    "objective_score": row.objective_score,
                    "num_records": row.num_records,
                    "error": row.error or "",
                }
            )
