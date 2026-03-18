"""Run a model-agnostic hyperparameter sweep for EuroEval finetuning benchmarks.

This script supports sweeping one or multiple models over learning rate, warmup
ratio, finetuning batch size, and max steps. Detailed benchmark records from all
trials are consolidated in a single JSONL file, while ranked summaries are
written to JSON and CSV.
"""

# ruff: noqa: I001

import argparse
import csv
import datetime as dt
import importlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from euroeval import Benchmarker


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

    model: str
    learning_rate: float
    warmup_ratio: float
    finetuning_batch_size: int
    max_steps: int


@dataclass
class TrialResult:
    """Sweep trial result."""

    config: TrialConfig
    trial_name: str
    objective_score: float | None
    num_records: int
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


def _parse_str_list(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one model value.")
    return values


def _import_wandb(enable_wandb: bool) -> object | None:
    if not enable_wandb:
        return None

    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "W&B logging requested but the 'wandb' package is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    return wandb


def _wandb_init(
    wandb_module: object | None,
    *,
    project: str,
    entity: str,
    name: str,
    group: str,
    notes: str,
    tags: list[str],
    job_type: str,
    mode: str,
    config: dict[str, Any],
) -> object | None:
    if wandb_module is None:
        return None

    init_fn = getattr(wandb_module, "init", None)
    if not callable(init_fn):
        return None

    return init_fn(
        project=project,
        entity=entity or None,
        name=name or None,
        group=group or None,
        notes=notes or None,
        tags=tags,
        job_type=job_type,
        mode=mode,
        config=config,
    )


def _wandb_define_metric(
    wandb_module: object | None,
    metric_name: str,
    step_metric: str | None = None,
) -> None:
    if wandb_module is None:
        return

    define_metric_fn = getattr(wandb_module, "define_metric", None)
    if not callable(define_metric_fn):
        return

    if step_metric is None:
        define_metric_fn(metric_name)
    else:
        define_metric_fn(metric_name, step_metric=step_metric)


def _wandb_log(wandb_run: object | None, payload: dict[str, Any]) -> None:
    if wandb_run is None:
        return

    log_fn = getattr(wandb_run, "log", None)
    if callable(log_fn):
        log_fn(payload)


def _wandb_set_summary(wandb_run: object | None, key: str, value: object) -> None:
    if wandb_run is None:
        return

    summary_obj = getattr(wandb_run, "summary", None)
    if summary_obj is not None:
        summary_obj[key] = value


def _wandb_new_table(wandb_module: object | None, columns: list[str]) -> object | None:
    if wandb_module is None:
        return None

    table_cls = getattr(wandb_module, "Table", None)
    if not callable(table_cls):
        return None

    return table_cls(columns=columns)


def _wandb_table_add_row(table: object | None, row_values: list[Any]) -> None:
    if table is None:
        return

    add_data_fn = getattr(table, "add_data", None)
    if callable(add_data_fn):
        add_data_fn(*row_values)


def _wandb_new_artifact(
    wandb_module: object | None, name: str, artifact_type: str
) -> object | None:
    if wandb_module is None:
        return None

    artifact_cls = getattr(wandb_module, "Artifact", None)
    if not callable(artifact_cls):
        return None

    return artifact_cls(name=name, type=artifact_type)


def _wandb_artifact_add_file(
    artifact: object | None, local_path: str, name: str
) -> None:
    if artifact is None:
        return

    add_file_fn = getattr(artifact, "add_file", None)
    if callable(add_file_fn):
        add_file_fn(local_path=local_path, name=name)


def _wandb_log_artifact(wandb_run: object | None, artifact: object | None) -> None:
    if wandb_run is None or artifact is None:
        return

    log_artifact_fn = getattr(wandb_run, "log_artifact", None)
    if callable(log_artifact_fn):
        log_artifact_fn(artifact)


def _wandb_finish(wandb_run: object | None) -> None:
    if wandb_run is None:
        return

    finish_fn = getattr(wandb_run, "finish", None)
    if callable(finish_fn):
        finish_fn()


def _flatten_numeric_dict(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, int | float):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def _parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def _generate_sweep_group(models: list[str]) -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_model_name(models[0])[:30]
    return f"sweep_{slug}_{timestamp}"


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


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace("@", "_at_")


def _build_trial_name(config: TrialConfig) -> str:
    model = _sanitize_model_name(config.model)
    lr = f"{config.learning_rate:.1e}".replace("+", "")
    wr = str(config.warmup_ratio).replace(".", "p")
    bs = str(config.finetuning_batch_size)
    ms = str(config.max_steps)
    return f"model_{model}_lr_{lr}_wr_{wr}_bs_{bs}_ms_{ms}"


def _trial_config_dict(config: TrialConfig) -> dict[str, int | float | str]:
    return {
        "model": config.model,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "finetuning_batch_size": config.finetuning_batch_size,
        "max_steps": config.max_steps,
    }


def _append_trial_results(
    results_path: Path,
    trial_name: str,
    config: TrialConfig,
    benchmark_results: list[Any],
) -> None:
    with results_path.open("a", encoding="utf-8") as handle:
        for benchmark_result in benchmark_results:
            result_dict = (
                benchmark_result.model_dump()
                if hasattr(benchmark_result, "model_dump")
                else dict(benchmark_result)
            )
            payload = {
                "trial_name": trial_name,
                "trial_config": _trial_config_dict(config),
                "benchmark_result": result_dict,
            }
            handle.write(json.dumps(payload) + "\n")


def _aggregate_by_task(
    benchmark_results: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    """Group benchmark results by task.

    Args:
        benchmark_results: List of benchmark result objects/dicts.

    Returns:
        Dict mapping task name to list of result dicts for that task.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}

    for benchmark_result in benchmark_results:
        result_dict = (
            benchmark_result.model_dump()
            if hasattr(benchmark_result, "model_dump")
            else dict(benchmark_result)
        )
        task = result_dict.get("task", "unknown")
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(result_dict)

    return by_task


def _compute_task_statistics(
    results_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float | None]]:
    """Compute mean and standard error for metrics per task.

    For each metric in the results of each task, compute mean over
    all datasets and standard error from the _se suffix values.

    Args:
        results_by_task: Dict mapping task to list of result dicts.

    Returns:
        Dict mapping task to dict of metric -> {mean, se}.
        The 'primary' key holds the mean of the primary scoring metric for that task.
    """
    task_stats: dict[str, dict[str, float | None]] = {}

    for task, task_results in results_by_task.items():
        task_stats[task] = {}
        metric_values: dict[str, list[float]] = {}
        metric_errors: dict[str, list[float]] = {}

        # Collect all metric values and their standard errors across datasets
        for result_dict in task_results:
            total_metrics = result_dict.get("results", {}).get("total", {})
            if not isinstance(total_metrics, dict):
                continue
            for key, value in total_metrics.items():
                if isinstance(value, int | float):
                    # Skip _se suffix metrics (we'll use them separately)
                    if key.endswith("_se"):
                        continue
                    if key not in metric_values:
                        metric_values[key] = []
                        metric_errors[key] = []
                    metric_values[key].append(float(value))
                    # Extract corresponding _se value if available
                    se_key = f"{key}_se"
                    se_value = total_metrics.get(se_key)
                    if se_value is not None and isinstance(se_value, int | float):
                        metric_errors[key].append(float(se_value))
                    else:
                        metric_errors[key].append(0.0)

        # Compute mean and combined standard error for each metric
        for metric_name, values in metric_values.items():
            if values:
                task_stats[task][f"{metric_name}_mean"] = sum(values) / len(values)
                
                # Combine standard errors: sqrt(sum(se_i^2)) / n
                errors = metric_errors.get(metric_name, [])
                if errors and any(e > 0 for e in errors):
                    combined_se = (sum(e**2 for e in errors) ** 0.5) / len(errors)
                    task_stats[task][f"{metric_name}_se"] = combined_se
                else:
                    task_stats[task][f"{metric_name}_se"] = 0.0

        # Compute primary score (average of test_* metrics for this task)
        primary_scores = []
        primary_errors = []
        for result_dict in task_results:
            total_metrics = result_dict.get("results", {}).get("total", {})
            score = _score_from_total_metrics(total_metrics)
            if score is not None:
                primary_scores.append(score)
                # Average the standard errors of the test_* metrics for this dataset
                test_errors = []
                for key, value in total_metrics.items():
                    if key.startswith("test_") and key.endswith("_se"):
                        if isinstance(value, int | float):
                            test_errors.append(float(value))
                if test_errors:
                    primary_errors.append(sum(test_errors) / len(test_errors))
                else:
                    primary_errors.append(0.0)

        if primary_scores:
            task_stats[task]["primary_mean"] = sum(primary_scores) / len(
                primary_scores
            )
            if primary_errors:
                combined_primary_se = (
                    sum(e**2 for e in primary_errors) ** 0.5
                ) / len(primary_errors)
                task_stats[task]["primary_se"] = combined_primary_se
            else:
                task_stats[task]["primary_se"] = 0.0
        else:
            task_stats[task]["primary_mean"] = None
            task_stats[task]["primary_se"] = None

    return task_stats



def _compute_total_benchmark_score(
    results_by_task: dict[str, list[dict[str, Any]]],
) -> float | None:
    """Compute total benchmark score averaging one metric per task.

    Computes the primary score for each task, then averages across tasks.

    Args:
        results_by_task: Dict mapping task to list of result dicts.

    Returns:
        Average of task-level primary scores, or None if no valid scores.
    """
    task_scores = []

    for task, task_results in results_by_task.items():
        primary_scores = []
        for result_dict in task_results:
            total_metrics = result_dict.get("results", {}).get("total", {})
            score = _score_from_total_metrics(total_metrics)
            if score is not None:
                primary_scores.append(score)

        if primary_scores:
            task_score = sum(primary_scores) / len(primary_scores)
            task_scores.append(task_score)

    if not task_scores:
        return None

    return sum(task_scores) / len(task_scores)


def _default_output_dir(first_model: str) -> Path:
    slug = _sanitize_model_name(first_model)
    return Path("sweep_runs") / slug


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sweep script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a EuroEval hyperparameter sweep for one or more models."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ltg/norbert4-xsmall",
        help="Comma-separated model IDs to benchmark.",
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="euroeval-sweeps",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="Optional W&B entity (team/user).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="Optional W&B group name.",
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
        "--wandb-notes",
        type=str,
        default="",
        help="Optional notes for the W&B run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run sweep trials and save ranked summary files."""
    args = parse_args()

    models = _parse_str_list(args.models)
    learning_rates = _parse_float_list(args.learning_rates)
    warmup_ratios = _parse_float_list(args.warmup_ratios)
    batch_sizes = _parse_int_list(args.batch_sizes)
    max_steps_values = _parse_int_list(args.max_steps)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()] or None
    wandb_module = _import_wandb(enable_wandb=args.wandb)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _default_output_dir(models[0])
    )

    configs = [
        TrialConfig(
            model=model,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            finetuning_batch_size=batch_size,
            max_steps=max_steps,
        )
        for model, learning_rate, warmup_ratio, batch_size, max_steps in (
            itertools.product(
                models, learning_rates, warmup_ratios, batch_sizes, max_steps_values
            )
        )
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[TrialResult] = []
    benchmark_table_rows: list[dict[str, Any]] = []
    detailed_results_jsonl = output_dir / "sweep_detailed_results.jsonl"
    detailed_results_jsonl.write_text("", encoding="utf-8")
    benchmark_record_index = 0
    sweep_group = args.wandb_group or _generate_sweep_group(models)
    if wandb_module is not None:
        print(f"W&B logging enabled. Sweep group: {sweep_group}")

    print(f"Running {len(configs)} sweep trials across {len(models)} model(s).")
    print(f"Detailed trial records will be saved to: {detailed_results_jsonl}")

    for index, config in enumerate(configs, start=1):
        trial_name = _build_trial_name(config)

        print(f"[{index}/{len(configs)}] Trial {trial_name}")

        trial_started = dt.datetime.now(dt.UTC)
        wandb_run: object | None = _wandb_init(
            wandb_module=wandb_module,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=trial_name,
            group=sweep_group,
            notes=args.wandb_notes,
            tags=_parse_tags(args.wandb_tags),
            job_type=args.wandb_job_type,
            mode=args.wandb_mode,
            config={
                "model": config.model,
                "learning_rate": config.learning_rate,
                "warmup_ratio": config.warmup_ratio,
                "finetuning_batch_size": config.finetuning_batch_size,
                "max_steps": config.max_steps,
                "language": args.language,
                "tasks": tasks,
                "num_iterations": args.num_iterations,
                "trust_remote_code": args.trust_remote_code,
                "prioritize_mask": args.prioritize_mask,
            },
        )

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
                save_results=False,
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir,
                force=args.force,
                raise_errors=True,
            )

            benchmark_results = list(
                benchmarker.benchmark(
                    model=[config.model],
                    prioritize_mask=args.prioritize_mask,
                )
            )
            _append_trial_results(
                results_path=detailed_results_jsonl,
                trial_name=trial_name,
                config=config,
                benchmark_results=benchmark_results,
            )

            objective_score, num_records = _aggregate_objective(benchmark_results)
            trial_finished = dt.datetime.now(dt.UTC)
            summary_rows.append(
                TrialResult(
                    config=config,
                    trial_name=trial_name,
                    objective_score=objective_score,
                    num_records=num_records,
                )
            )

            # Group results by task and compute statistics
            results_by_task = _aggregate_by_task(benchmark_results)
            task_stats = _compute_task_statistics(results_by_task)
            total_benchmark_score = _compute_total_benchmark_score(results_by_task)

            # Log task-level metrics and individual benchmark records
            for benchmark_result in benchmark_results:
                result_dict = (
                    benchmark_result.model_dump()
                    if hasattr(benchmark_result, "model_dump")
                    else dict(benchmark_result)
                )
                dataset = result_dict.get("dataset", "unknown")
                task = result_dict.get("task", "unknown")
                total_metrics = result_dict.get("results", {}).get("total", {})
                if wandb_run is not None:
                    if isinstance(total_metrics, dict):
                        for key, value in total_metrics.items():
                            if isinstance(value, int | float):
                                _wandb_set_summary(
                                    wandb_run,
                                    f"{dataset}/{key}",
                                    float(value),
                                )
                    _wandb_set_summary(wandb_run, f"{dataset}/task", task)
                    _wandb_set_summary(
                        wandb_run,
                        f"{dataset}/num_model_parameters",
                        result_dict.get("num_model_parameters", 0),
                    )
                benchmark_table_rows.append(
                    {
                        "trial_index": index,
                        "trial_name": trial_name,
                        "model": result_dict.get("model", config.model),
                        "dataset": dataset,
                        "task": task,
                        "languages": json.dumps(result_dict.get("languages", [])),
                        "objective_score": objective_score,
                        "result_json": json.dumps(result_dict),
                    }
                )
                benchmark_record_index += 1

            # Log task-level statistics
            if wandb_run is not None:
                for task, stats in task_stats.items():
                    for metric_name, value in stats.items():
                        if value is not None:
                            _wandb_set_summary(
                                wandb_run,
                                f"task/{task}/{metric_name}",
                                float(value),
                            )
                # Log total benchmark score
                if total_benchmark_score is not None:
                    _wandb_set_summary(
                        wandb_run, "total_benchmark_score", total_benchmark_score
                    )

            if wandb_run is not None:
                elapsed = (trial_finished - trial_started).total_seconds()
                _wandb_set_summary(wandb_run, "status", "success")
                _wandb_set_summary(wandb_run, "elapsed_seconds", elapsed)
                _wandb_set_summary(wandb_run, "num_datasets", num_records)
                if objective_score is not None:
                    _wandb_set_summary(
                        wandb_run, "objective_score", objective_score
                    )

        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            trial_finished = dt.datetime.now(dt.UTC)
            summary_rows.append(
                TrialResult(
                    config=config,
                    trial_name=trial_name,
                    objective_score=None,
                    num_records=0,
                    error=error_message,
                )
            )
            print(f"  Failed: {error_message}")

            if wandb_run is not None:
                elapsed = (trial_finished - trial_started).total_seconds()
                _wandb_set_summary(wandb_run, "status", "failed")
                _wandb_set_summary(wandb_run, "error", error_message)
                _wandb_set_summary(wandb_run, "elapsed_seconds", elapsed)

            if args.stop_on_error:
                break
        finally:
            _wandb_finish(wandb_run)

    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            float("-inf") if row.objective_score is None else row.objective_score
        ),
        reverse=True,
    )

    summary_json = output_dir / "sweep_summary.json"
    summary_csv = output_dir / "sweep_summary.csv"

    json_payload = [
        {
            "trial_name": row.trial_name,
            "model": row.config.model,
            "learning_rate": row.config.learning_rate,
            "warmup_ratio": row.config.warmup_ratio,
            "finetuning_batch_size": row.config.finetuning_batch_size,
            "max_steps": row.config.max_steps,
            "objective_score": row.objective_score,
            "num_records": row.num_records,
            "error": row.error,
        }
        for row in ranked_rows
    ]
    summary_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_name",
                "model",
                "learning_rate",
                "warmup_ratio",
                "finetuning_batch_size",
                "max_steps",
                "objective_score",
                "num_records",
                "error",
            ],
        )
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
                    "objective_score": row.objective_score,
                    "num_records": row.num_records,
                    "error": row.error or "",
                }
            )

    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved detailed records JSONL: {detailed_results_jsonl}")

    if wandb_module is not None:
        summary_run = _wandb_init(
            wandb_module=wandb_module,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="sweep-summary",
            group=sweep_group,
            notes=args.wandb_notes,
            tags=_parse_tags(args.wandb_tags),
            job_type="sweep-summary",
            mode=args.wandb_mode,
            config={
                "models": models,
                "learning_rates": learning_rates,
                "warmup_ratios": warmup_ratios,
                "batch_sizes": batch_sizes,
                "max_steps": max_steps_values,
                "language": args.language,
                "num_trials": len(ranked_rows),
            },
        )
        if summary_run is not None:
            best_summary = next(
                (row for row in ranked_rows if row.objective_score is not None),
                None,
            )
            _wandb_set_summary(summary_run, "sweep_group", sweep_group)
            _wandb_set_summary(summary_run, "num_trials", len(ranked_rows))
            _wandb_set_summary(
                summary_run,
                "num_successful_trials",
                sum(1 for row in ranked_rows if row.objective_score is not None),
            )
            _wandb_set_summary(
                summary_run,
                "num_failed_trials",
                sum(1 for row in ranked_rows if row.objective_score is None),
            )
            _wandb_set_summary(
                summary_run,
                "num_logged_benchmark_records",
                benchmark_record_index,
            )

            if best_summary is not None:
                _wandb_set_summary(
                    summary_run, "best_trial_name", best_summary.trial_name
                )
                _wandb_set_summary(
                    summary_run, "best_model", best_summary.config.model
                )
                _wandb_set_summary(
                    summary_run,
                    "best_learning_rate",
                    best_summary.config.learning_rate,
                )
                _wandb_set_summary(
                    summary_run,
                    "best_warmup_ratio",
                    best_summary.config.warmup_ratio,
                )
                _wandb_set_summary(
                    summary_run,
                    "best_finetuning_batch_size",
                    best_summary.config.finetuning_batch_size,
                )
                _wandb_set_summary(
                    summary_run,
                    "best_max_steps",
                    best_summary.config.max_steps,
                )
                _wandb_set_summary(
                    summary_run,
                    "best_objective_score",
                    best_summary.objective_score,
                )

            trials_table = _wandb_new_table(
                wandb_module,
                columns=[
                    "rank",
                    "trial_name",
                    "model",
                    "learning_rate",
                    "warmup_ratio",
                    "finetuning_batch_size",
                    "max_steps",
                    "objective_score",
                    "num_records",
                    "error",
                ],
            )
            for rank, row in enumerate(ranked_rows, start=1):
                _wandb_table_add_row(
                    trials_table,
                    [
                        rank,
                        row.trial_name,
                        row.config.model,
                        row.config.learning_rate,
                        row.config.warmup_ratio,
                        row.config.finetuning_batch_size,
                        row.config.max_steps,
                        row.objective_score,
                        row.num_records,
                        row.error or "",
                    ],
                )
            _wandb_log(summary_run, {"summary/trials_table": trials_table})

            if benchmark_table_rows:
                benchmarks_table = _wandb_new_table(
                    wandb_module,
                    columns=[
                        "trial_index",
                        "trial_name",
                        "model",
                        "dataset",
                        "task",
                        "languages",
                        "objective_score",
                        "result_json",
                    ],
                )
                for row in benchmark_table_rows:
                    _wandb_table_add_row(
                        benchmarks_table,
                        [
                            row["trial_index"],
                            row["trial_name"],
                            row["model"],
                            row["dataset"],
                            row["task"],
                            row["languages"],
                            row["objective_score"],
                            row["result_json"],
                        ],
                    )
                _wandb_log(
                    summary_run,
                    {"summary/benchmarks_table": benchmarks_table},
                )

            run_id = str(getattr(summary_run, "id", "unknown"))
            artifact_name = f"euroeval-sweep-{sweep_group}-{run_id}"
            artifact = _wandb_new_artifact(
                wandb_module,
                name=artifact_name,
                artifact_type="sweep-results",
            )
            _wandb_artifact_add_file(
                artifact,
                local_path=str(summary_json),
                name="sweep_summary.json",
            )
            _wandb_artifact_add_file(
                artifact,
                local_path=str(summary_csv),
                name="sweep_summary.csv",
            )
            _wandb_artifact_add_file(
                artifact,
                local_path=str(detailed_results_jsonl),
                name="sweep_detailed_results.jsonl",
            )
            _wandb_log_artifact(summary_run, artifact)
            _wandb_finish(summary_run)

    best = next((row for row in ranked_rows if row.objective_score is not None), None)
    if best is None:
        print("No successful trial produced an objective score.")
        return

    print("Best trial:")
    print(f"  trial_name={best.trial_name}")
    print(f"  model={best.config.model}")
    print(f"  learning_rate={best.config.learning_rate}")
    print(f"  warmup_ratio={best.config.warmup_ratio}")
    print(f"  finetuning_batch_size={best.config.finetuning_batch_size}")
    print(f"  max_steps={best.config.max_steps}")
    print(f"  objective_score={best.objective_score:.4f}")


if __name__ == "__main__":
    main()
