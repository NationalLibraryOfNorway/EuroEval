"""Metric computation and aggregation utilities."""

from typing import Any

from .utils import score_from_metrics


def aggregate_by_task(benchmark_results: list[Any]) -> dict[str, list[dict[str, Any]]]:
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


def compute_task_statistics(
    results_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float | None]]:
    """Compute mean and standard error for metrics per task.

    For each metric in the results of each task, compute mean over
    all datasets and standard error from the _se suffix values.

    Args:
        results_by_task: Dict mapping task to list of result dicts.

    Returns:
        Dict mapping task to dict of metric -> {mean, se}.
        The 'primary' key holds the mean of the primary scoring metric.
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
                if isinstance(value, (int, float)):
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
                    if se_value is not None and isinstance(se_value, (int, float)):
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
            score = score_from_metrics(total_metrics)
            if score is not None:
                primary_scores.append(score)
                # Average the standard errors of the test_* metrics for this dataset
                test_errors = []
                for key, value in total_metrics.items():
                    if key.startswith("test_") and key.endswith("_se"):
                        if isinstance(value, (int, float)):
                            test_errors.append(float(value))
                if test_errors:
                    primary_errors.append(sum(test_errors) / len(test_errors))
                else:
                    primary_errors.append(0.0)

        if primary_scores:
            task_stats[task]["primary_mean"] = sum(primary_scores) / len(primary_scores)
            if primary_errors:
                combined_primary_se = (sum(e**2 for e in primary_errors) ** 0.5) / len(
                    primary_errors
                )
                task_stats[task]["primary_se"] = combined_primary_se
            else:
                task_stats[task]["primary_se"] = 0.0
        else:
            task_stats[task]["primary_mean"] = None
            task_stats[task]["primary_se"] = None

    return task_stats


def compute_total_benchmark_score(
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
            score = score_from_metrics(total_metrics)
            if score is not None:
                primary_scores.append(score)

        if primary_scores:
            task_score = sum(primary_scores) / len(primary_scores)
            task_scores.append(task_score)

    if not task_scores:
        return None

    return sum(task_scores) / len(task_scores)


def aggregate_objective(results: list[Any]) -> tuple[float | None, int]:
    """Aggregate objective score from multiple benchmark results.

    Returns:
        Tuple of (average_score, num_records)
    """
    row_scores: list[float] = []

    for result in results:
        result_dict = (
            result.model_dump() if hasattr(result, "model_dump") else dict(result)
        )
        score_dict = result_dict.get("results", {})
        total_metrics = score_dict.get("total", {})
        if not isinstance(total_metrics, dict):
            continue

        row_score = score_from_metrics(total_metrics)
        if row_score is not None:
            row_scores.append(row_score)

    if not row_scores:
        return None, 0

    return sum(row_scores) / len(row_scores), len(row_scores)
