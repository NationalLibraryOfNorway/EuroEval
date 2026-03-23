"""Utility functions for hyperparameter sweep."""

import datetime as dt
import re
from typing import Any

# Metrics to ignore when computing objective scores
IGNORED_METRIC_SUFFIXES = ("_se",)
IGNORED_METRIC_KEYS = {
    "num_failed_instances",
    "test_loss",
    "test_runtime",
    "test_samples_per_second",
    "test_steps_per_second",
}


def parse_float_list(raw: str) -> list[float]:
    r"""Parse a comma, semicolon, or pipe-separated list of floats.

    Args:
        raw: String containing float values separated by ',', ';', or '|'.
             Escaped commas (\\,) are preserved.

    Returns:
        List of parsed float values.

    Raises:
        ValueError: If no valid float values are found.
    """
    cleaned = raw.replace("\\,", ",")
    values = [value.strip() for value in re.split(r"[,;|]", cleaned) if value.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(value) for value in values]


def parse_int_list(raw: str) -> list[int]:
    r"""Parse a comma, semicolon, or pipe-separated list of integers.

    Args:
        raw: String containing integer values separated by ',', ';', or '|'.
             Escaped commas (\\,) are preserved.

    Returns:
        List of parsed integer values.

    Raises:
        ValueError: If no valid integer values are found.
    """
    cleaned = raw.replace("\\,", ",")
    values = [value.strip() for value in re.split(r"[,;|]", cleaned) if value.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(value) for value in values]


def parse_str_list(raw: str) -> list[str]:
    r"""Parse a comma, semicolon, or pipe-separated list of strings.

    Args:
        raw: String containing values separated by ',', ';', or '|'.
             Escaped commas (\\,) are preserved.

    Returns:
        List of parsed string values.

    Raises:
        ValueError: If no valid string values are found.
    """
    cleaned = raw.replace("\\,", ",")
    values = [value.strip() for value in re.split(r"[,;|]", cleaned) if value.strip()]
    if not values:
        raise ValueError("Expected at least one model value.")
    return values


def parse_tags(raw: str) -> list[str]:
    """Parse comma-separated tags.

    Returns:
        List of stripped, non-empty tag strings.
    """
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in file/run names.

    Returns:
        Model name with path and symbol characters normalized for filenames.
    """
    return model.replace("/", "__").replace("@", "_at_")


def build_trial_name(
    model: str,
    learning_rate: float,
    warmup_ratio: float,
    finetuning_batch_size: int,
    max_steps: int,
    weight_decay: float,
    lr_scheduler_type: str,
) -> str:
    """Build a human-readable trial name from hyperparameters.

    Returns:
        Deterministic trial name including model and selected hyperparameters.
    """
    model_slug = sanitize_model_name(model)
    lr = f"{learning_rate:.1e}".replace("+", "")
    wr = str(warmup_ratio).replace(".", "p")
    bs = str(finetuning_batch_size)
    ms = str(max_steps)
    wd = str(weight_decay).replace(".", "p")
    sched = lr_scheduler_type
    return f"model_{model_slug}_lr_{lr}_wr_{wr}_bs_{bs}_ms_{ms}_wd_{wd}_sched_{sched}"


def score_from_metrics(metrics: dict[str, Any]) -> float | None:
    """Compute objective score from metrics dictionary.

    Averages all ``test_*`` metrics, ignoring specified keys and ``_se``
    (standard error) suffixes.

    Returns:
        Mean score over included metrics, or ``None`` if no valid metrics exist.
    """
    values: list[float] = []
    for key, value in metrics.items():
        if not key.startswith("test_"):
            continue
        if key in IGNORED_METRIC_KEYS:
            continue
        if any(key.endswith(suffix) for suffix in IGNORED_METRIC_SUFFIXES):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))

    if not values:
        return None
    return sum(values) / len(values)


def flatten_numeric_dict(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    """Flatten a nested dict keeping only numeric values with a prefix.

    Returns:
        Dictionary with prefixed keys and float-converted numeric values.
    """
    flat: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float)):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def generate_sweep_group(model: str) -> str:
    """Generate a unique sweep group name.

    Returns:
        Group name composed of model slug and UTC timestamp.
    """
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    slug = sanitize_model_name(model)[:30]
    return f"sweep_{slug}_{timestamp}"
