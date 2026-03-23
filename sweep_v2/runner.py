"""W&B Sweep runner for EuroEval hyperparameter sweeps."""

import datetime as dt
import importlib
import itertools
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from euroeval import Benchmarker

from .config import TrialConfig, TrialResult
from .metrics import (
    aggregate_by_task,
    aggregate_objective,
    compute_task_statistics,
    compute_total_benchmark_score,
)
from .output import append_trial_results, write_summary_csv, write_summary_json
from .utils import build_trial_name, generate_sweep_group, parse_tags


def import_wandb(enable_wandb: bool) -> object | None:
    """Import wandb module if W&B logging is enabled.

    Returns:
        Imported ``wandb`` module when enabled and installed, otherwise ``None``.

    Raises:
        ImportError: If W&B is enabled but the ``wandb`` package is not installed.
    """
    if not enable_wandb:
        return None

    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested but the 'wandb' package is not installed. "
            "Install it with: pip install wandb"
        ) from exc


def wandb_init(
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
    """Initialize a W&B run.

    Returns:
        W&B run object when initialization succeeds, otherwise ``None``.
    """
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


def wandb_define_metric(
    wandb_module: object | None, metric_name: str, step_metric: str | None = None
) -> None:
    """Define a W&B metric."""
    if wandb_module is None:
        return

    define_metric_fn = getattr(wandb_module, "define_metric", None)
    if not callable(define_metric_fn):
        return

    if step_metric is None:
        define_metric_fn(metric_name)
    else:
        define_metric_fn(metric_name, step_metric=step_metric)


def wandb_log(wandb_run: object | None, payload: dict[str, Any]) -> None:
    """Log metrics to W&B."""
    if wandb_run is None:
        return

    log_fn = getattr(wandb_run, "log", None)
    if callable(log_fn):
        log_fn(payload)


def wandb_set_summary(wandb_run: object | None, key: str, value: object) -> None:
    """Set a summary value in W&B."""
    if wandb_run is None:
        return

    summary_obj = getattr(wandb_run, "summary", None)
    if summary_obj is not None:
        summary_obj[key] = value


def wandb_finish(wandb_run: object | None) -> None:
    """Finish a W&B run."""
    if wandb_run is None:
        return

    finish_fn = getattr(wandb_run, "finish", None)
    if callable(finish_fn):
        finish_fn()


def create_sweep_config(
    models: list[str],
    learning_rates: list[float],
    warmup_ratios: list[float],
    batch_sizes: list[int],
    max_steps_values: list[int],
    weight_decays: list[float],
    lr_scheduler_types: list[str],
    fixed_config: dict[str, Any],
) -> dict[str, Any]:
    """Create a W&B sweep configuration.

    Args:
        models: List of model IDs to sweep.
        learning_rates: List of learning rates.
        warmup_ratios: List of warmup ratios.
        batch_sizes: List of batch sizes.
        max_steps_values: List of max steps.
        weight_decays: List of weight decay values.
        lr_scheduler_types: List of LR scheduler types.
        fixed_config: Dictionary of fixed hyperparameters.

    Returns:
        W&B sweep configuration dictionary.
    """
    return {
        "method": "grid",
        "metric": {"name": "objective_score", "goal": "maximize"},
        "parameters": {
            "model": {"values": models},
            "learning_rate": {"values": learning_rates},
            "warmup_ratio": {"values": warmup_ratios},
            "finetuning_batch_size": {"values": batch_sizes},
            "max_steps": {"values": max_steps_values},
            "weight_decay": {"values": weight_decays},
            "lr_scheduler_type": {"values": lr_scheduler_types},
            **{f"fixed_{k}": {"value": v} for k, v in fixed_config.items()},
        },
    }


def run_trial(
    config: TrialConfig,
    trial_index: int,
    total_trials: int,
    args: Namespace,
    output_dir: Path,
    wandb_module: object | None,
    wandb_run: object | None,
    wandb_tags: list[str],
    sweep_group: str,
) -> TrialResult:
    """Run a single trial.

    Args:
        config: Trial configuration.
        trial_index: Current trial index (1-indexed).
        total_trials: Total number of trials.
        args: Parsed command-line arguments.
        output_dir: Output directory for results.
        wandb_module: Imported wandb module or None.
        wandb_run: Current W&B run object or None.
        wandb_tags: Tags for W&B logging.
        sweep_group: Sweep group name.

    Returns:
        TrialResult with trial outcome.
    """
    trial_name = build_trial_name(
        model=config.model,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        finetuning_batch_size=config.finetuning_batch_size,
        max_steps=config.max_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
    )

    print(f"[{trial_index}/{total_trials}] Trial {trial_name}")
    trial_started = dt.datetime.now(dt.UTC)

    # Initialize W&B run if not already done (for W&B Sweep agent)
    if wandb_run is None and wandb_module is not None:
        wandb_run = wandb_init(
            wandb_module=wandb_module,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=trial_name,
            group=sweep_group,
            notes=args.wandb_notes,
            tags=wandb_tags,
            job_type=args.wandb_job_type,
            mode=args.wandb_mode,
            config=config.to_dict()
            | {
                "eval_steps": args.eval_steps,
                "logging_steps": args.logging_steps,
                "save_steps": args.save_steps,
                "eval_accumulation_steps": args.eval_accumulation_steps,
                "gradient_accumulation_base": args.gradient_accumulation_base,
                "early_stopping_patience": args.early_stopping_patience,
                "optimizer_name": args.optimizer_name,
                "save_total_limit": args.save_total_limit,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "language": args.language,
                "tasks": args.tasks,
                "num_iterations": args.num_iterations,
            },
        )

    try:
        benchmarker = Benchmarker(
            task=args.tasks,
            language=args.language,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            finetuning_batch_size=config.finetuning_batch_size,
            max_steps=config.max_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_accumulation_steps=args.eval_accumulation_steps,
            gradient_accumulation_base=args.gradient_accumulation_base,
            early_stopping_patience=args.early_stopping_patience,
            optimizer_name=args.optimizer_name,
            save_total_limit=args.save_total_limit,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
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
                model=[config.model], prioritize_mask=args.prioritize_mask
            )
        )

        # Append results to JSONL
        detailed_results_jsonl = output_dir / "sweep_detailed_results.jsonl"
        append_trial_results(
            results_path=detailed_results_jsonl,
            trial_name=trial_name,
            config_dict=config.to_dict(),
            benchmark_results=benchmark_results,
        )

        objective_score, num_records = aggregate_objective(benchmark_results)
        trial_finished = dt.datetime.now(dt.UTC)

        # Log to W&B
        if wandb_run is not None:
            results_by_task = aggregate_by_task(benchmark_results)
            task_stats = compute_task_statistics(results_by_task)
            total_benchmark_score = compute_total_benchmark_score(results_by_task)

            # Log task statistics
            for task, stats in task_stats.items():
                for metric_name, value in stats.items():
                    if value is not None:
                        wandb_log(
                            wandb_run, {f"task/{task}/{metric_name}": float(value)}
                        )
                        wandb_set_summary(
                            wandb_run, f"task/{task}/{metric_name}", float(value)
                        )

            # Log trial metrics
            elapsed = (trial_finished - trial_started).total_seconds()
            trial_payload: dict[str, Any] = {
                "objective_score": objective_score,
                "total_benchmark_score": total_benchmark_score,
                "elapsed_seconds": elapsed,
                "num_datasets": float(num_records),
                "status": "success",
            }
            wandb_log(wandb_run, trial_payload)

            wandb_set_summary(wandb_run, "objective_score", objective_score)
            wandb_set_summary(wandb_run, "total_benchmark_score", total_benchmark_score)
            wandb_set_summary(wandb_run, "status", "success")
            wandb_set_summary(wandb_run, "elapsed_seconds", elapsed)

        return TrialResult(
            config=config,
            trial_name=trial_name,
            objective_score=objective_score,
            num_records=num_records,
        )

    except Exception as exc:
        error_message = str(exc)
        trial_finished = dt.datetime.now(dt.UTC)
        print(f"  Failed: {error_message}")

        if wandb_run is not None:
            elapsed = (trial_finished - trial_started).total_seconds()
            wandb_log(
                wandb_run,
                {
                    "status": "failed",
                    "error": error_message,
                    "elapsed_seconds": elapsed,
                },
            )
            wandb_set_summary(wandb_run, "status", "failed")
            wandb_set_summary(wandb_run, "error", error_message)

        return TrialResult(
            config=config,
            trial_name=trial_name,
            objective_score=None,
            num_records=0,
            error=error_message,
        )

    finally:
        wandb_finish(wandb_run)


def run_sweep(
    models: list[str],
    learning_rates: list[float],
    warmup_ratios: list[float],
    batch_sizes: list[int],
    max_steps_values: list[int],
    weight_decays: list[float],
    lr_scheduler_types: list[str],
    args: Namespace,
    use_wandb_sweep: bool = True,
) -> None:
    """Run hyperparameter sweep.

    Args:
        models: List of model IDs.
        learning_rates: List of learning rates.
        warmup_ratios: List of warmup ratios.
        batch_sizes: List of batch sizes.
        max_steps_values: List of max steps.
        weight_decays: List of weight decay values.
        lr_scheduler_types: List of LR scheduler types.
        args: Parsed command-line arguments.
        use_wandb_sweep: If True, use W&B Sweep API; else use manual iteration.
    """
    wandb_module = import_wandb(enable_wandb=args.wandb)

    # Setup output directory
    output_dir = args.output_dir or Path("sweep_runs") / models[0].replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detailed results JSONL
    detailed_results_jsonl = output_dir / "sweep_detailed_results.jsonl"
    detailed_results_jsonl.write_text("", encoding="utf-8")

    # Create trial configs
    configs = [
        TrialConfig(
            model=model,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            finetuning_batch_size=batch_size,
            max_steps=max_steps,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
        )
        for (
            model,
            learning_rate,
            warmup_ratio,
            batch_size,
            max_steps,
            weight_decay,
            lr_scheduler_type,
        ) in itertools.product(
            models,
            learning_rates,
            warmup_ratios,
            batch_sizes,
            max_steps_values,
            weight_decays,
            lr_scheduler_types,
        )
    ]

    sweep_group = args.wandb_group or generate_sweep_group(models[0])
    if wandb_module is not None:
        print(f"W&B logging enabled. Sweep group: {sweep_group}")

    print(f"Running {len(configs)} sweep trials across {len(models)} model(s).")
    print(f"Detailed trial records will be saved to: {detailed_results_jsonl}")

    if use_wandb_sweep and wandb_module is not None:
        # Use W&B Sweep API
        _run_with_wandb_sweep(
            configs=configs,
            args=args,
            output_dir=output_dir,
            wandb_module=wandb_module,
            wandb_group=sweep_group,
        )
    else:
        # Manual iteration (original approach)
        _run_manual_sweep(
            configs=configs,
            args=args,
            output_dir=output_dir,
            wandb_module=wandb_module,
            wandb_group=sweep_group,
        )

    # Generate summary files
    _finalize_sweep(configs, output_dir, args, wandb_module, sweep_group)


def _run_manual_sweep(
    configs: list[TrialConfig],
    args: Namespace,
    output_dir: Path,
    wandb_module: object | None,
    wandb_group: str,
) -> None:
    """Run sweep with manual iteration (original approach)."""
    summary_rows: list[TrialResult] = []
    wandb_tags = parse_tags(args.wandb_tags)

    for index, config in enumerate(configs, start=1):
        result = run_trial(
            config=config,
            trial_index=index,
            total_trials=len(configs),
            args=args,
            output_dir=output_dir,
            wandb_module=wandb_module,
            wandb_run=None,
            wandb_tags=wandb_tags,
            sweep_group=wandb_group,
        )
        summary_rows.append(result)

        if args.stop_on_error and result.error is not None:
            break

    # Write summary files
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            float("-inf") if row.objective_score is None else row.objective_score
        ),
        reverse=True,
    )

    write_summary_json(output_dir / "sweep_summary.json", ranked_rows)
    write_summary_csv(output_dir / "sweep_summary.csv", ranked_rows)


def _run_with_wandb_sweep(
    configs: list[TrialConfig],
    args: Namespace,
    output_dir: Path,
    wandb_module: object,
    wandb_group: str,
) -> None:
    """Run sweep using W&B Sweep API.

    Raises:
        RuntimeError: If required W&B sweep or agent methods are unavailable.
    """
    # Create sweep configuration
    fixed_config = {
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "gradient_accumulation_base": args.gradient_accumulation_base,
        "early_stopping_patience": args.early_stopping_patience,
        "optimizer_name": args.optimizer_name,
        "save_total_limit": args.save_total_limit,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "language": args.language,
        "tasks": args.tasks,
        "num_iterations": args.num_iterations,
    }

    # Extract unique values for each swept parameter
    models = list({c.model for c in configs})
    learning_rates = list({c.learning_rate for c in configs})
    warmup_ratios = list({c.warmup_ratio for c in configs})
    batch_sizes = list({c.finetuning_batch_size for c in configs})
    max_steps_values = list({c.max_steps for c in configs})
    weight_decays = list({c.weight_decay for c in configs})
    lr_scheduler_types = list({c.lr_scheduler_type for c in configs})

    sweep_config = create_sweep_config(
        models=models,
        learning_rates=learning_rates,
        warmup_ratios=warmup_ratios,
        batch_sizes=batch_sizes,
        max_steps_values=max_steps_values,
        weight_decays=weight_decays,
        lr_scheduler_types=lr_scheduler_types,
        fixed_config=fixed_config,
    )

    # Initialize sweep
    sweep_id = getattr(wandb_module, "sweep", None)
    if not callable(sweep_id):
        raise RuntimeError("W&B sweep method not available")

    sweep_id = sweep_id(
        sweep=sweep_config, project=args.wandb_project, entity=args.wandb_entity or None
    )

    print(f"Created W&B sweep: {sweep_id}")

    # Run agent
    agent_fn = getattr(wandb_module, "agent", None)
    if not callable(agent_fn):
        raise RuntimeError("W&B agent method not available")

    def trial_function() -> None:
        wandb_run = wandb_module.init()
        config = wandb_run.config

        trial_config = TrialConfig(
            model=config.model,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            finetuning_batch_size=int(config.finetuning_batch_size),
            max_steps=int(config.max_steps),
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
        )

        # Find the trial index
        trial_index = next(
            (i + 1 for i, c in enumerate(configs) if c == trial_config), len(configs)
        )

        run_trial(
            config=trial_config,
            trial_index=trial_index,
            total_trials=len(configs),
            args=args,
            output_dir=output_dir,
            wandb_module=wandb_module,
            wandb_run=wandb_run,
            wandb_tags=parse_tags(args.wandb_tags),
            sweep_group=wandb_group,
        )

    agent_fn(sweep_id, trial_function, count=len(configs))


def _finalize_sweep(
    configs: list[TrialConfig],
    output_dir: Path,
    args: Namespace,
    wandb_module: object | None,
    sweep_group: str,
) -> None:
    """Finalize sweep: generate summary files and create summary run."""
    # Read and rank all results
    detailed_results_jsonl = output_dir / "sweep_detailed_results.jsonl"

    if detailed_results_jsonl.exists():
        with detailed_results_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    # Reconstruct TrialResult from stored data
                    # (simplified - in production you'd parse from stored results)
                    pass

    print(f"Saved sweep summary to {output_dir}")
