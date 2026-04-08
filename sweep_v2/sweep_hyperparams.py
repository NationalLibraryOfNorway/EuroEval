"""Refactored W&B-integrated hyperparameter sweep runner.

This script runs a hyperparameter sweep for EuroEval benchmarking with optional
W&B Sweep integration for distributed sweeping and metric tracking.

Usage:
    python sweep_hyperparams.py --models ltg/norbert4-xsmall --learning-rates 1e-5,2e-5
    ...

    With W&B Sweeps:
    python sweep_hyperparams.py --wandb --use-wandb-sweep ...

    Manual mode (legacy):
    python sweep_hyperparams.py --wandb --manual-only ...
"""

from cli import parse_args
from runner import run_sweep
from utils import parse_float_list, parse_int_list, parse_str_list


def main() -> None:
    """Main entry point for sweep runner."""
    args = parse_args()

    # Parse hyperparameter lists
    models = parse_str_list(args.models)
    learning_rates = parse_float_list(args.learning_rates)
    warmup_ratios = parse_float_list(args.warmup_ratios)
    batch_sizes = parse_int_list(args.batch_sizes)
    max_steps_values = parse_int_list(args.max_steps)
    weight_decays = parse_float_list(args.weight_decays)
    lr_scheduler_types = parse_str_list(args.lr_scheduler_types)

    # Parse tasks if provided
    tasks = (
        [task.strip() for task in args.tasks.split(",") if task.strip()]
        if args.tasks
        else None
    )
    args.tasks = tasks

    # Determine whether to use W&B Sweep API
    use_wandb_sweep = args.wandb and args.use_wandb_sweep and not args.manual_only

    # Run sweep
    run_sweep(
        models=models,
        learning_rates=learning_rates,
        warmup_ratios=warmup_ratios,
        batch_sizes=batch_sizes,
        max_steps_values=max_steps_values,
        weight_decays=weight_decays,
        lr_scheduler_types=lr_scheduler_types,
        args=args,
        use_wandb_sweep=use_wandb_sweep,
    )


if __name__ == "__main__":
    main()
