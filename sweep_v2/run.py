"""Run a single EuroEval model training.

Usage:
    python -m sweep_v2.run --model ltg/norbert4-xsmall
    python -m sweep_v2.run --model ltg/norbert4-xsmall --learning-rate 2e-5 --max-steps 320
    python -m sweep_v2.run --model ltg/norbert4-xsmall --language no --tasks sentiment-classification
"""

import argparse
import json
import sys
from pathlib import Path

from euroeval import Benchmarker


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single training run."""
    parser = argparse.ArgumentParser(
        description="Run a single EuroEval model training."
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID to benchmark (e.g. ltg/norbert4-xsmall).",
    )

    # Swept hyperparameters (single values)
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate."
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.01, help="Warmup ratio."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Finetuning batch size."
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum finetuning steps."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay."
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="LR scheduler type.",
    )

    # Fixed hyperparameters
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--logging-steps", type=int, default=30)
    parser.add_argument("--save-steps", type=int, default=30)
    parser.add_argument("--eval-accumulation-steps", type=int, default=32)
    parser.add_argument("--gradient-accumulation-base", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--optimizer-name", type=str, default="adamw_torch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)

    # Benchmark configuration
    parser.add_argument(
        "--language", type=str, default="no", help="Language filter."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated tasks to evaluate (empty = all).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=".euroeval_cache", help="Cache directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for result output. If omitted, results are only printed.",
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
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force evaluation even if existing records are present.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable progress bars.",
    )

    return parser.parse_args()


def main() -> None:
    """Run a single EuroEval training."""
    args = parse_args()

    tasks = args.tasks if args.tasks else None

    print(f"Model:          {args.model}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Warmup ratio:   {args.warmup_ratio}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Max steps:      {args.max_steps}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"LR scheduler:   {args.lr_scheduler_type}")
    print(f"Language:       {args.language}")
    print(f"Tasks:          {tasks or 'all'}")
    print(f"Iterations:     {args.num_iterations}")
    print()

    benchmarker = Benchmarker(
        task=tasks,
        language=args.language,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        finetuning_batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        gradient_accumulation_base=args.gradient_accumulation_base,
        early_stopping_patience=args.early_stopping_patience,
        optimizer_name=args.optimizer_name,
        save_total_limit=args.save_total_limit,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        num_iterations=args.num_iterations,
        progress_bar=not args.no_progress_bar,
        save_results=False,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        force=args.force,
        raise_errors=True,
        evaluate_test_split=True,
    )

    results = list(
        benchmarker.benchmark(
            model=[args.model], prioritize_mask=args.prioritize_mask
        )
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {args.model}")
    print(f"{'='*60}")
    for result in results:
        dataset = result.dataset
        task = result.task
        scores = result.results.get("total", {})
        test_metrics = {k: v for k, v in scores.items() if k.startswith("test_")}
        print(f"\n  {dataset} ({task}):")
        for metric, value in sorted(test_metrics.items()):
            print(f"    {metric}: {value:.4f}")

    # Save results if output dir specified
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "results.json"
        serializable = [
            {
                "model": r.model,
                "dataset": r.dataset,
                "task": r.task,
                "languages": r.languages,
                "results": r.results,
            }
            for r in results
        ]
        output_path.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
