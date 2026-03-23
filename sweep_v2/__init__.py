"""W&B-integrated hyperparameter sweep utilities for EuroEval.

Provides utilities for running hyperparameter sweeps with optional W&B Sweep
integration for distributed execution and comprehensive metric tracking.

Core modules:
    - cli: Argument parsing
    - config: Trial configuration dataclasses
    - metrics: Metric computation and aggregation
    - output: Output file generation (JSON, CSV, JSONL)
    - runner: Main sweep orchestration with W&B integration
    - utils: Utility functions (parsing, naming, etc.)
    - sweep_hyperparams: Main entry point script
"""

__version__ = "2.0.0"
