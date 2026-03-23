"""Trial configuration and result dataclasses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrialConfig:
    """Sweep trial configuration.

    Stores the core hyperparameters varied in the sweep. Additional EuroEval
    finetuning hyperparameters (eval_steps, optimizer_name, etc.) can be
    configured as fixed values that apply uniformly to all trials.
    """

    model: str
    learning_rate: float
    warmup_ratio: float
    finetuning_batch_size: int
    max_steps: int
    weight_decay: float
    lr_scheduler_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "finetuning_batch_size": self.finetuning_batch_size,
            "max_steps": self.max_steps,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
        }


@dataclass
class TrialResult:
    """Sweep trial result."""

    config: TrialConfig
    trial_name: str
    objective_score: float | None
    num_records: int
    error: str | None = None
