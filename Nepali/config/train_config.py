from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration for Nepali pretraining."""

    tokenized_dir: str = "Nepali/data/tokenized"
    validation_dir: str = "Nepali/data/tokenized/validation"
    test_dir: str = "Nepali/data/tokenized/test"
    checkpoint_dir: str = "Nepali/checkpoints"

    sequence_length: int = 512

    batch_size: int = 16
    num_workers: int = 0

    # 500,000,000 / (16 * 512) rounded up.
    max_steps: int = 61_036

    # Conservative peak learning rate for stable FP16 training.
    learning_rate: float = 2e-4
    min_learning_rate_ratio: float = 0.1

    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01

    warmup_steps: int = 2_000

    # Gradient clipping threshold.
    grad_clip: float = 1.0

    log_every: int = 50
    eval_every: int = 2_000
    validation_steps: int = 100
    checkpoint_every: int = 2_000

    seed: int = 42

    use_mixed_precision: bool = True

    def __post_init__(self):
        """Validate the training configuration."""

        if self.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError(
                "min_learning_rate_ratio must be in [0, 1]."
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative."
            )

        if self.grad_clip <= 0:
            raise ValueError(
                "grad_clip must be positive."
            )

        if self.log_every <= 0:
            raise ValueError(
                "log_every must be positive."
            )

        if self.eval_every <= 0:
            raise ValueError(
                "eval_every must be positive."
            )

        if self.validation_steps <= 0:
            raise ValueError(
                "validation_steps must be positive."
            )

        if self.checkpoint_every <= 0:
            raise ValueError(
                "checkpoint_every must be positive."
            )