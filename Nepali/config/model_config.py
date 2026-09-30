from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the Nepali decoder-only Transformer."""

    vocab_size: int = 8_000

    d_model: int = 448
    n_layers: int = 11
    n_heads: int = 8
    d_ff: int = 1_344

    max_seq_len: int = 512

    dropout: float = 0.1
    attention_dropout: float = 0.1

    pad_token_id: int = 0

    tie_embeddings: bool = True

    def __post_init__(self):
        """Validate the model configuration."""

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                "attention_dropout must be in [0, 1)."
            )

    @property
    def head_dim(self) -> int:
        """Return the dimensionality of each attention head."""
        return self.d_model // self.n_heads
