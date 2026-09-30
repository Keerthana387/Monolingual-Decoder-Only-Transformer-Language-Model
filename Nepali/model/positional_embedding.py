import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings."""

    def __init__(self, max_seq_len, d_model):
        """Create a trainable position embedding table."""

        super().__init__()

        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

    def forward(self, x):
        """Add position information to a ``(B, T, d_model)`` tensor."""

        _, sequence_length, _ = x.shape

        if sequence_length > self.max_seq_len:
            raise ValueError(
                "Sequence length exceeds configured maximum: "
                f"{sequence_length} > {self.max_seq_len}"
            )

        positions = torch.arange(
            sequence_length,
            device=x.device,
        )

        return x + self.embedding(positions)[None, :, :]
