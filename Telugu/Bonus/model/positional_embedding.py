import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned absolute positional embeddings.

    Each position from 0 to max_seq_len - 1 receives a learned
    d_model-dimensional embedding.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional information to token representations.

        Args:
            x:
                Tensor of shape (B, T, d_model).

        Returns:
            Tensor of shape (B, T, d_model) containing positional embeddings.
        """

        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum length {self.max_seq_len}."
            )

        positions = torch.arange(
            seq_len,
            device=x.device,
        )

        position_embeddings = self.embedding(positions)

        # (T, d_model) -> broadcast over batch
        return position_embeddings.unsqueeze(0)