import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    d_model -> d_ff -> GELU -> d_model
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            d_model,
            d_ff,
            bias=True,
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            d_ff,
            d_model,
            bias=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise feed-forward network.
        """

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """
    One pre-norm decoder-only Transformer block.

    Structure:

        x
        |
        LayerNorm
        |
        Causal Self-Attention
        |
        Residual Add
        |
        LayerNorm
        |
        Feed Forward
        |
        Residual Add
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Apply attention and feed-forward sublayers with
        pre-normalization and residual connections.
        """

        # ---------------------------------------------------------
        # Attention sublayer
        # ---------------------------------------------------------

        attention_output, attention_weights = self.attention(
            self.ln1(x),
            return_attention=return_attention,
        )

        x = x + attention_output

        # ---------------------------------------------------------
        # Feed-forward sublayer
        # ---------------------------------------------------------

        x = x + self.ffn(
            self.ln2(x)
        )

        return x, attention_weights