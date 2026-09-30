import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention implemented from primitive PyTorch
    layers.

    Returns attention probabilities when ``return_attention=True`` so
    Phase 2 attention analysis can inspect learned heads.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout,
        attention_dropout,
        max_seq_len,
    ):
        """Initialize Q/K/V projections, output projection and mask."""

        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )
        self.output_dropout = nn.Dropout(dropout)

        mask = torch.triu(
            torch.ones(
                max_seq_len,
                max_seq_len,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False,
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        """Compute causal multi-head attention.

        Input shape:
            ``(B, T, d_model)``

        Attention shape:
            ``(B, n_heads, T, T)``
        """

        batch_size, sequence_length, _ = x.shape

        if sequence_length > self.causal_mask.size(0):
            raise ValueError(
                "Sequence length exceeds configured maximum."
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            mask[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        attention_for_output = self.attention_dropout(
            attention
        )

        output = torch.matmul(
            attention_for_output,
            v,
        )

        output = output.transpose(
            1,
            2,
        ).contiguous().view(
            batch_size,
            sequence_length,
            self.d_model,
        )

        output = self.out_proj(output)
        output = self.output_dropout(output)

        if return_attention:
            return output, attention

        return output
