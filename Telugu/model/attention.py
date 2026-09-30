import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention implemented from scratch.

    Input:
        X: (B, T, d_model)

    Internally:
        Q, K, V: (B, n_heads, T, head_dim)

    Output:
        (B, T, d_model)

    Attention weights are optionally returned for attention analysis.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # One projection produces Q, K and V.
        self.qkv_proj = nn.Linear(
            d_model,
            3 * d_model,
            bias=True,
        )

        # Output projection after concatenating heads.
        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=True,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Causal mask:
        #
        # 0     = allowed
        # -inf  = future token
        #
        # Shape: (1, 1, T, T)
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len),
            diagonal=1,
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf"),
        )

        mask = mask.masked_fill(
            mask == 0,
            0.0,
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, max_seq_len, max_seq_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Perform causal multi-head self-attention.

        Args:
            x:
                Input tensor of shape (B, T, d_model).

            return_attention:
                If True, return attention probabilities for analysis.

        Returns:
            output:
                Tensor of shape (B, T, d_model).

            attention_weights:
                Tensor of shape (B, n_heads, T, T) if requested,
                otherwise None.
        """

        batch_size, seq_len, d_model = x.shape

        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, "
                f"received {d_model}."
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum length {self.max_seq_len}."
            )

        # ---------------------------------------------------------
        # 1. Project input into Q, K and V
        #
        # x:   (B, T, d)
        # qkv: (B, T, 3d)
        # ---------------------------------------------------------

        qkv = self.qkv_proj(x)

        # (B, T, 3d)
        q, k, v = qkv.chunk(3, dim=-1)

        # ---------------------------------------------------------
        # 2. Split into multiple heads
        #
        # (B, T, d)
        #      ->
        # (B, T, h, dk)
        #      ->
        # (B, h, T, dk)
        # ---------------------------------------------------------

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # q, k, v:
        # (B, h, T, dk)

        # ---------------------------------------------------------
        # 3. Scaled dot-product attention
        #
        # QK^T:
        # (B, h, T, dk)
        #     ×
        # (B, h, dk, T)
        #
        # ->
        # (B, h, T, T)
        # ---------------------------------------------------------

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # Scale by sqrt(dk)
        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        # ---------------------------------------------------------
        # 4. Apply causal mask
        #
        # Position t can only attend to positions <= t.
        # ---------------------------------------------------------

        attention_scores = (
            attention_scores
            + self.causal_mask[:, :, :seq_len, :seq_len]
        )

        # ---------------------------------------------------------
        # 5. Softmax
        # ---------------------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        # ---------------------------------------------------------
        # 6. Weighted sum of values
        #
        # (B, h, T, T)
        #     ×
        # (B, h, T, dk)
        #
        # ->
        # (B, h, T, dk)
        # ---------------------------------------------------------

        context = torch.matmul(
            attention_weights,
            v,
        )

        # ---------------------------------------------------------
        # 7. Merge heads
        #
        # (B, h, T, dk)
        #      ->
        # (B, T, h, dk)
        #      ->
        # (B, T, d)
        # ---------------------------------------------------------

        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        # ---------------------------------------------------------
        # 8. Output projection
        # ---------------------------------------------------------

        output = self.out_proj(context)

        output = self.output_dropout(output)

        if return_attention:
            return output, attention_weights

        return output, None