from torch import nn

from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """Pre-norm decoder Transformer block."""

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        attention_dropout,
        max_seq_len,
    ):
        """Create attention, feed-forward and normalization sublayers."""

        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model)

        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            max_seq_len=max_seq_len,
        )

        self.ln_2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        """Apply pre-norm causal attention and feed-forward sublayers."""

        if return_attention:
            attention_output, attention = self.attention(
                self.ln_1(x),
                return_attention=True,
            )
        else:
            attention_output = self.attention(
                self.ln_1(x)
            )

        x = x + attention_output

        x = x + self.feed_forward(
            self.ln_2(x)
        )

        if return_attention:
            return x, attention

        return x
