from torch import nn

from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    """Stack of decoder-only Transformer blocks."""

    def __init__(self, config):
        """Construct the configured stack of Transformer blocks."""

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):
        """Run all blocks and return the final normalized representation."""

        attentions = []

        for block in self.blocks:

            if return_attention:
                x, attention = block(
                    x,
                    return_attention=True,
                )
                attentions.append(attention)
            else:
                x = block(x)

        x = self.final_norm(x)

        if return_attention:
            return x, attentions

        return x
