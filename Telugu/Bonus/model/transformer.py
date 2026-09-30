import torch
import torch.nn as nn

# from .positional_embedding import LearnedPositionalEmbedding
from .transformer_block import TransformerBlock


class DecoderOnlyTransformer(nn.Module):
    """
    Stack of decoder-only Transformer blocks.

    This module does not contain the token embedding or LM head.
    Those are handled by the language-model wrapper.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        # self.position_embedding = LearnedPositionalEmbedding(
        #     max_seq_len=max_seq_len,
        #     d_model=d_model,
        # )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Run the input through the Transformer stack.

        Args:
            x:
                Token embeddings of shape (B, T, d_model).

        Returns:
            hidden_states:
                (B, T, d_model)

            all_attention_weights:
                List containing one tensor per layer if requested.
        """

        # Add positional embeddings.
        # x = x + self.position_embedding(x)

        x = self.embedding_dropout(x)

        all_attention_weights = []

        for block in self.blocks:
            x, attention_weights = block(
                x,
                return_attention=return_attention,
            )

            if return_attention:
                all_attention_weights.append(
                    attention_weights
                )

        x = self.final_layer_norm(x)

        if return_attention:
            return x, all_attention_weights

        return x, None