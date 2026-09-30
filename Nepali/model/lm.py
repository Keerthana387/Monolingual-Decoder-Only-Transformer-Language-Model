import torch
from torch import nn
from torch.nn import functional as F

from .positional_embedding import LearnedPositionalEmbedding
from .transformer import Transformer


class LanguageModel(nn.Module):
    """Decoder-only Nepali causal language model."""

    def __init__(self, config):
        """Construct embeddings, Transformer backbone and tied LM head."""

        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.position_embedding = (
            LearnedPositionalEmbedding(
                config.max_seq_len,
                config.d_model,
            )
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.transformer = Transformer(
            config
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        if config.tie_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Initialize trainable weights using GPT-style normal initialization."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids,
        targets=None,
        return_attention=False,
    ):
        """Run causal LM forward pass and optionally compute cross-entropy."""

        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape (B, T)."
            )

        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                "Sequence length exceeds model maximum."
            )

        x = self.token_embedding(
            input_ids
        )

        x = self.position_embedding(x)

        x = self.embedding_dropout(x)

        if return_attention:
            x, attentions = self.transformer(
                x,
                return_attention=True,
            )
        else:
            x = self.transformer(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have the same shape as input_ids."
                )

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        if return_attention:
            return logits, loss, attentions

        return logits, loss

    def count_parameters(self):
        """Return the number of trainable parameters."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
