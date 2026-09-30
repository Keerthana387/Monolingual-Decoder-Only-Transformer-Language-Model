import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig
from .transformer import DecoderOnlyTransformer


class TeluguLanguageModel(nn.Module):
    """
    Decoder-only Telugu language model.

    Components:

        Token Embedding
             +
        Learned Positional Embedding
             ↓
        Transformer Blocks
             ↓
        Final LayerNorm
             ↓
        Tied LM Head
             ↓
        Vocabulary Logits

    The model expects input_ids and targets to already be aligned
    for causal language modeling.

    Example:

        input_ids:
            [BOS, t1, t2, t3, t4]

        targets:
            [t1,  t2, t3, t4, t5]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # ---------------------------------------------------------
        # Token embedding
        # ---------------------------------------------------------

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        self.transformer = DecoderOnlyTransformer(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )

        # ---------------------------------------------------------
        # Language-model output head
        # ---------------------------------------------------------

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # ---------------------------------------------------------
        # Weight tying
        # ---------------------------------------------------------

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # ---------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights.
        """

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
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        Forward pass.

        Args:
            input_ids:
                Token IDs of shape (B, T).

            targets:
                Next-token targets of shape (B, T).

                These must already be shifted/aligned by the
                dataset or training pipeline.

            return_attention:
                Whether to return attention weights.

        Returns:
            logits:
                (B, T, vocab_size)

            loss:
                Scalar cross-entropy loss if targets are provided.

            attention_weights:
                Attention weights if requested.
        """

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum length {self.config.max_seq_len}."
            )

        # ---------------------------------------------------------
        # Token embeddings
        # ---------------------------------------------------------

        x = self.token_embedding(input_ids)

        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        hidden_states, attention_weights = self.transformer(
            x,
            return_attention=return_attention,
        )

        # ---------------------------------------------------------
        # LM head
        # ---------------------------------------------------------

        logits = self.lm_head(hidden_states)

        loss = None

        if targets is not None:

            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets and input_ids must have the same shape. "
                    f"Got input_ids={tuple(input_ids.shape)}, "
                    f"targets={tuple(targets.shape)}."
                )

            # -----------------------------------------------------
            # Cross-entropy over next-token targets.
            #
            # The shifting is intentionally NOT done here.
            # The data pipeline is responsible for constructing:
            #
            # input:
            #   [BOS, t1, t2, t3]
            #
            # target:
            #   [t1,  t2, t3, t4]
            # -----------------------------------------------------

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.config.vocab_size,
                ),
                targets.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss, attention_weights

    def count_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:
        """
        Count model parameters.
        """

        if trainable_only:

            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )