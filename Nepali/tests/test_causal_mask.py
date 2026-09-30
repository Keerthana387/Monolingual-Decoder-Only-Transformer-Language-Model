import torch

from Nepali.config.model_config import ModelConfig
from Nepali.model.attention import CausalSelfAttention


def test_causal_mask():
    config = ModelConfig(
        d_model=64,
        n_heads=4,
        max_seq_len=8,
    )

    attention = CausalSelfAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        dropout=0.0,
        attention_dropout=0.0,
        max_seq_len=config.max_seq_len,
    )

    x = torch.randn(1, 8, 64)

    _, weights = attention(
        x,
        return_attention=True,
    )

    future = torch.triu(
        torch.ones(8, 8, dtype=torch.bool),
        diagonal=1,
    )

    assert torch.all(
        weights[0, :, future] == 0
    )
