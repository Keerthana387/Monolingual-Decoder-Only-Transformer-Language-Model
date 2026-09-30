import torch

from Nepali.config.model_config import ModelConfig
from Nepali.model.lm import LanguageModel


def test_tiny_overfit():
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=8000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        attention_dropout=0.0,
        tie_embeddings=True,
    )

    model = LanguageModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 32),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 32),
    )

    model.train()

    with torch.no_grad():
        _, initial_loss = model(
            input_ids=input_ids,
            targets=targets,
        )

    for _ in range(100):
        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            input_ids=input_ids,
            targets=targets,
        )

        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert loss.item() < initial_loss.item()
