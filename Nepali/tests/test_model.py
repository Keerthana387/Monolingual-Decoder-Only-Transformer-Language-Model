import torch

from Nepali.config.model_config import ModelConfig
from Nepali.model.lm import LanguageModel


def test_model_forward():
    config = ModelConfig()

    model = LanguageModel(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )

    logits, loss = model(
        input_ids=input_ids,
        targets=targets,
    )

    assert logits.shape == (
        2,
        config.max_seq_len,
        config.vocab_size,
    )

    assert loss is not None
    assert torch.isfinite(loss)


def test_tied_embeddings():
    config = ModelConfig(tie_embeddings=True)
    model = LanguageModel(config)

    assert (
        model.lm_head.weight.data_ptr()
        == model.token_embedding.weight.data_ptr()
    )


def test_parameter_count():
    config = ModelConfig()
    model = LanguageModel(config)

    assert model.count_parameters() == 25_950_848
