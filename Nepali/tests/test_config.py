from Nepali.config.model_config import ModelConfig
from Nepali.config.train_config import TrainConfig


def test_model_config():
    config = ModelConfig()

    assert config.vocab_size == 8000
    assert config.max_seq_len == 512
    assert config.d_model == 448
    assert config.n_layers == 11
    assert config.n_heads == 8
    assert config.d_ff == 1344
    assert config.d_model % config.n_heads == 0


def test_train_config():
    config = TrainConfig()

    assert config.sequence_length == 512
    assert config.batch_size == 16
    assert config.max_steps == 61036
    assert config.learning_rate > 0
    assert config.warmup_steps > 0
    assert config.checkpoint_every > 0
