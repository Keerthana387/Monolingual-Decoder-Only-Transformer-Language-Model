from pathlib import Path

import numpy as np
import torch

from Nepali.config.train_config import TrainConfig
from Nepali.data.dataset import NepaliTrainDataset
from Nepali.data.validation_dataset import NepaliEvaluationDataset


ROOT = Path(__file__).resolve().parents[1]


def test_real_train_data_exists():
    config = TrainConfig()
    paths = sorted(
        Path(config.tokenized_dir).glob("train_*.bin")
    )

    assert len(paths) == 50


def test_real_validation_data_exists():
    config = TrainConfig()
    paths = sorted(
        Path(config.validation_dir).glob("val_*.bin")
    )

    assert len(paths) == 1


def test_real_test_data_exists():
    config = TrainConfig()
    paths = sorted(
        Path(config.test_dir).glob("test_*.bin")
    )

    assert len(paths) == 1


def test_real_token_ids_are_valid():
    config = TrainConfig()

    train = NepaliTrainDataset(
        data_dir=config.tokenized_dir,
        block_size=config.sequence_length,
    )

    for shard in train.shards[:2]:
        sample = np.asarray(
            shard[:100_000]
        )

        assert sample.min() >= 0
        assert sample.max() < 8000


def test_real_data_structure():
    config = TrainConfig()

    train = NepaliTrainDataset(
        data_dir=config.tokenized_dir,
        block_size=config.sequence_length,
    )

    x, y = train[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    assert x.shape == (
        config.sequence_length,
    )

    assert y.shape == (
        config.sequence_length,
    )

    assert torch.equal(
        y[:-1],
        x[1:],
    )
