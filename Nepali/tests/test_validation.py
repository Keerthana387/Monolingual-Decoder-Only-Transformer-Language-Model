import pytest
import torch

from Nepali.config.train_config import TrainConfig
from Nepali.data.validation_dataset import NepaliEvaluationDataset


def test_validation_dataset_length():
    config = TrainConfig()

    dataset = NepaliEvaluationDataset(
        data_dir=config.validation_dir,
        block_size=config.sequence_length,
        split="validation",
    )

    assert len(dataset) > 0


def test_validation_shapes():
    config = TrainConfig()

    dataset = NepaliEvaluationDataset(
        data_dir=config.validation_dir,
        block_size=config.sequence_length,
        split="validation",
    )

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.shape == (config.sequence_length,)
    assert y.shape == (config.sequence_length,)


def test_validation_shift():
    config = TrainConfig()

    dataset = NepaliEvaluationDataset(
        data_dir=config.validation_dir,
        block_size=config.sequence_length,
        split="validation",
    )

    x, y = dataset[0]

    assert torch.equal(
        y[:-1],
        x[1:],
    )


def test_test_split():
    config = TrainConfig()

    dataset = NepaliEvaluationDataset(
        data_dir=config.test_dir,
        block_size=config.sequence_length,
        split="test",
    )

    assert len(dataset) > 0


def test_invalid_split():
    config = TrainConfig()

    with pytest.raises(ValueError):
        NepaliEvaluationDataset(
            data_dir=config.validation_dir,
            block_size=config.sequence_length,
            split="invalid",
        )
