import torch

from Nepali.config.train_config import TrainConfig
from Nepali.data.dataset import NepaliTrainDataset
from Nepali.data.validation_dataset import NepaliEvaluationDataset


def test_train_dataset():
    config = TrainConfig()

    dataset = NepaliTrainDataset(
        data_dir=config.tokenized_dir,
        block_size=config.sequence_length,
    )

    assert len(dataset) > 0

    x, y = dataset[0]

    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    assert x.shape == (config.sequence_length,)
    assert y.shape == (config.sequence_length,)


def test_train_dataset_shift():
    config = TrainConfig()

    dataset = NepaliTrainDataset(
        data_dir=config.tokenized_dir,
        block_size=config.sequence_length,
    )

    x, y = dataset[0]

    assert torch.equal(
        y[:-1],
        x[1:],
    )


def test_validation_dataset():
    config = TrainConfig()

    dataset = NepaliEvaluationDataset(
        data_dir=config.validation_dir,
        block_size=config.sequence_length,
        split="validation",
    )

    x, y = dataset[0]

    assert x.shape == (config.sequence_length,)
    assert y.shape == (config.sequence_length,)


def test_test_dataset():
    config = TrainConfig()

    dataset = NepaliEvaluationDataset(
        data_dir=config.test_dir,
        block_size=config.sequence_length,
        split="test",
    )

    x, y = dataset[0]

    assert x.shape == (config.sequence_length,)
    assert y.shape == (config.sequence_length,)


def test_train_dataset_boundaries():
    config = TrainConfig()

    dataset = NepaliTrainDataset(
        data_dir=config.tokenized_dir,
        block_size=config.sequence_length,
    )

    x, y = dataset[len(dataset) - 1]

    assert x.shape == (config.sequence_length,)
    assert y.shape == (config.sequence_length,)
