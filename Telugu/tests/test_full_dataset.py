from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Telugu.config.model_config import ModelConfig
from Telugu.data.dataset import TokenShardDataset


# ============================================================
# SETTINGS
# ============================================================

TOKENIZED_DIR = Path(
    "Telugu/data/tokenized"
)

SEQUENCE_LENGTH = 512

BATCH_SIZE = 4


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TELUGU — FULL TOKENIZED DATASET TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = ModelConfig()

    print("\nConfiguration:")
    print(
        f"Vocabulary size : "
        f"{config.vocab_size}"
    )

    print(
        f"Context length  : "
        f"{config.max_seq_len}"
    )

    assert config.vocab_size == 6000

    assert (
        SEQUENCE_LENGTH
        == config.max_seq_len
    )

    # ---------------------------------------------------------
    # Locate shards
    # ---------------------------------------------------------

    shard_paths = sorted(
        TOKENIZED_DIR.glob(
            "train_*.bin"
        )
    )

    print(
        f"\nFound shards: "
        f"{len(shard_paths)}"
    )

    assert len(shard_paths) == 49

    for path in shard_paths:
        print(
            f"  {path.name}"
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = TokenShardDataset(
        shard_paths=shard_paths,
        sequence_length=SEQUENCE_LENGTH,
    )

    stats = dataset.get_statistics()

    print("\nDataset statistics:")
    print(
        f"Shards          : "
        f"{stats['num_shards']}"
    )

    print(
        f"Total tokens    : "
        f"{stats['total_tokens']:,}"
    )

    print(
        f"Sequence length : "
        f"{stats['sequence_length']}"
    )

    print(
        f"Block size      : "
        f"{stats['block_size']}"
    )

    print(
        f"Samples         : "
        f"{stats['num_samples']:,}"
    )

    print(
        f"Used tokens     : "
        f"{stats['used_tokens']:,}"
    )

    print(
        f"Discarded       : "
        f"{stats['discarded_tokens']:,}"
    )

    # ---------------------------------------------------------
    # Expected corpus size
    # ---------------------------------------------------------

    assert (
        stats["total_tokens"]
        == 486_031_477
    )

    # ---------------------------------------------------------
    # Inspect samples
    # ---------------------------------------------------------

    print("\nInspecting samples...")

    indices = [
        0,
        1,
        len(dataset) // 2,
        len(dataset) - 2,
        len(dataset) - 1,
    ]

    for index in indices:

        sample = dataset[index]

        input_ids = sample[
            "input_ids"
        ]

        targets = sample[
            "targets"
        ]

        assert input_ids.shape == (
            SEQUENCE_LENGTH,
        )

        assert targets.shape == (
            SEQUENCE_LENGTH,
        )

        assert input_ids.dtype == (
            torch.int64
        )

        assert targets.dtype == (
            torch.int64
        )

        assert (
            input_ids.min().item()
            >= 0
        )

        assert (
            input_ids.max().item()
            < config.vocab_size
        )

        assert (
            targets.min().item()
            >= 0
        )

        assert (
            targets.max().item()
            < config.vocab_size
        )

        print(
            f"  Sample {index:,}: "
            f"input={tuple(input_ids.shape)}, "
            f"target={tuple(targets.shape)}"
        )

    print(
        "✓ Individual samples valid"
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    print("\nTesting DataLoader...")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    batch = next(
        iter(dataloader)
    )

    input_ids = batch[
        "input_ids"
    ]

    targets = batch[
        "targets"
    ]

    print(
        f"Batch input shape  : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Batch target shape : "
        f"{tuple(targets.shape)}"
    )

    assert input_ids.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert targets.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert (
        input_ids.min().item()
        >= 0
    )

    assert (
        input_ids.max().item()
        < config.vocab_size
    )

    print(
        "✓ DataLoader batch valid"
    )

    # ---------------------------------------------------------
    # Verify causal alignment
    # ---------------------------------------------------------

    print(
        "\nChecking input/target alignment..."
    )

    sample = dataset[0]

    assert torch.equal(
        sample["input_ids"][1:],
        sample["targets"][:-1],
    )

    print(
        "✓ Input/target alignment correct"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FULL TOKENIZED DATASET TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()