import math

import torch
from torch.utils.data import DataLoader

from Telugu.config.model_config import ModelConfig
from Telugu.data.validation_dataset import (
    ValidationTokenDataset,
)
from Telugu.model.lm import TeluguLanguageModel


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_DIR = (
    "Telugu/data/tokenized/validation"
)

BATCH_SIZE = 4


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TELUGU — VALIDATION DATASET TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------

    config = ModelConfig()

    print("\nModel configuration:")

    print(
        f"Vocabulary size : "
        f"{config.vocab_size}"
    )

    print(
        f"Context length  : "
        f"{config.max_seq_len}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = ValidationTokenDataset(
        validation_dir=VALIDATION_DIR,
        sequence_length=config.max_seq_len,
    )

    stats = dataset.get_statistics()

    print("\nValidation dataset:")

    print(
        f"Shards          : "
        f"{stats['num_shards']}"
    )

    print(
        f"Total tokens    : "
        f"{stats['total_tokens']:,}"
    )

    print(
        f"Samples         : "
        f"{stats['num_samples']:,}"
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
        f"Discarded       : "
        f"{stats['discarded_tokens']:,}"
    )

    # ---------------------------------------------------------
    # Metadata consistency
    # ---------------------------------------------------------

    assert (
        dataset.metadata["vocab_size"]
        == config.vocab_size
    )

    assert (
        dataset.metadata["sequence_length"]
        == config.max_seq_len
    )

    assert (
        dataset.metadata["samples"]
        == stats["num_samples"]
    )

    assert (
        dataset.metadata["discarded_tokens"]
        == stats["discarded_tokens"]
    )

    print(
        "✓ Metadata consistent"
    )

    # ---------------------------------------------------------
    # Dataset length
    # ---------------------------------------------------------

    expected_samples = 5697

    assert len(dataset) == expected_samples, (
        f"Expected {expected_samples:,} "
        f"samples, got {len(dataset):,}"
    )

    print(
        "✓ Dataset length correct"
    )

    # ---------------------------------------------------------
    # Individual samples
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
            config.max_seq_len,
        )

        assert targets.shape == (
            config.max_seq_len,
        )

        assert input_ids.dtype == (
            torch.int64
        )

        assert targets.dtype == (
            torch.int64
        )

        print(
            f"  Sample {index}: "
            f"input={tuple(input_ids.shape)}, "
            f"target={tuple(targets.shape)}"
        )

    print(
        "✓ Individual samples valid"
    )

    # ---------------------------------------------------------
    # Token ID range
    # ---------------------------------------------------------

    print(
        "\nChecking validation token IDs..."
    )

    min_token = config.vocab_size
    max_token = -1

    # Check all samples.
    #
    # This is only ~2.9M tokens, so it is cheap
    # enough to perform locally.

    for index in range(len(dataset)):

        sample = dataset[index]

        input_ids = sample[
            "input_ids"
        ]

        targets = sample[
            "targets"
        ]

        local_min = min(
            input_ids.min().item(),
            targets.min().item(),
        )

        local_max = max(
            input_ids.max().item(),
            targets.max().item(),
        )

        min_token = min(
            min_token,
            local_min,
        )

        max_token = max(
            max_token,
            local_max,
        )

    print(
        f"Minimum token ID : "
        f"{min_token}"
    )

    print(
        f"Maximum token ID : "
        f"{max_token}"
    )

    assert min_token >= 0

    assert max_token < (
        config.vocab_size
    )

    print(
        "✓ All validation token IDs valid"
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(dataloader))

    input_ids = batch[
        "input_ids"
    ]

    targets = batch[
        "targets"
    ]

    print("\nDataLoader:")

    print(
        f"Input shape   : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Target shape  : "
        f"{tuple(targets.shape)}"
    )

    assert input_ids.shape == (
        BATCH_SIZE,
        config.max_seq_len,
    )

    assert targets.shape == (
        BATCH_SIZE,
        config.max_seq_len,
    )

    print(
        "✓ DataLoader batch correct"
    )

    # ---------------------------------------------------------
    # Input / target alignment
    # ---------------------------------------------------------

    print(
        "\nChecking input/target alignment..."
    )

    sample = dataset[0]

    sample_input = sample[
        "input_ids"
    ]

    sample_target = sample[
        "targets"
    ]

    assert torch.equal(
        sample_input[1:],
        sample_target[:-1],
    )

    print(
        "✓ Input/target alignment correct"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print(
        "\nCreating model..."
    )

    model = TeluguLanguageModel(
        config
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Validation forward pass
    # ---------------------------------------------------------

    input_ids = input_ids.to(device)

    targets = targets.to(device)

    print(
        "\nRunning validation forward pass..."
    )

    with torch.no_grad():

        logits, loss, _ = model(
            input_ids=input_ids,
            targets=targets,
        )

    print(
        f"Logits shape  : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Loss          : "
        f"{loss.item():.6f}"
    )

    assert logits.shape == (
        BATCH_SIZE,
        config.max_seq_len,
        config.vocab_size,
    )

    assert loss.ndim == 0

    assert torch.isfinite(loss)

    print(
        "✓ Validation forward pass correct"
    )

    # ---------------------------------------------------------
    # Perplexity
    # ---------------------------------------------------------

    perplexity = math.exp(
        min(loss.item(), 20)
    )

    print(
        f"Perplexity     : "
        f"{perplexity:.4f}"
    )

    assert math.isfinite(
        perplexity
    )

    assert perplexity > 0

    print(
        "✓ Validation perplexity valid"
    )

    # ---------------------------------------------------------
    # No gradients
    # ---------------------------------------------------------

    for parameter in model.parameters():

        assert parameter.grad is None

    print(
        "✓ No gradients produced during validation"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION DATASET TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()