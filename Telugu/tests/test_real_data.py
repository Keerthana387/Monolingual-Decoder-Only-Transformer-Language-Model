import torch
from torch.utils.data import DataLoader

from Telugu.config.model_config import ModelConfig
from Telugu.data.dataset import TokenShardDataset
from Telugu.model.lm import TeluguLanguageModel


# ============================================================
# SETTINGS
# ============================================================

SHARD_PATH = "Telugu/data/tokenized/train_000.bin"

SEQUENCE_LENGTH = 512
BATCH_SIZE = 2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TELUGU — REAL DATA SMOKE TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    # This test is intentionally CPU-first.
    #
    # We do NOT need GPU just to verify that the dataset and
    # model work together.
    #
    device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = ModelConfig()

    print("\nModel configuration:")
    print(
        f"Vocabulary size : "
        f"{config.vocab_size}"
    )

    print(
        f"d_model         : "
        f"{config.d_model}"
    )

    print(
        f"Layers          : "
        f"{config.n_layers}"
    )

    print(
        f"Heads           : "
        f"{config.n_heads}"
    )

    print(
        f"FFN dimension   : "
        f"{config.d_ff}"
    )

    print(
        f"Context length  : "
        f"{config.max_seq_len}"
    )

    # ---------------------------------------------------------
    # Configuration checks
    # ---------------------------------------------------------

    assert config.vocab_size == 6000

    assert SEQUENCE_LENGTH <= config.max_seq_len

    print("\n✓ Configuration is consistent")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print("\nLoading real Telugu shard:")
    print(SHARD_PATH)

    dataset = TokenShardDataset(
        shard_path=SHARD_PATH,
        sequence_length=SEQUENCE_LENGTH,
    )

    print("\nDataset:")
    print(
        f"Shard tokens    : "
        f"{len(dataset.tokens):,}"
    )

    print(
        f"Samples         : "
        f"{len(dataset):,}"
    )

    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH}"
    )

    # ---------------------------------------------------------
    # Inspect one sample
    # ---------------------------------------------------------

    sample = dataset[0]

    input_ids = sample["input_ids"]
    targets = sample["targets"]

    print("\nSingle sample:")
    print(
        f"Input shape     : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"Target shape    : "
        f"{tuple(targets.shape)}"
    )

    assert input_ids.shape == (
        SEQUENCE_LENGTH,
    )

    assert targets.shape == (
        SEQUENCE_LENGTH,
    )

    # ---------------------------------------------------------
    # Token ID validity
    # ---------------------------------------------------------

    print("\nChecking token IDs...")

    print(
        f"Minimum token ID : "
        f"{input_ids.min().item()}"
    )

    print(
        f"Maximum token ID : "
        f"{input_ids.max().item()}"
    )

    assert input_ids.min().item() >= 0

    assert input_ids.max().item() < config.vocab_size

    assert targets.min().item() >= 0

    assert targets.max().item() < config.vocab_size

    print("✓ All token IDs are valid")

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    batch_input_ids = batch["input_ids"]
    batch_targets = batch["targets"]

    print("\nDataLoader batch:")

    print(
        f"Input shape     : "
        f"{tuple(batch_input_ids.shape)}"
    )

    print(
        f"Target shape    : "
        f"{tuple(batch_targets.shape)}"
    )

    assert batch_input_ids.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert batch_targets.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    print("✓ DataLoader batch correct")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nCreating model...")

    model = TeluguLanguageModel(
        config
    ).to(device)

    model.train()

    print(
        f"Parameters      : "
        f"{model.count_parameters():,}"
    )

    assert (
        model.count_parameters()
        == 23_042_432
    )

    # ---------------------------------------------------------
    # Move batch to device
    # ---------------------------------------------------------

    batch_input_ids = batch_input_ids.to(
        device
    )

    batch_targets = batch_targets.to(
        device
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    print("\nRunning forward pass...")

    logits, loss, _ = model(
        input_ids=batch_input_ids,
        targets=batch_targets,
    )

    print(
        f"Logits shape    : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Loss            : "
        f"{loss.item():.6f}"
    )

    assert logits.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        config.vocab_size,
    )

    assert loss.ndim == 0

    assert torch.isfinite(loss)

    print("✓ Real-data forward pass correct")

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------

    print("\nRunning backward pass...")

    model.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradients_found = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            gradients_found += 1

    assert gradients_found > 0

    print(
        f"✓ Backward pass correct "
        f"({gradients_found} tensors with gradients)"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("REAL TELUGU DATA SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()