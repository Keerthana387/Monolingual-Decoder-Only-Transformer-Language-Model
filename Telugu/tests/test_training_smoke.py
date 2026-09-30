import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from Telugu.config.model_config import ModelConfig
from Telugu.config.train_config import TrainConfig
from Telugu.data.dataset import TokenShardDataset
from Telugu.data.validation_dataset import ValidationTokenDataset
from Telugu.model.lm import TeluguLanguageModel
from Telugu.train import (
    get_learning_rate,
    save_checkpoint,
    evaluate,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

SMOKE_STEPS = 20

BATCH_SIZE = 2

# Only use a small number of validation batches.
VALIDATION_STEPS = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TELUGU — REAL DATA CPU TRAINING SMOKE TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    model_config = ModelConfig()
    train_config = TrainConfig()

    print("\nModel configuration:")

    print(
        f"Vocabulary size : "
        f"{model_config.vocab_size}"
    )

    print(
        f"d_model         : "
        f"{model_config.d_model}"
    )

    print(
        f"Layers          : "
        f"{model_config.n_layers}"
    )

    print(
        f"Heads           : "
        f"{model_config.n_heads}"
    )

    print(
        f"Context length  : "
        f"{model_config.max_seq_len}"
    )

    # ---------------------------------------------------------
    # Training shards
    # ---------------------------------------------------------

    from pathlib import Path

    train_dir = Path(
        train_config.tokenized_dir
    )

    shard_paths = sorted(
        train_dir.glob(
            "train_*.bin"
        )
    )

    assert len(shard_paths) > 0, (
        f"No training shards found in "
        f"{train_dir}"
    )

    print(
        f"\nTraining shards : "
        f"{len(shard_paths)}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_dataset = TokenShardDataset(
        shard_paths=shard_paths,
        sequence_length=(
            model_config.max_seq_len
        ),
    )

    print(
        f"Training samples: "
        f"{len(train_dataset):,}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation_dataset = (
        ValidationTokenDataset(
            validation_dir=(
                train_config.validation_dir
            ),
            sequence_length=(
                model_config.max_seq_len
            ),
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nCreating model...")

    model = TeluguLanguageModel(
        model_config
    ).to(device)

    parameters = (
        model.count_parameters()
    )

    print(
        f"Parameters      : "
        f"{parameters:,}"
    )

    assert parameters == 23_042_432

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(
            train_config.beta1,
            train_config.beta2,
        ),
        weight_decay=(
            train_config.weight_decay
        ),
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print("\nTraining:")
    print("-" * 80)

    model.train()

    losses = []

    data_iterator = iter(
        train_loader
    )

    for step in range(
        1,
        SMOKE_STEPS + 1,
    ):

        # -----------------------------------------------------
        # Get batch
        # -----------------------------------------------------

        try:

            batch = next(
                data_iterator
            )

        except StopIteration:

            data_iterator = iter(
                train_loader
            )

            batch = next(
                data_iterator
            )

        input_ids = batch[
            "input_ids"
        ].to(device)

        targets = batch[
            "targets"
        ].to(device)

        # -----------------------------------------------------
        # Learning rate
        # -----------------------------------------------------

        lr = get_learning_rate(
            step,
            train_config,
        )

        for group in (
            optimizer.param_groups
        ):

            group["lr"] = lr

        # -----------------------------------------------------
        # Forward
        # -----------------------------------------------------

        logits, loss, _ = model(
            input_ids=input_ids,
            targets=targets,
        )

        assert logits.shape == (
            BATCH_SIZE,
            model_config.max_seq_len,
            model_config.vocab_size,
        )

        assert torch.isfinite(loss)

        # -----------------------------------------------------
        # Backward
        # -----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # -----------------------------------------------------
        # Gradient clipping
        # -----------------------------------------------------

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.grad_clip,
            )
        )

        assert torch.isfinite(
            grad_norm
        )

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------

        optimizer.step()

        loss_value = loss.item()

        losses.append(loss_value)

        perplexity = math.exp(
            min(
                loss_value,
                20,
            )
        )

        print(
            f"Step {step:2d}/{SMOKE_STEPS} | "
            f"LR {lr:.6e} | "
            f"Loss {loss_value:.5f} | "
            f"PPL {perplexity:.2f}"
        )

    # ---------------------------------------------------------
    # Check loss behaviour
    # ---------------------------------------------------------

    assert len(losses) == SMOKE_STEPS

    assert all(
        math.isfinite(x)
        for x in losses
    )

    print(
        "\n✓ All training losses finite"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\nRunning validation...")

    val_loss, val_ppl = evaluate(
        model=model,
        dataloader=validation_loader,
        device=device,
        max_batches=VALIDATION_STEPS,
    )

    assert math.isfinite(
        val_loss
    )

    assert math.isfinite(
        val_ppl
    )

    print(
        f"Validation loss : "
        f"{val_loss:.5f}"
    )

    print(
        f"Validation PPL  : "
        f"{val_ppl:.2f}"
    )

    print(
        "✓ Validation pass completed"
    )

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------

    checkpoint_path = (
        "Telugu/checkpoints/"
        "smoke_test.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=SMOKE_STEPS,
        loss=losses[-1],
        config=train_config,
        path=checkpoint_path,
    )

    print(
        f"\n✓ Checkpoint saved:"
        f"\n{checkpoint_path}"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("REAL DATA CPU TRAINING SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()