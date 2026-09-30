import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Nepali.config.model_config import ModelConfig
from Nepali.config.train_config import TrainConfig
from Nepali.data.dataset import NepaliTrainDataset
from Nepali.data.validation_dataset import NepaliEvaluationDataset
from Nepali.model.lm import LanguageModel
from Nepali.train import (
    get_learning_rate,
    save_checkpoint,
    load_checkpoint,
    evaluate,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

# Keep this deliberately small.
SMOKE_STEPS = 2

# Small batch to keep CPU memory usage low.
BATCH_SIZE = 1

# Only evaluate one validation batch.
VALIDATION_STEPS = 1


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
    print("NEPALI — REAL DATA CPU TRAINING SMOKE TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cpu")

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Verify expected Nepali model configuration
    # --------------------------------------------------------

    assert model_config.vocab_size == 8000

    assert model_config.d_model == 448

    assert model_config.n_layers == 11

    assert model_config.n_heads == 8

    assert model_config.max_seq_len == 512

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train_dir = Path(
        train_config.tokenized_dir
    )

    validation_dir = Path(
        train_config.validation_dir
    )

    print(
        f"\nTraining directory:"
        f"\n{train_dir}"
    )

    print(
        f"\nValidation directory:"
        f"\n{validation_dir}"
    )

    assert train_dir.exists(), (
        f"Training directory does not exist: "
        f"{train_dir}"
    )

    assert validation_dir.exists(), (
        f"Validation directory does not exist: "
        f"{validation_dir}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_dataset = NepaliTrainDataset(
        data_dir=train_dir,
        block_size=model_config.max_seq_len,
    )

    print(
        f"Training tokens  : "
        f"{train_dataset.total_tokens:,}"
    )

    print(
        f"Training samples : "
        f"{len(train_dataset):,}"
    )

    assert train_dataset.total_tokens == 500_000_000

    assert len(train_dataset) > 0

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    validation_dataset = NepaliEvaluationDataset(
        data_dir=validation_dir,
        block_size=model_config.max_seq_len,
        split="validation",
    )

    print(
        f"Validation tokens : "
        f"{validation_dataset.total_tokens:,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )

    assert validation_dataset.total_tokens == 10_000_000

    assert len(validation_dataset) > 0

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = LanguageModel(
        model_config
    ).to(device)

    parameters = model.count_parameters()

    print(
        f"Parameters      : "
        f"{parameters:,}"
    )

    # This is the expected Nepali model size.
    assert parameters == 25_950_848

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(
            train_config.beta1,
            train_config.beta2,
        ),
        weight_decay=train_config.weight_decay,
    )

    # --------------------------------------------------------
    # Training smoke test
    # --------------------------------------------------------

    print("\nTraining:")
    print("-" * 80)

    model.train()

    losses = []

    data_iterator = iter(train_loader)

    for step in range(
        1,
        SMOKE_STEPS + 1,
    ):

        # ----------------------------------------------------
        # Get batch
        # ----------------------------------------------------

        input_ids, targets = next(
            data_iterator
        )

        input_ids = input_ids.to(device)

        targets = targets.to(device)

        # ----------------------------------------------------
        # Check batch
        # ----------------------------------------------------

        assert input_ids.shape == (
            BATCH_SIZE,
            model_config.max_seq_len,
        )

        assert targets.shape == (
            BATCH_SIZE,
            model_config.max_seq_len,
        )

        assert input_ids.dtype == torch.int64

        assert targets.dtype == torch.int64

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_learning_rate(
            step,
            train_config,
        )

        assert math.isfinite(lr)

        for group in optimizer.param_groups:
            group["lr"] = lr

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, loss = model(
            input_ids=input_ids,
            targets=targets,
        )

        assert logits.shape == (
            BATCH_SIZE,
            model_config.max_seq_len,
            model_config.vocab_size,
        )

        assert loss is not None

        assert torch.isfinite(loss)

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.grad_clip,
            )
        )

        assert torch.isfinite(
            grad_norm
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Training checks
    # --------------------------------------------------------

    assert len(losses) == SMOKE_STEPS

    assert all(
        math.isfinite(loss)
        for loss in losses
    )

    print(
        "\n✓ Training forward/backward passed"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nRunning validation...")

    val_loss, val_ppl = evaluate(
        model=model,
        dataloader=validation_loader,
        device=device,
        max_batches=VALIDATION_STEPS,
    )

    assert math.isfinite(val_loss)

    assert math.isfinite(val_ppl)

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

    # --------------------------------------------------------
    # Checkpoint test
    # --------------------------------------------------------

    print("\nTesting checkpoint save/load...")

    smoke_checkpoint = (
        Path(train_config.checkpoint_dir)
        / "smoke_test.pt"
    )

    # Make sure an old smoke checkpoint cannot
    # accidentally affect this test.
    if smoke_checkpoint.exists():
        smoke_checkpoint.unlink()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=SMOKE_STEPS,
        loss=losses[-1],
        config=train_config,
        path=smoke_checkpoint,
        scheduler_state={
            "step": SMOKE_STEPS,
            "current_lr": optimizer.param_groups[0]["lr"],
            "schedule": "warmup_cosine",
        },
    )

    assert smoke_checkpoint.exists()

    print(
        f"✓ Checkpoint saved:"
        f"\n{smoke_checkpoint}"
    )

    # --------------------------------------------------------
    # Load checkpoint into a fresh model
    # --------------------------------------------------------

    restored_model = LanguageModel(
        model_config
    ).to(device)

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=train_config.learning_rate,
        betas=(
            train_config.beta1,
            train_config.beta2,
        ),
        weight_decay=train_config.weight_decay,
    )

    restored_step, restored_loss, scheduler_state = (
        load_checkpoint(
            smoke_checkpoint,
            restored_model,
            restored_optimizer,
            device,
        )
    )

    assert restored_step == SMOKE_STEPS

    assert math.isfinite(restored_loss)

    assert restored_loss == losses[-1]

    assert "current_lr" in scheduler_state

    # --------------------------------------------------------
    # Verify model weights match
    # --------------------------------------------------------

    original_state = model.state_dict()

    restored_state = restored_model.state_dict()

    assert original_state.keys() == restored_state.keys()

    for name in original_state:

        assert torch.equal(
            original_state[name],
            restored_state[name],
        ), (
            f"Checkpoint mismatch for "
            f"parameter: {name}"
        )

    print(
        "✓ Checkpoint restore verified"
    )

    # --------------------------------------------------------
    # Remove temporary checkpoint
    # --------------------------------------------------------

    smoke_checkpoint.unlink()

    assert not smoke_checkpoint.exists()

    print(
        "✓ Temporary smoke checkpoint removed"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("NEPALI — REAL DATA CPU TRAINING SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()