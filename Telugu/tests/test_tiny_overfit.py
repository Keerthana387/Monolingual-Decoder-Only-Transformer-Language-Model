import math
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from Telugu.config.model_config import ModelConfig
from Telugu.data.dataset import TinyLanguageModelDataset
from Telugu.model.lm import TeluguLanguageModel


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

NUM_SEQUENCES = 8
SEQUENCE_LENGTH = 33

BATCH_SIZE = 4

NUM_STEPS = 500

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

PRINT_EVERY = 25

OVERFIT_LOSS_THRESHOLD = 0.10

OUTPUT_PLOT = "Telugu/tests/tiny_overfit_loss.png"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE TINY DATASET
# ============================================================

def create_tiny_sequences(
    num_sequences: int,
    sequence_length: int,
    vocab_size: int,
) -> list[list[int]]:
    """
    Create deterministic synthetic token sequences.

    We intentionally use a small fixed dataset so the model
    should be able to memorize it.
    """

    generator = torch.Generator()

    generator.manual_seed(SEED)

    sequences = []

    for _ in range(num_sequences):

        sequence = torch.randint(
            low=1,
            high=vocab_size,
            size=(sequence_length,),
            generator=generator,
        )

        sequences.append(
            sequence.tolist()
        )

    return sequences


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TELUGU — TINY DATASET OVERFIT TEST")
    print("=" * 80)

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = ModelConfig()

    print("\nModel:")
    print(f"Vocabulary size : {config.vocab_size}")
    print(f"d_model         : {config.d_model}")
    print(f"Layers          : {config.n_layers}")
    print(f"Heads           : {config.n_heads}")
    print(f"FFN dimension   : {config.d_ff}")
    print(f"Context length  : {config.max_seq_len}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    sequences = create_tiny_sequences(
        num_sequences=NUM_SEQUENCES,
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=config.vocab_size,
    )

    dataset = TinyLanguageModelDataset(
        sequences
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print("\nDataset:")
    print(f"Sequences       : {len(dataset)}")
    print(f"Sequence length : {SEQUENCE_LENGTH}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(
        f"Input length    : "
        f"{SEQUENCE_LENGTH - 1}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = TeluguLanguageModel(
        config
    ).to(device)

    print(
        f"\nParameters      : "
        f"{model.count_parameters():,}"
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    model.train()

    losses = []

    step = 0

    print("\nTraining:")
    print("-" * 80)

    while step < NUM_STEPS:

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            logits, loss, _ = model(
                input_ids=input_ids,
                targets=targets,
            )

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # -------------------------------------------------
            # Optimizer step
            # -------------------------------------------------

            optimizer.step()

            step += 1

            loss_value = loss.item()

            losses.append(loss_value)

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if (
                step == 1
                or step % PRINT_EVERY == 0
                or step == NUM_STEPS
            ):

                perplexity = math.exp(
                    min(loss_value, 20)
                )

                print(
                    f"Step {step:4d}/{NUM_STEPS} | "
                    f"Loss: {loss_value:.6f} | "
                    f"Perplexity: {perplexity:.3f}"
                )

            if step >= NUM_STEPS:
                break

    # ---------------------------------------------------------
    # Final loss
    # ---------------------------------------------------------

    final_loss = losses[-1]

    print("\n" + "-" * 80)

    print(
        f"Initial loss : {losses[0]:.6f}"
    )

    print(
        f"Final loss   : {final_loss:.6f}"
    )

    print(
        f"Best loss    : {min(losses):.6f}"
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        losses,
        linewidth=1.5,
    )

    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Telugu Model — Tiny Dataset Overfit")

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150,
    )

    plt.close()

    print(
        f"\nSaved loss curve to:"
        f"\n{OUTPUT_PLOT}"
    )

    # ---------------------------------------------------------
    # Pass/fail
    # ---------------------------------------------------------

    if final_loss < OVERFIT_LOSS_THRESHOLD:

        print("\n✓ Tiny dataset successfully overfit")

        print(
            f"✓ Final loss "
            f"{final_loss:.6f} < "
            f"{OVERFIT_LOSS_THRESHOLD:.2f}"
        )

        print("\n" + "=" * 80)
        print("TINY OVERFIT TEST PASSED")
        print("=" * 80)

        return

    print("\n✗ Tiny dataset did not sufficiently overfit")

    print(
        f"✗ Final loss "
        f"{final_loss:.6f} >= "
        f"{OVERFIT_LOSS_THRESHOLD:.2f}"
    )

    raise RuntimeError(
        "Tiny overfit test failed."
    )


if __name__ == "__main__":
    main()