import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from Telugu.config.model_config import ModelConfig
from Telugu.config.train_config import TrainConfig
from Telugu.model.lm import TeluguLanguageModel

from Telugu.train import (
    save_checkpoint,
    load_checkpoint,
    get_learning_rate,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TELUGU — CHECKPOINT / RESUME TEST")
    print("=" * 80)

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    device = torch.device("cpu")

    print(
        f"\nDevice: {device}"
    )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    model_config = ModelConfig()

    train_config = TrainConfig()

    # Keep this test short.
    train_config.max_steps = 10

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = TeluguLanguageModel(
        model_config
    ).to(device)

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
    # Perform one actual optimizer step
    # ---------------------------------------------------------

    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        (
            2,
            model_config.max_seq_len,
        ),
        dtype=torch.long,
    )

    targets = torch.randint(
        0,
        model_config.vocab_size,
        (
            2,
            model_config.max_seq_len,
        ),
        dtype=torch.long,
    )

    logits, loss, _ = model(
        input_ids=input_ids,
        targets=targets,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    optimizer.step()

    step = 5

    # ---------------------------------------------------------
    # Temporary checkpoint
    # ---------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmp:

        checkpoint_path = (
            Path(tmp)
            / "test_checkpoint.pt"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=loss.item(),
            config=train_config,
            path=checkpoint_path,
        )

        assert checkpoint_path.exists()

        print(
            "\n✓ Checkpoint created"
        )

        # -----------------------------------------------------
        # Save original model state
        # -----------------------------------------------------

        original_state = {
            key: value.clone()
            for key, value
            in model.state_dict().items()
        }

        original_optimizer_state = (
            optimizer.state_dict()
        )

        # -----------------------------------------------------
        # Create fresh model
        # -----------------------------------------------------

        new_model = (
            TeluguLanguageModel(
                model_config
            ).to(device)
        )

        new_optimizer = (
            torch.optim.AdamW(
                new_model.parameters(),
                lr=train_config.learning_rate,
                betas=(
                    train_config.beta1,
                    train_config.beta2,
                ),
                weight_decay=(
                    train_config.weight_decay
                ),
            )
        )

        # -----------------------------------------------------
        # Load checkpoint
        # -----------------------------------------------------

        loaded_step = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            device=device,
        )

        assert loaded_step == step

        print(
            "✓ Training step restored"
        )

        # -----------------------------------------------------
        # Compare model parameters
        # -----------------------------------------------------

        for key, value in (
            new_model.state_dict().items()
        ):

            assert torch.equal(
                value,
                original_state[key],
            ), (
                f"Model parameter mismatch: "
                f"{key}"
            )

        print(
            "✓ Model weights restored exactly"
        )

        # -----------------------------------------------------
        # Compare optimizer
        # -----------------------------------------------------

        loaded_optimizer_state = (
            new_optimizer.state_dict()
        )

        assert (
            loaded_optimizer_state["state"]
            .keys()
            == original_optimizer_state[
                "state"
            ].keys()
        )

        print(
            "✓ Optimizer state restored"
        )

        # -----------------------------------------------------
        # LR schedule
        # -----------------------------------------------------

        lr_before = get_learning_rate(
            step,
            train_config,
        )

        lr_after = get_learning_rate(
            loaded_step,
            train_config,
        )

        assert math.isclose(
            lr_before,
            lr_after,
            rel_tol=1e-12,
        )

        print(
            "✓ Learning-rate schedule resumes correctly"
        )

        # -----------------------------------------------------
        # Continue training
        # -----------------------------------------------------

        new_model.train()

        logits, resumed_loss, _ = (
            new_model(
                input_ids=input_ids,
                targets=targets,
            )
        )

        new_optimizer.zero_grad(
            set_to_none=True
        )

        resumed_loss.backward()

        new_optimizer.step()

        assert torch.isfinite(
            resumed_loss
        )

        print(
            "✓ Training successfully continued after resume"
        )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHECKPOINT / RESUME TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()