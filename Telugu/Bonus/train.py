import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Telugu.Bonus.config.model_config import ModelConfig
from Telugu.Bonus.config.train_config import TrainConfig
from Telugu.data.dataset import TokenShardDataset
from Telugu.data.validation_dataset import ValidationTokenDataset
from Telugu.Bonus.model.lm import TeluguLanguageModel


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train Telugu language model."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps.",
    )

    return parser.parse_args()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(
    step,
    config,
):

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    if step < config.warmup_steps:

        return (
            config.learning_rate
            * step
            / max(
                1,
                config.warmup_steps,
            )
        )

    # ---------------------------------------------------------
    # Cosine decay
    # ---------------------------------------------------------

    decay_steps = (
        config.max_steps
        - config.warmup_steps
    )

    if decay_steps <= 0:

        return config.learning_rate

    progress = (
        step
        - config.warmup_steps
    ) / decay_steps

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    min_lr = (
        config.learning_rate
        * config.min_learning_rate_ratio
    )

    return (
        min_lr
        + (
            config.learning_rate
            - min_lr
        )
        * cosine
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    config,
    path,
):

    checkpoint = {

        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss": loss,

        "train_config":
            config.__dict__,
    }

    # ---------------------------------------------------------
    # RNG states
    #
    # These make resume more reproducible.
    # ---------------------------------------------------------

    checkpoint["python_rng_state"] = (
        random.getstate()
    )

    checkpoint["numpy_rng_state"] = (
        np.random.get_state()
    )

    checkpoint["torch_rng_state"] = (
        torch.get_rng_state()
    )

    if torch.cuda.is_available():

        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    device,
):

    print(
        f"\nLoading checkpoint:"
        f"\n{path}"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # ---------------------------------------------------------
    # Restore RNG states
    # ---------------------------------------------------------

    if (
        "python_rng_state"
        in checkpoint
    ):

        random.setstate(
            checkpoint[
                "python_rng_state"
            ]
        )

    if (
        "numpy_rng_state"
        in checkpoint
    ):

        np.random.set_state(
            checkpoint[
                "numpy_rng_state"
            ]
        )

    if (
        "torch_rng_state"
        in checkpoint
    ):

        torch.set_rng_state(
            checkpoint[
                "torch_rng_state"
            ]
        )

    if (
        torch.cuda.is_available()
        and "cuda_rng_state"
        in checkpoint
    ):

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_rng_state"
            ]
        )

    step = checkpoint["step"]

    loss = checkpoint["loss"]

    print(
        f"✓ Resumed from step {step:,}"
    )

    print(
        f"✓ Checkpoint loss: "
        f"{loss:.6f}"
    )

    return step


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    max_batches,
):

    model.eval()

    total_loss = 0.0

    batches = 0

    for batch in dataloader:

        if batches >= max_batches:

            break

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "targets"
        ].to(
            device,
            non_blocking=True,
        )

        logits, loss, _ = model(
            input_ids=input_ids,
            targets=targets,
        )

        total_loss += loss.item()

        batches += 1

    if batches == 0:

        raise RuntimeError(
            "Validation DataLoader "
            "returned no batches."
        )

    average_loss = (
        total_loss / batches
    )

    perplexity = math.exp(
        min(
            average_loss,
            20,
        )
    )

    model.train()

    return (
        average_loss,
        perplexity,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    train_config = TrainConfig()

    model_config = ModelConfig()

    # ---------------------------------------------------------
    # Optional command-line override
    # ---------------------------------------------------------

    if args.max_steps is not None:

        train_config.max_steps = (
            args.max_steps
        )

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------

    set_seed(
        train_config.seed
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("TELUGU MODEL H — TRAINING")
    print("=" * 80)

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    print("\nModel:")

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

    print(
        f"Parameters      : "
        f"{model_config}"
    )

    # ---------------------------------------------------------
    # Consistency
    # ---------------------------------------------------------

    assert (
        model_config.vocab_size
        == 6000
    )

    assert (
        train_config.sequence_length
        == model_config.max_seq_len
    )

    # ---------------------------------------------------------
    # Training shards
    # ---------------------------------------------------------

    tokenized_dir = Path(
        train_config.tokenized_dir
    )

    shard_paths = sorted(
        tokenized_dir.glob(
            "train_*.bin"
        )
    )

    if not shard_paths:

        raise FileNotFoundError(
            f"No training shards found in "
            f"{tokenized_dir}"
        )

    print(
        f"\nTraining shards: "
        f"{len(shard_paths)}"
    )

    # ---------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------

    train_dataset = TokenShardDataset(
        shard_paths=shard_paths,
        sequence_length=(
            train_config.sequence_length
        ),
    )

    train_stats = (
        train_dataset.get_statistics()
    )

    print("\nTraining dataset:")

    print(
        f"Tokens          : "
        f"{train_stats['total_tokens']:,}"
    )

    print(
        f"Samples         : "
        f"{train_stats['num_samples']:,}"
    )

    # ---------------------------------------------------------
    # Training DataLoader
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            train_config.batch_size
        ),
        shuffle=True,
        drop_last=True,
        num_workers=(
            train_config.num_workers
        ),
        pin_memory=(
            train_config.pin_memory
            and device.type == "cuda"
        ),
    )

    # ---------------------------------------------------------
    # Validation dataset
    # ---------------------------------------------------------

    validation_dataset = (
        ValidationTokenDataset(
            validation_dir=(
                train_config.validation_dir
            ),
            sequence_length=(
                train_config.sequence_length
            ),
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            train_config.batch_size
        ),
        shuffle=False,
        drop_last=False,
        num_workers=(
            train_config.num_workers
        ),
        pin_memory=(
            train_config.pin_memory
            and device.type == "cuda"
        ),
    )

    print("\nValidation dataset:")

    print(
        f"Tokens          : "
        f"{validation_dataset.metadata['tokens']:,}"
    )

    print(
        f"Samples         : "
        f"{len(validation_dataset):,}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = TeluguLanguageModel(
        model_config
    ).to(device)

    print(
        f"\nTrainable parameters: "
        f"{model.count_parameters():,}"
    )

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
    # Resume
    # ---------------------------------------------------------

    start_step = 0

    if args.resume is not None:

        start_step = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    # ---------------------------------------------------------
    # Mixed precision
    # ---------------------------------------------------------

    use_amp = (
        train_config.use_mixed_precision
        and device.type == "cuda"
    )

    print(
        f"\nMixed precision: "
        f"{'enabled' if use_amp else 'disabled'}"
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    model.train()

    step = start_step

    print("\nTraining:")
    print("-" * 80)

    while step < train_config.max_steps:

        for batch in train_loader:

            if (
                step
                >= train_config.max_steps
            ):

                break

            # -------------------------------------------------
            # Input
            # -------------------------------------------------

            input_ids = batch[
                "input_ids"
            ].to(
                device,
                non_blocking=True,
            )

            targets = batch[
                "targets"
            ].to(
                device,
                non_blocking=True,
            )

            # -------------------------------------------------
            # LR
            # -------------------------------------------------

            lr = get_learning_rate(
                step,
                train_config,
            )

            for group in (
                optimizer.param_groups
            ):

                group["lr"] = lr

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    logits, loss, _ = model(
                        input_ids=input_ids,
                        targets=targets,
                    )

            else:

                logits, loss, _ = model(
                    input_ids=input_ids,
                    targets=targets,
                )

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.grad_clip,
            )

            # -------------------------------------------------
            # Optimizer
            # -------------------------------------------------

            optimizer.step()

            step += 1

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if (
                step == 1
                or step
                % train_config.log_every
                == 0
            ):

                loss_value = loss.item()

                perplexity = math.exp(
                    min(
                        loss_value,
                        20,
                    )
                )

                print(
                    f"Step {step:6d} | "
                    f"LR {lr:.6e} | "
                    f"Loss {loss_value:.5f} | "
                    f"PPL {perplexity:.2f}"
                )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            if (
                step
                % train_config.eval_every
                == 0
            ):

                val_loss, val_ppl = evaluate(
                    model=model,
                    dataloader=validation_loader,
                    device=device,
                    max_batches=(
                        train_config.validation_steps
                    ),
                )

                print(
                    f"Validation | "
                    f"Loss {val_loss:.5f} | "
                    f"PPL {val_ppl:.2f}"
                )

            # -------------------------------------------------
            # Checkpoint
            # -------------------------------------------------

            if (
                step
                % train_config.checkpoint_every
                == 0
            ):

                checkpoint_path = (
                    Path(
                        train_config.checkpoint_dir
                    )
                    / f"step_{step:06d}.pt"
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=loss.item(),
                    config=train_config,
                    path=checkpoint_path,
                )

                print(
                    f"Saved checkpoint: "
                    f"{checkpoint_path}"
                )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    final_val_loss, final_val_ppl = (
        evaluate(
            model=model,
            dataloader=validation_loader,
            device=device,
            max_batches=(
                train_config.validation_steps
            ),
        )
    )

    # ---------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------

    final_path = (
        Path(
            train_config.checkpoint_dir
        )
        / "final.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        loss=loss.item(),
        config=train_config,
        path=final_path,
    )

    print(
        f"\nSaved final checkpoint:"
        f"\n{final_path}"
    )

    print(
        f"\nFinal validation loss : "
        f"{final_val_loss:.5f}"
    )

    print(
        f"Final validation PPL  : "
        f"{final_val_ppl:.2f}"
    )

    print("\n" + "=" * 80)
    print("TELUGU TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()