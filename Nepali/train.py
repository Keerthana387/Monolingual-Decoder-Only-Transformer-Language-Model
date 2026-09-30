# ============================================================
# LMA-Indiv — Phase 2
# NEPALI — PRETRAINING
# ============================================================

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from Nepali.config.model_config import ModelConfig
from Nepali.config.train_config import TrainConfig
from Nepali.data.dataset import NepaliTrainDataset
from Nepali.data.validation_dataset import NepaliEvaluationDataset
from Nepali.model.lm import LanguageModel


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    """Parse training command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the Nepali decoder-only language model."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps.",
    )

    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Override training shard directory.",
    )

    parser.add_argument(
        "--validation-dir",
        type=str,
        default=None,
        help="Override validation shard directory.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory.",
    )

    return parser.parse_args()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """Set Python, NumPy and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(step, config):
    """
    Return the warmup + cosine-decay learning rate
    for a given training step.
    """

    if step < config.warmup_steps:
        return (
            config.learning_rate
            * step
            / max(1, config.warmup_steps)
        )

    decay_steps = (
        config.max_steps
        - config.warmup_steps
    )

    if decay_steps <= 0:
        return config.learning_rate

    progress = (
        step - config.warmup_steps
    ) / decay_steps

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = 0.5 * (
        1.0
        + math.cos(math.pi * progress)
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
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    config,
    path,
    scheduler_state=None,
):
    """
    Save all state needed to resume training.

    Stores:
        - model weights
        - optimizer state
        - scheduler state
        - current step
        - loss
        - training configuration
        - model configuration
        - Python RNG state
        - NumPy RNG state
        - PyTorch RNG state
        - CUDA RNG state when available
    """

    checkpoint = {
        "step": int(step),
        "loss": float(loss),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict": (
            scheduler_state
            if scheduler_state is not None
            else {
                "step": int(step),
                "current_lr": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        ),

        "train_config":
            config.__dict__.copy(),

        "model_config":
            model.config.__dict__.copy(),

        "python_rng_state":
            random.getstate(),

        "numpy_rng_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    device,
):
    """
    Restore model, optimizer, RNG and training state
    from a checkpoint.
    """

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
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if "python_rng_state" in checkpoint:
        random.setstate(
            checkpoint["python_rng_state"]
        )

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(
            checkpoint["numpy_rng_state"]
        )

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )

    return (
        int(checkpoint["step"]),
        float(checkpoint["loss"]),
        checkpoint.get(
            "scheduler_state_dict",
            {},
        ),
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    max_batches,
):
    """Evaluate mean cross-entropy and perplexity."""

    model.eval()

    total_loss = 0.0
    batches = 0

    for input_ids, targets in dataloader:

        if batches >= max_batches:
            break

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        _, loss = model(
            input_ids=input_ids,
            targets=targets,
        )

        if (
            loss is None
            or not torch.isfinite(loss)
        ):
            raise RuntimeError(
                "Validation produced a "
                "non-finite loss."
            )

        total_loss += loss.item()

        batches += 1

    if batches == 0:
        raise RuntimeError(
            "Validation DataLoader returned "
            "no batches."
        )

    average_loss = (
        total_loss / batches
    )

    perplexity = math.exp(
        min(
            average_loss,
            20.0,
        )
    )

    model.train()

    return (
        average_loss,
        perplexity,
    )


# ============================================================
# MEMORY-SAFE DATALOADER
# ============================================================

def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    device,
):
    """
    Build a DataLoader.

    For training, use RandomSampler with replacement
    so PyTorch does not attempt to construct a complete
    permutation of the hundreds of millions of samples.
    """

    if shuffle:

        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                num_workers > 0
            ),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            num_workers > 0
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run Nepali pretraining from scratch
    or resume from a checkpoint.
    """

    args = parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    model_config = ModelConfig()
    train_config = TrainConfig()

    if args.max_steps is not None:
        train_config.max_steps = (
            args.max_steps
        )

    if args.train_dir is not None:
        train_config.tokenized_dir = (
            args.train_dir
        )

    if args.validation_dir is not None:
        train_config.validation_dir = (
            args.validation_dir
        )

    if args.checkpoint_dir is not None:
        train_config.checkpoint_dir = (
            args.checkpoint_dir
        )

    # --------------------------------------------------------
    # Configuration consistency
    # --------------------------------------------------------

    if (
        train_config.sequence_length
        != model_config.max_seq_len
    ):
        raise ValueError(
            "TrainConfig.sequence_length must equal "
            "ModelConfig.max_seq_len."
        )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        train_config.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 80)
    print("NEPALI — PRETRAINING")
    print("=" * 80)

    print(
        f"\nDevice          : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            "GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    print(
        f"\nVocabulary size : "
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
        f"FFN dimension   : "
        f"{model_config.d_ff}"
    )

    print(
        f"Context length  : "
        f"{model_config.max_seq_len}"
    )

    print(
        f"Peak learning rate : "
        f"{train_config.learning_rate:.2e}"
    )

    # --------------------------------------------------------
    # Data paths
    # --------------------------------------------------------

    print(
        f"\nTraining data   : "
        f"{train_config.tokenized_dir}"
    )

    print(
        f"Validation data : "
        f"{train_config.validation_dir}"
    )

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_dataset = NepaliTrainDataset(
        data_dir=(
            train_config.tokenized_dir
        ),
        block_size=(
            model_config.max_seq_len
        ),
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    validation_dataset = (
        NepaliEvaluationDataset(
            data_dir=(
                train_config.validation_dir
            ),
            block_size=(
                model_config.max_seq_len
            ),
            split="validation",
        )
    )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    print(
        f"\nTraining tokens : "
        f"{train_dataset.total_tokens:,}"
    )

    print(
        f"Training samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation tokens : "
        f"{validation_dataset.total_tokens:,}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------

    train_loader = build_dataloader(
        train_dataset,
        train_config.batch_size,
        True,
        train_config.num_workers,
        device,
    )

    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    validation_loader = build_dataloader(
        validation_dataset,
        train_config.batch_size,
        False,
        train_config.num_workers,
        device,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LanguageModel(
        model_config
    ).to(device)

    parameters = (
        model.count_parameters()
    )

    print(
        f"\nTrainable parameters: "
        f"{parameters:,}"
    )

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
        weight_decay=(
            train_config.weight_decay
        ),
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    use_amp = (
        device.type == "cuda"
        and train_config.use_mixed_precision
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------------
    # Initial training state
    # --------------------------------------------------------

    step = 0

    last_loss = float("inf")

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if args.resume is not None:

        (
            step,
            last_loss,
            scheduler_state,
        ) = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
        )

        print(
            f"\nResumed from step "
            f"{step:,}"
        )

        saved_lr = (
            scheduler_state.get(
                "current_lr"
            )
        )

        if saved_lr is not None:

            for group in (
                optimizer.param_groups
            ):
                group["lr"] = (
                    saved_lr
                )

    # --------------------------------------------------------
    # Training iterator
    # --------------------------------------------------------

    data_iterator = iter(
        train_loader
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining:")
    print("-" * 80)

    while (
        step
        < train_config.max_steps
    ):

        try:

            input_ids, targets = next(
                data_iterator
            )

        except StopIteration:

            data_iterator = iter(
                train_loader
            )

            input_ids, targets = next(
                data_iterator
            )

        # ----------------------------------------------------
        # Device transfer
        # ----------------------------------------------------

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Step
        # ----------------------------------------------------

        next_step = (
            step + 1
        )

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_learning_rate(
            next_step,
            train_config,
        )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = lr

        # ----------------------------------------------------
        # Gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            _, loss = model(
                input_ids=input_ids,
                targets=targets,
            )

        # ----------------------------------------------------
        # Loss validation
        # ----------------------------------------------------

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss at "
                f"step {next_step}: "
                f"{loss.item()}"
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.grad_clip,
            )
        )

        # ----------------------------------------------------
        # AMP overflow handling
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # A non-finite gradient can occur during FP16
        # training even when the forward loss is finite.
        #
        # Do NOT update the model in that case.
        #
        # GradScaler will detect the overflow during
        # scaler.step() and reduce its scale.
        #
        # We therefore:
        #
        #   1. discard this update
        #   2. call scaler.update()
        #   3. keep the same logical training step
        #   4. continue with the next batch
        #
        # This prevents one FP16 overflow from terminating
        # the entire multi-hour pretraining run.
        # ----------------------------------------------------

        if not torch.isfinite(
            grad_norm
        ):

            print(
                f"WARNING: non-finite gradient "
                f"at training step {next_step}. "
                f"Skipping optimizer update and "
                f"reducing AMP scale."
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # scaler.step() performs no optimizer
            # update when its internal inf/NaN check
            # detects an overflow.
            scaler.step(
                optimizer
            )

            scaler.update()

            continue

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        step = next_step

        last_loss = loss.item()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            step == 1
            or step
            % train_config.log_every
            == 0
        ):

            ppl = math.exp(
                min(
                    last_loss,
                    20.0,
                )
            )

            print(
                f"Step {step:6d} | "
                f"LR {lr:.6e} | "
                f"Loss {last_loss:.5f} | "
                f"PPL {ppl:.2f}"
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            step
            % train_config.eval_every
            == 0
        ):

            (
                val_loss,
                val_ppl,
            ) = evaluate(
                model=model,
                dataloader=(
                    validation_loader
                ),
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

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

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
                loss=last_loss,
                config=train_config,
                path=checkpoint_path,
                scheduler_state={
                    "step": step,
                    "current_lr": lr,
                    "schedule":
                        "warmup_cosine",
                },
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

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
        loss=last_loss,
        config=train_config,
        path=final_path,
        scheduler_state={
            "step": step,
            "current_lr":
                get_learning_rate(
                    step,
                    train_config,
                ),
            "schedule":
                "warmup_cosine",
        },
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    (
        val_loss,
        val_ppl,
    ) = evaluate(
        model=model,
        dataloader=validation_loader,
        device=device,
        max_batches=(
            train_config.validation_steps
        ),
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        f"\nFinal validation loss : "
        f"{val_loss:.5f}"
    )

    print(
        f"Final validation PPL  : "
        f"{val_ppl:.2f}"
    )

    print(
        f"Final checkpoint      : "
        f"{final_path}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "NEPALI PRETRAINING COMPLETE"
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()