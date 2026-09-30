import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from Telugu.config.model_config import ModelConfig
from Telugu.config.train_config import TrainConfig
from Telugu.data.dataset import TokenShardDataset
from Telugu.data.validation_dataset import ValidationTokenDataset
from Telugu.model.lm import TeluguLanguageModel


# ============================================================
# COLAB PATHS
# ============================================================
#
# The large tokenized inputs are copied to Colab local storage
# by the notebook.
#
# Checkpoints remain on Google Drive.
# ============================================================

LOCAL_TRAIN_DIR = Path(
    "/content/LMA_IP_DATA/train"
)

LOCAL_VALIDATION_DIR = Path(
    "/content/LMA_IP_DATA/validation"
)

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/LMA_IP"
)

CHECKPOINT_DIR = (
    DRIVE_ROOT / "Telugu" / "checkpoints"
)

# ============================================================
# COLAB TRAINING OVERRIDES
# ============================================================

BATCH_SIZE = 16

# One pass over the 486M-token corpus
MAX_STEPS =  59_214 

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

WARMUP_STEPS = 2_000
MIN_LEARNING_RATE_RATIO = 0.1

LOG_EVERY = 50

# Keep validation/checkpoints reasonably frequent
EVAL_EVERY = 2_000
VALIDATION_STEPS = 100
CHECKPOINT_EVERY = 2_000

USE_MIXED_PRECISION = True

# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train Telugu Model H using "
            "Colab GPU + FP16 AMP."
        )
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

    if step < config.warmup_steps:

        return (
            config.learning_rate
            * step
            / max(
                1,
                config.warmup_steps,
            )
        )

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
        max(
            progress,
            0.0,
        ),
        1.0,
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
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
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scaler,
    step,
    loss,
    config,
    path,
):

    checkpoint = {

        # -----------------------------------------------------
        # Training state
        # -----------------------------------------------------

        "step": step,

        "loss": loss,

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------

        "optimizer_state_dict":
            optimizer.state_dict(),

        # -----------------------------------------------------
        # Training configuration
        # -----------------------------------------------------

        "train_config":
            config.__dict__,

        # -----------------------------------------------------
        # AMP
        # -----------------------------------------------------

        "scaler_state_dict":
            scaler.state_dict(),

        # -----------------------------------------------------
        # RNG states
        # -----------------------------------------------------

        "python_rng_state":
            random.getstate(),

        "numpy_rng_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        checkpoint[
            "cuda_rng_state"
        ] = torch.cuda.get_rng_state_all()

    # ---------------------------------------------------------
    # Ensure directory exists
    # ---------------------------------------------------------

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

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
    scaler,
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

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # ---------------------------------------------------------
    # AMP scaler
    # ---------------------------------------------------------

    if (
        "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

        print(
            "✓ AMP scaler state restored"
        )

    else:

        print(
            "⚠ No AMP scaler state "
            "found in checkpoint"
        )

    # ---------------------------------------------------------
    # Python RNG
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

    # ---------------------------------------------------------
    # NumPy RNG
    # ---------------------------------------------------------

    if (
        "numpy_rng_state"
        in checkpoint
    ):

        np.random.set_state(
            checkpoint[
                "numpy_rng_state"
            ]
        )

    # ---------------------------------------------------------
    # PyTorch RNG
    # ---------------------------------------------------------

    if (
        "torch_rng_state"
        in checkpoint
    ):

        torch.set_rng_state(
            checkpoint[
                "torch_rng_state"
            ]
        )

    # ---------------------------------------------------------
    # CUDA RNG
    # ---------------------------------------------------------

    if (
        torch.cuda.is_available()
        and
        "cuda_rng_state"
        in checkpoint
    ):

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_rng_state"
            ]
        )

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------

    step = checkpoint[
        "step"
    ]

    loss = checkpoint[
        "loss"
    ]

    print(
        f"✓ Resumed from step "
        f"{step:,}"
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

        if (
            batches
            >= max_batches
        ):

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

        # -----------------------------------------------------
        # Validation also uses FP16 AMP
        # -----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            _, loss, _ = model(
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
        total_loss
        / batches
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

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    train_config = TrainConfig()

    model_config = ModelConfig()

    # ---------------------------------------------------------
    # Apply Colab training configuration
    # ---------------------------------------------------------
    
    train_config.batch_size = BATCH_SIZE
    train_config.max_steps = MAX_STEPS
    
    train_config.learning_rate = LEARNING_RATE
    train_config.weight_decay = WEIGHT_DECAY
    
    train_config.warmup_steps = WARMUP_STEPS
    train_config.min_learning_rate_ratio = (
        MIN_LEARNING_RATE_RATIO
    )
    
    train_config.log_every = LOG_EVERY
    train_config.eval_every = EVAL_EVERY
    train_config.validation_steps = VALIDATION_STEPS
    train_config.checkpoint_every = CHECKPOINT_EVERY
    
    train_config.use_mixed_precision = (
        USE_MIXED_PRECISION
    )
    
    train_config.checkpoint_dir = str(
        CHECKPOINT_DIR
    )
    
    # ---------------------------------------------------------
    # Optional command-line override
    # ---------------------------------------------------------
    
    if args.max_steps is not None:
    
        train_config.max_steps = args.max_steps

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    set_seed(
        train_config.seed
    )

    # ---------------------------------------------------------
    # GPU
    # ---------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is not available. "
            "This script is intended "
            "for Colab GPU training."
        )

    device = torch.device(
        "cuda"
    )

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    print("=" * 80)
    print(
        "TELUGU MODEL H — "
        "COLAB GPU TRAINING"
    )
    print("=" * 80)

    print(
        f"\nGPU:"
        f"\n{torch.cuda.get_device_name(0)}"
    )

    total_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory: "
        f"{total_memory:.2f} GB"
    )

    # ---------------------------------------------------------
    # Model configuration
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
        f"FFN dimension   : "
        f"{model_config.d_ff}"
    )

    print(
        f"Context length  : "
        f"{model_config.max_seq_len}"
    )

    # ---------------------------------------------------------
    # Configuration checks
    # ---------------------------------------------------------

    assert (
        model_config.vocab_size
        == 6000
    )

    assert (
        model_config.max_seq_len
        == 512
    )

    assert (
        train_config.sequence_length
        == model_config.max_seq_len
    )

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------

    # Use the benchmarked batch size.
    train_config.batch_size = 16

    print(
        "\nTraining configuration:"
    )

    print(
        f"Batch size      : "
        f"{train_config.batch_size}"
    )

    print(
        f"Sequence length : "
        f"{train_config.sequence_length}"
    )

    print(
        f"Maximum steps   : "
        f"{train_config.max_steps:,}"
    )

    print(
        f"Learning rate   : "
        f"{train_config.learning_rate:.6e}"
    )

    print(
        f"Weight decay    : "
        f"{train_config.weight_decay}"
    )

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    print(
        "\nMixed precision : FP16"
    )

    print(
        "GradScaler      : enabled"
    )

    # =========================================================
    # LOCAL TRAINING DATA
    # =========================================================

    print(
        "\nChecking local training data..."
    )

    if not LOCAL_TRAIN_DIR.exists():

        raise FileNotFoundError(
            "Local training cache does "
            "not exist:\n"
            f"{LOCAL_TRAIN_DIR}\n\n"
            "Run the Colab notebook's "
            "data-cache cell first."
        )

    train_shards = sorted(
        LOCAL_TRAIN_DIR.glob(
            "train_*.bin"
        )
    )

    if not train_shards:

        raise FileNotFoundError(
            "No training shards found in:\n"
            f"{LOCAL_TRAIN_DIR}"
        )

    print(
        f"Training data:"
        f"\n{LOCAL_TRAIN_DIR}"
    )

    print(
        f"Training shards: "
        f"{len(train_shards)}"
    )

    # ---------------------------------------------------------
    # Verify expected corpus
    # ---------------------------------------------------------

    if len(train_shards) != 49:

        raise RuntimeError(
            f"Expected 49 training shards, "
            f"found {len(train_shards)}."
        )

    # =========================================================
    # DATASET
    # =========================================================

    train_dataset = TokenShardDataset(
        shard_paths=train_shards,
        sequence_length=(
            train_config.sequence_length
        ),
    )

    train_stats = (
        train_dataset.get_statistics()
    )

    print(
        "\nTraining dataset:"
    )

    print(
        f"Tokens          : "
        f"{train_stats['total_tokens']:,}"
    )

    print(
        f"Samples         : "
        f"{train_stats['num_samples']:,}"
    )

    # =========================================================
    # DATALOADER
    # =========================================================

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
        pin_memory=True,
        persistent_workers=(
            train_config.num_workers > 0
        ),
    )

    # =========================================================
    # LOCAL VALIDATION DATA
    # =========================================================

    print(
        "\nChecking local validation data..."
    )

    if not LOCAL_VALIDATION_DIR.exists():

        raise FileNotFoundError(
            "Local validation cache does "
            "not exist:\n"
            f"{LOCAL_VALIDATION_DIR}\n\n"
            "Run the Colab notebook's "
            "data-cache cell first."
        )

    val_shards = sorted(
        LOCAL_VALIDATION_DIR.glob(
            "val_*.bin"
        )
    )

    if not val_shards:

        raise FileNotFoundError(
            "No validation shards found in:\n"
            f"{LOCAL_VALIDATION_DIR}"
        )

    print(
        f"Validation data:"
        f"\n{LOCAL_VALIDATION_DIR}"
    )

    print(
        f"Validation shards: "
        f"{len(val_shards)}"
    )

    # =========================================================
    # VALIDATION DATASET
    # =========================================================

    validation_dataset = (
        ValidationTokenDataset(
            validation_dir=(
                LOCAL_VALIDATION_DIR
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
        pin_memory=True,
        persistent_workers=(
            train_config.num_workers > 0
        ),
    )

    print(
        "\nValidation dataset:"
    )

    print(
        f"Tokens          : "
        f"{validation_dataset.metadata['tokens']:,}"
    )

    print(
        f"Samples         : "
        f"{len(validation_dataset):,}"
    )

    # =========================================================
    # MODEL
    # =========================================================

    print(
        "\nCreating model..."
    )

    model = TeluguLanguageModel(
        model_config
    ).to(device)

    print(
        f"Trainable parameters: "
        f"{model.count_parameters():,}"
    )

    # =========================================================
    # OPTIMIZER
    # =========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            train_config.learning_rate
        ),
        betas=(
            train_config.beta1,
            train_config.beta2,
        ),
        weight_decay=(
            train_config.weight_decay
        ),
    )

    # =========================================================
    # RESUME
    # =========================================================

    start_step = 0

    if args.resume is not None:

        start_step = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    # =========================================================
    # TRAINING
    # =========================================================

    model.train()

    step = start_step

    tokens_processed = 0

    training_start = (
        time.perf_counter()
    )

    print(
        "\nTraining:"
    )

    print(
        "-" * 80
    )

    while (
        step
        < train_config.max_steps
    ):

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
            # Learning rate
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
            # Clear gradients
            # -------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # FP16 AMP forward
            # -------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                _, loss, _ = model(
                    input_ids=input_ids,
                    targets=targets,
                )

            # -------------------------------------------------
            # Backward
            # -------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            # -------------------------------------------------
            # Unscale before clipping
            # -------------------------------------------------

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.grad_clip,
            )

            # -------------------------------------------------
            # Optimizer step
            # -------------------------------------------------

            scaler.step(
                optimizer
            )

            scaler.update()

            # -------------------------------------------------
            # Step
            # -------------------------------------------------

            step += 1

            tokens_processed += (
                input_ids.numel()
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            if (
                step == 1
                or
                step
                % train_config.log_every
                == 0
            ):

                loss_value = (
                    loss.item()
                )

                perplexity = math.exp(
                    min(
                        loss_value,
                        20,
                    )
                )

                elapsed = (
                    time.perf_counter()
                    - training_start
                )

                tokens_per_second = (
                    tokens_processed
                    / max(
                        elapsed,
                        1e-6,
                    )
                )

                print(
                    f"Step {step:7d} | "
                    f"LR {lr:.6e} | "
                    f"Loss {loss_value:.5f} | "
                    f"PPL {perplexity:.2f} | "
                    f"{tokens_per_second:,.0f} tok/s"
                )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            if (
                step
                % train_config.eval_every
                == 0
            ):

                val_loss, val_ppl = (
                    evaluate(
                        model=model,
                        dataloader=(
                            validation_loader
                        ),
                        device=device,
                        max_batches=(
                            train_config.validation_steps
                        ),
                    )
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
                    scaler=scaler,
                    step=step,
                    loss=loss.item(),
                    config=train_config,
                    path=checkpoint_path,
                )

                print(
                    f"Saved checkpoint:"
                    f"\n{checkpoint_path}"
                )

    # =========================================================
    # FINAL VALIDATION
    # =========================================================

    print(
        "\nRunning final validation..."
    )

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

    # =========================================================
    # FINAL CHECKPOINT
    # =========================================================

    final_path = (
        Path(
            train_config.checkpoint_dir
        )
        / "final.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        loss=loss.item(),
        config=train_config,
        path=final_path,
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    total_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"\nFinal validation loss : "
        f"{final_val_loss:.5f}"
    )

    print(
        f"Final validation PPL  : "
        f"{final_val_ppl:.2f}"
    )

    print(
        f"Training time         : "
        f"{total_time / 3600:.2f} hours"
    )

    print(
        f"Final checkpoint:"
        f"\n{final_path}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "TELUGU COLAB TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()