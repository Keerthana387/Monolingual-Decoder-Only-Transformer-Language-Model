"""
NEPALI — GOOGLE COLAB TRAINING LAUNCHER

This file does not implement a second training loop.

It launches the existing Nepali.train training pipeline while
redirecting large repeatedly-read input data to Colab's local
runtime storage.

Persistent:
    Google Drive
    LMA_IP/Nepali/checkpoints/

Temporary:
    /content/LMA_IP_DATA/train/
    /content/LMA_IP_DATA/validation/

Run from the project root:

    python -m Nepali.train_colab

Short test:

    python -m Nepali.train_colab --max-steps 100

Resume:

    python -m Nepali.train_colab \
        --resume /content/drive/MyDrive/LMA_IP/Nepali/checkpoints/step_XXXXXX.pt
"""

import argparse
import sys
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/LMA_IP"
)

NEPALI_ROOT = (
    PROJECT_ROOT / "Nepali"
)

# ------------------------------------------------------------
# Temporary Colab input cache
# ------------------------------------------------------------

LOCAL_DATA_ROOT = Path(
    "/content/LMA_IP_DATA"
)

LOCAL_TRAIN_DIR = (
    LOCAL_DATA_ROOT / "train"
)

LOCAL_VALIDATION_DIR = (
    LOCAL_DATA_ROOT / "validation"
)

# ------------------------------------------------------------
# Persistent checkpoint directory
# ------------------------------------------------------------

DRIVE_CHECKPOINT_DIR = (
    NEPALI_ROOT / "checkpoints"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    """Parse Colab-specific training arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch Nepali training using "
            "Colab-local input data."
        )
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Override maximum training steps."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint path to resume from."
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

def validate_environment():
    """Verify Drive project and local cached inputs."""

    print("=" * 80)
    print("NEPALI — COLAB TRAINING LAUNCHER")
    print("=" * 80)

    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root not found:\n"
            f"{PROJECT_ROOT}\n\n"
            "Make sure Google Drive is mounted and "
            "the project is located at MyDrive/LMA_IP."
        )

    if not NEPALI_ROOT.exists():
        raise FileNotFoundError(
            f"Nepali project directory not found:\n"
            f"{NEPALI_ROOT}"
        )

    # --------------------------------------------------------
    # Local training data
    # --------------------------------------------------------

    if not LOCAL_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Local training cache not found:\n"
            f"{LOCAL_TRAIN_DIR}\n\n"
            "Run the notebook cell that copies the "
            "training shards to /content/LMA_IP_DATA."
        )

    train_shards = sorted(
        LOCAL_TRAIN_DIR.glob(
            "train_*.bin"
        )
    )

    if len(train_shards) == 0:
        raise FileNotFoundError(
            f"No training shards found in:\n"
            f"{LOCAL_TRAIN_DIR}"
        )

    # --------------------------------------------------------
    # Local validation data
    # --------------------------------------------------------

    if not LOCAL_VALIDATION_DIR.exists():
        raise FileNotFoundError(
            f"Local validation cache not found:\n"
            f"{LOCAL_VALIDATION_DIR}"
        )

    validation_shards = sorted(
        LOCAL_VALIDATION_DIR.glob(
            "*.bin"
        )
    )

    if len(validation_shards) == 0:
        raise FileNotFoundError(
            f"No validation shards found in:\n"
            f"{LOCAL_VALIDATION_DIR}"
        )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    DRIVE_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    args = parse_args()

    if args.resume is not None:

        resume_path = Path(
            args.resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found:\n"
                f"{resume_path}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nProject root        : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Local training data: "
        f"{LOCAL_TRAIN_DIR}"
    )

    print(
        f"Local validation   : "
        f"{LOCAL_VALIDATION_DIR}"
    )

    print(
        f"Drive checkpoints  : "
        f"{DRIVE_CHECKPOINT_DIR}"
    )

    print(
        f"Training shards    : "
        f"{len(train_shards)}"
    )

    print(
        f"Validation shards  : "
        f"{len(validation_shards)}"
    )

    if args.resume is not None:
        print(
            f"Resume checkpoint  : "
            f"{args.resume}"
        )
    else:
        print(
            "Resume checkpoint  : none"
        )

    print(
        "\n✓ Colab environment verified"
    )

    return args


# ============================================================
# LAUNCH TRAINING
# ============================================================

def main():
    """
    Launch the existing Nepali training pipeline.

    train.py remains the single source of truth for:
        - model creation
        - optimizer
        - AMP
        - learning-rate schedule
        - training loop
        - validation
        - checkpoint save/load
        - RNG restoration
    """

    args = validate_environment()

    # --------------------------------------------------------
    # Make sure imports resolve from project root.
    # --------------------------------------------------------

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(PROJECT_ROOT),
        )

    # --------------------------------------------------------
    # Import the actual training implementation.
    # --------------------------------------------------------

    from Nepali import train

    # --------------------------------------------------------
    # Translate Colab paths into train.py arguments.
    #
    # train.py already supports:
    #
    #   --train-dir
    #   --validation-dir
    #   --checkpoint-dir
    #   --resume
    #   --max-steps
    #
    # --------------------------------------------------------

    argv = [
        "Nepali.train",
        "--train-dir",
        str(LOCAL_TRAIN_DIR),
        "--validation-dir",
        str(LOCAL_VALIDATION_DIR),
        "--checkpoint-dir",
        str(DRIVE_CHECKPOINT_DIR),
    ]

    if args.max_steps is not None:

        argv.extend(
            [
                "--max-steps",
                str(args.max_steps),
            ]
        )

    if args.resume is not None:

        argv.extend(
            [
                "--resume",
                str(args.resume),
            ]
        )

    # --------------------------------------------------------
    # Replace argv temporarily.
    # --------------------------------------------------------

    original_argv = sys.argv

    try:

        sys.argv = argv

        print("\n" + "=" * 80)
        print("STARTING NEPALI TRAINING")
        print("=" * 80)

        print(
            "\nInput data:"
        )

        print(
            f"  Train      : "
            f"{LOCAL_TRAIN_DIR}"
        )

        print(
            f"  Validation : "
            f"{LOCAL_VALIDATION_DIR}"
        )

        print(
            "\nPersistent output:"
        )

        print(
            f"  Checkpoints: "
            f"{DRIVE_CHECKPOINT_DIR}"
        )

        print(
            "\nTraining implementation:"
        )

        print(
            "  Nepali/train.py"
        )

        print(
            "\n" + "-" * 80
        )

        train.main()

    finally:

        sys.argv = original_argv


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()