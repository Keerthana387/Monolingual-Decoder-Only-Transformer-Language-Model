"""
===============================================================================
TELUGU — PHASE 3 REASONING FINE-TUNING — COLAB
===============================================================================

Colab launcher for finetune.py.

Expected setup:
    /content/LMA_IP_DATA/reasoning/
        train.jsonl
        val.jsonl

The PRETRAINED MODEL CHECKPOINT and the existing Telugu tokenizer are supplied
by command-line arguments. The tokenizer is the existing Telugu BPE model
under Telugu/data/tokenizer/. The model checkpoint is never copied/retrained into
a new architecture; its weights initialize the Phase 3 fine-tuning run.

Example:

python Telugu/finetune/finetune_colab.py \
    --model-path "/content/drive/MyDrive/.../pretrained/final.pt" \
    --tokenizer "/content/drive/MyDrive/.../telugu.model" \
    --data-dir "/content/LMA_IP_DATA/reasoning" \
    --output-dir "/content/drive/MyDrive/.../Telugu/finetune" \
    --max-steps 2000

Resume:

python Telugu/finetune/finetune_colab.py \
    --resume "/content/drive/MyDrive/.../Telugu/finetune/step_000500.pt" \
    --tokenizer "/content/drive/MyDrive/.../telugu.model" \
    --data-dir "/content/LMA_IP_DATA/reasoning" \
    --output-dir "/content/drive/MyDrive/.../Telugu/finetune"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_LOCAL_REASONING_DIR = Path(
    "/content/LMA_IP_DATA/reasoning"
)


def parse_args() -> argparse.Namespace:
    """Parse Colab fine-tuning arguments."""

    parser = argparse.ArgumentParser(
        description="Run Telugu Phase 3 reasoning fine-tuning on Colab GPU."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the Phase 2 pretrained Telugu checkpoint.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a Phase 3 fine-tuning checkpoint.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_LOCAL_REASONING_DIR),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    """Validate the Colab environment and invoke the main fine-tuner."""

    args = parse_args()

    if args.model_path is None and args.resume is None:
        raise ValueError(
            "Provide --model-path for the pretrained checkpoint, "
            "or --resume for a Phase 3 checkpoint."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. "
            "This script is intended for Colab GPU fine-tuning."
        )

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Reasoning data directory not found:\n{data_dir}\n\n"
            "Copy Telugu/reasoning/data to this location first."
        )

    for filename in ("train.jsonl", "val.jsonl"):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing reasoning file:\n{path}"
            )

    print("=" * 80)
    print("TELUGU — COLAB PHASE 3 REASONING FINE-TUNING")
    print("=" * 80)

    print(
        f"\nGPU: {torch.cuda.get_device_name(0)}"
    )

    memory_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory: {memory_gb:.2f} GB"
    )

    print(
        f"Reasoning data: {data_dir}"
    )

    # Rebuild argv for the shared implementation. This keeps the actual
    # fine-tuning logic in one language-specific file while retaining a
    # dedicated Colab entry point.
    from finetune import main as finetune_main

    sys.argv = [
        "finetune.py",
        "--tokenizer", args.tokenizer,
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ]

    if args.model_path is not None:
        sys.argv += [
            "--model-path",
            args.model_path,
        ]

    if args.resume is not None:
        sys.argv += [
            "--resume",
            args.resume,
        ]

    finetune_main()


if __name__ == "__main__":
    main()
