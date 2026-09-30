"""
===============================================================================
NEPALI — PHASE 3 REASONING FINE-TUNING — COLAB
===============================================================================

Expected:
    /content/LMA_IP_DATA/reasoning/
        train.jsonl
        val.jsonl

Persistent outputs:
    /content/drive/MyDrive/LMA_IP/Nepali/finetune/

The Phase 2 pretrained checkpoint and existing 8000-token Nepali SentencePiece
tokenizer are supplied explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_LOCAL_REASONING_DIR = Path("/content/LMA_IP_DATA/reasoning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nepali Phase 3 reasoning fine-tuning on Colab GPU."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_LOCAL_REASONING_DIR),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_path is None and args.resume is None:
        raise ValueError(
            "Provide --model-path for the Phase 2 pretrained checkpoint, "
            "or --resume for a Phase 3 checkpoint."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This script is intended for Colab GPU."
        )

    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Reasoning data directory not found:\n{data_dir}")

    for filename in ("train.jsonl", "val.jsonl"):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing reasoning file:\n{path}")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found:\n{tokenizer_path}")

    if args.model_path is not None and not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found:\n{args.model_path}"
        )

    if args.resume is not None and not Path(args.resume).exists():
        raise FileNotFoundError(
            f"Phase 3 resume checkpoint not found:\n{args.resume}"
        )

    print("=" * 80)
    print("NEPALI — COLAB PHASE 3 REASONING FINE-TUNING")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Reasoning data: {data_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Output: {output_dir}")

    # Make both the finetune directory and the LMA_IP project root importable.
    # The latter is required for imports such as `Nepali.config...`.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    for path in (script_dir, project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from finetune import main as finetune_main

    sys.argv = [
        "finetune.py",
        "--tokenizer", str(tokenizer_path),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ]

    if args.model_path is not None:
        sys.argv += ["--model-path", args.model_path]

    if args.resume is not None:
        sys.argv += ["--resume", args.resume]

    finetune_main()


if __name__ == "__main__":
    main()
