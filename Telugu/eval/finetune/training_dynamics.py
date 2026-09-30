"""
TELUGU — PHASE 3 MIXED-DATA TRAINING DYNAMICS

Reads the resumable Phase 3 checkpoints and produces, for every run:
    - training loss vs step
    - validation loss vs step
    - training PPL vs step
    - validation PPL vs step
    - learning rate vs step
    - combined loss and PPL summaries

Important:
The current fine-tuning implementation stores the latest training loss and
validation loss in each checkpoint. It does not need to be retrained just
to make these plots. The manual warmup+cosine learning-rate schedule is
reconstructed from finetune_config.json.

Run from project root:
    python -m Telugu.eval.finetune.training_dynamics \
        --finetune-root Telugu/finetune/mixed_data_size \
        --output-root Telugu/results/finetune
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


RUN_ORDER = [
    "reasoning_2k_real_18k",
    "reasoning_5k_real_15k",
    "reasoning_8k_real_12k",
    "reasoning_12k_real_8k",
    "reasoning_16k_real_4k",
    "reasoning_20k_real_0k",
]


def parse_args() -> argparse.Namespace:
    """Parse training-dynamics paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run names. Default: all six runs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_step(path: Path) -> int | None:
    """Extract the training step from a step_XXXXXX.pt filename."""
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_checkpoint_summary(path: Path) -> dict:
    """Read only the checkpoint fields needed for plotting."""
    import torch

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint: {path}")

    step = checkpoint.get("step")
    if step is None:
        step = checkpoint_step(path)

    return {
        "step": int(step),
        "loss": checkpoint.get("loss"),
        "validation_loss": checkpoint.get("validation_loss"),
    }


def reconstruct_lr(step: int, config: dict) -> float:
    """
    Reconstruct the exact manual warmup+cosine LR used by finetune.py.
    """
    base = float(config["learning_rate"])
    max_steps = int(config["max_steps"])
    warmup = int(config["warmup_steps"])
    min_ratio = float(config["min_learning_rate_ratio"])

    if step < warmup:
        return base * step / max(1, warmup)

    if step >= max_steps:
        return base * min_ratio

    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base * min_ratio
    return min_lr + (base - min_lr) * cosine


def save_plot(
    path: Path,
    x: list[int],
    y1: list[float],
    y2: list[float] | None,
    label1: str,
    label2: str | None,
    title: str,
    ylabel: str,
) -> None:
    """Create one fully labelled plot."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y1, label=label1)

    if y2 is not None and label2 is not None:
        ax.plot(x, y2, label=label2)

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Generate training dynamics for every selected mixed-data run."""
    args = parse_args()

    if not args.finetune_root.exists():
        raise FileNotFoundError(args.finetune_root)

    selected = set(args.runs) if args.runs else set(RUN_ORDER)

    global_rows = []

    for run_name in RUN_ORDER:
        if run_name not in selected:
            continue

        run_dir = args.finetune_root / run_name
        config_path = run_dir / "finetune_config.json"
        final_path = run_dir / "final.pt"

        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not final_path.exists():
            raise FileNotFoundError(final_path)

        config = load_json(config_path)

        checkpoints = []
        for path in sorted(run_dir.glob("step_*.pt")):
            if checkpoint_step(path) is not None:
                checkpoints.append(load_checkpoint_summary(path))

        # Add final checkpoint if it is not already represented by a step
        # checkpoint. This gives the plot the actual final validation value.
        final_summary = load_checkpoint_summary(final_path)
        if not any(r["step"] == final_summary["step"] for r in checkpoints):
            checkpoints.append(final_summary)

        checkpoints.sort(key=lambda r: r["step"])

        train = [
            r for r in checkpoints
            if r["loss"] is not None
        ]
        val = [
            r for r in checkpoints
            if r["validation_loss"] is not None
        ]

        if not train:
            raise RuntimeError(f"No training-loss checkpoints found in {run_dir}")

        run_output = args.output_root / run_name / "training"
        run_output.mkdir(parents=True, exist_ok=True)

        steps_train = [r["step"] for r in train]
        train_loss = [float(r["loss"]) for r in train]
        train_ppl = [math.exp(min(v, 20.0)) for v in train_loss]
        train_lr = [reconstruct_lr(s, config) for s in steps_train]

        steps_val = [r["step"] for r in val]
        val_loss = [float(r["validation_loss"]) for r in val]
        val_ppl = [math.exp(min(v, 20.0)) for v in val_loss]

        # Exact per-run plots required for the report.
        save_plot(
            run_output / "loss_curve.png",
            steps_train,
            train_loss,
            None if not val else val_loss,
            "Training loss",
            None if not val else "Validation loss",
            f"Telugu fine-tuning loss — {run_name}",
            "Cross-entropy loss",
        )

        if val:
            # Loss and PPL plots use their own axes/scales; don't combine
            # unrelated quantities into one chart.
            save_plot(
                run_output / "perplexity_curve.png",
                steps_train,
                train_ppl,
                val_ppl,
                "Training PPL",
                "Validation PPL",
                f"Telugu fine-tuning perplexity — {run_name}",
                "Perplexity",
            )
        else:
            save_plot(
                run_output / "perplexity_curve.png",
                steps_train,
                train_ppl,
                None,
                "Training PPL",
                None,
                f"Telugu fine-tuning perplexity — {run_name}",
                "Perplexity",
            )

        save_plot(
            run_output / "learning_rate_curve.png",
            steps_train,
            train_lr,
            None,
            "Learning rate",
            None,
            f"Telugu fine-tuning learning rate — {run_name}",
            "Learning rate",
        )

        # Save a compact machine-readable history reconstructed from the
        # resumable checkpoints.
        history = {
            "run": run_name,
            "config": config,
            "checkpoints": checkpoints,
            "note": (
                "Training/validation losses are read from the saved "
                "resumable checkpoints. Learning rate is reconstructed "
                "from the exact manual warmup+cosine schedule in the config."
            ),
        }
        with (run_output / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        summary = {
            "run": run_name,
            "max_steps": config["max_steps"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "warmup_steps": config["warmup_steps"],
            "final_step": final_summary["step"],
            "final_training_loss": final_summary["loss"],
            "final_training_ppl": (
                math.exp(min(float(final_summary["loss"]), 20.0))
                if final_summary["loss"] is not None else None
            ),
            "final_validation_loss": final_summary["validation_loss"],
            "final_validation_ppl": (
                math.exp(min(float(final_summary["validation_loss"]), 20.0))
                if final_summary["validation_loss"] is not None else None
            ),
        }

        with (run_output / "training_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        global_rows.append(summary)

    with (args.output_root / "mixture_training_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({"runs": global_rows}, f, ensure_ascii=False, indent=2)

    print("\nTraining-dynamics evaluation complete.")


if __name__ == "__main__":
    main()
