"""
NEPALI — PHASE 3 MIXED-DATA ATTENTION COMPARISON

Runs the detailed Nepali attention_compare.py once for each of the six
mixed-data fine-tuning checkpoints.

The same held-out reasoning test set is used for every run.

Run from project root:

    python -m Nepali.eval.finetune.attention_compare_mixed \
        --pretrained Nepali/checkpoints/final.pt \
        --finetune-root Nepali/finetune/mixed_data_size \
        --tokenizer Nepali/data/tokenizer/nepali_bpe_8000.json \
        --test Nepali/reasoning/data/test.jsonl \
        --output-root Nepali/results/finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUN_ORDER = [
    ("reasoning_2k_real_18k", 2_000, 18_000),
    ("reasoning_5k_real_15k", 5_000, 15_000),
    ("reasoning_8k_real_12k", 8_000, 12_000),
    ("reasoning_12k_real_8k", 12_000, 8_000),
    ("reasoning_16k_real_4k", 16_000, 4_000),
    ("reasoning_20k_real_0k", 20_000, 0),
]

RUN_NAMES = {x[0] for x in RUN_ORDER}
TOTAL_EXAMPLES = 20_000


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--finetune-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--representative-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", nargs="*", default=None)

    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: object) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run attention comparison for all six mixed-data runs."""

    args = parse_args()

    for label, path in [
        ("pretrained checkpoint", args.pretrained),
        ("tokenizer", args.tokenizer),
        ("reasoning test", args.test),
        ("fine-tuning root", args.finetune_root),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found:\n{path}"
            )

    selected = set(args.runs) if args.runs else None

    if selected:
        unknown = selected - RUN_NAMES
        if unknown:
            raise ValueError(
                f"Unknown run(s): {sorted(unknown)}"
            )

    from .attention_compare import main as single_run_main

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    original_argv = sys.argv[:]

    try:
        for run_name, reasoning_n, real_n in RUN_ORDER:

            if selected is not None and run_name not in selected:
                continue

            checkpoint = (
                args.finetune_root
                / run_name
                / "final.pt"
            )

            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint:\n{checkpoint}"
                )

            run_output = (
                args.output_root
                / run_name
                / "attention"
            )

            run_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            print("\n" + "=" * 90)
            print("NEPALI PHASE 3 — ATTENTION COMPARISON")
            print("=" * 90)
            print(f"Run:                {run_name}")
            print(f"Reasoning examples: {reasoning_n:,}")
            print(f"Real examples:      {real_n:,}")
            print(f"Checkpoint:         {checkpoint}")
            print(f"Output:             {run_output}")
            print("=" * 90)

            sys.argv = [
                "attention_compare.py",
                "--pretrained",
                str(args.pretrained),
                "--finetuned",
                str(checkpoint),
                "--tokenizer",
                str(args.tokenizer),
                "--test",
                str(args.test),
                "--output-dir",
                str(run_output),
                "--num-prompts",
                str(args.num_prompts),
                "--sequence-length",
                str(args.sequence_length),
                "--representative-index",
                str(args.representative_index),
            ]

            single_run_main()

            comparison_path = (
                run_output
                / "attention_comparison.json"
            )

            if not comparison_path.exists():
                raise RuntimeError(
                    "attention_comparison.json was not produced:\n"
                    f"{comparison_path}"
                )

            comparison = load_json(
                comparison_path
            )

            row = {
                "run": run_name,
                "reasoning_examples": reasoning_n,
                "real_examples": real_n,
                "reasoning_fraction_of_examples":
                    reasoning_n / TOTAL_EXAMPLES,
                "real_fraction_of_examples":
                    real_n / TOTAL_EXAMPLES,
                "num_prompts":
                    comparison.get(
                        "summary",
                        {},
                    ).get(
                        "pretrained",
                        {},
                    ).get(
                        "num_prompts",
                        args.num_prompts,
                    ),
                "pretrained_entropy_mean":
                    comparison.get(
                        "summary",
                        {},
                    ).get(
                        "pretrained",
                        {},
                    ).get(
                        "entropy_mean",
                    ),
                "finetuned_entropy_mean":
                    comparison.get(
                        "summary",
                        {},
                    ).get(
                        "finetuned",
                        {},
                    ).get(
                        "entropy_mean",
                    ),
                "pretrained_distance_mean":
                    comparison.get(
                        "summary",
                        {},
                    ).get(
                        "pretrained",
                        {},
                    ).get(
                        "distance_mean",
                    ),
                "finetuned_distance_mean":
                    comparison.get(
                        "summary",
                        {},
                    ).get(
                        "finetuned",
                        {},
                    ).get(
                        "distance_mean",
                    ),
                "attention_comparison_file":
                    str(comparison_path),
            }

            if (
                row["pretrained_entropy_mean"] is not None
                and row["finetuned_entropy_mean"] is not None
            ):
                row["entropy_change"] = (
                    row["finetuned_entropy_mean"]
                    - row["pretrained_entropy_mean"]
                )

            if (
                row["pretrained_distance_mean"] is not None
                and row["finetuned_distance_mean"] is not None
            ):
                row["distance_change"] = (
                    row["finetuned_distance_mean"]
                    - row["pretrained_distance_mean"]
                )

            summary_rows.append(row)

    finally:
        sys.argv = original_argv

    if not summary_rows:
        raise ValueError("No runs were selected.")

    write_csv(
        args.output_root / "mixture_attention_summary.csv",
        summary_rows,
    )

    write_json(
        args.output_root / "mixture_attention_summary.json",
        {
            "experiment": {
                "language": "Nepali",
                "total_training_examples_per_run": TOTAL_EXAMPLES,
                "same_pretrained_checkpoint": str(args.pretrained),
                "same_held_out_reasoning_test": str(args.test),
                "tokenizer": str(args.tokenizer),
                "num_prompts": args.num_prompts,
                "sequence_length": args.sequence_length,
                "seed": args.seed,
                "note": (
                    "Attention analysis uses held-out reasoning prompts "
                    "only. The mixture fraction refers to training examples."
                ),
            },
            "runs": summary_rows,
        },
    )

    print("\n" + "=" * 90)
    print("NEPALI MIXED-DATA ATTENTION EVALUATION COMPLETE")
    print("=" * 90)

    for row in summary_rows:
        print(
            f"{row['run']}: "
            f"entropy "
            f"{row['pretrained_entropy_mean']} -> "
            f"{row['finetuned_entropy_mean']}; "
            f"distance "
            f"{row['pretrained_distance_mean']} -> "
            f"{row['finetuned_distance_mean']}"
        )

    print(
        f"\nSummary: "
        f"{args.output_root / 'mixture_attention_summary.csv'}"
    )


if __name__ == "__main__":
    main()
