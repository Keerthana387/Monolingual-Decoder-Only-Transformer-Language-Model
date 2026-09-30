"""
NEPALI — PHASE 3 MIXED-DATA REASONING EVALUATION

Runs the detailed Nepali reasoning_eval.py once for each of the six
mixed-data fine-tuning checkpoints.

Run from project root:

    python -m Nepali.eval.finetune.reasoning_eval_mixed \
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
    """Parse mixed-data evaluation arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--finetune-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--test",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--qualitative-limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
    )

    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    """Load metrics.json."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: object) -> None:
    """Write UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""
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
    """Evaluate all selected Nepali mixed-data runs."""

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

    from .reasoning_eval import main as single_run_main

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
                / "reasoning"
            )

            run_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            print("\n" + "=" * 90)
            print("NEPALI PHASE 3 — REASONING EVALUATION")
            print("=" * 90)
            print(f"Run:                {run_name}")
            print(f"Reasoning examples: {reasoning_n:,}")
            print(f"Real examples:      {real_n:,}")
            print(
                f"Reasoning fraction: "
                f"{reasoning_n / TOTAL_EXAMPLES:.2%}"
            )
            print(f"Checkpoint:         {checkpoint}")
            print(f"Output:             {run_output}")
            print("=" * 90)

            sys.argv = [
                "reasoning_eval.py",
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
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--seed",
                str(args.seed),
                "--qualitative-per-group",
                str(args.qualitative_limit),
            ]

            single_run_main()

            metrics_path = run_output / "metrics.json"

            if not metrics_path.exists():
                raise RuntimeError(
                    f"metrics.json was not produced:\n{metrics_path}"
                )

            metrics = load_metrics(metrics_path)

            pretrained = metrics["pretrained"]
            finetuned = metrics["finetuned"]
            improvement = metrics["improvement"]

            summary_rows.append(
                {
                    "run": run_name,
                    "reasoning_examples": reasoning_n,
                    "real_examples": real_n,
                    "total_training_examples": TOTAL_EXAMPLES,
                    "reasoning_fraction_of_examples":
                        reasoning_n / TOTAL_EXAMPLES,
                    "real_fraction_of_examples":
                        real_n / TOTAL_EXAMPLES,
                    "pretrained_accuracy_percent":
                        pretrained["exact_match_accuracy"] * 100.0,
                    "finetuned_accuracy_percent":
                        finetuned["exact_match_accuracy"] * 100.0,
                    "absolute_improvement_pp":
                        improvement["absolute_accuracy_points"],
                    "pretrained_token_accuracy_percent":
                        pretrained["mean_answer_token_accuracy"] * 100.0,
                    "finetuned_token_accuracy_percent":
                        finetuned["mean_answer_token_accuracy"] * 100.0,
                    "metrics_file": str(metrics_path),
                }
            )

    finally:
        sys.argv = original_argv

    if not summary_rows:
        raise ValueError("No runs were selected.")

    write_csv(
        args.output_root / "mixture_reasoning_summary.csv",
        summary_rows,
    )

    write_json(
        args.output_root / "mixture_reasoning_summary.json",
        {
            "experiment": {
                "language": "Nepali",
                "total_training_examples_per_run": TOTAL_EXAMPLES,
                "same_pretrained_checkpoint": str(args.pretrained),
                "same_held_out_test_set": str(args.test),
                "tokenizer": str(args.tokenizer),
                "seed": args.seed,
                "reasoning_sizes": [x[1] for x in RUN_ORDER],
                "real_sizes": [x[2] for x in RUN_ORDER],
                "note": (
                    "Reasoning fraction is fraction of training examples, "
                    "not fraction of target tokens. Real pretraining examples "
                    "are longer than synthetic reasoning examples."
                ),
            },
            "runs": summary_rows,
        },
    )

    print("\n" + "=" * 90)
    print("NEPALI MIXED-DATA REASONING EVALUATION COMPLETE")
    print("=" * 90)

    for row in summary_rows:
        print(
            f"{row['run']}: "
            f"{row['pretrained_accuracy_percent']:.2f}% -> "
            f"{row['finetuned_accuracy_percent']:.2f}% "
            f"({row['absolute_improvement_pp']:+.2f} pp)"
        )

    print(
        f"\nSummary: "
        f"{args.output_root / 'mixture_reasoning_summary.csv'}"
    )


if __name__ == "__main__":
    main()
