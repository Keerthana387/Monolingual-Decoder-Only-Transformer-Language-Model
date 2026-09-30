"""
TELUGU — PHASE 3 MIXED-DATA REASONING EVALUATION

Runs the existing Telugu reasoning_eval.py once for each mixed-data
fine-tuning checkpoint and stores results separately.

Expected fine-tuning directory layout:

    Telugu/finetune/mixed_data_size/
        reasoning_2k_real_18k/final.pt
        reasoning_5k_real_15k/final.pt
        reasoning_8k_real_12k/final.pt
        reasoning_12k_real_8k/final.pt
        reasoning_16k_real_4k/final.pt
        reasoning_20k_real_0k/final.pt

The existing detailed reasoning_eval.py is reused so that the full
reasoning evaluation logic, metrics, plots, qualitative examples,
and paired pretrained-vs-finetuned analysis remain unchanged.

Run from project root:

    python -m Telugu.eval.finetune.reasoning_eval_mixed \
        --pretrained Telugu/checkpoints/final.pt \
        --finetune-root Telugu/finetune/mixed_data_size \
        --tokenizer Telugu/data/tokenizer/telugu_bpe_6000.model \
        --test Telugu/reasoning/data/test.jsonl \
        --output-root Telugu/results/finetune

The six runs are:

    2k reasoning  + 18k real
    5k reasoning  + 15k real
    8k reasoning  + 12k real
    12k reasoning + 8k real
    16k reasoning + 4k real
    20k reasoning + 0k real

All runs use the same pretrained Phase 2 checkpoint and the same
held-out reasoning test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# =============================================================================
# PATH / RUN CONFIGURATION
# =============================================================================

# This file is:
#
#   Telugu/eval/finetune/reasoning_eval_mixed.py
#
# parents[0] = Telugu/eval/finetune
# parents[1] = Telugu/eval
# parents[2] = Telugu
#
# We add Telugu/ to sys.path because the existing reasoning evaluator may
# import language-specific modules using imports such as:
#
#   from model.lm import TeluguLanguageModel
#
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MIXTURE DEFINITIONS
# =============================================================================

RUN_ORDER = [
    ("reasoning_2k_real_18k", 2_000, 18_000),
    ("reasoning_5k_real_15k", 5_000, 15_000),
    ("reasoning_8k_real_12k", 8_000, 12_000),
    ("reasoning_12k_real_8k", 12_000, 8_000),
    ("reasoning_16k_real_4k", 16_000, 4_000),
    ("reasoning_20k_real_0k", 20_000, 0),
]

RUN_NAMES = {
    run_name
    for run_name, _, _ in RUN_ORDER
}

TOTAL_TRAINING_EXAMPLES = 20_000


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse paths and evaluation settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Telugu Phase 3 mixed-data fine-tuning runs "
            "against the same pretrained checkpoint."
        )
    )

    parser.add_argument(
        "--pretrained",
        required=True,
        type=Path,
        help="Path to the Phase 2 pretrained Telugu checkpoint.",
    )

    parser.add_argument(
        "--finetune-root",
        required=True,
        type=Path,
        help=(
            "Root directory containing the six mixed-data "
            "fine-tuning run directories."
        ),
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Path to the existing Telugu tokenizer.",
    )

    parser.add_argument(
        "--test",
        required=True,
        type=Path,
        help="Held-out Telugu reasoning test JSONL.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root directory where evaluation results are written.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated answer tokens.",
    )

    parser.add_argument(
        "--qualitative-limit",
        type=int,
        default=100,
        help="Maximum number of qualitative examples to save.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the detailed evaluator.",
    )

    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Optional subset of run names to evaluate. "
            "Default: all six runs."
        ),
    )

    return parser.parse_args()


# =============================================================================
# FILE HELPERS
# =============================================================================

def load_metrics(path: Path) -> dict:
    """Load the metrics.json emitted by reasoning_eval.py."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    path: Path,
    obj: object,
) -> None:
    """Write an object as UTF-8 JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a list of dictionaries to a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_selected_runs(
    selected: set[str] | None,
) -> None:
    """Validate that all requested run names belong to the experiment."""
    if selected is None:
        return

    unknown = selected - RUN_NAMES

    if unknown:
        raise ValueError(
            "Unknown run name(s):\n"
            + "\n".join(
                f"  - {name}"
                for name in sorted(unknown)
            )
            + "\n\nValid run names are:\n"
            + "\n".join(
                f"  - {name}"
                for name, _, _ in RUN_ORDER
            )
        )


def validate_paths(
    args: argparse.Namespace,
) -> None:
    """Verify that all required input paths exist."""
    required_paths = [
        (
            "Pretrained checkpoint",
            args.pretrained,
        ),
        (
            "Tokenizer",
            args.tokenizer,
        ),
        (
            "Reasoning test set",
            args.test,
        ),
        (
            "Fine-tuning root",
            args.finetune_root,
        ),
    ]

    missing = []

    for label, path in required_paths:
        if not path.exists():
            missing.append(
                f"{label}: {path}"
            )

    if missing:
        raise FileNotFoundError(
            "The following required paths do not exist:\n\n"
            + "\n".join(missing)
        )


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main() -> None:
    """
    Evaluate every selected mixed-data checkpoint.

    The detailed evaluation is delegated to reasoning_eval.py. This wrapper
    changes only the checkpoint/output directory for each mixture and then
    collects the resulting metrics into one experiment-level summary.
    """
    args = parse_args()

    validate_paths(args)

    selected = (
        set(args.runs)
        if args.runs
        else None
    )

    validate_selected_runs(selected)

    # -------------------------------------------------------------------------
    # Import the existing detailed evaluator.
    #
    # IMPORTANT:
    # This MUST be a relative import because this script is executed as:
    #
    #   python -m Telugu.eval.finetune.reasoning_eval_mixed
    #
    # and reasoning_eval.py is in the same package directory.
    # -------------------------------------------------------------------------

    from .reasoning_eval import main as single_run_main

    # -------------------------------------------------------------------------
    # Prepare output directory.
    # -------------------------------------------------------------------------

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[dict] = []

    # Save the original command-line arguments so that calling the existing
    # evaluator does not permanently modify sys.argv.
    original_argv = sys.argv[:]

    try:

        # =====================================================================
        # RUN EACH MIXTURE
        # =====================================================================

        for (
            run_name,
            reasoning_n,
            real_n,
        ) in RUN_ORDER:

            if (
                selected is not None
                and run_name not in selected
            ):
                continue

            # -------------------------------------------------------------
            # Check fine-tuned checkpoint.
            # -------------------------------------------------------------

            checkpoint = (
                args.finetune_root
                / run_name
                / "final.pt"
            )

            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing final checkpoint for "
                    f"{run_name}:\n{checkpoint}"
                )

            # -------------------------------------------------------------
            # Per-run output directory.
            #
            # Example:
            #
            # Telugu/results/finetune/
            #     reasoning_8k_real_12k/
            #         reasoning/
            #             metrics.json
            #             predictions.jsonl
            #             ...
            # -------------------------------------------------------------

            run_output = (
                args.output_root
                / run_name
                / "reasoning"
            )

            run_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            print("\n")
            print("=" * 90)
            print(
                f"TELUGU PHASE 3 — REASONING EVALUATION"
            )
            print("=" * 90)
            print(
                f"Run:                 {run_name}"
            )
            print(
                f"Reasoning examples:  {reasoning_n:,}"
            )
            print(
                f"Real examples:       {real_n:,}"
            )
            print(
                f"Reasoning fraction:  "
                f"{reasoning_n / TOTAL_TRAINING_EXAMPLES:.2%}"
            )
            print(
                f"Fine-tuned checkpoint:\n"
                f"  {checkpoint}"
            )
            print(
                f"Output directory:\n"
                f"  {run_output}"
            )
            print("=" * 90)

            # -------------------------------------------------------------
            # Build argv for the existing detailed evaluator.
            #
            # This preserves the original reasoning_eval.py implementation
            # instead of duplicating its evaluation logic here.
            # -------------------------------------------------------------

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

                "--qualitative-limit",
                str(args.qualitative_limit),

                "--seed",
                str(args.seed),
            ]

            # -------------------------------------------------------------
            # Run detailed evaluation.
            # -------------------------------------------------------------

            single_run_main()

            # -------------------------------------------------------------
            # Verify that the detailed evaluator produced metrics.
            # -------------------------------------------------------------

            metrics_path = (
                run_output
                / "metrics.json"
            )

            if not metrics_path.exists():
                raise RuntimeError(
                    "The detailed reasoning evaluator completed, "
                    "but metrics.json was not produced:\n"
                    f"{metrics_path}"
                )

            metrics = load_metrics(
                metrics_path
            )

            # -------------------------------------------------------------
            # Extract the standard fields produced by reasoning_eval.py.
            # -------------------------------------------------------------

            if "pretrained" not in metrics:
                raise RuntimeError(
                    f"'pretrained' section missing from:\n"
                    f"{metrics_path}"
                )

            if "finetuned" not in metrics:
                raise RuntimeError(
                    f"'finetuned' section missing from:\n"
                    f"{metrics_path}"
                )

            if "improvement" not in metrics:
                raise RuntimeError(
                    f"'improvement' section missing from:\n"
                    f"{metrics_path}"
                )

            pretrained = metrics["pretrained"]
            finetuned = metrics["finetuned"]
            improvement = metrics["improvement"]

            # -------------------------------------------------------------
            # Collect experiment-level summary.
            # -------------------------------------------------------------

            summary_rows.append(
                {
                    "run": run_name,

                    "reasoning_examples":
                        reasoning_n,

                    "real_examples":
                        real_n,

                    "total_training_examples":
                        TOTAL_TRAINING_EXAMPLES,

                    "reasoning_fraction_of_examples":
                        reasoning_n
                        / TOTAL_TRAINING_EXAMPLES,

                    "real_fraction_of_examples":
                        real_n
                        / TOTAL_TRAINING_EXAMPLES,

                    "pretrained_accuracy_percent":
                        pretrained[
                            "accuracy_percent"
                        ],

                    "finetuned_accuracy_percent":
                        finetuned[
                            "accuracy_percent"
                        ],

                    "absolute_improvement_pp":
                        improvement[
                            "absolute_percentage_points"
                        ],

                    "pretrained_answer_token_accuracy_percent":
                        pretrained[
                            "mean_answer_token_accuracy_percent"
                        ],

                    "finetuned_answer_token_accuracy_percent":
                        finetuned[
                            "mean_answer_token_accuracy_percent"
                        ],

                    "metrics_file":
                        str(metrics_path),
                }
            )

            print("\nCompleted:")
            print(
                f"  {run_name}"
            )

    finally:
        # -------------------------------------------------------------
        # Restore sys.argv even if one of the six evaluations fails.
        # -------------------------------------------------------------
        sys.argv = original_argv

    # =========================================================================
    # FINAL MIXTURE SUMMARY
    # =========================================================================

    if not summary_rows:
        raise ValueError(
            "No runs were selected for evaluation."
        )

    summary_csv = (
        args.output_root
        / "mixture_reasoning_summary.csv"
    )

    summary_json = (
        args.output_root
        / "mixture_reasoning_summary.json"
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    write_csv(
        summary_csv,
        summary_rows,
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    write_json(
        summary_json,
        {
            "experiment": {
                "name":
                    "Telugu Phase 3 mixed-data reasoning fine-tuning",

                "total_training_examples_per_run":
                    TOTAL_TRAINING_EXAMPLES,

                "reasoning_sizes":
                    [
                        reasoning_n
                        for _, reasoning_n, _
                        in RUN_ORDER
                    ],

                "real_sizes":
                    [
                        real_n
                        for _, _, real_n
                        in RUN_ORDER
                    ],

                "same_pretrained_checkpoint":
                    str(args.pretrained),

                "same_held_out_test_set":
                    str(args.test),

                "tokenizer":
                    str(args.tokenizer),

                "seed":
                    args.seed,

                "max_new_tokens":
                    args.max_new_tokens,

                "qualitative_limit":
                    args.qualitative_limit,

                "evaluation_note":
                    (
                        "Each mixed-data run contains exactly 20,000 "
                        "training examples. The x-axis represented by "
                        "reasoning_fraction_of_examples is the fraction "
                        "of training examples that are reasoning examples, "
                        "not the fraction of target tokens. Real pretraining "
                        "examples are longer than synthetic reasoning "
                        "examples, so equal example counts do not imply "
                        "equal target-token exposure."
                    ),
            },

            "runs":
                summary_rows,
        },
    )

    # =========================================================================
    # FINAL MESSAGE
    # =========================================================================

    print("\n")
    print("=" * 90)
    print(
        "TELUGU PHASE 3 MIXED-DATA "
        "REASONING EVALUATION COMPLETE"
    )
    print("=" * 90)

    print(
        f"\nRuns evaluated: "
        f"{len(summary_rows)}"
    )

    print(
        f"\nSummary CSV:\n"
        f"  {summary_csv}"
    )

    print(
        f"\nSummary JSON:\n"
        f"  {summary_json}"
    )

    print("\nPer-run results:")

    for row in summary_rows:
        print(
            f"  {row['run']}: "
            f"{row['pretrained_accuracy_percent']:.2f}% "
            f"-> "
            f"{row['finetuned_accuracy_percent']:.2f}% "
            f"("
            f"{row['absolute_improvement_pp']:+.2f} pp"
            f")"
        )

    print("=" * 90)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()