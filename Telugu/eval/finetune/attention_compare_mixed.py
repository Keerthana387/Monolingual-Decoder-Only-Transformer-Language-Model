"""
TELUGU — PHASE 3 MIXED-DATA ATTENTION EVALUATION

Runs attention_compare.py for every mixed-data checkpoint.

Attention is evaluated on comparative reasoning prompts from the fixed
held-out reasoning test set. Results are stored independently per run.

The existing attention_compare.py is reused so its attention collection,
entropy, distance, head metrics and heatmaps remain consistent.

Run from project root:

    python -m Telugu.eval.finetune.attention_compare_mixed \
        --pretrained Telugu/checkpoints/final.pt \
        --finetune-root Telugu/finetune/mixed_data_size \
        --tokenizer Telugu/data/tokenizer/telugu_bpe_6000.model \
        --test Telugu/reasoning/data/test.jsonl \
        --output-root Telugu/results/finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# MIXTURE RUNS
# ============================================================

RUN_ORDER = [
    ("reasoning_2k_real_18k", 2000, 18000),
    ("reasoning_5k_real_15k", 5000, 15000),
    ("reasoning_8k_real_12k", 8000, 12000),
    ("reasoning_12k_real_8k", 12000, 8000),
    ("reasoning_16k_real_4k", 16000, 4000),
    ("reasoning_20k_real_0k", 20000, 0),
]


# ============================================================
# PREFERRED REASONING TASKS
# ============================================================

PREFERRED_TASKS = {
    "transitive_2",
    "transitive_3",
    "mixed_transitive",
    "ordering",
    "distractor_reasoning",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse attention-evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Telugu Phase 3 attention comparison for "
            "all mixed-data fine-tuning checkpoints."
        )
    )

    parser.add_argument(
        "--pretrained",
        required=True,
        type=Path,
        help="Phase 2 pretrained checkpoint.",
    )

    parser.add_argument(
        "--finetune-root",
        required=True,
        type=Path,
        help="Root directory containing all mixed-data checkpoints.",
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Project-trained Telugu SentencePiece tokenizer.",
    )

    parser.add_argument(
        "--test",
        required=True,
        type=Path,
        help="Held-out reasoning test JSONL.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root directory for Phase 3 evaluation results.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of reasoning test examples for attention analysis.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help=(
            "Optional run names to evaluate. "
            "Default: all six runs."
        ),
    )

    return parser.parse_args()


# ============================================================
# JSON / CSV UTILITIES
# ============================================================

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


def load_json(
    path: Path,
) -> dict:
    """Load a JSON file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write summary rows to CSV."""

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

    fields = list(
        rows[0].keys()
    )

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


# ============================================================
# TEST SET
# ============================================================

def load_test_examples(
    test_path: Path,
) -> list[dict]:
    """
    Load the reasoning test set.

    Only examples explicitly marked as non-test are excluded.
    """

    rows: list[dict] = []

    with test_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            split = row.get(
                "split"
            )

            if split not in (
                None,
                "test",
            ):
                continue

            rows.append(row)

    if not rows:
        raise ValueError(
            f"No usable reasoning test examples found in {test_path}"
        )

    return rows


# ============================================================
# REPRESENTATIVE PROMPT SELECTION
# ============================================================

def choose_representative(
    test_path: Path,
    seed: int,
) -> dict:
    """
    Select a comparative-reasoning example for documentation.

    Preference is given to transitive/order/comparison reasoning.
    Among a short list of readable prompts, selection is deterministic
    with the supplied seed.
    """

    rows = load_test_examples(
        test_path
    )

    preferred = [
        row
        for row in rows
        if row.get("task_type")
        in PREFERRED_TASKS
    ]

    pool = (
        preferred
        if preferred
        else rows
    )

    if not pool:
        raise ValueError(
            "Reasoning test set is empty."
        )

    # Prefer prompts of a readable size.
    pool = sorted(
        pool,
        key=lambda row: abs(
            len(
                str(
                    row.get(
                        "prompt",
                        "",
                    )
                )
            )
            - 180
        ),
    )

    top = pool[
        : min(
            25,
            len(pool),
        )
    ]

    rng = random.Random(
        seed
    )

    return rng.choice(
        top
    )


# ============================================================
# REPRESENTATIVE HEATMAPS
# ============================================================

def overwrite_representative_heatmaps(
    attention_module,
    representative: dict,
    pretrained_path: Path,
    finetuned_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
) -> None:
    """
    Recompute representative heatmaps using the selected
    comparative-reasoning prompt.

    Aggregate statistics remain those computed by attention_compare.py
    over the requested attention sample.
    """

    import matplotlib.pyplot as plt
    import torch

    # --------------------------------------------------------
    # Load tokenizer.
    # --------------------------------------------------------

    tokenizer = (
        attention_module.load_tokenizer(
            tokenizer_path
        )
    )

    # --------------------------------------------------------
    # Device.
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Load models.
    # --------------------------------------------------------

    pretrained_model = (
        attention_module.build_model(
            pretrained_path,
            tokenizer,
            device,
        )
    )

    finetuned_model = (
        attention_module.build_model(
            finetuned_path,
            tokenizer,
            device,
        )
    )

    # --------------------------------------------------------
    # Representative prompt.
    # --------------------------------------------------------

    prompt = str(
        representative.get(
            "prompt",
            "",
        )
    )

    if not prompt:
        raise ValueError(
            "Representative example has an empty prompt."
        )

    # --------------------------------------------------------
    # Collect attention.
    # --------------------------------------------------------

    pretrained = (
        attention_module.collect_attention(
            pretrained_model,
            tokenizer,
            prompt,
            device,
        )
    )

    finetuned = (
        attention_module.collect_attention(
            finetuned_model,
            tokenizer,
            prompt,
            device,
        )
    )

    # --------------------------------------------------------
    # Number of common layers.
    # --------------------------------------------------------

    n_layers = min(
        len(
            pretrained[
                "attention"
            ]
        ),
        len(
            finetuned[
                "attention"
            ]
        ),
    )

    if n_layers == 0:
        raise RuntimeError(
            "Representative prompt produced no attention layers."
        )

    early_layer = 0

    late_layer = n_layers - 1

    tokens = pretrained[
        "tokens"
    ]

    # ========================================================
    # EARLY / LATE HEATMAPS
    # ========================================================

    attention_module.save_heatmap(
        output_dir
        / "attention_pretrained_early_layer.png",

        pretrained[
            "attention"
        ][early_layer],

        pretrained[
            "tokens"
        ],

        (
            "Telugu pretrained — "
            "representative comparative prompt — "
            f"early layer {early_layer}"
        ),
    )

    attention_module.save_heatmap(
        output_dir
        / "attention_finetuned_early_layer.png",

        finetuned[
            "attention"
        ][early_layer],

        finetuned[
            "tokens"
        ],

        (
            "Telugu fine-tuned — "
            "representative comparative prompt — "
            f"early layer {early_layer}"
        ),
    )

    attention_module.save_heatmap(
        output_dir
        / "attention_pretrained_late_layer.png",

        pretrained[
            "attention"
        ][late_layer],

        pretrained[
            "tokens"
        ],

        (
            "Telugu pretrained — "
            "representative comparative prompt — "
            f"late layer {late_layer}"
        ),
    )

    attention_module.save_heatmap(
        output_dir
        / "attention_finetuned_late_layer.png",

        finetuned[
            "attention"
        ][late_layer],

        finetuned[
            "tokens"
        ],

        (
            "Telugu fine-tuned — "
            "representative comparative prompt — "
            f"late layer {late_layer}"
        ),
    )

    # ========================================================
    # DIFFERENCE HEATMAPS
    # ========================================================

    for name, layer in (
        (
            "early_layer",
            early_layer,
        ),
        (
            "late_layer",
            late_layer,
        ),
    ):

        pretrained_mean = (
            pretrained[
                "attention"
            ][layer]
            .mean(
                axis=0
            )
        )

        finetuned_mean = (
            finetuned[
                "attention"
            ][layer]
            .mean(
                axis=0
            )
        )

        difference = (
            finetuned_mean
            - pretrained_mean
        )

        fig_size = max(
            8,
            len(tokens) * 0.18,
        )

        fig, ax = plt.subplots(
            figsize=(
                fig_size,
                fig_size,
            )
        )

        im = ax.imshow(
            difference,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
        )

        # ----------------------------------------------------
        # Token labels.
        # ----------------------------------------------------

        if len(tokens) <= 60:

            ax.set_xticks(
                range(
                    len(tokens)
                )
            )

            ax.set_yticks(
                range(
                    len(tokens)
                )
            )

            ax.set_xticklabels(
                tokens,
                rotation=90,
                fontsize=7,
            )

            ax.set_yticklabels(
                tokens,
                fontsize=7,
            )

        else:

            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(
            (
                "Telugu fine-tuned − pretrained "
                "attention difference — "
                f"{name}"
            )
        )

        ax.set_xlabel(
            "Key token"
        )

        ax.set_ylabel(
            "Query token"
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Attention change",
        )

        fig.tight_layout()

        fig.savefig(
            output_dir
            / f"attention_difference_{name}.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run attention analysis for each mixed-data checkpoint."""

    args = parse_args()

    # --------------------------------------------------------
    # Validate top-level inputs.
    # --------------------------------------------------------

    required_paths = (
        args.pretrained,
        args.tokenizer,
        args.test,
        args.finetune_root,
    )

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )

    if args.num_examples <= 0:

        raise ValueError(
            "--num-examples must be greater than zero."
        )

    # --------------------------------------------------------
    # Select runs.
    # --------------------------------------------------------

    selected = (
        set(args.runs)
        if args.runs
        else None
    )

    # Validate explicitly requested run names.
    if selected is not None:

        known_runs = {
            run_name
            for run_name, _, _
            in RUN_ORDER
        }

        unknown = (
            selected
            - known_runs
        )

        if unknown:

            raise ValueError(
                "Unknown run name(s): "
                + ", ".join(
                    sorted(
                        unknown
                    )
                )
            )

    # ========================================================
    # IMPORTANT:
    # Import BOTH:
    #
    #   attention_module
    #   single_run_main
    #
    # The wrapper needs the module because it directly uses:
    #
    #   load_tokenizer()
    #   build_model()
    #   collect_attention()
    #   save_heatmap()
    #
    # It also needs main() to run the standard evaluator.
    # ========================================================

    try:

        from . import attention_compare as attention_module

        single_run_main = (
            attention_module.main
        )

    except Exception as exc:

        raise ImportError(
            "Could not import the sibling "
            "Telugu.eval.finetune.attention_compare module."
        ) from exc

    # ========================================================
    # REPRESENTATIVE PROMPT
    # ========================================================

    representative = (
        choose_representative(
            args.test,
            args.seed,
        )
    )

    write_json(
        args.output_root
        / "attention_representative_prompt.json",
        {
            "selection_rule": (
                "Prefer transitive/order comparative-reasoning tasks; "
                "among a short list of readable prompts, choose "
                "deterministically using the supplied seed."
            ),

            "seed":
                args.seed,

            "example":
                representative,
        },
    )

    # ========================================================
    # RUN EVALUATIONS
    # ========================================================

    summary_rows: list[dict] = []

    for (
        run_name,
        reasoning_n,
        real_n,
    ) in RUN_ORDER:

        # ----------------------------------------------------
        # Skip non-selected runs.
        # ----------------------------------------------------

        if (
            selected is not None
            and run_name not in selected
        ):
            continue

        # ----------------------------------------------------
        # Check checkpoint.
        # ----------------------------------------------------

        checkpoint = (
            args.finetune_root
            / run_name
            / "final.pt"
        )

        if not checkpoint.exists():

            raise FileNotFoundError(
                (
                    f"Missing final checkpoint for "
                    f"{run_name}:\n{checkpoint}"
                )
            )

        # ----------------------------------------------------
        # Per-run output.
        #
        # This matches the Telugu results structure:
        #
        # results/finetune/
        #     reasoning_2k_real_18k/
        #         attention/
        #
        # ----------------------------------------------------

        run_output = (
            args.output_root
            / run_name
            / "attention"
        )

        run_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"ATTENTION RUN: {run_name}"
        )

        print(
            "=" * 90
        )

        print(
            f"Reasoning examples: {reasoning_n:,}"
        )

        print(
            f"Real examples: {real_n:,}"
        )

        print(
            f"Reasoning fraction: "
            f"{reasoning_n / 20000.0:.2%}"
        )

        print(
            f"Checkpoint: {checkpoint}"
        )

        print(
            f"Output: {run_output}"
        )

        # ====================================================
        # CALL STANDARD SINGLE-RUN EVALUATOR
        # ====================================================

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

            "--num-examples",
            str(args.num_examples),

            "--seed",
            str(args.seed),
        ]

        # This uses the actual Telugu attention evaluator.
        single_run_main()

        # ====================================================
        # REPRESENTATIVE HEATMAPS
        # ====================================================

        # The standard evaluator analyzes a reproducible random sample.
        # For the report, replace only the displayed representative
        # heatmaps with a deliberately selected comparative-reasoning
        # example.
        #
        # Aggregate metrics remain based on the full requested sample.
        # ====================================================

        overwrite_representative_heatmaps(
            attention_module=attention_module,

            representative=representative,

            pretrained_path=args.pretrained,

            finetuned_path=checkpoint,

            tokenizer_path=args.tokenizer,

            output_dir=run_output,
        )

        # ====================================================
        # LOAD STANDARD SUMMARY
        # ====================================================

        comparison_path = (
            run_output
            / "attention_comparison.json"
        )

        if not comparison_path.exists():

            raise RuntimeError(
                (
                    "Expected attention comparison was not produced:\n"
                    f"{comparison_path}"
                )
            )

        comparison = load_json(
            comparison_path
        )

        # ----------------------------------------------------
        # Extract pretrained and fine-tuned layer summaries.
        # ----------------------------------------------------

        pre_layers = comparison.get(
            "pretrained",
            {},
        )

        ft_layers = comparison.get(
            "finetuned",
            {},
        )

        common_layers = sorted(
            (
                set(
                    pre_layers.keys()
                )
                &
                set(
                    ft_layers.keys()
                )
            ),
            key=lambda x: int(x),
        )

        if not common_layers:

            raise RuntimeError(
                (
                    f"No common attention layers "
                    f"were found for {run_name}."
                )
            )

        # ----------------------------------------------------
        # Early and late layers.
        # ----------------------------------------------------

        early = common_layers[0]

        late = common_layers[-1]

        # ----------------------------------------------------
        # Build mixture summary row.
        # ----------------------------------------------------

        summary_rows.append(
            {
                "run":
                    run_name,

                "reasoning_examples":
                    reasoning_n,

                "real_examples":
                    real_n,

                "total_examples":
                    reasoning_n + real_n,

                "reasoning_fraction_of_examples":
                    reasoning_n / 20000.0,

                "real_fraction_of_examples":
                    real_n / 20000.0,

                "num_attention_examples":
                    args.num_examples,

                "early_layer":
                    int(early),

                "late_layer":
                    int(late),

                "pretrained_early_entropy":
                    pre_layers[
                        early
                    ][
                        "mean_entropy"
                    ],

                "finetuned_early_entropy":
                    ft_layers[
                        early
                    ][
                        "mean_entropy"
                    ],

                "pretrained_late_entropy":
                    pre_layers[
                        late
                    ][
                        "mean_entropy"
                    ],

                "finetuned_late_entropy":
                    ft_layers[
                        late
                    ][
                        "mean_entropy"
                    ],

                "pretrained_early_distance":
                    pre_layers[
                        early
                    ][
                        "mean_attention_distance"
                    ],

                "finetuned_early_distance":
                    ft_layers[
                        early
                    ][
                        "mean_attention_distance"
                    ],

                "pretrained_late_distance":
                    pre_layers[
                        late
                    ][
                        "mean_attention_distance"
                    ],

                "finetuned_late_distance":
                    ft_layers[
                        late
                    ][
                        "mean_attention_distance"
                    ],
            }
        )

        print(
            f"\nCompleted attention evaluation: "
            f"{run_name}"
        )

    # ========================================================
    # VALIDATE THAT AT LEAST ONE RUN COMPLETED
    # ========================================================

    if not summary_rows:

        raise ValueError(
            "No runs were selected for evaluation."
        )

    # ========================================================
    # GLOBAL MIXTURE CSV
    # ========================================================

    summary_csv = (
        args.output_root
        / "mixture_attention_summary.csv"
    )

    write_csv(
        summary_csv,
        summary_rows,
    )

    # ========================================================
    # GLOBAL MIXTURE JSON
    # ========================================================

    summary_json = (
        args.output_root
        / "mixture_attention_summary.json"
    )

    write_json(
        summary_json,
        {
            "num_examples_per_run":
                args.num_examples,

            "seed":
                args.seed,

            "pretrained_checkpoint":
                str(args.pretrained),

            "tokenizer":
                str(args.tokenizer),

            "test_set":
                str(args.test),

            "finetune_root":
                str(args.finetune_root),

            "output_root":
                str(args.output_root),

            "representative_prompt":
                representative,

            "runs":
                summary_rows,
        },
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "TELUGU PHASE 3 MIXED-DATA "
        "ATTENTION EVALUATION COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        f"Runs completed: "
        f"{len(summary_rows)}"
    )

    print(
        f"Examples per run: "
        f"{args.num_examples}"
    )

    print(
        f"Results root:\n"
        f"  {args.output_root}"
    )

    print(
        f"\nSummary CSV:\n"
        f"  {summary_csv}"
    )

    print(
        f"\nSummary JSON:\n"
        f"  {summary_json}"
    )

    print(
        "\nPer-run results are stored as:"
    )

    print(
        "  results/finetune/"
        "<run_name>/attention/"
    )

    print(
        "=" * 90
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()