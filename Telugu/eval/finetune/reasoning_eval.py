"""
TELUGU — PHASE 3 REASONING TEST-SET EVALUATION

Compares the Phase 2 pretrained Telugu model against the Phase 3
fine-tuned Telugu model on the held-out 3,000-example reasoning test set.

Primary metrics:
    - exact-match reasoning accuracy
    - accuracy by task type
    - accuracy by difficulty
    - accuracy by attribute
    - accuracy by entity type
    - pretrained -> fine-tuned improvement
    - paired outcome counts

Additional diagnostics:
    - answer-token accuracy
    - qualitative successes
    - qualitative regressions
    - qualitative failures
    - answer-length distribution
    - paired correctness matrix

Run from the project root ONLY:

    python -m Telugu.eval.finetune.reasoning_eval \
        --pretrained Telugu/checkpoints/final.pt \
        --finetuned Telugu/finetune/final.pt \
        --tokenizer Telugu/data/tokenizer/telugu_bpe_6000.model \
        --test Telugu/reasoning/data/test.jsonl \
        --output-dir Telugu/results/finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ============================================================
# MODULE IMPORT SUPPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch

from Telugu.config.model_config import ModelConfig
from Telugu.model.lm import TeluguLanguageModel


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_SEED = 42
DEFAULT_QUALITATIVE_LIMIT = 100


# ============================================================
# FILE UTILITIES
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}"
                ) from exc

    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Conservative normalization.

    We intentionally do NOT remove Telugu punctuation or words.
    This prevents different answers from being incorrectly treated
    as identical.
    """

    text = str(text).strip()

    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def answer_is_correct(
    prediction: str,
    gold: str,
) -> bool:
    return (
        normalize_text(prediction)
        == normalize_text(gold)
    )


def extract_answer_region(
    generated: str,
    prompt: str,
) -> str:
    """
    Removes the prompt from the decoded generated sequence.
    """

    generated = str(generated)

    if generated.startswith(prompt):
        answer = generated[len(prompt):]

    else:
        position = generated.find(prompt)

        if position >= 0:
            answer = generated[
                position + len(prompt):
            ]
        else:
            answer = generated

    for marker in (
        "<eos>",
        "</s>",
    ):
        if marker in answer:
            answer = answer.split(
                marker,
                1,
            )[0]

    return normalize_text(answer)


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(
    path: Path,
) -> spm.SentencePieceProcessor:

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.Load(str(path)):
        raise RuntimeError(
            f"Could not load SentencePiece tokenizer: {path}"
        )

    return tokenizer


# ============================================================
# MODEL LOADING
# ============================================================

def build_model(
    checkpoint_path: Path,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
) -> TeluguLanguageModel:

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint_path}"
        )

    saved_config = checkpoint.get(
        "model_config"
    )

    if isinstance(saved_config, dict):
        config = ModelConfig(
            **saved_config
        )
    else:
        config = ModelConfig()

    # Match the actual tokenizer PAD ID.
    pad_id = tokenizer.pad_id()

    if pad_id >= 0:
        config.pad_token_id = int(pad_id)

    # Vocabulary consistency check.
    model_vocab = getattr(
        config,
        "vocab_size",
        None,
    )

    tokenizer_vocab = tokenizer.GetPieceSize()

    if (
        model_vocab is not None
        and model_vocab != tokenizer_vocab
    ):
        raise ValueError(
            "Model/tokenizer vocabulary mismatch: "
            f"model={model_vocab}, "
            f"tokenizer={tokenizer_vocab}"
        )

    model = TeluguLanguageModel(
        config
    ).to(device)

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        state_dict = checkpoint.get(
            "model"
        )

    if state_dict is None:
        # Support raw state-dict checkpoints.
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# GREEDY GENERATION
# ============================================================

@torch.no_grad()
def greedy_generate(
    model: TeluguLanguageModel,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:

    input_ids = tokenizer.EncodeAsIds(
        prompt
    )

    if not input_ids:
        return ""

    context_limit = int(
        getattr(
            model.config,
            "max_seq_len",
            512,
        )
    )

    x = torch.tensor(
        [input_ids[-context_limit:]],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.eos_id()

    for _ in range(max_new_tokens):

        x_cond = x[:, -context_limit:]

        output = model(
            x_cond
        )

        # Telugu LanguageModel returns:
        # (logits, loss)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        x = torch.cat(
            [
                x,
                next_token,
            ],
            dim=1,
        )

        if (
            eos_id >= 0
            and int(next_token.item()) == eos_id
        ):
            break

    return tokenizer.DecodeIds(
        x[0]
        .detach()
        .cpu()
        .tolist()
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model: TeluguLanguageModel,
    tokenizer: spm.SentencePieceProcessor,
    examples: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:

    results = []

    total = len(examples)

    for idx, example in enumerate(
        examples,
        1,
    ):

        prompt = str(
            example["prompt"]
        )

        gold = str(
            example["answer"]
        )

        generated = greedy_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        prediction = extract_answer_region(
            generated,
            prompt,
        )

        correct = answer_is_correct(
            prediction,
            gold,
        )

        # Secondary token-level diagnostic.
        prediction_tokens = tokenizer.EncodeAsIds(
            prediction
        )

        gold_tokens = tokenizer.EncodeAsIds(
            normalize_text(gold)
        )

        common = min(
            len(prediction_tokens),
            len(gold_tokens),
        )

        matching = sum(
            prediction_tokens[i]
            == gold_tokens[i]
            for i in range(common)
        )

        token_accuracy = (
            matching / len(gold_tokens)
            if gold_tokens
            else float(correct)
        )

        row = {
            "id": example.get("id"),
            "model": model_name,
            "task_type": example.get("task_type"),
            "difficulty": example.get("difficulty"),
            "attribute": example.get("attribute"),
            "entity_type": example.get("entity_type"),
            "prompt": prompt,
            "gold_answer": gold,
            "prediction": prediction,
            "correct": bool(correct),
            "gold_token_count": len(gold_tokens),
            "prediction_token_count": len(
                prediction_tokens
            ),
            "answer_token_accuracy": float(
                token_accuracy
            ),
        }

        for key in (
            "entities",
            "values",
            "template_ids",
            "language",
        ):
            if key in example:
                row[key] = example[key]

        results.append(row)

        if (
            idx == 1
            or idx % 250 == 1
            or idx == total
        ):
            print(
                f"{idx:7d}/{total:5d}"
            )

    return results


# ============================================================
# BASIC METRICS
# ============================================================

def accuracy(
    rows: list[dict[str, Any]]
) -> float:

    if not rows:
        return 0.0

    return (
        100.0
        * sum(
            bool(row["correct"])
            for row in rows
        )
        / len(rows)
    )


def mean_token_accuracy(
    rows: list[dict[str, Any]]
) -> float:

    if not rows:
        return 0.0

    return (
        100.0
        * float(
            np.mean(
                [
                    float(
                        row[
                            "answer_token_accuracy"
                        ]
                    )
                    for row in rows
                ]
            )
        )
    )


# ============================================================
# GROUPED METRICS
# ============================================================

def paired_group_metrics(
    pretrained_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:

    fine_by_id = {
        row["id"]: row
        for row in finetuned_rows
    }

    groups = defaultdict(list)

    for pretrained in pretrained_rows:

        finetuned = fine_by_id.get(
            pretrained["id"]
        )

        if finetuned is None:
            continue

        value = pretrained.get(key)

        if value is None:
            value = "unknown"

        groups[
            str(value)
        ].append(
            (
                pretrained,
                finetuned,
            )
        )

    output = []

    for group, pairs in sorted(
        groups.items()
    ):

        n = len(pairs)

        pretrained_correct = sum(
            p["correct"]
            for p, _ in pairs
        )

        finetuned_correct = sum(
            f["correct"]
            for _, f in pairs
        )

        output.append(
            {
                key: group,
                "n": n,
                "pretrained_correct":
                    pretrained_correct,
                "pretrained_accuracy_percent":
                    (
                        100.0
                        * pretrained_correct
                        / n
                        if n
                        else 0.0
                    ),
                "finetuned_correct":
                    finetuned_correct,
                "finetuned_accuracy_percent":
                    (
                        100.0
                        * finetuned_correct
                        / n
                        if n
                        else 0.0
                    ),
                "improvement_percentage_points":
                    (
                        100.0
                        * (
                            finetuned_correct
                            - pretrained_correct
                        )
                        / n
                        if n
                        else 0.0
                    ),
                "newly_correct":
                    sum(
                        not p["correct"]
                        and f["correct"]
                        for p, f in pairs
                    ),
                "regressed":
                    sum(
                        p["correct"]
                        and not f["correct"]
                        for p, f in pairs
                    ),
            }
        )

    return output


# ============================================================
# PAIRED CORRECTNESS MATRIX
# ============================================================

def build_relation_matrix(
    pretrained_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str]]:

    fine_by_id = {
        row["id"]: row
        for row in finetuned_rows
    }

    # Rows = pretrained:
    #       wrong / correct
    #
    # Columns = fine-tuned:
    #       wrong / correct
    matrix = np.zeros(
        (2, 2),
        dtype=int,
    )

    for pretrained in pretrained_rows:

        finetuned = fine_by_id.get(
            pretrained["id"]
        )

        if finetuned is None:
            continue

        row = int(
            bool(pretrained["correct"])
        )

        col = int(
            bool(finetuned["correct"])
        )

        matrix[row, col] += 1

    return matrix, [
        "Wrong",
        "Correct",
    ]


def save_confusion(
    path: Path,
    matrix: Any,
    labels: list[str],
    title: str,
    xlabel: str = "Fine-tuned",
    ylabel: str = "Pretrained",
) -> None:

    # IMPORTANT:
    # Convert lists to NumPy arrays so matrix[i, j]
    # is always valid.
    matrix = np.asarray(
        matrix,
        dtype=int,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.imshow(
        matrix,
        interpolation="nearest",
    )

    ax.set(
        xticks=np.arange(
            len(labels)
        ),
        yticks=np.arange(
            len(labels)
        ),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            ax.text(
                j,
                i,
                int(
                    matrix[i, j]
                ),
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PLOTS
# ============================================================

def save_accuracy_comparison(
    path: Path,
    pretrained_accuracy: float,
    finetuned_accuracy: float,
) -> None:

    names = [
        "Pretrained",
        "Fine-tuned",
    ]

    values = [
        pretrained_accuracy,
        finetuned_accuracy,
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        names,
        values,
    )

    ax.set_ylabel(
        "Exact-match accuracy (%)"
    )

    ax.set_title(
        "Telugu reasoning test-set accuracy"
    )

    ax.set_ylim(
        0,
        max(
            100.0,
            max(values) * 1.15,
        ),
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_task_accuracy_comparison(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        return

    categories = [
        row["task_type"]
        for row in rows
    ]

    pretrained = [
        row[
            "pretrained_accuracy_percent"
        ]
        for row in rows
    ]

    finetuned = [
        row[
            "finetuned_accuracy_percent"
        ]
        for row in rows
    ]

    x = np.arange(
        len(categories)
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(
            max(
                9,
                len(categories) * 1.5,
            ),
            6,
        )
    )

    ax.bar(
        x - width / 2,
        pretrained,
        width,
        label="Pretrained",
    )

    ax.bar(
        x + width / 2,
        finetuned,
        width,
        label="Fine-tuned",
    )

    ax.set_ylabel(
        "Exact-match accuracy (%)"
    )

    ax.set_title(
        "Telugu reasoning accuracy by task type"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories,
        rotation=35,
        ha="right",
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_answer_length_distribution(
    path: Path,
    pretrained_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
) -> None:

    pretrained = [
        row[
            "prediction_token_count"
        ]
        for row in pretrained_rows
    ]

    finetuned = [
        row[
            "prediction_token_count"
        ]
        for row in finetuned_rows
    ]

    all_lengths = (
        pretrained
        + finetuned
    )

    max_length = (
        max(all_lengths)
        if all_lengths
        else 1
    )

    bins = np.arange(
        -0.5,
        max_length + 1.5,
        1,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        pretrained,
        bins=bins,
        alpha=0.6,
        label="Pretrained",
    )

    ax.hist(
        finetuned,
        bins=bins,
        alpha=0.6,
        label="Fine-tuned",
    )

    ax.set_xlabel(
        "Predicted answer length (tokens)"
    )

    ax.set_ylabel(
        "Number of test examples"
    )

    ax.set_title(
        "Telugu predicted answer-length distribution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# QUALITATIVE EXAMPLES
# ============================================================

def collect_qualitative(
    pretrained_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:

    fine_by_id = {
        row["id"]: row
        for row in finetuned_rows
    }

    newly_correct = []
    regressions = []
    both_wrong = []

    for pretrained in pretrained_rows:

        finetuned = fine_by_id.get(
            pretrained["id"]
        )

        if finetuned is None:
            continue

        row = {
            "id": pretrained.get("id"),
            "task_type":
                pretrained.get("task_type"),
            "difficulty":
                pretrained.get("difficulty"),
            "attribute":
                pretrained.get("attribute"),
            "entity_type":
                pretrained.get("entity_type"),
            "prompt":
                pretrained.get("prompt"),
            "gold_answer":
                pretrained.get("gold_answer"),
            "pretrained_prediction":
                pretrained.get("prediction"),
            "finetuned_prediction":
                finetuned.get("prediction"),
        }

        if (
            not pretrained["correct"]
            and finetuned["correct"]
        ):
            newly_correct.append(row)

        elif (
            pretrained["correct"]
            and not finetuned["correct"]
        ):
            regressions.append(row)

        elif (
            not pretrained["correct"]
            and not finetuned["correct"]
        ):
            both_wrong.append(row)

    return {
        "newly_correct":
            newly_correct[:limit],
        "regressions":
            regressions[:limit],
        "both_wrong":
            both_wrong[:limit],
    }


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Telugu Phase 2 "
            "vs Phase 3 reasoning."
        )
    )

    parser.add_argument(
        "--pretrained",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--finetuned",
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
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--qualitative-limit",
        type=int,
        default=DEFAULT_QUALITATIVE_LIMIT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    for path in (
        args.pretrained,
        args.finetuned,
        args.tokenizer,
        args.test,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "TELUGU — PHASE 3 REASONING TEST-SET EVALUATION"
    )
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    examples = load_jsonl(
        args.test
    )

    if not examples:
        raise ValueError(
            "Test set is empty."
        )

    # Ensure this is actually the held-out test split.
    invalid = [
        example
        for example in examples
        if example.get("split")
        not in (
            None,
            "test",
        )
    ]

    if invalid:
        raise ValueError(
            "The supplied file contains examples "
            "not marked as split='test'."
        )

    print(
        f"Test examples: {len(examples):,}"
    )

    tokenizer = load_tokenizer(
        args.tokenizer
    )

    print(
        "Tokenizer vocabulary: "
        f"{tokenizer.GetPieceSize():,}"
    )

    print(
        f"PAD token ID: {tokenizer.pad_id()}"
    )

    print(
        f"EOS token ID: {tokenizer.eos_id()}"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "\nLoading pretrained..."
    )

    pretrained_model = build_model(
        args.pretrained,
        tokenizer,
        device,
    )

    print(
        "Loading fine-tuned..."
    )

    finetuned_model = build_model(
        args.finetuned,
        tokenizer,
        device,
    )

    pretrained_params = sum(
        parameter.numel()
        for parameter
        in pretrained_model.parameters()
    )

    finetuned_params = sum(
        parameter.numel()
        for parameter
        in finetuned_model.parameters()
    )

    print(
        f"Pretrained parameters: "
        f"{pretrained_params:,}"
    )

    print(
        f"Fine-tuned parameters: "
        f"{finetuned_params:,}"
    )

    # --------------------------------------------------------
    # FULL TEST SET
    # --------------------------------------------------------

    print(
        "\nEvaluating pretrained..."
    )

    print("-" * 80)

    pretrained_rows = evaluate_model(
        "pretrained",
        pretrained_model,
        tokenizer,
        examples,
        device,
        args.max_new_tokens,
    )

    print(
        "\nEvaluating finetuned..."
    )

    print("-" * 80)

    finetuned_rows = evaluate_model(
        "finetuned",
        finetuned_model,
        tokenizer,
        examples,
        device,
        args.max_new_tokens,
    )

    # --------------------------------------------------------
    # RAW PREDICTIONS
    # --------------------------------------------------------

    write_jsonl(
        args.output_dir
        / "predictions.jsonl",
        pretrained_rows
        + finetuned_rows,
    )

    write_csv(
        args.output_dir
        / "predictions.csv",
        pretrained_rows
        + finetuned_rows,
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    pretrained_accuracy = accuracy(
        pretrained_rows
    )

    finetuned_accuracy = accuracy(
        finetuned_rows
    )

    relative_improvement = (
        (
            finetuned_accuracy
            - pretrained_accuracy
        )
        / pretrained_accuracy
        * 100.0
        if pretrained_accuracy > 0
        else None
    )

    paired_outcomes = {
        "both_correct": sum(
            p["correct"]
            and f["correct"]
            for p, f in zip(
                pretrained_rows,
                finetuned_rows,
            )
        ),
        "pretrained_only_correct": sum(
            p["correct"]
            and not f["correct"]
            for p, f in zip(
                pretrained_rows,
                finetuned_rows,
            )
        ),
        "finetuned_only_correct": sum(
            not p["correct"]
            and f["correct"]
            for p, f in zip(
                pretrained_rows,
                finetuned_rows,
            )
        ),
        "both_wrong": sum(
            not p["correct"]
            and not f["correct"]
            for p, f in zip(
                pretrained_rows,
                finetuned_rows,
            )
        ),
    }

    metrics = {
        "test_examples":
            len(examples),

        "pretrained": {
            "correct":
                sum(
                    r["correct"]
                    for r in pretrained_rows
                ),
            "accuracy_percent":
                pretrained_accuracy,
            "mean_answer_token_accuracy_percent":
                mean_token_accuracy(
                    pretrained_rows
                ),
        },

        "finetuned": {
            "correct":
                sum(
                    r["correct"]
                    for r in finetuned_rows
                ),
            "accuracy_percent":
                finetuned_accuracy,
            "mean_answer_token_accuracy_percent":
                mean_token_accuracy(
                    finetuned_rows
                ),
        },

        "improvement": {
            "absolute_percentage_points":
                finetuned_accuracy
                - pretrained_accuracy,
            "relative_percent":
                relative_improvement,
        },

        "paired_outcomes":
            paired_outcomes,

        "configuration": {
            "test_file":
                str(args.test),
            "pretrained_checkpoint":
                str(args.pretrained),
            "finetuned_checkpoint":
                str(args.finetuned),
            "tokenizer":
                str(args.tokenizer),
            "max_new_tokens":
                args.max_new_tokens,
            "seed":
                args.seed,
            "device":
                str(device),
            "tokenizer_vocab_size":
                tokenizer.GetPieceSize(),
        },

        "model_parameters": {
            "pretrained":
                pretrained_params,
            "finetuned":
                finetuned_params,
        },
    }

    write_json(
        args.output_dir
        / "metrics.json",
        metrics,
    )

    # --------------------------------------------------------
    # GROUPED METRICS
    # --------------------------------------------------------

    task_metrics = paired_group_metrics(
        pretrained_rows,
        finetuned_rows,
        "task_type",
    )

    difficulty_metrics = paired_group_metrics(
        pretrained_rows,
        finetuned_rows,
        "difficulty",
    )

    attribute_metrics = paired_group_metrics(
        pretrained_rows,
        finetuned_rows,
        "attribute",
    )

    entity_metrics = paired_group_metrics(
        pretrained_rows,
        finetuned_rows,
        "entity_type",
    )

    write_csv(
        args.output_dir
        / "task_metrics.csv",
        task_metrics,
    )

    write_csv(
        args.output_dir
        / "difficulty_metrics.csv",
        difficulty_metrics,
    )

    write_csv(
        args.output_dir
        / "attribute_metrics.csv",
        attribute_metrics,
    )

    write_csv(
        args.output_dir
        / "entity_type_metrics.csv",
        entity_metrics,
    )

    write_json(
        args.output_dir
        / "grouped_metrics.json",
        {
            "task_type":
                task_metrics,
            "difficulty":
                difficulty_metrics,
            "attribute":
                attribute_metrics,
            "entity_type":
                entity_metrics,
        },
    )

    # --------------------------------------------------------
    # QUALITATIVE ANALYSIS
    # --------------------------------------------------------

    qualitative = collect_qualitative(
        pretrained_rows,
        finetuned_rows,
        args.qualitative_limit,
    )

    write_jsonl(
        args.output_dir
        / "qualitative_successes.jsonl",
        qualitative["newly_correct"],
    )

    write_jsonl(
        args.output_dir
        / "qualitative_regressions.jsonl",
        qualitative["regressions"],
    )

    write_jsonl(
        args.output_dir
        / "qualitative_failures.jsonl",
        qualitative["both_wrong"],
    )

    # --------------------------------------------------------
    # PAIRED MATRIX
    # --------------------------------------------------------

    matrix, labels = build_relation_matrix(
        pretrained_rows,
        finetuned_rows,
    )

    save_confusion(
        args.output_dir
        / "relation_confusion.png",
        matrix,
        labels,
        "Telugu pretrained vs fine-tuned reasoning outcomes",
    )

    write_json(
        args.output_dir
        / "relation_confusion.json",
        {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    save_accuracy_comparison(
        args.output_dir
        / "accuracy_comparison.png",
        pretrained_accuracy,
        finetuned_accuracy,
    )

    save_task_accuracy_comparison(
        args.output_dir
        / "task_accuracy_comparison.png",
        task_metrics,
    )

    save_answer_length_distribution(
        args.output_dir
        / "answer_length_distribution.png",
        pretrained_rows,
        finetuned_rows,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = f"""
TELUGU PHASE 3 REASONING EVALUATION
====================================

Test examples: {len(examples):,}

Pretrained:
    Correct:  {sum(r["correct"] for r in pretrained_rows):,}
    Accuracy: {pretrained_accuracy:.2f}%
    Mean answer-token accuracy:
              {mean_token_accuracy(pretrained_rows):.2f}%

Fine-tuned:
    Correct:  {sum(r["correct"] for r in finetuned_rows):,}
    Accuracy: {finetuned_accuracy:.2f}%
    Mean answer-token accuracy:
              {mean_token_accuracy(finetuned_rows):.2f}%

Improvement:
    Absolute: {finetuned_accuracy - pretrained_accuracy:.2f}
              percentage points
    Relative: {
        f"{relative_improvement:.2f}%"
        if relative_improvement is not None
        else "N/A"
    }

Paired outcomes:
    Both correct:          {paired_outcomes["both_correct"]:,}
    Pretrained only:       {paired_outcomes["pretrained_only_correct"]:,}
    Fine-tuned only:       {paired_outcomes["finetuned_only_correct"]:,}
    Both wrong:            {paired_outcomes["both_wrong"]:,}

Primary metric:
    Exact-match reasoning accuracy on the held-out test set.
""".strip()

    (args.output_dir / "summary.txt").write_text(
        summary + "\n",
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(summary)

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()