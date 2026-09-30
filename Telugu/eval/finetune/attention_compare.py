"""
TELUGU — PHASE 3 ATTENTION COMPARISON

Compares Phase 2 pretrained and Phase 3 fine-tuned Telugu attention
on held-out reasoning TEST examples.

The analysis reports:
    - early-layer attention
    - late-layer attention
    - attention entropy
    - mean attention distance
    - head-level statistics
    - representative heatmaps
    - fine-tuned minus pretrained attention differences

Run from the project root ONLY:

    python -m Telugu.eval.finetune.attention_compare \
        --pretrained Telugu/checkpoints/final.pt \
        --finetuned Telugu/finetune/final.pt \
        --tokenizer Telugu/data/tokenizer/telugu_bpe_6000.model \
        --test Telugu/reasoning/data/test.jsonl \
        --output-dir Telugu/results/finetune
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch

from Telugu.config.model_config import ModelConfig
from Telugu.model.lm import TeluguLanguageModel


DEFAULT_NUM_EXAMPLES = 50
DEFAULT_SEED = 42


# ============================================================
# FILE UTILITIES
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write_json(
    path: Path,
    obj: Any,
) -> None:

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


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(
    path: Path,
) -> spm.SentencePieceProcessor:

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.Load(
        str(path)
    ):
        raise RuntimeError(
            f"Could not load tokenizer: {path}"
        )

    return tokenizer


# ============================================================
# MODEL
# ============================================================

def build_model(
    checkpoint_path: Path,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
) -> LanguageModel:

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            f"Unsupported checkpoint: "
            f"{checkpoint_path}"
        )

    saved_config = checkpoint.get(
        "model_config"
    )

    if isinstance(
        saved_config,
        dict,
    ):
        config = ModelConfig(
            **saved_config
        )
    else:
        config = ModelConfig()

    pad_id = tokenizer.pad_id()

    if pad_id >= 0:
        config.pad_token_id = int(
            pad_id
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
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# ATTENTION COLLECTION
# ============================================================

@torch.no_grad()
def collect_attention(
    model: TeluguLanguageModel,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run one prompt through the Telugu model and collect the attention
    probabilities returned by the model's existing Phase 2 attention API.

    The actual Telugu implementation exposes attention through:

        model(
            input_ids,
            targets=None,
            return_attention=True
        )

    which returns:

        logits, loss, attention_weights

    where attention_weights is a list containing one tensor per
    Transformer layer, each with shape:

        [batch, heads, sequence, sequence]

    No hooks or guessed module attributes are used.
    """

    # --------------------------------------------------------
    # Tokenize prompt.
    # --------------------------------------------------------

    input_ids = tokenizer.EncodeAsIds(prompt)

    if not input_ids:
        raise RuntimeError(
            "Prompt produced zero tokens."
        )

    # --------------------------------------------------------
    # Respect the model context length.
    # --------------------------------------------------------

    max_seq_len = int(
        getattr(
            model.config,
            "max_seq_len",
            512,
        )
    )

    input_ids = input_ids[-max_seq_len:]

    x = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Use the model's actual Phase 2 attention interface.
    # --------------------------------------------------------

    logits, loss, attention_weights = model(
        x,
        targets=None,
        return_attention=True,
    )

    if attention_weights is None:
        raise RuntimeError(
            "The Telugu model returned no attention weights when "
            "return_attention=True was requested."
        )

    if not isinstance(attention_weights, (list, tuple)):
        raise RuntimeError(
            "Unexpected Telugu attention output type: "
            f"{type(attention_weights).__name__}. "
            "Expected a list/tuple containing one tensor per layer."
        )

    # --------------------------------------------------------
    # Convert every layer to:
    #
    # [heads, sequence, sequence]
    #
    # by removing the batch dimension.
    # --------------------------------------------------------

    layer_attentions: list[np.ndarray] = []

    for layer_idx, attention in enumerate(attention_weights):

        if attention is None:
            raise RuntimeError(
                f"Layer {layer_idx} returned None attention weights."
            )

        if not torch.is_tensor(attention):
            raise RuntimeError(
                f"Layer {layer_idx} returned attention of type "
                f"{type(attention).__name__}, expected torch.Tensor."
            )

        attention = (
            attention
            .detach()
            .float()
            .cpu()
        )

        if attention.dim() == 4:
            # [batch, heads, seq, seq]
            if attention.shape[0] != 1:
                raise RuntimeError(
                    f"Expected batch size 1 for attention collection, "
                    f"got {attention.shape[0]} at layer {layer_idx}."
                )
            attention = attention[0]

        elif attention.dim() != 3:
            raise RuntimeError(
                f"Unexpected attention shape at layer {layer_idx}: "
                f"{tuple(attention.shape)}. Expected "
                "[batch, heads, seq, seq] or [heads, seq, seq]."
            )

        # ----------------------------------------------------
        # Validate the attention matrix.
        # ----------------------------------------------------

        if attention.shape[-1] != attention.shape[-2]:
            raise RuntimeError(
                f"Attention matrix at layer {layer_idx} is not square: "
                f"{tuple(attention.shape)}."
            )

        layer_attentions.append(
            attention.numpy()
        )

    if not layer_attentions:
        raise RuntimeError(
            "The Telugu model returned an empty attention list."
        )

    return {
        "input_ids": input_ids,

        "tokens": [
            tokenizer.IdToPiece(int(token_id))
            for token_id in input_ids
        ],

        "attention": layer_attentions,
    }


# ============================================================
# ATTENTION METRICS
# ============================================================

def attention_entropy(
    attention: np.ndarray,
) -> float:

    probabilities = np.clip(
        attention.astype(
            np.float64
        ),
        1e-12,
        1.0,
    )

    return float(
        -(
            probabilities
            * np.log(
                probabilities
            )
        )
        .sum(
            axis=-1
        )
        .mean()
    )


def attention_distance(
    attention: np.ndarray,
) -> float:
    """
    Mean absolute token distance weighted by attention.

    attention shape:
        [heads, sequence, sequence]
    """

    _, sequence_length, _ = (
        attention.shape
    )

    positions = np.arange(
        sequence_length
    )

    distances = np.abs(
        positions[:, None]
        - positions[None, :]
    )

    return float(
        (
            attention
            * distances[
                None,
                :,
                :
            ]
        )
        .sum(
            axis=-1
        )
        .mean()
    )


def aggregate_attention(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    layer_stats = defaultdict(list)

    for record in records:

        for layer_idx, attention in enumerate(
            record["attention"]
        ):

            layer_stats[
                layer_idx
            ].append(
                {
                    "entropy":
                        attention_entropy(
                            attention
                        ),
                    "distance":
                        attention_distance(
                            attention
                        ),
                    "heads":
                        int(
                            attention.shape[0]
                        ),
                    "sequence_length":
                        int(
                            attention.shape[1]
                        ),
                }
            )

    output = {}

    for layer_idx, values in sorted(
        layer_stats.items()
    ):

        output[str(layer_idx)] = {
            "mean_entropy":
                float(
                    np.mean(
                        [
                            v["entropy"]
                            for v in values
                        ]
                    )
                ),

            "mean_attention_distance":
                float(
                    np.mean(
                        [
                            v["distance"]
                            for v in values
                        ]
                    )
                ),

            "num_examples":
                len(values),

            "num_heads":
                values[0]["heads"],

            "mean_sequence_length":
                float(
                    np.mean(
                        [
                            v["sequence_length"]
                            for v in values
                        ]
                    )
                ),
        }

    return output


# ============================================================
# HEATMAP
# ============================================================

def save_heatmap(
    path: Path,
    attention: np.ndarray,
    tokens: list[str],
    title: str,
) -> None:

    # Average attention over heads.
    matrix = attention.mean(
        axis=0
    )

    size = max(
        8,
        len(tokens) * 0.18,
    )

    fig, ax = plt.subplots(
        figsize=(
            size,
            size,
        )
    )

    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    if len(tokens) <= 60:

        ax.set_xticks(
            np.arange(
                len(tokens)
            )
        )

        ax.set_yticks(
            np.arange(
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

    ax.set_xlabel(
        "Key token"
    )

    ax.set_ylabel(
        "Query token"
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# LAYER METRIC PLOTS
# ============================================================

def save_metric_plot(
    path: Path,
    pretrained: dict[str, Any],
    finetuned: dict[str, Any],
    metric: str,
    ylabel: str,
    title: str,
) -> None:

    common_layers = sorted(
        set(
            pretrained.keys()
        )
        & set(
            finetuned.keys()
        ),
        key=int,
    )

    if not common_layers:
        return

    x = np.arange(
        len(common_layers)
    )

    pretrained_values = [
        pretrained[layer][metric]
        for layer in common_layers
    ]

    finetuned_values = [
        finetuned[layer][metric]
        for layer in common_layers
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        x,
        pretrained_values,
        marker="o",
        label="Pretrained",
    )

    ax.plot(
        x,
        finetuned_values,
        marker="o",
        label="Fine-tuned",
    )

    ax.set_xlabel(
        "Layer"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        common_layers
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
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

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
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "TELUGU — PHASE 3 ATTENTION COMPARISON"
    )
    print("=" * 80)

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

    examples = [
        example
        for example in examples
        if example.get("split")
        in (
            None,
            "test",
        )
    ]

    if not examples:
        raise ValueError(
            "No test examples found."
        )

    rng = random.Random(
        args.seed
    )

    if (
        len(examples)
        > args.num_examples
    ):

        examples = rng.sample(
            examples,
            args.num_examples,
        )

    tokenizer = load_tokenizer(
        args.tokenizer
    )

    pretrained_model = build_model(
        args.pretrained,
        tokenizer,
        device,
    )

    finetuned_model = build_model(
        args.finetuned,
        tokenizer,
        device,
    )

    print(
        f"Attention test examples: "
        f"{len(examples):,}"
    )

    pretrained_records = []
    finetuned_records = []

    # --------------------------------------------------------
    # COLLECT ATTENTION
    # --------------------------------------------------------

    for idx, example in enumerate(
        examples,
        1,
    ):

        prompt = str(
            example["prompt"]
        )

        try:

            pretrained = collect_attention(
                pretrained_model,
                tokenizer,
                prompt,
                device,
            )

            finetuned = collect_attention(
                finetuned_model,
                tokenizer,
                prompt,
                device,
            )

            pretrained_records.append(
                {
                    "id":
                        example.get("id"),
                    "task_type":
                        example.get(
                            "task_type"
                        ),
                    "prompt":
                        prompt,
                    **pretrained,
                }
            )

            finetuned_records.append(
                {
                    "id":
                        example.get("id"),
                    "task_type":
                        example.get(
                            "task_type"
                        ),
                    "prompt":
                        prompt,
                    **finetuned,
                }
            )

        except RuntimeError as exc:

            print(
                f"Skipping "
                f"{example.get('id')}: "
                f"{exc}"
            )

        if (
            idx == 1
            or idx % 10 == 0
            or idx == len(examples)
        ):
            print(
                f"{idx:4d}/"
                f"{len(examples):4d}"
            )

    if not pretrained_records:
        raise RuntimeError(
            "No attention records were collected."
        )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    pretrained_summary = aggregate_attention(
        pretrained_records
    )

    finetuned_summary = aggregate_attention(
        finetuned_records
    )

    write_json(
        args.output_dir
        / "attention_comparison.json",
        {
            "num_examples_requested":
                args.num_examples,
            "num_examples_evaluated":
                len(pretrained_records),
            "pretrained":
                pretrained_summary,
            "finetuned":
                finetuned_summary,
        },
    )

    # --------------------------------------------------------
    # HEAD-LEVEL METRICS
    # --------------------------------------------------------

    head_rows = []

    for model_name, records in (
        (
            "pretrained",
            pretrained_records,
        ),
        (
            "finetuned",
            finetuned_records,
        ),
    ):

        per_head = defaultdict(list)

        for record in records:

            for layer_idx, attention in enumerate(
                record["attention"]
            ):

                for head_idx in range(
                    attention.shape[0]
                ):

                    head_attention = (
                        attention[
                            head_idx:
                            head_idx + 1
                        ]
                    )

                    per_head[
                        (
                            layer_idx,
                            head_idx,
                        )
                    ].append(
                        {
                            "entropy":
                                attention_entropy(
                                    head_attention
                                ),
                            "distance":
                                attention_distance(
                                    head_attention
                                ),
                        }
                    )

        for (
            layer_idx,
            head_idx,
        ), values in sorted(
            per_head.items()
        ):

            head_rows.append(
                {
                    "model":
                        model_name,
                    "layer":
                        layer_idx,
                    "head":
                        head_idx,
                    "mean_entropy":
                        float(
                            np.mean(
                                [
                                    v["entropy"]
                                    for v in values
                                ]
                            )
                        ),
                    "mean_attention_distance":
                        float(
                            np.mean(
                                [
                                    v["distance"]
                                    for v in values
                                ]
                            )
                        ),
                    "num_examples":
                        len(values),
                }
            )

    write_json(
        args.output_dir
        / "attention_head_metrics.json",
        head_rows,
    )

    # --------------------------------------------------------
    # METRIC PLOTS
    # --------------------------------------------------------

    save_metric_plot(
        args.output_dir
        / "attention_entropy_comparison.png",
        pretrained_summary,
        finetuned_summary,
        "mean_entropy",
        "Attention entropy",
        "Telugu attention entropy: pretrained vs fine-tuned",
    )

    save_metric_plot(
        args.output_dir
        / "attention_distance_comparison.png",
        pretrained_summary,
        finetuned_summary,
        "mean_attention_distance",
        "Mean attention distance (tokens)",
        "Telugu attention distance: pretrained vs fine-tuned",
    )

    # --------------------------------------------------------
    # REPRESENTATIVE EXAMPLE
    # --------------------------------------------------------

    pretrained_rep = (
        pretrained_records[0]
    )

    finetuned_rep = (
        finetuned_records[0]
    )

    write_json(
        args.output_dir
        / "representative_prompt.json",
        {
            "id":
                pretrained_rep["id"],
            "task_type":
                pretrained_rep[
                    "task_type"
                ],
            "prompt":
                pretrained_rep[
                    "prompt"
                ],
            "tokens":
                pretrained_rep[
                    "tokens"
                ],
        },
    )

    n_layers = min(
        len(
            pretrained_rep[
                "attention"
            ]
        ),
        len(
            finetuned_rep[
                "attention"
            ]
        ),
    )

    early_layer = 0
    late_layer = max(
        0,
        n_layers - 1,
    )

    # --------------------------------------------------------
    # EARLY/LATE HEATMAPS
    # --------------------------------------------------------

    save_heatmap(
        args.output_dir
        / "attention_pretrained_early_layer.png",
        pretrained_rep[
            "attention"
        ][early_layer],
        pretrained_rep[
            "tokens"
        ],
        (
            "Telugu pretrained — "
            f"early layer {early_layer}"
        ),
    )

    save_heatmap(
        args.output_dir
        / "attention_finetuned_early_layer.png",
        finetuned_rep[
            "attention"
        ][early_layer],
        finetuned_rep[
            "tokens"
        ],
        (
            "Telugu fine-tuned — "
            f"early layer {early_layer}"
        ),
    )

    save_heatmap(
        args.output_dir
        / "attention_pretrained_late_layer.png",
        pretrained_rep[
            "attention"
        ][late_layer],
        pretrained_rep[
            "tokens"
        ],
        (
            "Telugu pretrained — "
            f"late layer {late_layer}"
        ),
    )

    save_heatmap(
        args.output_dir
        / "attention_finetuned_late_layer.png",
        finetuned_rep[
            "attention"
        ][late_layer],
        finetuned_rep[
            "tokens"
        ],
        (
            "Telugu fine-tuned — "
            f"late layer {late_layer}"
        ),
    )

    # --------------------------------------------------------
    # DIFFERENCE HEATMAPS
    # --------------------------------------------------------

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

        pretrained_attention = (
            pretrained_rep[
                "attention"
            ][layer]
            .mean(axis=0)
        )

        finetuned_attention = (
            finetuned_rep[
                "attention"
            ][layer]
            .mean(axis=0)
        )

        difference = (
            finetuned_attention
            - pretrained_attention
        )

        tokens = pretrained_rep[
            "tokens"
        ]

        size = max(
            8,
            len(tokens) * 0.18,
        )

        fig, ax = plt.subplots(
            figsize=(
                size,
                size,
            )
        )

        ax.imshow(
            difference,
            aspect="auto",
            interpolation="nearest",
        )

        if len(tokens) <= 60:

            ax.set_xticks(
                np.arange(
                    len(tokens)
                )
            )

            ax.set_yticks(
                np.arange(
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

        ax.set_xlabel(
            "Key token"
        )

        ax.set_ylabel(
            "Query token"
        )

        ax.set_title(
            "Telugu fine-tuned − pretrained "
            f"attention — {name}"
        )

        fig.tight_layout()

        fig.savefig(
            args.output_dir
            / f"attention_difference_{name}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    print("\n" + "=" * 80)
    print(
        "TELUGU PHASE 3 ATTENTION COMPARISON COMPLETE"
    )
    print("=" * 80)

    print(
        f"Examples analyzed: "
        f"{len(pretrained_records):,}"
    )

    print(
        f"Results: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()