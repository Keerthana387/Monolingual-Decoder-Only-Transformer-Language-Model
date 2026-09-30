"""
TELUGU — PHASE 3 PRETRAINING RETENTION EVALUATION

Evaluates each Phase 3 fine-tuned checkpoint on the ORIGINAL Telugu
held-out language-modeling test set.

Purpose:
    Measure retention / catastrophic forgetting after reasoning fine-tuning.

This is intentionally NOT a second reasoning evaluation. The reasoning
test answers "did the model learn reasoning?", while this test answers
"how much of its original language-modeling ability was retained?"

Metrics:
    - cross-entropy loss
    - perplexity
    - BPB
    - BLEU-4
    - chrF
    - ROUGE-L
    - repetition rate
    - Distinct-1 / Distinct-2
    - generated samples at greedy, T=0.5, 1.0, 1.5

Input format:
    The evaluator reads the project's original raw uint16 tokenized held-out
    test shard directly (test/test_000.bin). The same token IDs are used for
    the pretrained baseline and every fine-tuned run.

Run from project root:
    python -m Telugu.eval.finetune.pretrain_eval_mixed \
        --pretrained Telugu/checkpoints/final.pt \
        --finetune-root Telugu/finetune/mixed_data_size \
        --tokenizer Telugu/data/tokenizer/telugu_bpe_6000.model \
        --test-bin /content/LMA_IP_DATA/train/test/test_000.bin \
        --output-root Telugu/results/finetune
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Telugu.config.model_config import ModelConfig
from Telugu.model.lm import TeluguLanguageModel


RUN_ORDER = [
    ("reasoning_2k_real_18k", 2000, 18000),
    ("reasoning_5k_real_15k", 5000, 15000),
    ("reasoning_8k_real_12k", 8000, 12000),
    ("reasoning_12k_real_8k", 12000, 8000),
    ("reasoning_16k_real_4k", 16000, 4000),
    ("reasoning_20k_real_0k", 20000, 0),
]



def parse_args() -> argparse.Namespace:
    """Parse LM-retention evaluation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Telugu Phase 2/3 checkpoints on the original "
            "tokenized held-out LM test set."
        )
    )
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--finetune-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--test-bin",
        required=True,
        type=Path,
        help="Tokenized held-out test file, e.g. test/test_000.bin",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--eval-sequences",
        type=int,
        default=2000,
        help="Maximum number of 512-token evaluation sequences.",
    )
    parser.add_argument("--generation-samples", type=int, default=100)
    parser.add_argument("--prefix-ratio", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run names. Default: all six runs.",
    )
    return parser.parse_args()


def load_token_ids(path: Path, vocab_size: int) -> np.ndarray:
    """
    Load the project's raw uint16 tokenized held-out test file.

    The Telugu tokenized shards are raw token-ID binaries, so no text
    conversion is required before language-model evaluation.
    """
    if path.suffix.lower() != ".bin":
        raise ValueError(f"Expected a .bin token file, got: {path}")

    ids = np.memmap(path, dtype=np.uint16, mode="r")
    if ids.size == 0:
        raise ValueError(f"Token file is empty: {path}")

    max_id = int(ids.max())
    if max_id >= vocab_size:
        raise ValueError(
            f"Token ID {max_id} is outside vocabulary size {vocab_size}."
        )

    return ids


def load_tokenizer(path: Path):
    """Load the existing student-trained Telugu SentencePiece tokenizer."""
    import sentencepiece as spm

    tok = spm.SentencePieceProcessor()
    if not tok.Load(str(path)):
        raise RuntimeError(f"Could not load tokenizer: {path}")
    return tok


def build_model(
    checkpoint_path: Path,
    tokenizer,
    device: torch.device,
) -> TeluguLanguageModel:
    """
    Build the exact student Telugu model architecture and load its checkpoint.

    Phase 3 checkpoints store model weights under `model_state_dict` and the
    Phase 3 configuration separately. The architecture itself is recreated
    using the project's ModelConfig defaults, exactly as the fine-tuning code
    does.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")

    config = ModelConfig()

    if hasattr(config, "vocab_size"):
        config.vocab_size = int(tokenizer.GetPieceSize())

    if hasattr(config, "max_seq_len"):
        config.max_seq_len = int(getattr(config, "max_seq_len", 512))

    model = TeluguLanguageModel(config).to(device)

    state = checkpoint.get("model_state_dict")
    if state is None:
        state = checkpoint.get("model_state_dicts")
    if state is None:
        raise RuntimeError(
            f"{checkpoint_path} does not contain 'model_state_dict'."
        )

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_eval_sequences(
    token_ids: np.ndarray,
    max_seq_len: int,
    max_sequences: int,
) -> np.ndarray:
    """
    Create deterministic non-overlapping causal-LM sequences from the .bin file.

    Each returned row contains max_seq_len + 1 tokens, allowing the first
    max_seq_len tokens to predict the following token at every position.
    """
    if max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive.")
    if max_sequences <= 0:
        raise ValueError("--eval-sequences must be positive.")

    usable = min(
        max_sequences * max_seq_len,
        max(0, token_ids.size - 1),
    )

    if usable < 2:
        raise RuntimeError("Held-out test file does not contain enough tokens.")

    rows = []
    start = 0

    while start + 1 < token_ids.size and len(rows) < max_sequences:
        end = min(start + max_seq_len + 1, token_ids.size)
        chunk = np.asarray(token_ids[start:end], dtype=np.int64)

        if chunk.size >= 2:
            rows.append(chunk)

        start += max_seq_len

    if not rows:
        raise RuntimeError("Could not create evaluation sequences.")

    return np.asarray(rows, dtype=object)


@torch.no_grad()
def language_model_loss(
    model: TeluguLanguageModel,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[float, int]:
    """
    Compute token-weighted causal-LM cross-entropy over held-out .bin tokens.
    """
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start:start + batch_size]

        max_len = max(len(x) for x in batch_sequences)
        x_batch = np.zeros((len(batch_sequences), max_len - 1), dtype=np.int64)
        y_batch = np.zeros((len(batch_sequences), max_len - 1), dtype=np.int64)

        for i, seq in enumerate(batch_sequences):
            x_batch[i, :len(seq) - 1] = seq[:-1]
            y_batch[i, :len(seq) - 1] = seq[1:]

        x = torch.tensor(x_batch, dtype=torch.long, device=device)
        y = torch.tensor(y_batch, dtype=torch.long, device=device)

        # All sequences are full-length except possibly the final one.
        valid_lengths = torch.tensor(
            [len(seq) - 1 for seq in batch_sequences],
            dtype=torch.long,
            device=device,
        )

        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output

        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape(y.shape)

        mask = (
            torch.arange(y.shape[1], device=device)[None, :]
            < valid_lengths[:, None]
        )

        total_nll += float(losses.masked_select(mask).sum().item())
        total_tokens += int(mask.sum().item())

    if total_tokens == 0:
        raise RuntimeError("No valid tokens were available for LM evaluation.")

    return total_nll / total_tokens, total_tokens


@torch.no_grad()
def generate(
    model: TeluguLanguageModel,
    tokenizer,
    prefix_ids: list[int],
    device: torch.device,
    temperature: float,
    max_new_tokens: int,
) -> list[int]:
    """Generate token IDs after a held-out token prefix."""
    if not prefix_ids:
        return []

    context_limit = int(getattr(model.config, "max_seq_len", 512))
    x = torch.tensor(
        [prefix_ids[-context_limit:]],
        dtype=torch.long,
        device=device,
    )

    eos = tokenizer.eos_id()
    generated = []

    for _ in range(max_new_tokens):
        output = model(x[:, -context_limit:])
        logits = output[0] if isinstance(output, tuple) else output
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )
        else:
            probs = torch.softmax(
                next_logits / temperature,
                dim=-1,
            )
            next_id = torch.multinomial(
                probs,
                num_samples=1,
            )

        token_id = int(next_id.item())
        generated.append(token_id)

        x = torch.cat([x, next_id], dim=1)

        if eos >= 0 and token_id == eos:
            break

    return generated


def corpus_bleu4(references: list[str], hypotheses: list[str]) -> float:
    """Compute corpus-level BLEU-4 with clipped n-gram precision."""
    if not references or not hypotheses:
        return 0.0

    clipped_totals = [0, 0, 0, 0]
    possible_totals = [0, 0, 0, 0]
    ref_len = 0
    hyp_len = 0

    for ref_text, hyp_text in zip(references, hypotheses):
        ref = ref_text.split()
        hyp = hyp_text.split()

        ref_len += len(ref)
        hyp_len += len(hyp)

        for n in range(1, 5):
            ref_counts = Counter(
                tuple(ref[i:i+n])
                for i in range(max(0, len(ref) - n + 1))
            )
            hyp_counts = Counter(
                tuple(hyp[i:i+n])
                for i in range(max(0, len(hyp) - n + 1))
            )

            clipped_totals[n - 1] += sum(
                min(count, ref_counts[gram])
                for gram, count in hyp_counts.items()
            )
            possible_totals[n - 1] += sum(hyp_counts.values())

    precisions = [
        clipped_totals[i] / max(1, possible_totals[i])
        for i in range(4)
    ]

    if any(p == 0.0 for p in precisions):
        return 0.0

    geo = math.exp(
        sum(math.log(p) for p in precisions) / 4.0
    )

    if hyp_len == 0:
        return 0.0

    bp = (
        1.0
        if hyp_len > ref_len
        else math.exp(1.0 - ref_len / max(1, hyp_len))
    )

    return 100.0 * bp * geo


def chrf(
    reference: str,
    hypothesis: str,
    order: int = 6,
) -> float:
    """Compute character n-gram F-score."""
    def grams(text: str, n: int) -> Counter:
        text = text.strip()
        return Counter(
            text[i:i+n]
            for i in range(max(0, len(text) - n + 1))
        )

    scores = []

    for n in range(1, order + 1):
        r = grams(reference, n)
        h = grams(hypothesis, n)

        overlap = sum(
            min(v, h[g])
            for g, v in r.items()
        )

        precision = overlap / max(1, sum(h.values()))
        recall = overlap / max(1, sum(r.values()))

        score = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

        scores.append(score)

    return 100.0 * float(np.mean(scores))


def lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the longest common subsequence."""
    prev = [0] * (len(b) + 1)

    for x in a:
        cur = [0]

        for j, y in enumerate(b, 1):
            cur.append(
                prev[j - 1] + 1
                if x == y
                else max(prev[j], cur[-1])
            )

        prev = cur

    return prev[-1]


def rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 over whitespace tokens."""
    ref = reference.split()
    hyp = hypothesis.split()

    if not ref or not hyp:
        return 0.0

    lcs = lcs_length(ref, hyp)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)

    return (
        0.0
        if precision + recall == 0
        else 100.0 * 2 * precision * recall / (precision + recall)
    )


def repetition_rate(tokens: list[str], n: int = 3) -> float:
    """Return the fraction of n-gram occurrences that are repeats."""
    if len(tokens) < n:
        return 0.0

    grams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(tokens: list[str], n: int) -> float:
    """Return unique n-grams divided by all n-grams."""
    if len(tokens) < n:
        return 0.0

    grams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    return len(set(grams)) / len(grams)


def evaluate_generation(
    model: TeluguLanguageModel,
    tokenizer,
    sequences: np.ndarray,
    device: torch.device,
    samples: int,
    prefix_ratio: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Generate held-out continuations at greedy and T=0.5/1.0/1.5."""
    if not 0.0 < prefix_ratio < 1.0:
        raise ValueError("--prefix-ratio must be between 0 and 1.")

    rng = random.Random(seed)
    count = min(samples, len(sequences))
    indices = rng.sample(range(len(sequences)), count)

    temperatures = {
        "greedy": 0.0,
        "temperature_0.5": 0.5,
        "temperature_1.0": 1.0,
        "temperature_1.5": 1.5,
    }

    all_rows = []
    metric_rows = []

    for sample_index, seq_index in enumerate(indices):
        seq = sequences[seq_index].tolist()

        cut = max(
            1,
            min(
                len(seq) - 1,
                int(len(seq) * prefix_ratio),
            ),
        )

        prefix_ids = [int(x) for x in seq[:cut]]
        reference_ids = [int(x) for x in seq[cut:]]

        prefix_text = tokenizer.DecodeIds(prefix_ids).strip()
        reference_text = tokenizer.DecodeIds(reference_ids).strip()

        for label, temperature in temperatures.items():
            generated_ids = generate(
                model,
                tokenizer,
                prefix_ids,
                device,
                temperature,
                max_new_tokens,
            )

            generated_text = tokenizer.DecodeIds(
                generated_ids
            ).strip()

            all_rows.append({
                "sample_index": sample_index,
                "temperature": label,
                "prefix": prefix_text,
                "reference_continuation": reference_text,
                "generated_continuation": generated_text,
            })

            tokens = generated_text.split()

            metric_rows.append({
                "bleu4": 0.0,
                "chrf": chrf(reference_text, generated_text),
                "rouge_l": rouge_l(
                    reference_text,
                    generated_text,
                ),
                "repetition_rate": repetition_rate(tokens, 3),
                "distinct_1": distinct_n(tokens, 1),
                "distinct_2": distinct_n(tokens, 2),
            })

    summary = {}

    for label in temperatures:
        rows = [
            (r, m)
            for r, m in zip(all_rows, metric_rows)
            if r["temperature"] == label
        ]

        if not rows:
            continue

        refs = [
            r["reference_continuation"]
            for r, _ in rows
        ]
        hyps = [
            r["generated_continuation"]
            for r, _ in rows
        ]

        summary[label] = {
            "samples": len(rows),
            "BLEU4_corpus": corpus_bleu4(refs, hyps),
            "chrF_mean": float(
                np.mean([m["chrf"] for _, m in rows])
            ),
            "ROUGE_L_mean": float(
                np.mean([m["rouge_l"] for _, m in rows])
            ),
            "repetition_rate_mean": float(
                np.mean(
                    [m["repetition_rate"] for _, m in rows]
                )
            ),
            "distinct_1_mean": float(
                np.mean(
                    [m["distinct_1"] for _, m in rows]
                )
            ),
            "distinct_2_mean": float(
                np.mean(
                    [m["distinct_2"] for _, m in rows]
                )
            ),
        }

    return summary, all_rows


def utf8_bytes_for_sequences(
    tokenizer,
    sequences: np.ndarray,
) -> int:
    """Calculate UTF-8 bytes represented by the evaluated token sequences."""
    total = 0

    for seq in sequences:
        text = tokenizer.DecodeIds(
            [int(x) for x in seq]
        )
        total += len(text.encode("utf-8"))

    return total


def write_summary_plots(
    output_root: Path,
    summary_rows: list[dict],
    pretrained_ppl: float,
    pretrained_bpb: float,
) -> None:
    """Save required mixture-level held-out LM retention plots."""
    if not summary_rows:
        return

    xs = [
        r["reasoning_fraction_of_examples"]
        for r in summary_rows
    ]

    def plot(
        filename: str,
        ys: list[float],
        baseline: float,
        ylabel: str,
        title: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            xs,
            ys,
            marker="o",
            label="Fine-tuned",
        )
        ax.axhline(
            baseline,
            linestyle="--",
            label="Pretrained baseline",
        )
        ax.set_title(title)
        ax.set_xlabel(
            "Reasoning fraction of training examples"
        )
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_root / filename,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    plot(
        "mixture_pretrain_ppl.png",
        [r["ppl"] for r in summary_rows],
        pretrained_ppl,
        "Perplexity",
        "Telugu held-out LM perplexity after reasoning fine-tuning",
    )

    plot(
        "mixture_pretrain_bpb.png",
        [r["bpb"] for r in summary_rows],
        pretrained_bpb,
        "Bits per byte",
        "Telugu held-out LM BPB after reasoning fine-tuning",
    )


def main() -> None:
    """Evaluate pretrained and fine-tuned Telugu checkpoints."""
    args = parse_args()

    required = [
        args.pretrained,
        args.finetune_root,
        args.tokenizer,
        args.test_bin,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.GetPieceSize()

    print("=" * 80)
    print("TELUGU — PHASE 3 PRETRAINING RETENTION EVALUATION")
    print("=" * 80)
    print(f"\nHeld-out token file: {args.test_bin}")
    print(f"Tokenizer vocabulary: {vocab_size:,}")

    token_ids = load_token_ids(
        args.test_bin,
        vocab_size,
    )

    print(f"Held-out tokens available: {token_ids.size:,}")

    sequences = make_eval_sequences(
        token_ids,
        args.max_seq_len,
        args.eval_sequences,
    )

    print(f"Evaluation sequences: {len(sequences):,}")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    evaluated_bytes = utf8_bytes_for_sequences(
        tokenizer,
        sequences,
    )

    # -------------------------------------------------------------------------
    # Pretrained baseline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PRETRAINED BASELINE")
    print("=" * 80)

    pretrained_model = build_model(
        args.pretrained,
        tokenizer,
        device,
    )

    pre_loss, pre_tokens = language_model_loss(
        pretrained_model,
        sequences,
        device,
        args.batch_size,
    )

    pre_ppl = math.exp(
        min(pre_loss, 20.0)
    )

    pre_bpb = (
        pre_loss
        * pre_tokens
        / math.log(2.0)
        / max(1, evaluated_bytes)
    )

    pre_gen_metrics, pre_samples = evaluate_generation(
        pretrained_model,
        tokenizer,
        sequences,
        device,
        args.generation_samples,
        args.prefix_ratio,
        args.max_new_tokens,
        args.seed,
    )

    base_out = (
        args.output_root
        / "pretrain_test"
        / "pretrained"
    )

    write_json(
        base_out / "metrics.json",
        {
            "checkpoint": str(args.pretrained),
            "test_bin": str(args.test_bin),
            "loss": pre_loss,
            "ppl": pre_ppl,
            "bpb": pre_bpb,
            "evaluated_tokens": pre_tokens,
            "evaluated_sequences": len(sequences),
            "evaluated_utf8_bytes": evaluated_bytes,
            "generation": pre_gen_metrics,
        },
    )

    write_jsonl(
        base_out / "generated_samples.jsonl",
        pre_samples,
    )

    del pretrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selected = (
        set(args.runs)
        if args.runs
        else None
    )

    summary_rows = []

    for run_name, reasoning_n, real_n in RUN_ORDER:
        if selected is not None and run_name not in selected:
            continue

        checkpoint = (
            args.finetune_root
            / run_name
            / "final.pt"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        print("\n" + "=" * 80)
        print(f"LM RETENTION: {run_name}")
        print("=" * 80)

        model = build_model(
            checkpoint,
            tokenizer,
            device,
        )

        loss, tokens = language_model_loss(
            model,
            sequences,
            device,
            args.batch_size,
        )

        ppl = math.exp(
            min(loss, 20.0)
        )

        bpb = (
            loss
            * tokens
            / math.log(2.0)
            / max(1, evaluated_bytes)
        )

        gen_metrics, samples = evaluate_generation(
            model,
            tokenizer,
            sequences,
            device,
            args.generation_samples,
            args.prefix_ratio,
            args.max_new_tokens,
            args.seed,
        )

        out = (
            args.output_root
            / run_name
            / "pretrain_test"
        )

        write_json(
            out / "metrics.json",
            {
                "checkpoint": str(checkpoint),
                "test_bin": str(args.test_bin),
                "loss": loss,
                "ppl": ppl,
                "bpb": bpb,
                "evaluated_tokens": tokens,
                "evaluated_sequences": len(sequences),
                "evaluated_utf8_bytes": evaluated_bytes,
                "generation": gen_metrics,
            },
        )

        write_jsonl(
            out / "generated_samples.jsonl",
            samples,
        )

        greedy = gen_metrics["greedy"]

        summary_rows.append({
            "run": run_name,
            "reasoning_examples": reasoning_n,
            "real_examples": real_n,
            "reasoning_fraction_of_examples": (
                reasoning_n / 20000.0
            ),
            "loss": loss,
            "ppl": ppl,
            "bpb": bpb,
            "BLEU4_greedy": greedy[
                "BLEU4_corpus"
            ],
            "chrF_greedy": greedy[
                "chrF_mean"
            ],
            "ROUGE_L_greedy": greedy[
                "ROUGE_L_mean"
            ],
            "repetition_greedy": greedy[
                "repetition_rate_mean"
            ],
            "distinct1_greedy": greedy[
                "distinct_1_mean"
            ],
            "distinct2_greedy": greedy[
                "distinct_2_mean"
            ],
            "delta_ppl_vs_pretrained": (
                ppl - pre_ppl
            ),
            "delta_bpb_vs_pretrained": (
                bpb - pre_bpb
            ),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pretrain_out = (
        args.output_root
        / "pretrain_test"
    )

    write_json(
        pretrain_out / "mixture_summary.json",
        {
            "held_out_token_file": str(args.test_bin),
            "pretrained_baseline": {
                "loss": pre_loss,
                "ppl": pre_ppl,
                "bpb": pre_bpb,
                "generation": pre_gen_metrics,
            },
            "runs": summary_rows,
            "evaluation_note": (
                "The held-out set is the project's original tokenized "
                "test/test_000.bin. The same token IDs are evaluated for "
                "the Phase 2 pretrained checkpoint and every Phase 3 "
                "fine-tuned checkpoint. BPB uses UTF-8 bytes obtained by "
                "decoding the evaluated token sequences with the existing "
                "Telugu tokenizer."
            ),
        },
    )

    write_csv(
        pretrain_out / "mixture_summary.csv",
        summary_rows,
    )

    write_summary_plots(
        args.output_root,
        summary_rows,
        pre_ppl,
        pre_bpb,
    )

    print("\n" + "=" * 80)
    print("PRETRAINING-RETENTION EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Baseline PPL: {pre_ppl:.4f}")
    print(f"Baseline BPB: {pre_bpb:.4f}")
    print(
        f"Results: {args.output_root}"
    )


if __name__ == "__main__":
    main()


def write_json(path: Path, obj: object) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    path: Path,
    xs: list[float],
    ys: list[float],
    baseline: float,
    ylabel: str,
    title: str,
) -> None:
    """Plot a mixture metric against reasoning-example fraction."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", label="Fine-tuned")
    ax.axhline(baseline, linestyle="--", label="Pretrained baseline")
    ax.set_title(title)
    ax.set_xlabel("Reasoning fraction of training examples")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Evaluate pretrained and all selected fine-tuned models on held-out LM text."""
    args = parse_args()

    for p in (
        args.pretrained,
        args.finetune_root,
        args.tokenizer,
        args.test_text,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    texts = load_texts(args.test_text)
    tokenizer = load_tokenizer(args.tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Evaluate the pretrained baseline once.
    pretrained_model = build_model(args.pretrained, tokenizer, device)
    pre_loss, pre_tokens = language_model_loss(
        pretrained_model,
        tokenizer,
        texts,
        device,
        args.max_seq_len,
        args.eval_sequences,
    )
    pre_ppl = math.exp(min(pre_loss, 20.0))

    # BPB = total negative log-likelihood in bits / total UTF-8 bytes
    # for the text actually evaluated.
    eval_texts = texts if len(texts) <= args.eval_sequences else random.Random(42).sample(
        texts, args.eval_sequences
    )
    eval_bytes = sum(len(t.encode("utf-8")) for t in eval_texts)
    pre_bpb = pre_loss * pre_tokens / math.log(2.0) / max(1, eval_bytes)

    pre_gen_metrics, pre_samples = evaluate_generation(
        pretrained_model,
        tokenizer,
        texts,
        device,
        args.generation_samples,
        args.prefix_ratio,
        args.max_new_tokens,
        args.seed,
    )

    base_out = args.output_root / "pretrain_test" / "pretrained"
    write_json(
        base_out / "metrics.json",
        {
            "loss": pre_loss,
            "ppl": pre_ppl,
            "bpb": pre_bpb,
            "evaluated_tokens": pre_tokens,
            "generation": pre_gen_metrics,
        },
    )
    write_jsonl(base_out / "generated_samples.jsonl", pre_samples)

    selected = set(args.runs) if args.runs else None
    summary_rows = []

    for run_name, reasoning_n, real_n in RUN_ORDER:
        if selected is not None and run_name not in selected:
            continue

        checkpoint = args.finetune_root / run_name / "final.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        print("\n" + "=" * 80)
        print(f"LM RETENTION: {run_name}")
        print("=" * 80)

        model = build_model(checkpoint, tokenizer, device)

        loss, tokens = language_model_loss(
            model,
            tokenizer,
            texts,
            device,
            args.max_seq_len,
            args.eval_sequences,
        )
        ppl = math.exp(min(loss, 20.0))

        bpb = loss * tokens / math.log(2.0) / max(1, eval_bytes)

        gen_metrics, samples = evaluate_generation(
            model,
            tokenizer,
            texts,
            device,
            args.generation_samples,
            args.prefix_ratio,
            args.max_new_tokens,
            args.seed,
        )

        out = args.output_root / run_name / "pretrain_test"
        write_json(
            out / "metrics.json",
            {
                "loss": loss,
                "ppl": ppl,
                "bpb": bpb,
                "evaluated_tokens": tokens,
                "generation": gen_metrics,
            },
        )
        write_jsonl(out / "generated_samples.jsonl", samples)

        # Use greedy generation as the primary comparable generation row.
        greedy = gen_metrics["greedy"]

        summary_rows.append({
            "run": run_name,
            "reasoning_examples": reasoning_n,
            "real_examples": real_n,
            "reasoning_fraction_of_examples": reasoning_n / 20000.0,
            "loss": loss,
            "ppl": ppl,
            "bpb": bpb,
            "BLEU4_greedy": greedy["BLEU4_corpus"],
            "chrF_greedy": greedy["chrF_mean"],
            "ROUGE_L_greedy": greedy["ROUGE_L_mean"],
            "repetition_greedy": greedy["repetition_rate_mean"],
            "distinct1_greedy": greedy["distinct_1_mean"],
            "distinct2_greedy": greedy["distinct_2_mean"],
            "delta_ppl_vs_pretrained": ppl - pre_ppl,
            "delta_bpb_vs_pretrained": bpb - pre_bpb,
        })

    write_json(
        args.output_root / "pretrain_test" / "mixture_summary.json",
        {
            "held_out_text": str(args.test_text),
            "pretrained_baseline": {
                "loss": pre_loss,
                "ppl": pre_ppl,
                "bpb": pre_bpb,
                "generation": pre_gen_metrics,
            },
            "runs": summary_rows,
            "note": (
                "This is an LM-retention test, not a reasoning test. "
                "The same held-out text is used for the pretrained baseline "
                "and every fine-tuned checkpoint."
            ),
        },
    )
    write_csv(
        args.output_root / "pretrain_test" / "mixture_summary.csv",
        summary_rows,
    )

    if summary_rows:
        xs = [r["reasoning_fraction_of_examples"] for r in summary_rows]
        plot_metric(
            args.output_root / "mixture_pretrain_ppl.png",
            xs,
            [r["ppl"] for r in summary_rows],
            pre_ppl,
            "Perplexity",
            "Telugu held-out LM perplexity after reasoning fine-tuning",
        )
        plot_metric(
            args.output_root / "mixture_pretrain_bpb.png",
            xs,
            [r["bpb"] for r in summary_rows],
            pre_bpb,
            "Bits per byte",
            "Telugu held-out LM BPB after reasoning fine-tuning",
        )

    print("\nPretraining-retention evaluation complete.")


if __name__ == "__main__":
    main()
