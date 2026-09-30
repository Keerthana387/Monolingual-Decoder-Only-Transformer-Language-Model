"""
NEPALI — PHASE 3 REASONING TEST-SET EVALUATION

Evaluates the SAME held-out synthetic reasoning test set with:
  1. Phase 2 pretrained Nepali checkpoint
  2. Phase 3 fine-tuned Nepali checkpoint

Outputs:
  - predictions.jsonl
  - metrics.json
  - task_metrics.csv
  - qualitative_successes.jsonl
  - qualitative_failures.jsonl
  - accuracy_comparison.png
  - task_accuracy_comparison.png
  - confusion_matrix.png (for relation-label tasks)
  - answer_length_distribution.png

The primary metric is exact-match accuracy on the answer string after
Unicode/whitespace normalization. Token-level answer accuracy is also
reported as a secondary diagnostic.

No training/validation examples are used for the reported metrics.
The entire test.jsonl is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Nepali.config.model_config import ModelConfig
from Nepali.model.lm import LanguageModel


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tokenizer(path: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size() != 8000:
        raise RuntimeError(f"Expected 8000-token Nepali tokenizer, got {tok.get_vocab_size()}")
    return tok


def load_checkpoint(path: Path, device: torch.device):
    config = ModelConfig()
    model = LanguageModel(config).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Normalization / generation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_eos(text: str) -> str:
    return normalize_text(text)


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int):
    prompt_ids = tokenizer.encode(prompt.strip() + "\n").ids
    if not prompt_ids:
        raise ValueError("Empty prompt after tokenization")

    # Keep the most recent context if a prompt ever exceeds the model limit.
    prompt_ids = prompt_ids[-model.config.max_seq_len:]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):
        context = ids[:, -model.config.max_seq_len:]
        logits, _ = model(input_ids=context, targets=None)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    generated_ids = ids[0, len(prompt_ids):].tolist()
    if eos_id is not None and eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]
    return normalize_text(tokenizer.decode(generated_ids))


def answer_token_accuracy(tokenizer, prediction: str, answer: str) -> float:
    pred = tokenizer.encode(prediction).ids
    gold = tokenizer.encode(answer).ids
    if not gold:
        return 1.0 if not pred else 0.0
    correct = sum(a == b for a, b in zip(pred, gold))
    return correct / max(len(pred), len(gold))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def group_metrics(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, "unknown")].append(r)
    out = {}
    for name, items in sorted(groups.items(), key=lambda x: str(x[0])):
        out[str(name)] = {
            "count": len(items),
            "exact_match_accuracy": sum(x["exact_match"] for x in items) / len(items),
            "token_accuracy": sum(x["token_accuracy"] for x in items) / len(items),
        }
    return out


def relation_label(record):
    """Best-effort relation label from generator metadata."""
    for key in ("relation", "correct_relation", "label", "answer_relation"):
        if key in record:
            return str(record[key])
    ans = normalize_text(record.get("answer", ""))
    # Nepali answer labels used by the generator.
    if "भन्दा ठूलो" in ans or "ठूलो" in ans:
        return "greater"
    if "भन्दा सानो" in ans or "सानो" in ans:
        return "smaller"
    if ans in {"बराबर", "हो"} or "बराबर" in ans:
        return "equal"
    if "सबैभन्दा अग्लो" in ans or "सबैभन्दा ठूलो" in ans:
        return "highest"
    if "सबैभन्दा होचो" in ans or "सबैभन्दा सानो" in ans:
        return "lowest"
    return "other"


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_bar(path, labels, values, title, ylabel, ylim=None):
    plt.figure(figsize=(8, 5))
    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion(path, matrix, labels, title):
    """
    Save a confusion-style matrix visualization.

    Parameters
    ----------
    path : Path
        Output image path.
    matrix : array-like
        Confusion matrix values.
    labels : list[str]
        Labels for rows/columns.
    title : str
        Plot title.
    """
    matrix = np.asarray(matrix, dtype=int)

    fig, ax = plt.subplots(
        figsize=(
            max(6, len(labels) * 1.2),
            max(5, len(labels) * 1.0)
        )
    )

    ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted / Outcome")
    ax.set_ylabel("Actual / Outcome")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="Phase 2 pretrained final.pt")
    ap.add_argument("--finetuned", required=True, help="Phase 3 fine-tuned final.pt")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--test", required=True, help="Held-out reasoning test.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qualitative-per-group", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(Path(args.tokenizer))
    with Path(args.test).open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if len(records) == 0:
        raise RuntimeError("Test set is empty")

    print("=" * 80)
    print("NEPALI — PHASE 3 REASONING TEST-SET EVALUATION")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test examples: {len(records):,}")

    models = {
        "pretrained": load_checkpoint(Path(args.pretrained), device),
        "finetuned": load_checkpoint(Path(args.finetuned), device),
    }

    rows = []
    by_model = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        model_rows = []
        for idx, rec in enumerate(records):
            gold = normalize_text(rec["answer"])
            prediction = generate_answer(
                model, tokenizer, rec["prompt"], device, args.max_new_tokens
            )
            exact = prediction == gold
            token_acc = answer_token_accuracy(tokenizer, prediction, gold)
            row = {
                "id": rec.get("id", idx),
                "task_type": rec.get("task_type", "unknown"),
                "difficulty": rec.get("difficulty", "unknown"),
                "attribute": rec.get("attribute", "unknown"),
                "entity_type": rec.get("entity_type", "unknown"),
                "prompt": rec["prompt"],
                "gold_answer": gold,
                "prediction": prediction,
                "exact_match": int(exact),
                "token_accuracy": token_acc,
                "gold_relation": relation_label(rec),
                "model": model_name,
            }
            model_rows.append(row)
            if idx % 250 == 0 or idx == len(records) - 1:
                print(f"  {idx + 1:5d}/{len(records):5d}")
        by_model[model_name] = model_rows
        rows.extend(model_rows)

    write_csv(
        out / "predictions.csv",
        rows,
        ["id", "model", "task_type", "difficulty", "attribute", "entity_type",
         "gold_relation", "exact_match", "token_accuracy", "prompt",
         "gold_answer", "prediction"],
    )
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = {
        "language": "Nepali",
        "phase": "phase3_reasoning",
        "test_examples": len(records),
        "primary_metric": "exact_match_accuracy",
        "secondary_metric": "mean_answer_token_accuracy",
        "pretrained": {},
        "finetuned": {},
    }

    for name, model_rows in by_model.items():
        metrics[name] = {
            "count": len(model_rows),
            "exact_match_accuracy": sum(r["exact_match"] for r in model_rows) / len(model_rows),
            "exact_matches": sum(r["exact_match"] for r in model_rows),
            "mean_answer_token_accuracy": sum(r["token_accuracy"] for r in model_rows) / len(model_rows),
            "by_task_type": group_metrics(model_rows, "task_type"),
            "by_difficulty": group_metrics(model_rows, "difficulty"),
            "by_attribute": group_metrics(model_rows, "attribute"),
            "by_entity_type": group_metrics(model_rows, "entity_type"),
        }

    metrics["improvement"] = {
        "absolute_accuracy_points": (
            metrics["finetuned"]["exact_match_accuracy"]
            - metrics["pretrained"]["exact_match_accuracy"]
        ) * 100.0,
        "relative_accuracy_change_percent": (
            ((metrics["finetuned"]["exact_match_accuracy"]
              / metrics["pretrained"]["exact_match_accuracy"]) - 1.0) * 100.0
            if metrics["pretrained"]["exact_match_accuracy"] > 0 else None
        ),
        "token_accuracy_points": (
            metrics["finetuned"]["mean_answer_token_accuracy"]
            - metrics["pretrained"]["mean_answer_token_accuracy"]
        ) * 100.0,
    }

    # Paired outcome table: every test example has both predictions.
    p = {r["id"]: r for r in by_model["pretrained"]}
    q = {r["id"]: r for r in by_model["finetuned"]}
    paired = Counter()
    for rid in p:
        paired[(p[rid]["exact_match"], q[rid]["exact_match"])] += 1
    metrics["paired_outcomes"] = {
        "both_correct": paired[(1, 1)],
        "pretrained_only": paired[(1, 0)],
        "finetuned_only": paired[(0, 1)],
        "both_wrong": paired[(0, 0)],
    }

    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Task CSV for easy inclusion in the report.
    task_names = sorted(metrics["pretrained"]["by_task_type"].keys())
    task_rows = []
    for task in task_names:
        a = metrics["pretrained"]["by_task_type"][task]
        b = metrics["finetuned"]["by_task_type"].get(task, {})
        task_rows.append({
            "task_type": task,
            "count": a["count"],
            "pretrained_accuracy": a["exact_match_accuracy"],
            "finetuned_accuracy": b.get("exact_match_accuracy", 0.0),
            "accuracy_gain_points": (b.get("exact_match_accuracy", 0.0) - a["exact_match_accuracy"]) * 100.0,
            "pretrained_token_accuracy": a["token_accuracy"],
            "finetuned_token_accuracy": b.get("token_accuracy", 0.0),
        })
    write_csv(out / "task_metrics.csv", task_rows, list(task_rows[0].keys()) if task_rows else ["task_type"])

    # Overall comparison.
    labels = ["Pretrained", "Fine-tuned"]
    vals = [metrics["pretrained"]["exact_match_accuracy"] * 100,
            metrics["finetuned"]["exact_match_accuracy"] * 100]
    save_bar(out / "accuracy_comparison.png", labels, vals,
             "Nepali reasoning test-set exact-match accuracy", "Accuracy (%)", (0, 100))

    # Task comparison.
    plt.figure(figsize=(11, 6))
    x = list(range(len(task_names)))
    width = 0.38
    pre = [metrics["pretrained"]["by_task_type"][t]["exact_match_accuracy"] * 100 for t in task_names]
    ft = [metrics["finetuned"]["by_task_type"][t]["exact_match_accuracy"] * 100 for t in task_names]
    plt.bar([i - width / 2 for i in x], pre, width, label="Pretrained")
    plt.bar([i + width / 2 for i in x], ft, width, label="Fine-tuned")
    plt.xticks(x, task_names, rotation=35, ha="right")
    plt.ylabel("Exact-match accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Nepali reasoning accuracy by task type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "task_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Relation confusion matrices, only for records where a simple relation label exists.
    labels_rel = ["greater", "smaller", "equal", "highest", "lowest", "other"]
    label_index = {x: i for i, x in enumerate(labels_rel)}
    for model_name in ("pretrained", "finetuned"):
        cm = [[0 for _ in labels_rel] for _ in labels_rel]
        for r, rec in zip(by_model[model_name], records):
            gold = r["gold_relation"]
            pred_text = normalize_text(r["prediction"])
            # Map predicted text to the same coarse labels when possible.
            if pred_text == normalize_text(rec["answer"]):
                pred = gold
            elif "ठूलो" in pred_text:
                pred = "greater"
            elif "सानो" in pred_text:
                pred = "smaller"
            elif "बराबर" in pred_text or pred_text == "हो":
                pred = "equal"
            else:
                pred = "other"
            if gold not in label_index:
                gold = "other"
            if pred not in label_index:
                pred = "other"
            cm[label_index[gold]][label_index[pred]] += 1
        save_confusion(out / f"relation_confusion_{model_name}.png", cm, labels_rel,
                       f"Nepali reasoning relation outcomes — {model_name}")

    # Answer length distribution.
    plt.figure(figsize=(9, 5))
    for model_name, rows_m in by_model.items():
        lengths = [len(tokenizer.encode(r["prediction"]).ids) for r in rows_m]
        plt.hist(lengths, bins=range(0, max(lengths + [1]) + 2), alpha=0.55, label=model_name)
    plt.xlabel("Predicted answer length (tokens)")
    plt.ylabel("Number of test examples")
    plt.title("Predicted answer-length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "answer_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Qualitative samples: paired improvements and failures, deterministic.
    improvements = []
    regressions = []
    both_wrong = []
    for rid in p:
        a, b = p[rid], q[rid]
        item = {
            "id": rid,
            "task_type": a["task_type"],
            "difficulty": a["difficulty"],
            "prompt": a["prompt"],
            "gold_answer": a["gold_answer"],
            "pretrained_prediction": a["prediction"],
            "finetuned_prediction": b["prediction"],
        }
        if a["exact_match"] == 0 and b["exact_match"] == 1:
            improvements.append(item)
        elif a["exact_match"] == 1 and b["exact_match"] == 0:
            regressions.append(item)
        elif a["exact_match"] == 0 and b["exact_match"] == 0:
            both_wrong.append(item)

    for path, data in [
        ("qualitative_successes.jsonl", improvements),
        ("qualitative_regressions.jsonl", regressions),
        ("qualitative_failures.jsonl", both_wrong),
    ]:
        with (out / path).open("w", encoding="utf-8") as f:
            for item in data[:100]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for name in ("pretrained", "finetuned"):
        print(f"{name:12s}: {metrics[name]['exact_match_accuracy'] * 100:7.2f}% exact match | "
              f"{metrics[name]['mean_answer_token_accuracy'] * 100:7.2f}% token accuracy")
    print(f"Accuracy gain: {metrics['improvement']['absolute_accuracy_points']:+.2f} percentage points")
    print(f"Paired: FT-only={metrics['paired_outcomes']['finetuned_only']}, "
          f"PT-only={metrics['paired_outcomes']['pretrained_only']}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
