import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter

try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError as e:
    raise ImportError(
        "SacreBLEU is required. Install with: pip install sacrebleu"
    ) from e


def tokenize(text):
    # Whitespace tokenization is used for BLEU/ROUGE/diversity reporting.
    # chrF++ operates directly on character sequences through SacreBLEU.
    return text.split()


def ngrams(tokens, n):
    return list(zip(*(tokens[i:] for i in range(n))))


def repetition_rate(texts, n=3):
    total = 0
    repeated = 0
    for text in texts:
        grams = ngrams(tokenize(text), n)
        total += len(grams)
        repeated += len(grams) - len(set(grams))
    return repeated / total if total else 0.0


def distinct_n(texts, n):
    all_grams = []
    for text in texts:
        all_grams.extend(ngrams(tokenize(text), n))
    return len(set(all_grams)) / len(all_grams) if all_grams else 0.0


def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(candidates, references):
    scores = []
    for cand, ref in zip(candidates, references):
        c = tokenize(cand)
        r = tokenize(ref)
        if not c or not r:
            scores.append(0.0)
            continue
        lcs = lcs_length(c, r)
        p = lcs / len(c)
        q = lcs / len(r)
        scores.append(2 * p * q / (p + q) if p + q else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise RuntimeError("No generation records found.")

    grouped = {}
    for r in records:
        grouped.setdefault(r["setting"], {"candidates": [], "references": []})
        grouped[r["setting"]]["candidates"].append(r["generated"])
        grouped[r["setting"]]["references"].append(r["reference"])

    bleu = BLEU(tokenize="none", effective_order=True)
    chrfpp = CHRF(word_order=2)

    results = {}
    for setting, data in grouped.items():
        candidates = data["candidates"]
        references = data["references"]

        # SacreBLEU expects a list of hypotheses and a list of reference streams.
        bleu_score = bleu.corpus_score(
            candidates,
            [references],
        ).score

        chrf_score = chrfpp.corpus_score(
            candidates,
            [references],
        ).score

        results[setting] = {
            "num_samples": len(candidates),
            "bleu_4": bleu_score,
            "chrf_plus_plus": chrf_score,
            "rouge_l": rouge_l(candidates, references),
            "repetition_rate_3gram": repetition_rate(candidates, 3),
            "distinct_1": distinct_n(candidates, 1),
            "distinct_2": distinct_n(candidates, 2),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "language": "Telugu",
                "phase": 2,
                "metric_definitions": {
                    "bleu": "SacreBLEU corpus BLEU-4",
                    "chrf": "SacreBLEU chrF++",
                    "rouge_l": "Token-level average F1 based on LCS",
                    "repetition": "Fraction of repeated 3-grams",
                    "distinct_1": "Unique unigrams / all unigrams",
                    "distinct_2": "Unique bigrams / all bigrams",
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("TELUGU — GENERATION METRICS")
    print("=" * 80)
    for setting, values in results.items():
        print(f"\n{setting}")
        print(f"  BLEU-4       : {values['bleu_4']:.4f}")
        print(f"  chrF++       : {values['chrf_plus_plus']:.4f}")
        print(f"  ROUGE-L      : {values['rouge_l']:.4f}")
        print(f"  Repetition   : {values['repetition_rate_3gram']:.4f}")
        print(f"  Distinct-1   : {values['distinct_1']:.4f}")
        print(f"  Distinct-2   : {values['distinct_2']:.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
