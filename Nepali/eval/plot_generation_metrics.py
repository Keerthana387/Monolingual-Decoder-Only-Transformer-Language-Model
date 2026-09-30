import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = data["results"]

    settings = list(results.keys())
    metrics = [
        ("bleu_4", "BLEU-4"),
        ("chrf_plus_plus", "chrF++"),
        ("rouge_l", "ROUGE-L"),
        ("distinct_1", "Distinct-1"),
        ("distinct_2", "Distinct-2"),
        ("repetition_rate_3gram", "Repetition Rate"),
    ]

    x = np.arange(len(settings))
    width = 0.12

    plt.figure(figsize=(12, 6))
    for i, (key, label) in enumerate(metrics):
        values = [results[s][key] for s in settings]
        plt.bar(x + (i - 2.5) * width, values, width, label=label)

    plt.xlabel("Decoding Setting")
    plt.ylabel("Score")
    plt.title("Nepali Generation Quality and Diversity by Decoding Setting")
    plt.xticks(x, settings, rotation=15)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "generation_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out / 'generation_metrics.png'}")


if __name__ == "__main__":
    main()
