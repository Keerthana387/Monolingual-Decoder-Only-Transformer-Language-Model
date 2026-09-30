import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def matrix_from_rows(rows, key):
    layers = max(r["layer"] for r in rows) + 1
    heads = max(r["head"] for r in rows) + 1
    matrix = np.zeros((layers, heads), dtype=float)
    for r in rows:
        matrix[r["layer"], r["head"]] = r[key]
    return matrix


def heatmap(matrix, title, xlabel, ylabel, path):
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Value")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-dir", required=True)
    args = parser.parse_args()

    d = Path(args.attention_dir)
    entropy_rows = json.loads(
        (d / "attention_entropy.json").read_text(encoding="utf-8")
    )
    distance_rows = json.loads(
        (d / "attention_distance.json").read_text(encoding="utf-8")
    )

    entropy = matrix_from_rows(
        entropy_rows, "attention_entropy"
    )
    distance = matrix_from_rows(
        distance_rows, "mean_attention_distance"
    )

    heatmap(
        entropy,
        "Telugu Attention Entropy by Layer and Head",
        "Head",
        "Layer",
        d / "attention_entropy_layer_head.png",
    )

    heatmap(
        distance,
        "Telugu Mean Attention Distance by Layer and Head",
        "Head",
        "Layer",
        d / "attention_distance_layer_head.png",
    )

    print(f"Saved attention summary plots to: {d}")


if __name__ == "__main__":
    main()
