import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    e = {(r["layer"], r["head"]): r["attention_entropy"]
         for r in entropy_rows}
    z = {(r["layer"], r["head"]): r["mean_attention_distance"]
         for r in distance_rows}

    layers = sorted({k[0] for k in e})
    max_entropy = max(e.values())

    plt.figure(figsize=(9, 6))

    for layer in layers:
        xs = []
        ys = []
        for (l, h), entropy in e.items():
            if l == layer:
                xs.append(z[(l, h)])
                ys.append(entropy / max_entropy if max_entropy else 0.0)
        plt.scatter(xs, ys, label=f"Layer {layer}")

    plt.xlabel("Mean Attention Distance")
    plt.ylabel("Normalized Attention Entropy")
    plt.title("Nepali Attention: Local vs Long-Range Behavior")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        ncol=1,
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        d / "attention_local_global_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {d / 'attention_local_global_scatter.png'}")


if __name__ == "__main__":
    main()
