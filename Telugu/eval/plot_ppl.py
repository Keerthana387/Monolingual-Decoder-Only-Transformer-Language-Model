#!/usr/bin/env python3
"""Plot Telugu Phase 2 validation perplexity from validation-loss history."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
TELUGU_DIR = SCRIPT_DIR.parent
RESULTS = TELUGU_DIR / "results"


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse(obj):
    if isinstance(obj, dict):
        for key in ("history", "logs", "records", "metrics"):
            if isinstance(obj.get(key), list):
                return parse(obj[key])
        steps = next((obj[k] for k in ("steps", "step", "global_steps")
                      if isinstance(obj.get(k), list)), None)
        vals = next((obj[k] for k in ("val_loss", "validation_loss", "eval_loss")
                     if isinstance(obj.get(k), list)), None)
        if steps is not None and vals is not None:
            return [(float(s), float(v)) for s, v in zip(steps, vals)]
        for k in ("validation_loss", "val_loss", "eval_loss"):
            if k in obj:
                return [(0.0, float(obj[k]))]

    if isinstance(obj, list):
        out = []
        for r in obj:
            if isinstance(r, dict):
                s = next((r.get(k) for k in ("step", "global_step") if r.get(k) is not None), None)
                v = next((r.get(k) for k in ("val_loss", "validation_loss", "eval_loss") if r.get(k) is not None), None)
                if s is not None and v is not None:
                    out.append((float(s), float(v)))
        return out

    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=RESULTS / "validation_ppl.png")
    args = p.parse_args()

    data = parse(load(args.input))
    if not data:
        raise ValueError("No validation-loss values found.")

    steps = [s for s, _ in data]
    ppl = [math.exp(v) for _, v in data]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ppl, marker="o" if len(steps) < 30 else None,
             linewidth=2, label="Validation perplexity")
    plt.yscale("log")
    plt.xlabel("Optimizer step")
    plt.ylabel("Perplexity (log scale)")
    plt.title("Telugu Phase 2 Validation Perplexity")
    plt.legend()
    plt.grid(alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
