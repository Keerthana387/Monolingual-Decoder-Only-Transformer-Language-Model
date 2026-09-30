import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    training = {}
    validation = {}
    current_step = None
    final_val = None

    step_re = re.compile(
        r"^\s*Step\s+(\d+)\s*\|.*?\|\s*Loss\s+([0-9.eE+-]+)\s*\|",
        re.I,
    )
    val_re = re.compile(
        r"^\s*Validation\s*\|\s*Loss\s+([0-9.eE+-]+)",
        re.I,
    )
    final_re = re.compile(
        r"^\s*Final\s+validation\s+loss\s*:\s*([0-9.eE+-]+)",
        re.I,
    )

    for line in text.splitlines():
        m = step_re.search(line)
        if m:
            current_step = int(m.group(1))
            training[current_step] = float(m.group(2))
            continue

        m = val_re.search(line)
        if m and current_step is not None:
            validation[current_step] = float(m.group(1))
            continue

        m = final_re.search(line)
        if m:
            final_val = float(m.group(1))

    if final_val is not None and training:
        validation[max(training)] = final_val

    rows = []
    for step in sorted(set(training) | set(validation)):
        rows.append({
            "step": step,
            "train_loss": training.get(step),
            "validation_loss": validation.get(step),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=25)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = parse_log(args.log)
    if df.empty:
        raise RuntimeError("No training/validation loss found in log.")

    df["train_loss_rolling"] = (
        df["train_loss"].rolling(
            args.rolling_window, min_periods=1
        ).mean()
    )

    df.to_csv(out / "loss_history.csv", index=False)

    plt.figure(figsize=(10, 6))
    train = df.dropna(subset=["train_loss"])
    val = df.dropna(subset=["validation_loss"])

    if not train.empty:
        plt.plot(
            train["step"], train["train_loss"],
            linewidth=0.8, alpha=0.35,
            label="Training Loss",
        )
        plt.plot(
            train["step"], train["train_loss_rolling"],
            linewidth=1.8,
            label=f"Training Loss ({args.rolling_window}-point rolling mean)",
        )

    if not val.empty:
        plt.plot(
            val["step"], val["validation_loss"],
            marker="o", markersize=3, linewidth=1.6,
            label="Validation Loss",
        )

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Nepali Language Model — Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "train_vs_validation_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training points: {len(train)}")
    print(f"Validation points: {len(val)}")
    print(f"Saved: {out / 'train_vs_validation_loss.png'}")


if __name__ == "__main__":
    main()
