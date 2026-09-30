from pathlib import Path

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# LOG PARSING
# ============================================================

STEP_RE = re.compile(
    r"^\s*Step\s+(\d+)\s*\|.*?\|\s*Loss\s+([0-9.eE+-]+)\s*\|",
    re.I,
)

VAL_RE = re.compile(
    r"^\s*Validation\s*\|\s*Loss\s+([0-9.eE+-]+)",
    re.I,
)

FINAL_RE = re.compile(
    r"^\s*Final\s+validation\s+loss\s*:\s*([0-9.eE+-]+)",
    re.I,
)


def parse_log(log_path):
    train = []
    validation = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        current_step = None

        for line in f:
            # Training loss
            m = STEP_RE.search(line)
            if m:
                current_step = int(m.group(1))
                loss = float(m.group(2))

                if math.isfinite(loss):
                    train.append((current_step, loss))

                continue

            # Validation loss
            m = VAL_RE.search(line)
            if m and current_step is not None:
                loss = float(m.group(1))

                if math.isfinite(loss):
                    validation.append((current_step, loss))

    return train, validation


# ============================================================
# PLOT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot training and validation perplexity."
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to training_output.txt",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the plot will be saved.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window for smoothing training perplexity.",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    output_dir = Path(args.output_dir)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train, validation = parse_log(log_path)

    if not train:
        raise RuntimeError("No training loss values were found in the log.")

    if not validation:
        raise RuntimeError("No validation loss values were found in the log.")

    # Convert loss -> perplexity
    train_steps = [x[0] for x in train]
    train_ppl = [math.exp(x[1]) for x in train]

    val_steps = [x[0] for x in validation]
    val_ppl = [math.exp(x[1]) for x in validation]

    # Optional rolling smoothing for training PPL
    window = max(1, args.rolling_window)

    if window > 1 and len(train_ppl) >= window:
        smoothed_steps = []
        smoothed_ppl = []

        running_sum = 0.0

        for i, value in enumerate(train_ppl):
            running_sum += value

            if i >= window:
                running_sum -= train_ppl[i - window]

            if i >= window - 1:
                smoothed_steps.append(train_steps[i])
                smoothed_ppl.append(running_sum / window)
    else:
        smoothed_steps = train_steps
        smoothed_ppl = train_ppl

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        smoothed_steps,
        smoothed_ppl,
        label=f"Training perplexity (rolling {window})",
    )

    plt.plot(
        val_steps,
        val_ppl,
        marker="o",
        markersize=3,
        label="Validation perplexity",
    )

    plt.xlabel("Training step")
    plt.ylabel("Perplexity")
    plt.title("Training and Validation Perplexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "perplexity.png"

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Training points: {len(train)}")
    print(f"Validation points: {len(validation)}")
    print(f"Final training perplexity: {train_ppl[-1]:.4f}")
    print(f"Final validation perplexity: {val_ppl[-1]:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
