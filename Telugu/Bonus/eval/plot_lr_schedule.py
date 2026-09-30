
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# LOG PARSING
# ============================================================

STEP_PATTERN = re.compile(
    r"^\s*Step\s+(\d+)\s*\|\s*LR\s+([0-9.eE+-]+)",
    re.IGNORECASE,
)

GRAD_PATTERN = re.compile(
    r"(?:Grad(?:ient)?[-\s]?norm|grad_norm)\s*[:=]?\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)


def parse_log(log_file):
    text = Path(log_file).read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lr_steps = []
    lr_values = []

    grad_steps = []
    grad_values = []

    current_step = None

    for line in text.splitlines():

        # ----------------------------------------------------
        # Step + learning rate
        # ----------------------------------------------------

        step_match = STEP_PATTERN.search(line)

        if step_match:
            current_step = int(step_match.group(1))
            lr = float(step_match.group(2))

            if (
                lr == lr
                and lr != float("inf")
                and lr != float("-inf")
            ):
                lr_steps.append(current_step)
                lr_values.append(lr)

        # ----------------------------------------------------
        # Gradient norm, if present
        # ----------------------------------------------------

        grad_match = GRAD_PATTERN.search(line)

        if grad_match and current_step is not None:
            grad = float(grad_match.group(1))

            if (
                grad == grad
                and grad != float("inf")
                and grad != float("-inf")
            ):
                grad_steps.append(current_step)
                grad_values.append(grad)

    return (
        lr_steps,
        lr_values,
        grad_steps,
        grad_values,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plot learning-rate schedule from Telugu training log."
    )

    parser.add_argument(
        "--log",
        required=True,
        help="Path to training_output.txt",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output plot.",
    )

    args = parser.parse_args()

    log_file = Path(args.log)
    output_dir = Path(args.output_dir)

    if not log_file.exists():
        raise FileNotFoundError(
            f"Log file does not exist:\n{log_file}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("TELUGU — LEARNING RATE SCHEDULE")
    print("=" * 80)

    print(f"\nLog file: {log_file}")

    (
        lr_steps,
        lr_values,
        grad_steps,
        grad_values,
    ) = parse_log(log_file)

    print(f"\nLearning-rate records : {len(lr_steps)}")
    print(f"Gradient-norm records : {len(grad_steps)}")

    # --------------------------------------------------------
    # LR plot
    # --------------------------------------------------------

    if not lr_steps:
        raise RuntimeError(
            "No learning-rate records were found in the log."
        )

    plt.figure(figsize=(10, 6))

    plt.plot(
        lr_steps,
        lr_values,
        linewidth=1.5,
        label="Learning Rate",
    )

    plt.xlabel(
        "Training Step",
        fontsize=12,
    )

    plt.ylabel(
        "Learning Rate",
        fontsize=12,
    )

    plt.title(
        "Telugu Language Model — Learning Rate Schedule",
        fontsize=13,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    lr_output = output_dir / "learning_rate_schedule.png"

    plt.savefig(
        lr_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\n✓ Learning-rate plot saved to:\n"
        f"  {lr_output}"
    )

    # --------------------------------------------------------
    # Gradient norm plot, only if available
    # --------------------------------------------------------

    if grad_steps:

        plt.figure(figsize=(10, 6))

        plt.plot(
            grad_steps,
            grad_values,
            linewidth=1.2,
            label="Gradient Norm",
        )

        plt.xlabel(
            "Training Step",
            fontsize=12,
        )

        plt.ylabel(
            "Gradient Norm",
            fontsize=12,
        )

        plt.title(
            "Telugu Language Model — Gradient Norm",
            fontsize=13,
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        grad_output = output_dir / "gradient_norm.png"

        plt.savefig(
            grad_output,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"\n✓ Gradient-norm plot saved to:\n"
            f"  {grad_output}"
        )

    else:

        print(
            "\nNo gradient-norm records found."
            "\nSkipping gradient-norm plot."
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
