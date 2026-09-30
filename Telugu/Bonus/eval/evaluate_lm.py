import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Telugu.data.test_dataset import TestTokenDataset
from common import load_model, assert_telugu_config


# ============================================================
# TEST TEXT BYTE COUNT
# ============================================================

def count_test_text_bytes(path):

    total = 0

    with open(path, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:

                raise RuntimeError(
                    f"Invalid JSONL at line {line_no}: {e}"
                )

            if "text" not in record:

                raise KeyError(
                    f"Test record at line {line_no} "
                    f"has no 'text' field."
                )

            total += len(
                str(record["text"]).encode("utf-8")
            )

    return total


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):

    total_nll = 0.0
    total_tokens = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc="Evaluating test set",
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )

        targets = batch["targets"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            _, loss, _ = model(
                input_ids=input_ids,
                targets=targets,
            )

        n = targets.numel()

        total_nll += (
            float(loss.item()) * n
        )

        total_tokens += n

        # ----------------------------------------------------
        # Running metrics
        # ----------------------------------------------------

        running_loss = (
            total_nll / total_tokens
        )

        try:
            running_ppl = math.exp(
                running_loss
            )
        except OverflowError:
            running_ppl = float("inf")

        progress.set_postfix(
            loss=f"{running_loss:.4f}",
            ppl=f"{running_ppl:.2f}",
            tokens=f"{total_tokens:,}",
        )

    progress.close()

    if total_tokens == 0:

        raise RuntimeError(
            "No test target tokens were evaluated."
        )

    loss = (
        total_nll / total_tokens
    )

    perplexity = math.exp(loss)

    return (
        loss,
        perplexity,
        total_nll,
        total_tokens,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    parser.add_argument(
        "--test-dir",
        required=True,
        help="Directory containing tokenized test shards.",
    )

    parser.add_argument(
        "--test-text",
        required=True,
        help="Path to the Telugu test JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("TELUGU — PHASE 2 TEST SET EVALUATION")
    print("=" * 80)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, config, checkpoint_path = load_model(
        args.checkpoint_dir,
        device,
    )

    assert_telugu_config(config)

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = TestTokenDataset(
        test_dir=Path(args.test_dir),
        sequence_length=config.max_seq_len,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset):,}"
    )

    if hasattr(test_dataset, "metadata"):

        print(
            f"Test tokens: "
            f"{test_dataset.metadata['tokens']:,}"
        )

    print(
        f"Evaluation batches: "
        f"{len(loader):,}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    loss, ppl, total_nll, total_tokens = evaluate(
        model,
        loader,
        device,
    )

    # --------------------------------------------------------
    # BPB
    # --------------------------------------------------------

    print("\nCalculating UTF-8 byte count...")

    utf8_bytes = count_test_text_bytes(
        args.test_text
    )

    if utf8_bytes <= 0:

        raise RuntimeError(
            "Test text contains zero UTF-8 bytes."
        )

    bpb = (
        total_nll
        / math.log(2.0)
        / utf8_bytes
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "language": "Telugu",

        "phase": 2,

        "split": "test",

        "checkpoint": str(
            checkpoint_path
        ),

        "parameters": model.count_parameters(),

        "vocab_size": config.vocab_size,

        "context_length": config.max_seq_len,

        "test_loss": loss,

        "perplexity": ppl,

        "bpb": bpb,

        "evaluated_target_tokens": (
            total_tokens
        ),

        "test_utf8_bytes": (
            utf8_bytes
        ),
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(
        f"Checkpoint      : "
        f"{checkpoint_path}"
    )

    print(
        f"Parameters      : "
        f"{results['parameters']:,}"
    )

    print(
        f"Test loss       : "
        f"{loss:.6f}"
    )

    print(
        f"Perplexity      : "
        f"{ppl:.4f}"
    )

    print(
        f"BPB             : "
        f"{bpb:.6f}"
    )

    print(
        f"Target tokens   : "
        f"{total_tokens:,}"
    )

    print(
        f"UTF-8 bytes     : "
        f"{utf8_bytes:,}"
    )

    print(
        f"\nSaved: {out}"
    )


if __name__ == "__main__":
    main()