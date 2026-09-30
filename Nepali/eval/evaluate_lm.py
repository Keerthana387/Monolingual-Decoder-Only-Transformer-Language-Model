"""
NEPALI — PHASE 2 INTRINSIC EVALUATION

Efficient test-set evaluation for the Nepali Phase 2 model.

Important:
The normal NepaliEvaluationDataset creates a sliding-window sample for
every token. That gives ~10 million samples for a 10M-token test
set and makes evaluation unnecessarily expensive.

This file instead evaluates contiguous, non-overlapping blocks. The
test stream is therefore processed in roughly 10M / 512 blocks.

The code follows the actual Nepali model/common.py interfaces:
    model, config, checkpoint_path = load_model(...)
    logits, loss = model(input_ids=..., targets=...)

CUDA + FP16 autocast is used when CUDA is available, with CPU fallback.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import load_model, assert_nepali_config, load_jsonl_texts


# ============================================================
# CONTIGUOUS TEST DATASET
# ============================================================

class ContiguousTestDataset(Dataset):
    """
    Test dataset using contiguous blocks rather than sliding windows.

    Each sample consists of:
        input_ids = tokens[start : start + block_size]
        targets   = tokens[start + 1 : start + block_size + 1]

    Thus each target token is evaluated once, except for the final
    incomplete portion of the test stream.

    Blocks do not cross shard boundaries.
    """

    def __init__(self, data_dir, block_size):
        self.data_dir = Path(data_dir)
        self.block_size = int(block_size)

        shard_paths = sorted(self.data_dir.glob("test_*.bin"))

        if not shard_paths:
            raise FileNotFoundError(
                f"No test_*.bin shards found in {self.data_dir}"
            )

        self.shards = []
        self.block_counts = []

        for path in shard_paths:
            size_bytes = path.stat().st_size

            if size_bytes % 2 != 0:
                raise ValueError(
                    f"Invalid uint16 shard size: {path}"
                )

            token_count = size_bytes // np.dtype(np.uint16).itemsize

            # Need block_size input tokens + 1 target token.
            if token_count <= self.block_size:
                continue

            self.shards.append(
                (
                    path,
                    token_count,
                    np.memmap(
                        path,
                        dtype=np.uint16,
                        mode="r",
                    ),
                )
            )

            # Number of complete contiguous (x,y) blocks.
            self.block_counts.append(
                (token_count - 1) // self.block_size
            )

        if not self.shards:
            raise ValueError(
                f"No test shard contains enough tokens for "
                f"block_size={self.block_size}"
            )

        self.cumulative_blocks = np.cumsum(
            self.block_counts,
            dtype=np.int64,
        )

        self.num_blocks = int(self.cumulative_blocks[-1])

        self.total_tokens = int(
            sum(token_count for _, token_count, _ in self.shards)
        )

        self.evaluated_target_tokens = (
            self.num_blocks * self.block_size
        )

    def __len__(self):
        return self.num_blocks

    def _locate_block(self, index):
        if index < 0:
            index += self.num_blocks

        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)

        shard_index = int(
            np.searchsorted(
                self.cumulative_blocks,
                index,
                side="right",
            )
        )

        previous = (
            0
            if shard_index == 0
            else int(self.cumulative_blocks[shard_index - 1])
        )

        local_block = index - previous

        return shard_index, local_block

    def __getitem__(self, index):
        shard_index, local_block = self._locate_block(index)

        _, _, shard = self.shards[shard_index]

        start = local_block * self.block_size
        end = start + self.block_size + 1

        tokens = np.asarray(
            shard[start:end],
            dtype=np.int64,
        )

        # tokens has block_size + 1 elements.
        input_ids = torch.from_numpy(tokens[:-1].copy())
        targets = torch.from_numpy(tokens[1:].copy())

        return input_ids, targets


# ============================================================
# TEST
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    start_time = time.time()
    last_report = start_time
    total_batches = len(loader)

    for batch_idx, (input_ids, targets) in enumerate(
        loader,
        start=1,
    ):
        input_ids = input_ids.to(
            device,
            non_blocking=(device.type == "cuda"),
        )
        targets = targets.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        if use_amp:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                _, loss = model(
                    input_ids=input_ids,
                    targets=targets,
                )
        else:
            _, loss = model(
                input_ids=input_ids,
                targets=targets,
            )

        n = targets.numel()

        # model loss is mean CE over the batch.
        total_nll += float(loss.item()) * n
        total_tokens += n

        now = time.time()

        if (
            batch_idx == 1
            or batch_idx == total_batches
            or now - last_report >= 5.0
        ):
            elapsed = max(now - start_time, 1e-9)
            tok_per_sec = total_tokens / elapsed
            pct = 100.0 * batch_idx / total_batches

            print(
                f"\rEvaluation: {batch_idx:,}/{total_batches:,} "
                f"({pct:6.2f}%) | "
                f"{total_tokens:,} target tokens | "
                f"{tok_per_sec:,.0f} tok/s",
                end="",
                flush=True,
            )

            last_report = now

    print()

    if total_tokens == 0:
        raise RuntimeError(
            "No test target tokens were evaluated."
        )

    loss = total_nll / total_tokens
    perplexity = math.exp(loss)

    return loss, perplexity, total_nll, total_tokens


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
                ) from e

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
    )
    parser.add_argument(
        "--test-text",
        required=True,
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

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
    else:
        device = torch.device("cpu")
        use_amp = False

    print("=" * 80)
    print("NEPALI — PHASE 2 INTRINSIC EVALUATION")
    print("=" * 80)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        print("Mixed precision: FP16 autocast")
    else:
        print("GPU not available; using CPU")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    # IMPORTANT:
    # Nepali common.load_model() returns THREE values:
    # model, config, checkpoint_path
    model, config, checkpoint_path = load_model(
        args.checkpoint_dir,
        device,
    )

    assert_nepali_config(config)

    print(f"Using final checkpoint: {checkpoint_path}")

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test_dataset = ContiguousTestDataset(
        args.test_dir,
        config.max_seq_len,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Test tokens: "
        f"{test_dataset.total_tokens:,}"
    )
    print(
        f"Test blocks: "
        f"{len(test_dataset):,}"
    )
    print(
        f"Evaluated target tokens: "
        f"{test_dataset.evaluated_target_tokens:,}"
    )
    print(
        f"Batch size: {args.batch_size}"
    )
    print(
        f"Test batches: {len(loader):,}"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("Evaluation:")

    loss, ppl, total_nll, total_tokens = evaluate(
        model,
        loader,
        device,
        use_amp,
    )

    # --------------------------------------------------------
    # BPB
    # --------------------------------------------------------

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
    # OUTPUT
    # --------------------------------------------------------

    results = {
        "language": "Nepali",
        "phase": 2,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "mixed_precision": (
            "FP16 autocast"
            if use_amp
            else None
        ),
        "checkpoint": str(checkpoint_path),
        "parameters": model.count_parameters(),
        "vocab_size": config.vocab_size,
        "context_length": config.max_seq_len,
        "test_loss": loss,
        "perplexity": ppl,
        "bpb": bpb,
        "evaluated_target_tokens": total_tokens,
        "test_tokens": test_dataset.total_tokens,
        "test_blocks": len(test_dataset),
        "test_utf8_bytes": utf8_bytes,
    }

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

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(
        f"Checkpoint      : {checkpoint_path}"
    )
    print(
        f"Parameters      : "
        f"{results['parameters']:,}"
    )
    print(
        f"Test loss : {loss:.6f}"
    )
    print(
        f"Perplexity      : {ppl:.4f}"
    )
    print(
        f"BPB             : {bpb:.6f}"
    )
    print(
        f"Target tokens   : {total_tokens:,}"
    )
    print(
        f"UTF-8 bytes     : {utf8_bytes:,}"
    )
    print(f"\nSaved: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
