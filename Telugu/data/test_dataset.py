# ============================================================
# TELUGU — TEST TOKEN DATASET
# Phase 2 held-out test evaluation
#
# Separate from validation_dataset.py so the original
# validation pipeline remains unchanged.
# ============================================================

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class TestTokenDataset(Dataset):
    """
    Dataset for tokenized Telugu test data.

    Expected directory structure:

        test/
        ├── test_000.bin
        ├── test_001.bin
        ├── ...
        └── metadata.json

    Each .bin file contains uint16 token IDs, matching the
    format produced by the Telugu test-tokenization pipeline.
    """

    def __init__(
        self,
        test_dir: Path,
        sequence_length: int,
    ):
        self.test_dir = Path(test_dir)
        self.sequence_length = sequence_length

        if not self.test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: {self.test_dir}"
            )

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        self.metadata_path = self.test_dir / "metadata.json"

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Test metadata not found:\n{self.metadata_path}"
            )

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # --------------------------------------------------------
        # Test shards
        # --------------------------------------------------------

        self.shard_paths = sorted(
            self.test_dir.glob("test_*.bin")
        )

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No test shards found in {self.test_dir}"
            )

        # --------------------------------------------------------
        # Load shard information
        # --------------------------------------------------------

        self.shards = []
        self.cumulative_samples = []

        total_samples = 0

        for shard_path in self.shard_paths:
            tokens = np.memmap(
                shard_path,
                dtype=np.uint16,
                mode="r",
            )

            num_tokens = len(tokens)

            # Need sequence_length + 1 tokens because:
            #
            # input  = tokens[i : i + sequence_length]
            # target = tokens[i + 1 : i + sequence_length + 1]
            #
            num_samples = max(
                0,
                num_tokens - sequence_length,
            )

            self.shards.append(
                {
                    "path": shard_path,
                    "tokens": tokens,
                    "num_tokens": num_tokens,
                    "num_samples": num_samples,
                }
            )

            total_samples += num_samples
            self.cumulative_samples.append(total_samples)

        self.total_samples = total_samples

        if self.total_samples == 0:
            raise ValueError(
                f"Test data contains insufficient tokens for "
                f"sequence length {sequence_length}"
            )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        if index < 0:
            index += self.total_samples

        if index < 0 or index >= self.total_samples:
            raise IndexError(
                f"Test dataset index out of range: {index}"
            )

        # --------------------------------------------------------
        # Find the shard containing this sample
        # --------------------------------------------------------

        shard_index = np.searchsorted(
            self.cumulative_samples,
            index,
            side="right",
        )

        previous_samples = (
            0
            if shard_index == 0
            else self.cumulative_samples[shard_index - 1]
        )

        local_index = index - previous_samples

        shard = self.shards[shard_index]
        tokens = shard["tokens"]

        # --------------------------------------------------------
        # Input / target
        # --------------------------------------------------------

        start = local_index
        end = start + self.sequence_length

        x = np.asarray(
            tokens[start:end],
            dtype=np.int64,
        )

        y = np.asarray(
            tokens[start + 1:end + 1],
            dtype=np.int64,
        )

        return {
            "input_ids": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
        }