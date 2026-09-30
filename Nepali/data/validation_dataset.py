# ============================================================
# LMA-Indiv — Phase 2
# NEPALI — VALIDATION / TEST DATASET
#
# Memory-mapped uint16 token shards.
#
# ============================================================

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NepaliEvaluationDataset(Dataset):
    """
    Dataset for validation or test token shards.

    Expected directory:

        validation/
            val_000.bin

    or:

        test/
            test_000.bin
    """

    def __init__(
        self,
        data_dir,
        block_size,
        split,
    ):

        self.data_dir = Path(
            data_dir
        )

        self.block_size = block_size
        self.split = split.lower()

        if self.split == "validation":

            pattern = "val_*.bin"

        elif self.split == "test":

            pattern = "test_*.bin"

        else:

            raise ValueError(
                "split must be either "
                "'validation' or 'test'"
            )

        # ----------------------------------------------------
        # FIND SHARDS
        # ----------------------------------------------------

        self.shard_paths = sorted(
            self.data_dir.glob(
                pattern
            )
        )

        if not self.shard_paths:

            raise FileNotFoundError(
                f"No {pattern} files found in:\n"
                f"{self.data_dir}"
            )

        # ----------------------------------------------------
        # OPEN MEMORY MAPS
        # ----------------------------------------------------

        self.shards = []

        self.shard_token_counts = []

        for path in self.shard_paths:

            size_bytes = path.stat().st_size

            if size_bytes % 2 != 0:

                raise ValueError(
                    f"Invalid uint16 shard:\n{path}"
                )

            token_count = (
                size_bytes // 2
            )

            if token_count <= block_size:

                raise ValueError(
                    f"Shard is too small for "
                    f"block_size={block_size}:\n"
                    f"{path}"
                )

            data = np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
            )

            self.shards.append(data)

            self.shard_token_counts.append(
                token_count
            )

        # ----------------------------------------------------
        # SAMPLE COUNTS
        # ----------------------------------------------------

        self.samples_per_shard = [
            count - block_size
            for count in self.shard_token_counts
        ]

        self.cumulative_samples = np.cumsum(
            self.samples_per_shard
        )

        self.total_tokens = sum(
            self.shard_token_counts
        )

        self.total_samples = int(
            self.cumulative_samples[-1]
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return self.total_samples

    # ========================================================
    # LOCATE SAMPLE
    # ========================================================

    def _locate_sample(
        self,
        index,
    ):

        if index < 0:

            index += self.total_samples

        if (
            index < 0
            or index >= self.total_samples
        ):

            raise IndexError(
                f"Dataset index out of range: "
                f"{index}"
            )

        shard_index = int(
            np.searchsorted(
                self.cumulative_samples,
                index,
                side="right",
            )
        )

        previous_samples = 0

        if shard_index > 0:

            previous_samples = int(
                self.cumulative_samples[
                    shard_index - 1
                ]
            )

        local_index = (
            index
            - previous_samples
        )

        return (
            shard_index,
            local_index,
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        shard_index, local_index = (
            self._locate_sample(index)
        )

        data = self.shards[
            shard_index
        ]

        start = local_index

        end = (
            start
            + self.block_size
            + 1
        )

        tokens = np.asarray(
            data[start:end],
            dtype=np.int64,
        )

        if len(tokens) != (
            self.block_size + 1
        ):

            raise RuntimeError(
                "Unexpected sequence length."
            )

        x = torch.from_numpy(
            tokens[:-1]
        )

        y = torch.from_numpy(
            tokens[1:]
        )

        return x, y

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        for shard in self.shards:

            del shard

        self.shards.clear()

    # ========================================================
    # CLEANUP
    # ========================================================

    def __del__(self):

        try:
            self.close()
        except Exception:
            pass