import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# VALIDATION DATASET
# ============================================================

class ValidationTokenDataset(Dataset):
    """
    Dataset for tokenized Telugu validation shards.

    Each shard contains a continuous stream of uint16 token IDs.

    Samples are constructed as:

        block = [x0, x1, ..., x512]

        input_ids = [x0, ..., x511]
        targets   = [x1, ..., x512]

    Therefore:

        sequence_length = 512
        block_size       = 513
    """

    def __init__(
        self,
        validation_dir,
        sequence_length=512,
    ):

        self.validation_dir = Path(
            validation_dir
        )

        self.sequence_length = (
            sequence_length
        )

        self.block_size = (
            sequence_length + 1
        )

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        metadata_path = (
            self.validation_dir
            / "metadata.json"
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Validation metadata not found:\n"
                f"{metadata_path}"
            )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(f)

        # -----------------------------------------------------
        # Shards
        # -----------------------------------------------------

        self.shard_paths = sorted(
            self.validation_dir.glob(
                "val_*.bin"
            )
        )

        if not self.shard_paths:

            raise FileNotFoundError(
                f"No validation shards found "
                f"in {self.validation_dir}"
            )

        # -----------------------------------------------------
        # Build shard information
        # -----------------------------------------------------

        self.shard_token_counts = []

        self.shard_sample_counts = []

        for shard_path in self.shard_paths:

            token_count = (
                shard_path.stat().st_size
                // np.dtype(np.uint16).itemsize
            )

            sample_count = (
                token_count
                // self.block_size
            )

            self.shard_token_counts.append(
                token_count
            )

            self.shard_sample_counts.append(
                sample_count
            )

        # -----------------------------------------------------
        # Cumulative sample offsets
        # -----------------------------------------------------

        self.cumulative_samples = []

        total = 0

        for count in self.shard_sample_counts:

            self.cumulative_samples.append(
                total
            )

            total += count

        self.num_samples = total

        # -----------------------------------------------------
        # Active mmap
        #
        # We keep only the currently accessed shard mapped.
        # -----------------------------------------------------

        self._current_shard_index = None

        self._current_data = None

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return self.num_samples

    # ========================================================
    # FIND SHARD
    # ========================================================

    def _find_shard(
        self,
        sample_index,
    ):

        for shard_index in range(
            len(self.shard_paths) - 1,
            -1,
            -1,
        ):

            if (
                sample_index
                >= self.cumulative_samples[
                    shard_index
                ]
            ):

                return shard_index

        raise IndexError(
            "Invalid validation sample index."
        )

    # ========================================================
    # LOAD SHARD
    # ========================================================

    def _load_shard(
        self,
        shard_index,
    ):

        if (
            self._current_shard_index
            == shard_index
        ):

            return

        self._current_data = np.memmap(
            self.shard_paths[
                shard_index
            ],
            dtype=np.uint16,
            mode="r",
        )

        self._current_shard_index = (
            shard_index
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        if index < 0:

            index += self.num_samples

        if (
            index < 0
            or index >= self.num_samples
        ):

            raise IndexError(
                "Validation dataset index "
                "out of range."
            )

        # -----------------------------------------------------
        # Determine shard
        # -----------------------------------------------------

        shard_index = self._find_shard(
            index
        )

        # -----------------------------------------------------
        # Determine sample inside shard
        # -----------------------------------------------------

        local_sample_index = (
            index
            - self.cumulative_samples[
                shard_index
            ]
        )

        # -----------------------------------------------------
        # Load shard
        # -----------------------------------------------------

        self._load_shard(
            shard_index
        )

        # -----------------------------------------------------
        # Extract block
        # -----------------------------------------------------

        start = (
            local_sample_index
            * self.block_size
        )

        end = (
            start
            + self.block_size
        )

        block = self._current_data[
            start:end
        ]

        if len(block) != self.block_size:

            raise RuntimeError(
                "Validation shard does not "
                "contain a complete block."
            )

        # -----------------------------------------------------
        # Input / target split
        # -----------------------------------------------------

        input_ids = torch.from_numpy(
            block[:-1].astype(
                np.int64,
                copy=False,
            )
        )

        targets = torch.from_numpy(
            block[1:].astype(
                np.int64,
                copy=False,
            )
        )

        return {
            "input_ids": input_ids,
            "targets": targets,
        }

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(self):

        return {
            "num_shards": len(
                self.shard_paths
            ),

            "total_tokens": sum(
                self.shard_token_counts
            ),

            "num_samples": self.num_samples,

            "sequence_length": (
                self.sequence_length
            ),

            "block_size": self.block_size,

            "discarded_tokens": sum(
                self.shard_token_counts
            ) - (
                self.num_samples
                * self.block_size
            ),
        }