from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# TINY SYNTHETIC DATASET
# ============================================================

class TinyLanguageModelDataset(Dataset):
    """
    Small synthetic dataset used for model sanity and
    overfitting tests.
    """

    def __init__(
        self,
        sequences: list[list[int]],
    ):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int):

        sequence = torch.tensor(
            self.sequences[index],
            dtype=torch.long,
        )

        return {
            "input_ids": sequence[:-1],
            "targets": sequence[1:],
        }


# ============================================================
# REAL TOKEN SHARD DATASET
# ============================================================

class TokenShardDataset(Dataset):
    """
    Dataset over the complete pre-tokenized Telugu corpus.

    The token shards are treated as one continuous stream.

    Training samples are NON-OVERLAPPING blocks:

        tokens:
        [0 ... 511]
        [512 ... 1023]
        [1024 ... 1535]
        ...

    For each block:

        input_ids = block[:-1]
        targets   = block[1:]

    Therefore each sample contains sequence_length input tokens
    and sequence_length target tokens.

    IMPORTANT:
    The dataset does not load the complete corpus into RAM.
    Individual shards are memory-mapped from disk.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        sequence_length: int,
    ):

        self.shard_paths = [
            Path(path)
            for path in shard_paths
        ]

        self.sequence_length = sequence_length

        if not self.shard_paths:
            raise ValueError(
                "No token shards were provided."
            )

        # -----------------------------------------------------
        # Memory-map all shards.
        #
        # The actual token data remains on disk.
        # -----------------------------------------------------

        self.shards = []

        for path in self.shard_paths:

            if not path.exists():
                raise FileNotFoundError(
                    f"Shard not found: {path}"
                )

            tokens = np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
            )

            self.shards.append(tokens)

        # -----------------------------------------------------
        # Calculate total corpus size.
        # -----------------------------------------------------

        self.total_tokens = sum(
            len(tokens)
            for tokens in self.shards
        )

        # -----------------------------------------------------
        # Number of complete non-overlapping samples.
        #
        # Each sample requires:
        #
        # sequence_length + 1 tokens
        #
        # because:
        #
        # input  = tokens[0 : sequence_length]
        # target = tokens[1 : sequence_length + 1]
        #
        # We intentionally use a continuous global stream.
        # -----------------------------------------------------

        self.block_size = (
            sequence_length + 1
        )

        self.num_samples = (
            self.total_tokens
            // self.block_size
        )

        self.used_tokens = (
            self.num_samples
            * self.block_size
        )

        self.discarded_tokens = (
            self.total_tokens
            - self.used_tokens
        )

        # -----------------------------------------------------
        # Build global token offsets.
        #
        # offsets[i] = global starting position of shard i
        # -----------------------------------------------------

        self.shard_offsets = []

        offset = 0

        for tokens in self.shards:

            self.shard_offsets.append(
                offset
            )

            offset += len(tokens)

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return self.num_samples

    # ========================================================
    # LOCATE GLOBAL TOKEN
    # ========================================================

    def _locate_token(
        self,
        global_index: int,
    ):

        """
        Convert a global token index into:

            shard index
            local token index
        """

        # -----------------------------------------------------
        # Binary search over shard offsets.
        # -----------------------------------------------------

        left = 0
        right = len(
            self.shard_offsets
        ) - 1

        while left <= right:

            mid = (
                left + right
            ) // 2

            if (
                self.shard_offsets[mid]
                <= global_index
            ):

                left = mid + 1

            else:

                right = mid - 1

        shard_index = right

        local_index = (
            global_index
            - self.shard_offsets[
                shard_index
            ]
        )

        return (
            shard_index,
            local_index,
        )

    # ========================================================
    # GET SAMPLE
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ):

        if index < 0:
            index += len(self)

        if (
            index < 0
            or index >= len(self)
        ):
            raise IndexError(
                "Dataset index out of range."
            )

        # -----------------------------------------------------
        # Global position of the first token.
        #
        # Samples are non-overlapping.
        # -----------------------------------------------------

        global_start = (
            index
            * self.block_size
        )

        global_end = (
            global_start
            + self.block_size
        )

        # -----------------------------------------------------
        # Locate starting shard.
        # -----------------------------------------------------

        start_shard, start_local = (
            self._locate_token(
                global_start
            )
        )

        # -----------------------------------------------------
        # Locate ending shard.
        # -----------------------------------------------------

        end_shard, end_local = (
            self._locate_token(
                global_end - 1
            )
        )

        # -----------------------------------------------------
        # Fast path:
        #
        # Entire sample is inside one shard.
        # -----------------------------------------------------

        if start_shard == end_shard:

            sequence = self.shards[
                start_shard
            ][
                start_local :
                start_local
                + self.block_size
            ]

            sequence = np.asarray(
                sequence
            )

        # -----------------------------------------------------
        # Cross-shard sample.
        #
        # This can happen because the shards are treated as
        # one continuous token stream.
        # -----------------------------------------------------

        else:

            pieces = []

            # Tokens from starting shard

            pieces.append(
                self.shards[
                    start_shard
                ][
                    start_local:
                ]
            )

            # Complete intermediate shards

            for shard_index in range(
                start_shard + 1,
                end_shard,
            ):

                pieces.append(
                    self.shards[
                        shard_index
                    ][:]
                )

            # Tokens from ending shard

            pieces.append(
                self.shards[
                    end_shard
                ][
                    : end_local + 1
                ]
            )

            sequence = np.concatenate(
                pieces
            )

        # -----------------------------------------------------
        # Convert to PyTorch tensors.
        # -----------------------------------------------------

        input_ids = torch.from_numpy(
            sequence[:-1].astype(
                np.int64,
                copy=True,
            )
        )

        targets = torch.from_numpy(
            sequence[1:].astype(
                np.int64,
                copy=True,
            )
        )

        return {
            "input_ids": input_ids,
            "targets": targets,
        }

    # ========================================================
    # INFORMATION
    # ========================================================

    def get_statistics(self):

        return {
            "num_shards": len(
                self.shards
            ),
            "total_tokens": (
                self.total_tokens
            ),
            "sequence_length": (
                self.sequence_length
            ),
            "block_size": (
                self.block_size
            ),
            "num_samples": (
                self.num_samples
            ),
            "used_tokens": (
                self.used_tokens
            ),
            "discarded_tokens": (
                self.discarded_tokens
            ),
        }