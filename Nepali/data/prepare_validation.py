# ============================================================
# LMA-Indiv — Phase 2
# NEPALI — PREPARE TRAIN / VALIDATION / TEST SPLITS
#
# Runs LOCALLY on CPU.
#
# Input:
#   Nepali/data/tokenized/train_*.bin
#
# Output:
#
#   Nepali/data/tokenized/
#   ├── train_000.bin
#   ├── train_001.bin
#   ├── ...
#   ├── validation/
#   │   └── val_000.bin
#   ├── test/
#   │   └── test_000.bin
#   └── split_metadata.json
#
# IMPORTANT:
#   The original train_*.bin files are replaced with new
#   training-only shards containing exactly TRAIN_TOKENS.
#
#   Validation and test tokens are taken AFTER the training
#   token range, so their token positions cannot overlap with
#   the training data.
#
# ============================================================

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_TRAIN_TOKENS = 500_000_000
DEFAULT_VALIDATION_TOKENS = 10_000_000
DEFAULT_TEST_TOKENS = 10_000_000

DEFAULT_SHARD_SIZE = 10_000_000


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare Nepali Phase 2 train, validation, "
            "and test splits from tokenized shards."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Directory containing the existing "
            "tokenized train_*.bin shards."
        ),
    )

    parser.add_argument(
        "--train-tokens",
        type=int,
        default=DEFAULT_TRAIN_TOKENS,
        help=(
            "Number of tokens to retain for training. "
            "Default: 500,000,000"
        ),
    )

    parser.add_argument(
        "--validation-tokens",
        type=int,
        default=DEFAULT_VALIDATION_TOKENS,
        help=(
            "Number of tokens to reserve for validation. "
            "Default: 10,000,000"
        ),
    )

    parser.add_argument(
        "--test-tokens",
        type=int,
        default=DEFAULT_TEST_TOKENS,
        help=(
            "Number of tokens to reserve for testing. "
            "Default: 10,000,000"
        ),
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=(
            "Number of tokens per training shard. "
            "Default: 10,000,000"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow existing validation/test directories "
            "and existing split metadata to be replaced."
        ),
    )

    return parser.parse_args()


# ============================================================
# COUNT INPUT TOKENS
# ============================================================

def get_input_shards(input_dir):

    shard_paths = sorted(
        input_dir.glob("train_*.bin")
    )

    if not shard_paths:

        raise FileNotFoundError(
            "No train_*.bin files found in:\n"
            f"{input_dir}"
        )

    return shard_paths


def count_tokens(shard_paths):

    total_tokens = 0

    print("\nInspecting input shards...\n")

    for path in tqdm(
        shard_paths,
        desc="Counting tokens",
        unit="shard",
    ):

        size_bytes = path.stat().st_size

        # uint16 = 2 bytes
        if size_bytes % 2 != 0:

            raise ValueError(
                f"Invalid uint16 shard size:\n{path}\n"
                f"File size: {size_bytes} bytes"
            )

        total_tokens += size_bytes // 2

    return total_tokens


# ============================================================
# STREAM TOKENS FROM EXISTING SHARDS
# ============================================================

def iter_token_chunks(
    shard_paths,
    chunk_size=1_000_000,
):

    for shard_path in shard_paths:

        print(
            f"Reading: {shard_path.name}"
        )

        data = np.memmap(
            shard_path,
            dtype=np.uint16,
            mode="r",
        )

        for start in range(
            0,
            len(data),
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                len(data),
            )

            yield np.asarray(
                data[start:end]
            )

        del data


# ============================================================
# BINARY SHARD WRITER
# ============================================================

class BinaryShardWriter:

    def __init__(
        self,
        output_dir,
        prefix,
        shard_size,
    ):

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.prefix = prefix
        self.shard_size = shard_size

        self.buffer = np.empty(
            self.shard_size,
            dtype=np.uint16,
        )

        self.position = 0
        self.shard_index = 0
        self.total_tokens = 0
        self.shards = []

    # --------------------------------------------------------
    # ADD TOKENS
    # --------------------------------------------------------

    def add(self, tokens):

        tokens = np.asarray(
            tokens,
            dtype=np.uint16,
        )

        start = 0

        while start < len(tokens):

            remaining = (
                self.shard_size
                - self.position
            )

            take = min(
                remaining,
                len(tokens) - start,
            )

            self.buffer[
                self.position:
                self.position + take
            ] = tokens[
                start:
                start + take
            ]

            self.position += take
            start += take

            self.total_tokens += take

            if (
                self.position
                == self.shard_size
            ):

                self.flush()

    # --------------------------------------------------------
    # FLUSH SHARD
    # --------------------------------------------------------

    def flush(self):

        if self.position == 0:
            return

        shard_path = (
            self.output_dir
            / f"{self.prefix}_{self.shard_index:03d}.bin"
        )

        self.buffer[
            :self.position
        ].tofile(shard_path)

        self.shards.append(
            {
                "file": shard_path.name,
                "tokens": self.position,
                "bytes": self.position * 2,
            }
        )

        print(
            f"Saved {shard_path.name} "
            f"({self.position:,} tokens)"
        )

        self.position = 0
        self.shard_index += 1

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        self.flush()


# ============================================================
# SAFE OUTPUT PREPARATION
# ============================================================

def prepare_output_directories(
    input_dir,
    overwrite,
):

    validation_dir = (
        input_dir / "validation"
    )

    test_dir = (
        input_dir / "test"
    )

    split_metadata = (
        input_dir / "split_metadata.json"
    )

    if not overwrite:

        if validation_dir.exists():

            raise FileExistsError(
                f"\nValidation directory already exists:\n"
                f"{validation_dir}\n\n"
                "If you intentionally want to regenerate "
                "the split, rerun with --overwrite."
            )

        if test_dir.exists():

            raise FileExistsError(
                f"\nTest directory already exists:\n"
                f"{test_dir}\n\n"
                "If you intentionally want to regenerate "
                "the split, rerun with --overwrite."
            )

        if split_metadata.exists():

            raise FileExistsError(
                f"\nSplit metadata already exists:\n"
                f"{split_metadata}\n\n"
                "If you intentionally want to regenerate "
                "the split, rerun with --overwrite."
            )

    if overwrite:

        if validation_dir.exists():

            print(
                f"Removing existing:\n"
                f"{validation_dir}"
            )

            shutil.rmtree(
                validation_dir
            )

        if test_dir.exists():

            print(
                f"Removing existing:\n"
                f"{test_dir}"
            )

            shutil.rmtree(
                test_dir
            )

        if split_metadata.exists():

            print(
                f"Removing existing:\n"
                f"{split_metadata}"
            )

            split_metadata.unlink()

    validation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        validation_dir,
        test_dir,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_dir = Path(
        args.input
    ).resolve()

    print("=" * 100)
    print("NEPALI — PHASE 2 DATA SPLIT")
    print("=" * 100)

    print(
        f"Input directory    : {input_dir}"
    )

    print(
        f"Training tokens    : "
        f"{args.train_tokens:,}"
    )

    print(
        f"Validation tokens  : "
        f"{args.validation_tokens:,}"
    )

    print(
        f"Test tokens        : "
        f"{args.test_tokens:,}"
    )

    print(
        f"Shard size         : "
        f"{args.shard_size:,}"
    )

    print("=" * 100)

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_dir}"
        )

    # ========================================================
    # FIND ORIGINAL SHARDS
    # ========================================================

    shard_paths = get_input_shards(
        input_dir
    )

    print(
        f"\nFound {len(shard_paths)} "
        f"input token shards."
    )

    # ========================================================
    # COUNT TOTAL INPUT TOKENS
    # ========================================================

    total_input_tokens = count_tokens(
        shard_paths
    )

    print(
        f"\nTotal input tokens: "
        f"{total_input_tokens:,}"
    )

    # ========================================================
    # CALCULATE SPLIT BOUNDARIES
    # ========================================================

    train_start = 0

    train_end = (
        train_start
        + args.train_tokens
    )

    validation_start = train_end

    validation_end = (
        validation_start
        + args.validation_tokens
    )

    test_start = validation_end

    test_end = (
        test_start
        + args.test_tokens
    )

    required_tokens = test_end

    unused_tokens = (
        total_input_tokens
        - required_tokens
    )

    if total_input_tokens < required_tokens:

        raise ValueError(
            "\nNot enough tokens for requested split.\n\n"
            f"Available tokens : "
            f"{total_input_tokens:,}\n"
            f"Required tokens  : "
            f"{required_tokens:,}\n"
            f"Missing tokens   : "
            f"{required_tokens - total_input_tokens:,}"
        )

    print("\n")
    print("-" * 100)
    print("SPLIT BOUNDARIES")
    print("-" * 100)

    print(
        f"TRAIN      : "
        f"[{train_start:,}, {train_end:,})"
    )

    print(
        f"VALIDATION : "
        f"[{validation_start:,}, "
        f"{validation_end:,})"
    )

    print(
        f"TEST       : "
        f"[{test_start:,}, {test_end:,})"
    )

    print(
        f"UNUSED     : "
        f"{unused_tokens:,} tokens"
    )

    print("-" * 100)

    # ========================================================
    # IMPORTANT OVERLAP CHECK
    # ========================================================

    assert train_start == 0

    assert (
        train_end
        == validation_start
    )

    assert (
        validation_end
        == test_start
    )

    assert (
        train_end
        <= validation_start
    )

    assert (
        validation_end
        <= test_start
    )

    assert (
        train_end
        == args.train_tokens
    )

    assert (
        validation_end - validation_start
        == args.validation_tokens
    )

    assert (
        test_end - test_start
        == args.test_tokens
    )

    print(
        "\n✓ Train / validation / test ranges "
        "are strictly non-overlapping."
    )

    # ========================================================
    # PREPARE TEMPORARY OUTPUT
    #
    # We first write the new training shards into a temporary
    # directory. This prevents us from deleting the existing
    # train shards before the new split has been successfully
    # created.
    # ========================================================

    temp_train_dir = (
        input_dir / ".train_split_tmp"
    )

    if temp_train_dir.exists():

        print(
            f"\nRemoving previous temporary directory:\n"
            f"{temp_train_dir}"
        )

        shutil.rmtree(
            temp_train_dir
        )

    temp_train_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_dir, test_dir = (
        prepare_output_directories(
            input_dir,
            args.overwrite,
        )
    )

    # ========================================================
    # WRITERS
    # ========================================================

    train_writer = BinaryShardWriter(
        output_dir=temp_train_dir,
        prefix="train",
        shard_size=args.shard_size,
    )

    validation_writer = BinaryShardWriter(
        output_dir=validation_dir,
        prefix="val",
        shard_size=args.validation_tokens,
    )

    test_writer = BinaryShardWriter(
        output_dir=test_dir,
        prefix="test",
        shard_size=args.test_tokens,
    )

    # ========================================================
    # STREAM AND SPLIT
    # ========================================================

    processed_tokens = 0

    print("\n")
    print("=" * 100)
    print("CREATING SPLITS")
    print("=" * 100)

    for chunk in iter_token_chunks(
        shard_paths
    ):

        chunk_start = (
            processed_tokens
        )

        chunk_end = (
            processed_tokens
            + len(chunk)
        )

        # ====================================================
        # TRAIN RANGE
        # ====================================================

        train_chunk_start = max(
            chunk_start,
            train_start,
        )

        train_chunk_end = min(
            chunk_end,
            train_end,
        )

        if (
            train_chunk_start
            < train_chunk_end
        ):

            local_start = (
                train_chunk_start
                - chunk_start
            )

            local_end = (
                train_chunk_end
                - chunk_start
            )

            train_writer.add(
                chunk[
                    local_start:
                    local_end
                ]
            )

        # ====================================================
        # VALIDATION RANGE
        # ====================================================

        validation_chunk_start = max(
            chunk_start,
            validation_start,
        )

        validation_chunk_end = min(
            chunk_end,
            validation_end,
        )

        if (
            validation_chunk_start
            < validation_chunk_end
        ):

            local_start = (
                validation_chunk_start
                - chunk_start
            )

            local_end = (
                validation_chunk_end
                - chunk_start
            )

            validation_writer.add(
                chunk[
                    local_start:
                    local_end
                ]
            )

        # ====================================================
        # TEST RANGE
        # ====================================================

        test_chunk_start = max(
            chunk_start,
            test_start,
        )

        test_chunk_end = min(
            chunk_end,
            test_end,
        )

        if (
            test_chunk_start
            < test_chunk_end
        ):

            local_start = (
                test_chunk_start
                - chunk_start
            )

            local_end = (
                test_chunk_end
                - chunk_start
            )

            test_writer.add(
                chunk[
                    local_start:
                    local_end
                ]
            )

        # ====================================================
        # UPDATE POSITION
        # ====================================================

        processed_tokens = chunk_end

        # Once we've passed the test range, nothing else
        # needs to be read.
        if processed_tokens >= test_end:

            break

    # ========================================================
    # CLOSE WRITERS
    # ========================================================

    train_writer.close()
    validation_writer.close()
    test_writer.close()

    # ========================================================
    # VERIFY EXACT TOKEN COUNTS
    # ========================================================

    print("\n")
    print("=" * 100)
    print("VERIFYING SPLITS")
    print("=" * 100)

    print(
        f"Training tokens produced   : "
        f"{train_writer.total_tokens:,}"
    )

    print(
        f"Validation tokens produced : "
        f"{validation_writer.total_tokens:,}"
    )

    print(
        f"Test tokens produced       : "
        f"{test_writer.total_tokens:,}"
    )

    if (
        train_writer.total_tokens
        != args.train_tokens
    ):

        raise RuntimeError(
            "Training token count mismatch:\n"
            f"Expected: {args.train_tokens:,}\n"
            f"Actual:   {train_writer.total_tokens:,}"
        )

    if (
        validation_writer.total_tokens
        != args.validation_tokens
    ):

        raise RuntimeError(
            "Validation token count mismatch:\n"
            f"Expected: {args.validation_tokens:,}\n"
            f"Actual:   {validation_writer.total_tokens:,}"
        )

    if (
        test_writer.total_tokens
        != args.test_tokens
    ):

        raise RuntimeError(
            "Test token count mismatch:\n"
            f"Expected: {args.test_tokens:,}\n"
            f"Actual:   {test_writer.total_tokens:,}"
        )

    print(
        "\n✓ Exact token counts verified."
    )

    # ========================================================
    # REPLACE ORIGINAL TRAIN SHARDS
    #
    # At this point the new train/validation/test data has
    # been successfully written and verified.
    #
    # Only now do we remove the original train shards.
    # ========================================================

    print("\n")
    print("=" * 100)
    print("INSTALLING TRAINING-ONLY SHARDS")
    print("=" * 100)

    # Re-identify the original shards because the list was
    # created before writing the temporary output.
    original_train_shards = sorted(
        input_dir.glob("train_*.bin")
    )

    print(
        f"Removing {len(original_train_shards)} "
        f"original train shards..."
    )

    for path in original_train_shards:

        path.unlink()

    # Move temporary train shards into tokenized/
    new_train_shards = sorted(
        temp_train_dir.glob(
            "train_*.bin"
        )
    )

    for source_path in new_train_shards:

        destination_path = (
            input_dir
            / source_path.name
        )

        shutil.move(
            str(source_path),
            str(destination_path),
        )

    # Remove temporary directory
    shutil.rmtree(
        temp_train_dir
    )

    print(
        "✓ Training-only shards installed."
    )

    # ========================================================
    # FINAL OUTPUT VERIFICATION
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINAL OUTPUT VERIFICATION")
    print("=" * 100)

    final_train_shards = sorted(
        input_dir.glob("train_*.bin")
    )

    final_validation_shards = sorted(
        validation_dir.glob("val_*.bin")
    )

    final_test_shards = sorted(
        test_dir.glob("test_*.bin")
    )

    final_train_tokens = sum(
        path.stat().st_size // 2
        for path in final_train_shards
    )

    final_validation_tokens = sum(
        path.stat().st_size // 2
        for path in final_validation_shards
    )

    final_test_tokens = sum(
        path.stat().st_size // 2
        for path in final_test_shards
    )

    print(
        f"Training shards    : "
        f"{len(final_train_shards)}"
    )

    print(
        f"Training tokens    : "
        f"{final_train_tokens:,}"
    )

    print(
        f"Validation shards  : "
        f"{len(final_validation_shards)}"
    )

    print(
        f"Validation tokens  : "
        f"{final_validation_tokens:,}"
    )

    print(
        f"Test shards        : "
        f"{len(final_test_shards)}"
    )

    print(
        f"Test tokens        : "
        f"{final_test_tokens:,}"
    )

    # Exact final checks
    assert (
        final_train_tokens
        == args.train_tokens
    )

    assert (
        final_validation_tokens
        == args.validation_tokens
    )

    assert (
        final_test_tokens
        == args.test_tokens
    )

    # There should be no train shards in the validation/test
    # directories and no validation/test shards in tokenized/.
    assert all(
        path.parent == input_dir
        for path in final_train_shards
    )

    assert all(
        path.parent == validation_dir
        for path in final_validation_shards
    )

    assert all(
        path.parent == test_dir
        for path in final_test_shards
    )

    print(
        "\n✓ Final shard structure verified."
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "language": "Nepali",

        "source_directory": str(
            input_dir
        ),

        "input_tokens_before_split": (
            total_input_tokens
        ),

        "train_tokens": (
            final_train_tokens
        ),

        "validation_tokens": (
            final_validation_tokens
        ),

        "test_tokens": (
            final_test_tokens
        ),

        "unused_tokens": (
            unused_tokens
        ),

        "dtype": "uint16",

        "bytes_per_token": 2,

        "train_shard_size": (
            args.shard_size
        ),

        "split_method": (
            "strict sequential token-stream split"
        ),

        "overlap": False,

        "train_range": {
            "start": train_start,
            "end_exclusive": train_end,
            "tokens": args.train_tokens,
        },

        "validation_range": {
            "start": validation_start,
            "end_exclusive": validation_end,
            "tokens": args.validation_tokens,
        },

        "test_range": {
            "start": test_start,
            "end_exclusive": test_end,
            "tokens": args.test_tokens,
        },

        "train_shards": [
            {
                "file": path.name,
                "tokens": path.stat().st_size // 2,
            }
            for path in final_train_shards
        ],

        "validation_shards": [
            {
                "file": path.name,
                "tokens": path.stat().st_size // 2,
            }
            for path in final_validation_shards
        ],

        "test_shards": [
            {
                "file": path.name,
                "tokens": path.stat().st_size // 2,
            }
            for path in final_test_shards
        ],
    }

    metadata_path = (
        input_dir
        / "split_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 100)
    print("NEPALI — PHASE 2 DATA SPLIT COMPLETE")
    print("=" * 100)

    print(
        f"Original tokens    : "
        f"{total_input_tokens:,}"
    )

    print(
        f"Training tokens    : "
        f"{final_train_tokens:,}"
    )

    print(
        f"Validation tokens  : "
        f"{final_validation_tokens:,}"
    )

    print(
        f"Test tokens        : "
        f"{final_test_tokens:,}"
    )

    print(
        f"Unused tokens      : "
        f"{unused_tokens:,}"
    )

    print()
    print(
        "Train/validation/test overlap: NONE"
    )

    print()
    print(
        f"Metadata           : "
        f"{metadata_path}"
    )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()