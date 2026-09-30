# ============================================================
# LMA-Indiv — Phase 2
# NEPALI — TOKENIZED DATA VALIDATION
#
# Runs locally on CPU.
#
# Verifies:
#   - shard structure
#   - uint16 integrity
#   - token counts
#   - vocabulary range
#   - metadata consistency
#   - train / validation / test separation
#
# ============================================================

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Validate Nepali Phase 2 tokenized data."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to Nepali/data/tokenized",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Tokenizer vocabulary size.",
    )

    return parser.parse_args()


# ============================================================
# COUNT / VALIDATE SHARD
# ============================================================

def validate_shard(
    path,
    vocab_size,
):

    size_bytes = path.stat().st_size

    if size_bytes % 2 != 0:

        raise ValueError(
            f"Invalid uint16 file size:\n{path}\n"
            f"Size: {size_bytes} bytes"
        )

    tokens = size_bytes // 2

    data = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )

    if len(data) != tokens:

        raise RuntimeError(
            f"Token count mismatch while reading:\n{path}"
        )

    # --------------------------------------------------------
    # TOKEN RANGE
    # --------------------------------------------------------

    if len(data) > 0:

        minimum = int(data.min())
        maximum = int(data.max())

        if minimum < 0:

            raise ValueError(
                f"Negative token ID found in {path}: "
                f"{minimum}"
            )

        if maximum >= vocab_size:

            raise ValueError(
                f"Token ID outside vocabulary in {path}:\n"
                f"Maximum token ID: {maximum}\n"
                f"Vocabulary size: {vocab_size}"
            )

    else:

        minimum = None
        maximum = None

    del data

    return {
        "file": path.name,
        "tokens": tokens,
        "bytes": size_bytes,
        "min_token_id": minimum,
        "max_token_id": maximum,
    }


# ============================================================
# VALIDATE DIRECTORY
# ============================================================

def validate_directory(
    directory,
    pattern,
    vocab_size,
    description,
):

    paths = sorted(
        directory.glob(pattern)
    )

    if not paths:

        raise FileNotFoundError(
            f"No {pattern} files found in "
            f"{description}:\n{directory}"
        )

    print(
        f"\n{description}: {len(paths)} shard(s)"
    )

    results = []

    total_tokens = 0

    for path in tqdm(
        paths,
        desc=f"Validating {description}",
        unit="shard",
    ):

        result = validate_shard(
            path,
            vocab_size,
        )

        results.append(result)

        total_tokens += result["tokens"]

    return results, total_tokens


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    data_dir = Path(
        args.data_dir
    ).resolve()

    print("=" * 100)
    print("NEPALI — TOKENIZED DATA VALIDATION")
    print("=" * 100)

    print(
        f"Data directory : {data_dir}"
    )

    print(
        f"Vocabulary size : {args.vocab_size:,}"
    )

    print("=" * 100)

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Data directory does not exist:\n"
            f"{data_dir}"
        )

    # ========================================================
    # REQUIRED DIRECTORIES
    # ========================================================

    validation_dir = (
        data_dir / "validation"
    )

    test_dir = (
        data_dir / "test"
    )

    if not validation_dir.exists():

        raise FileNotFoundError(
            f"Validation directory missing:\n"
            f"{validation_dir}"
        )

    if not test_dir.exists():

        raise FileNotFoundError(
            f"Test directory missing:\n"
            f"{test_dir}"
        )

    # ========================================================
    # TRAIN
    # ========================================================

    train_results, train_tokens = (
        validate_directory(
            data_dir,
            "train_*.bin",
            args.vocab_size,
            "Training data",
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_results, validation_tokens = (
        validate_directory(
            validation_dir,
            "val_*.bin",
            args.vocab_size,
            "Validation data",
        )
    )

    # ========================================================
    # TEST
    # ========================================================

    test_results, test_tokens = (
        validate_directory(
            test_dir,
            "test_*.bin",
            args.vocab_size,
            "Test data",
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata_path = (
        data_dir / "split_metadata.json"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Missing split metadata:\n"
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    # ========================================================
    # CHECK EXPECTED COUNTS
    # ========================================================

    expected_train = metadata[
        "train_tokens"
    ]

    expected_validation = metadata[
        "validation_tokens"
    ]

    expected_test = metadata[
        "test_tokens"
    ]

    if train_tokens != expected_train:

        raise AssertionError(
            "Training token count mismatch:\n"
            f"Actual:   {train_tokens:,}\n"
            f"Expected: {expected_train:,}"
        )

    if validation_tokens != expected_validation:

        raise AssertionError(
            "Validation token count mismatch:\n"
            f"Actual:   {validation_tokens:,}\n"
            f"Expected: {expected_validation:,}"
        )

    if test_tokens != expected_test:

        raise AssertionError(
            "Test token count mismatch:\n"
            f"Actual:   {test_tokens:,}\n"
            f"Expected: {expected_test:,}"
        )

    print(
        "\n✓ Token counts match split_metadata.json"
    )

    # ========================================================
    # CHECK SHARD LOCATIONS
    # ========================================================

    root_train_files = sorted(
        data_dir.glob("train_*.bin")
    )

    nested_train_files = sorted(
        validation_dir.glob("train_*.bin")
    ) + sorted(
        test_dir.glob("train_*.bin")
    )

    if nested_train_files:

        raise AssertionError(
            "Unexpected train_*.bin files found "
            "inside validation/test directories."
        )

    if not root_train_files:

        raise AssertionError(
            "No training shards found directly "
            "inside tokenized/."
        )

    # ========================================================
    # CHECK SPLIT RANGES
    # ========================================================

    train_range = metadata[
        "train_range"
    ]

    validation_range = metadata[
        "validation_range"
    ]

    test_range = metadata[
        "test_range"
    ]

    train_start = train_range["start"]
    train_end = train_range["end_exclusive"]

    validation_start = (
        validation_range["start"]
    )

    validation_end = (
        validation_range["end_exclusive"]
    )

    test_start = test_range["start"]
    test_end = test_range["end_exclusive"]

    if train_start != 0:

        raise AssertionError(
            "Training range does not start at zero."
        )

    if train_end != validation_start:

        raise AssertionError(
            "Training and validation ranges overlap "
            "or contain a gap."
        )

    if validation_end != test_start:

        raise AssertionError(
            "Validation and test ranges overlap "
            "or contain a gap."
        )

    if not (
        train_end
        <= validation_start
        <= validation_end
        <= test_start
        <= test_end
    ):

        raise AssertionError(
            "Invalid split ordering."
        )

    if metadata.get(
        "overlap"
    ) is not False:

        raise AssertionError(
            "Metadata does not explicitly indicate "
            "overlap=False."
        )

    print(
        "✓ Train / validation / test ranges "
        "are strictly non-overlapping"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 100)
    print("NEPALI — DATA VALIDATION PASSED")
    print("=" * 100)

    print(
        f"Train tokens       : {train_tokens:,}"
    )

    print(
        f"Validation tokens  : {validation_tokens:,}"
    )

    print(
        f"Test tokens        : {test_tokens:,}"
    )

    print(
        f"Train shards       : {len(train_results)}"
    )

    print(
        f"Validation shards  : "
        f"{len(validation_results)}"
    )

    print(
        f"Test shards        : "
        f"{len(test_results)}"
    )

    print()
    print("✓ All token IDs are within vocabulary range")
    print("✓ All binary files are valid uint16 shards")
    print("✓ Token counts match metadata")
    print("✓ Directory structure is correct")
    print("✓ Train/validation/test ranges do not overlap")

    print("=" * 100)


if __name__ == "__main__":
    main()