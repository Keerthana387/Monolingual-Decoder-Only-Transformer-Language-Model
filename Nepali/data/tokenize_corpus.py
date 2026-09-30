# ============================================================
# NEPALI — TOKENIZE CORPUS
# Phase 2 — Data Preparation
#
# Runs LOCALLY on CPU.
#
# Input:
#   - Already deduplicated + Unicode-normalized Nepali TXT
#   - Existing Nepali 8K BPE tokenizer JSON
#
# Output:
#   - Tokenized uint16 binary shards
#   - metadata.json
#
# Example:
#
# python tokenize_corpus.py \
#     --input /path/to/nepali_500M_tokenizer_training.txt \
#     --tokenizer /path/to/nepali_bpe_8000_sp.json \
#     --output /path/to/Nepali/data/tokenized
#
# ============================================================

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_SHARD_SIZE = 10_000_000


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize the Nepali corpus into uint16 binary shards."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Nepali input TXT corpus."
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the trained Nepali BPE tokenizer JSON."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory where tokenized shards will be written."
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of tokens per shard."
    )

    return parser.parse_args()


# ============================================================
# SHARD WRITER
# ============================================================

class ShardWriter:
    """
    Write token IDs into fixed-size uint16 binary shards.
    """

    def __init__(self, output_dir, shard_size):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.shard_size = shard_size

        self.buffer = np.empty(
            self.shard_size,
            dtype=np.uint16
        )

        self.position = 0
        self.shard_index = 0
        self.total_tokens = 0
        self.shards = []

    def add_tokens(self, token_ids):
        """
        Add token IDs to the current shard buffer.
        """

        token_ids = np.asarray(
            token_ids,
            dtype=np.uint16
        )

        start = 0

        while start < len(token_ids):

            remaining = self.shard_size - self.position

            take = min(
                remaining,
                len(token_ids) - start
            )

            self.buffer[
                self.position:
                self.position + take
            ] = token_ids[
                start:
                start + take
            ]

            self.position += take
            start += take
            self.total_tokens += take

            if self.position == self.shard_size:
                self.flush()

    def flush(self):
        """
        Write the current shard to disk.
        """

        if self.position == 0:
            return

        shard_path = (
            self.output_dir
            / f"train_{self.shard_index:03d}.bin"
        )

        self.buffer[
            :self.position
        ].tofile(shard_path)

        shard_tokens = self.position

        self.shards.append(
            {
                "file": shard_path.name,
                "tokens": shard_tokens
            }
        )

        print(
            f"Saved {shard_path.name} "
            f"({shard_tokens:,} tokens)"
        )

        self.shard_index += 1
        self.position = 0

    def close(self):
        """
        Flush the final partial shard.
        """

        self.flush()


# ============================================================
# MAIN TOKENIZATION
# ============================================================

def main():

    args = parse_args()

    input_path = Path(args.input)
    tokenizer_path = Path(args.tokenizer)
    output_dir = Path(args.output)

    print("=" * 100)
    print("NEPALI — CORPUS TOKENIZATION")
    print("=" * 100)

    print(f"Input corpus : {input_path}")
    print(f"Tokenizer    : {tokenizer_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Shard size   : {args.shard_size:,}")
    print("=" * 100)

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input corpus does not exist:\n{input_path}"
        )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist:\n{tokenizer_path}"
        )

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    vocab_size = tokenizer.get_vocab_size()

    print(
        f"Tokenizer vocabulary: {vocab_size:,}"
    )

    if vocab_size != 8000:
        raise ValueError(
            f"Expected Nepali 8K tokenizer, "
            f"but found vocabulary size {vocab_size:,}"
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SHARD WRITER
    # --------------------------------------------------------

    writer = ShardWriter(
        output_dir=output_dir,
        shard_size=args.shard_size
    )

    # --------------------------------------------------------
    # PROCESS CORPUS
    # --------------------------------------------------------

    print("\nTokenizing corpus...\n")

    total_lines = 0
    total_characters = 0
    unknown_tokens = 0

    file_size = input_path.stat().st_size

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as f:

        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc="Tokenizing"
        ) as pbar:

            for line in f:

                line_bytes = len(
                    line.encode("utf-8")
                )

                pbar.update(line_bytes)

                line = line.strip()

                if not line:
                    continue

                total_lines += 1
                total_characters += len(line)

                encoding = tokenizer.encode(line)

                token_ids = encoding.ids

                # ------------------------------------------------
                # UNKNOWN TOKEN CHECK
                # ------------------------------------------------

                unknown_id = tokenizer.token_to_id(
                    "<unk>"
                )

                if unknown_id is not None:
                    unknown_tokens += token_ids.count(
                        unknown_id
                    )

                # ------------------------------------------------
                # WRITE TOKENS
                # ------------------------------------------------

                writer.add_tokens(token_ids)

    # --------------------------------------------------------
    # FINALIZE
    # --------------------------------------------------------

    writer.close()

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "language": "Nepali",
        "vocab_size": vocab_size,
        "tokenizer": str(tokenizer_path),
        "input_file": str(input_path),

        "dtype": "uint16",

        "shard_size_tokens": args.shard_size,

        "total_tokens": writer.total_tokens,

        "total_lines": total_lines,

        "total_characters": total_characters,

        "unknown_tokens": unknown_tokens,

        "unknown_rate_percent": (
            100.0 * unknown_tokens / writer.total_tokens
            if writer.total_tokens > 0
            else 0.0
        ),

        "num_shards": len(writer.shards),

        "shards": writer.shards
    }

    metadata_path = (
        output_dir / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("NEPALI TOKENIZATION COMPLETE")
    print("=" * 100)

    print(
        f"Lines             : {total_lines:,}"
    )

    print(
        f"Characters        : {total_characters:,}"
    )

    print(
        f"Tokens            : {writer.total_tokens:,}"
    )

    print(
        f"Unknown tokens    : {unknown_tokens:,}"
    )

    print(
        f"Unknown rate      : "
        f"{metadata['unknown_rate_percent']:.6f}%"
    )

    print(
        f"Shards            : {len(writer.shards):,}"
    )

    print(
        f"Metadata          : {metadata_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()