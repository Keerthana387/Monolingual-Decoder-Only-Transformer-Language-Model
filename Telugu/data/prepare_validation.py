import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm.auto import tqdm


# ============================================================
# CONSTANTS
# ============================================================

SEQUENCE_LENGTH = 512
BLOCK_SIZE = SEQUENCE_LENGTH + 1

DEFAULT_OUTPUT_DIR = (
    "Telugu/data/tokenized/validation"
)

DEFAULT_TOKENIZER = (
    "Telugu/data/tokenizer/model/telugu_bpe_6000.model"
)

SHARD_SIZE = 10_000_000


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Tokenize Telugu validation JSONL "
            "using the final SentencePiece tokenizer."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Telugu validation JSONL file.",
    )

    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=(
            "Path to the final SentencePiece "
            "model file."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where validation token "
            "shards and metadata are saved."
        ),
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=(
            "Maximum number of tokens per output "
            "binary shard."
        ),
    )

    return parser.parse_args()


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(record):

    # ---------------------------------------------------------
    # Most of the project's JSONL data uses "text".
    # ---------------------------------------------------------

    if "text" in record:

        text = record["text"]

        if isinstance(text, str):

            return text

    # ---------------------------------------------------------
    # Fallbacks for common validation formats.
    # ---------------------------------------------------------

    for key in (
        "content",
        "document",
        "body",
    ):

        if key in record:

            value = record[key]

            if isinstance(value, str):

                return value

    return ""


# ============================================================
# WRITE SHARD
# ============================================================

def write_shard(
    token_buffer,
    output_dir,
    shard_index,
):

    if not token_buffer:

        return 0, None

    output_path = (
        Path(output_dir)
        / f"val_{shard_index:03d}.bin"
    )

    array = np.asarray(
        token_buffer,
        dtype=np.uint16,
    )

    array.tofile(output_path)

    return len(array), output_path


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_path = Path(args.input)

    tokenizer_path = Path(args.tokenizer)

    output_dir = Path(args.output)

    # ---------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{input_path}"
        )

    if not tokenizer_path.exists():

        raise FileNotFoundError(
            f"Tokenizer not found:\n"
            f"{tokenizer_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------

    print("=" * 80)
    print("TELUGU — VALIDATION TOKENIZATION")
    print("=" * 80)

    print(
        f"\nInput:"
        f"\n{input_path}"
    )

    print(
        f"\nTokenizer:"
        f"\n{tokenizer_path}"
    )

    print(
        f"\nOutput:"
        f"\n{output_dir}"
    )

    tokenizer = spm.SentencePieceProcessor()

    tokenizer.load(
        str(tokenizer_path)
    )

    vocab_size = tokenizer.get_piece_size()

    print(
        f"\nVocabulary size: "
        f"{vocab_size:,}"
    )

    # ---------------------------------------------------------
    # Remove old validation shards
    # ---------------------------------------------------------

    old_shards = sorted(
        output_dir.glob("val_*.bin")
    )

    if old_shards:

        print(
            f"\nRemoving {len(old_shards)} "
            f"existing validation shards..."
        )

        for path in old_shards:

            path.unlink()

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    documents = 0

    characters = 0

    tokens = 0

    unknown_tokens = 0

    shard_index = 0

    token_buffer = []

    shard_token_counts = []

    # ---------------------------------------------------------
    # Process JSONL
    # ---------------------------------------------------------

    print("\nTokenizing validation set...")
    print("-" * 80)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Validation",
            unit="docs",
        ):

            line = line.strip()

            if not line:

                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                continue

            text = extract_text(record)

            if not text:

                continue

            documents += 1

            characters += len(text)

            ids = tokenizer.encode(
                text,
                out_type=int,
            )

            tokens += len(ids)

            unknown_id = (
                tokenizer.unk_id()
            )

            if unknown_id >= 0:

                unknown_tokens += sum(
                    token_id == unknown_id
                    for token_id in ids
                )

            token_buffer.extend(ids)

            # -------------------------------------------------
            # Write full shards
            # -------------------------------------------------

            while len(token_buffer) >= args.shard_size:

                shard_tokens = token_buffer[
                    :args.shard_size
                ]

                token_buffer = token_buffer[
                    args.shard_size:
                ]

                count, path = write_shard(
                    shard_tokens,
                    output_dir,
                    shard_index,
                )

                shard_token_counts.append(
                    count
                )

                print(
                    f"\nSaved "
                    f"{path.name}: "
                    f"{count:,} tokens"
                )

                shard_index += 1

    # ---------------------------------------------------------
    # Final partial shard
    # ---------------------------------------------------------

    if token_buffer:

        count, path = write_shard(
            token_buffer,
            output_dir,
            shard_index,
        )

        shard_token_counts.append(
            count
        )

        print(
            f"\nSaved "
            f"{path.name}: "
            f"{count:,} tokens"
        )

        shard_index += 1

    # ---------------------------------------------------------
    # Validation block statistics
    # ---------------------------------------------------------

    usable_tokens = (
        (tokens // BLOCK_SIZE)
        * BLOCK_SIZE
    )

    discarded_tokens = (
        tokens - usable_tokens
    )

    num_samples = (
        usable_tokens // BLOCK_SIZE
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {

        "input_file": str(
            input_path
        ),

        "tokenizer": str(
            tokenizer_path
        ),

        "vocab_size": vocab_size,

        "sequence_length": (
            SEQUENCE_LENGTH
        ),

        "block_size": BLOCK_SIZE,

        "documents": documents,

        "characters": characters,

        "tokens": tokens,

        "unknown_tokens": (
            unknown_tokens
        ),

        "unk_rate_percent": (
            100.0
            * unknown_tokens
            / max(tokens, 1)
        ),

        "usable_tokens": (
            usable_tokens
        ),

        "discarded_tokens": (
            discarded_tokens
        ),

        "samples": num_samples,

        "num_shards": (
            len(shard_token_counts)
        ),

        "shard_token_counts": (
            shard_token_counts
        ),
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TELUGU VALIDATION TOKENIZATION COMPLETE")
    print("=" * 80)

    print(
        f"\nDocuments       : "
        f"{documents:,}"
    )

    print(
        f"Characters      : "
        f"{characters:,}"
    )

    print(
        f"Tokens          : "
        f"{tokens:,}"
    )

    print(
        f"Tokens (M)      : "
        f"{tokens / 1_000_000:.3f}"
    )

    print(
        f"Vocabulary      : "
        f"{vocab_size:,}"
    )

    print(
        f"Unknown tokens  : "
        f"{unknown_tokens:,}"
    )

    print(
        f"UNK rate        : "
        f"{metadata['unk_rate_percent']:.6f}%"
    )

    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Block size      : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Samples         : "
        f"{num_samples:,}"
    )

    print(
        f"Discarded       : "
        f"{discarded_tokens:,}"
    )

    print(
        f"Shards          : "
        f"{len(shard_token_counts)}"
    )

    print(
        f"\nMetadata saved:"
        f"\n{metadata_path}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()