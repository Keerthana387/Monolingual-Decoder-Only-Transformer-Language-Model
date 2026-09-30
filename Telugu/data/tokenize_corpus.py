import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

TOKENS_PER_SHARD = 10_000_000

DTYPE = np.uint16


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(
    tokenizer_path: Path,
) -> spm.SentencePieceProcessor:

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.load(
        str(tokenizer_path)
    ):
        raise RuntimeError(
            f"Could not load tokenizer: "
            f"{tokenizer_path}"
        )

    return tokenizer


# ============================================================
# MAIN TOKENIZATION
# ============================================================

def tokenize_corpus(
    input_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
):
    """
    Tokenize the final deduplicated Telugu corpus.

    The input is processed line-by-line so that the entire
    4.6 GB corpus is never loaded into RAM.

    Token IDs are stored as uint16 because the selected
    vocabulary is 10,000 tokens.
    """

    tokenizer = load_tokenizer(
        tokenizer_path
    )

    vocab_size = tokenizer.get_piece_size()

    if vocab_size > np.iinfo(DTYPE).max:
        raise ValueError(
            f"Vocabulary size {vocab_size} cannot be "
            f"stored using {DTYPE}."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_tokens = []
    shard_token_count = 0

    shard_index = 0

    total_documents = 0
    total_characters = 0
    total_tokens = 0

    shard_metadata = []

    # --------------------------------------------------------
    # Process corpus
    # --------------------------------------------------------

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as f:

        for line in tqdm(
            f,
            desc="Tokenizing Telugu corpus",
            unit="documents",
        ):

            text = line.rstrip("\n\r")

            if not text.strip():
                continue

            total_documents += 1
            total_characters += len(text)

            token_ids = tokenizer.encode(
                text,
                out_type=int,
            )

            if not token_ids:
                continue

            total_tokens += len(token_ids)

            shard_tokens.extend(
                token_ids
            )

            shard_token_count += len(
                token_ids
            )

            # ------------------------------------------------
            # Write shard
            # ------------------------------------------------

            if shard_token_count >= TOKENS_PER_SHARD:

                shard_array = np.asarray(
                    shard_tokens,
                    dtype=DTYPE,
                )

                shard_path = (
                    output_dir
                    / f"train_{shard_index:03d}.bin"
                )

                shard_array.tofile(
                    shard_path
                )

                actual_count = len(
                    shard_array
                )

                shard_metadata.append(
                    {
                        "file": shard_path.name,
                        "tokens": actual_count,
                    }
                )

                print(
                    f"\nSaved {shard_path.name}: "
                    f"{actual_count:,} tokens"
                )

                shard_index += 1

                shard_tokens = []
                shard_token_count = 0

    # --------------------------------------------------------
    # Final shard
    # --------------------------------------------------------

    if shard_tokens:

        shard_array = np.asarray(
            shard_tokens,
            dtype=DTYPE,
        )

        shard_path = (
            output_dir
            / f"train_{shard_index:03d}.bin"
        )

        shard_array.tofile(
            shard_path
        )

        actual_count = len(
            shard_array
        )

        shard_metadata.append(
            {
                "file": shard_path.name,
                "tokens": actual_count,
            }
        )

        print(
            f"\nSaved {shard_path.name}: "
            f"{actual_count:,} tokens"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "input_file": str(input_path),
        "tokenizer": str(tokenizer_path),
        "vocab_size": vocab_size,
        "dtype": str(DTYPE),
        "documents": total_documents,
        "characters": total_characters,
        "tokens": total_tokens,
        "tokens_millions": (
            total_tokens / 1_000_000
        ),
        "tokens_per_character": (
            total_tokens / total_characters
            if total_characters > 0
            else 0.0
        ),
        "tokens_per_shard": TOKENS_PER_SHARD,
        "num_shards": len(shard_metadata),
        "shards": shard_metadata,
    }

    metadata_path = (
        output_dir / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TELUGU TOKENIZATION COMPLETE")
    print("=" * 80)

    print(
        f"\nDocuments       : "
        f"{total_documents:,}"
    )

    print(
        f"Characters      : "
        f"{total_characters:,}"
    )

    print(
        f"Tokens          : "
        f"{total_tokens:,}"
    )

    print(
        f"Tokens (M)      : "
        f"{total_tokens / 1_000_000:.3f}"
    )

    print(
        f"Tokens/char     : "
        f"{total_tokens / total_characters:.4f}"
    )

    print(
        f"Vocabulary      : "
        f"{vocab_size:,}"
    )

    print(
        f"Shards          : "
        f"{len(shard_metadata)}"
    )

    print(
        f"\nMetadata saved:"
        f"\n{metadata_path}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    tokenize_corpus(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()