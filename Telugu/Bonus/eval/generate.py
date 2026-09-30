import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm
import torch

from common import (
    load_model,
    assert_telugu_config,
    load_jsonl_texts,
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature,
):

    generated = input_ids.clone()

    for _ in range(max_new_tokens):

        context = generated[
            :,
            -model.config.max_seq_len:
        ]

        logits, _, _ = model(
            input_ids=context,
            targets=None,
        )

        next_logits = logits[:, -1, :]

        if temperature == 0.0:

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True,
            )

        else:

            probs = torch.softmax(
                next_logits / temperature,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                1,
            )

        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=1,
        )

    return generated


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
        "--tokenizer",
        required=True,
    )

    parser.add_argument(
        "--test-text",
        required=True,
        help="Path to Telugu test JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    set_seed(args.seed)

    device = torch.device("cuda")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.load(
        args.tokenizer
    ):

        raise RuntimeError(
            f"Could not load tokenizer: "
            f"{args.tokenizer}"
        )

    if tokenizer.vocab_size() != 6000:

        raise RuntimeError(
            "Expected 6000-token Telugu tokenizer, "
            f"got {tokenizer.vocab_size()}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, config, checkpoint_path = load_model(
        args.checkpoint_dir,
        device,
    )

    assert_telugu_config(config)

    # --------------------------------------------------------
    # TEST TEXT
    # --------------------------------------------------------

    texts = load_jsonl_texts(
        args.test_text
    )

    settings = {

        "greedy": 0.0,

        "temperature_0.5": 0.5,

        "temperature_1.0": 1.0,

        "temperature_1.5": 1.5,
    }

    required = (
        args.prefix_tokens
        + args.generation_tokens
    )

    records = []

    used = 0

    # --------------------------------------------------------
    # Generate from TEST documents
    # --------------------------------------------------------

    for doc_index, text in enumerate(texts):

        if used >= args.num_samples:

            break

        ids = tokenizer.encode(
            text,
            out_type=int,
        )

        if len(ids) < required:

            continue

        prefix_ids = ids[
            :args.prefix_tokens
        ]

        reference_ids = ids[
            args.prefix_tokens:
            args.prefix_tokens
            + args.generation_tokens
        ]

        prefix = torch.tensor(
            [prefix_ids],
            dtype=torch.long,
            device=device,
        )

        for setting, temperature in settings.items():

            generated = generate(
                model,
                prefix,
                args.generation_tokens,
                temperature,
            )

            continuation_ids = generated[
                0,
                len(prefix_ids):
            ].tolist()

            records.append({

                "sample_id": used,

                "source_document": doc_index,

                "split": "test",

                "setting": setting,

                "temperature": temperature,

                "prefix": tokenizer.decode(
                    prefix_ids
                ),

                "reference": tokenizer.decode(
                    reference_ids
                ),

                "generated": tokenizer.decode(
                    continuation_ids
                ),
            })

        used += 1

    if not records:

        raise RuntimeError(
            "No usable test documents were long enough "
            "for the requested prefix + generation length."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 80)
    print("TELUGU — PHASE 2 TEST GENERATION")
    print("=" * 80)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Test samples: {used}"
    )

    print(
        f"Records: {len(records)}"
    )

    print(
        f"Saved: {out}"
    )


if __name__ == "__main__":
    main()