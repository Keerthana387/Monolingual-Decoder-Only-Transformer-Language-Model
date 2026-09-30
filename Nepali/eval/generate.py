import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .common import load_model, assert_nepali_config, load_jsonl_texts


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_batch(model, prefixes, max_new_tokens, temperatures):
    """
    Generate all requested temperature settings in one batched inference stream.

    prefixes: [B, T]
    temperatures: list of floats, one per setting.

    The four settings are expanded along the batch dimension, so the model
    performs one forward pass per generation step instead of one pass per
    sample/temperature combination.
    """
    bsz = prefixes.size(0)
    n_settings = len(temperatures)

    generated = prefixes.unsqueeze(1).expand(
        bsz, n_settings, prefixes.size(1)
    ).reshape(bsz * n_settings, prefixes.size(1)).clone()

    temperature_tensor = torch.tensor(
        temperatures, dtype=torch.float32, device=prefixes.device
    ).repeat(bsz)

    for _ in range(max_new_tokens):
        context = generated[:, -model.config.max_seq_len:]

        logits, _ = model(input_ids=context, targets=None)
        next_logits = logits[:, -1, :].float()

        next_token = torch.empty(
            (bsz * n_settings, 1),
            dtype=torch.long,
            device=generated.device,
        )

        greedy_mask = temperature_tensor == 0.0

        if greedy_mask.any():
            next_token[greedy_mask] = torch.argmax(
                next_logits[greedy_mask], dim=-1, keepdim=True
            )

        sample_mask = ~greedy_mask
        if sample_mask.any():
            scaled = next_logits[sample_mask] / temperature_tensor[
                sample_mask
            ].unsqueeze(1)
            probs = torch.softmax(scaled, dim=-1)
            next_token[sample_mask] = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_token], dim=1)

    return generated.reshape(bsz, n_settings, -1)


def load_nepali_tokenizer(path):
    tokenizer = Tokenizer.from_file(path)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != 8000:
        raise RuntimeError(
            f"Expected 8000-token Nepali tokenizer, got {vocab_size}"
        )
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    set_seed(args.seed)
    device = torch.device("cuda")

    tokenizer = load_nepali_tokenizer(args.tokenizer)

    model, config, checkpoint_path = load_model(
        args.checkpoint_dir, device
    )
    assert_nepali_config(config)

    texts = load_jsonl_texts(args.test_text)

    required = args.prefix_tokens + args.generation_tokens
    settings = [
        ("greedy", 0.0),
        ("temperature_0.5", 0.5),
        ("temperature_1.0", 1.0),
        ("temperature_1.5", 1.5),
    ]

    samples = []
    for doc_index, text in enumerate(texts):
        if len(samples) >= args.num_samples:
            break

        ids = tokenizer.encode(text).ids
        if len(ids) < required:
            continue

        samples.append({
            "source_document": doc_index,
            "prefix_ids": ids[:args.prefix_tokens],
            "reference_ids": ids[
                args.prefix_tokens:
                args.prefix_tokens + args.generation_tokens
            ],
        })

    if not samples:
        raise RuntimeError(
            "No usable test documents were long enough for "
            "the requested prefix + generation length."
        )

    records = []

    # Batched generation reduces the expensive Python/model-call overhead
    # without changing the Phase 2 experiment: same samples, same prefix,
    # same four generation settings, same 64-token continuation.
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        prefix_ids = torch.tensor(
            [s["prefix_ids"] for s in batch],
            dtype=torch.long,
            device=device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            generated = generate_batch(
                model,
                prefix_ids,
                args.generation_tokens,
                [temperature for _, temperature in settings],
            )

        generated = generated.cpu()

        for i, sample in enumerate(batch):
            prefix_len = len(sample["prefix_ids"])

            for setting_index, (setting, temperature) in enumerate(settings):
                continuation_ids = generated[
                    i, setting_index, prefix_len:
                ].tolist()

                records.append({
                    "sample_id": start + i,
                    "source_document": sample["source_document"],
                    "setting": setting,
                    "temperature": temperature,
                    "prefix": tokenizer.decode(
                        sample["prefix_ids"]
                    ),
                    "reference": tokenizer.decode(
                        sample["reference_ids"]
                    ),
                    "generated": tokenizer.decode(
                        continuation_ids
                    ),
                })

        print(
            f"Generated samples: "
            f"{min(start + args.batch_size, len(samples))}/{len(samples)}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("=" * 80)
    print("NEPALI — PHASE 2 GENERATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Mixed precision: FP16 autocast")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {len(samples)}")
    print(f"Records: {len(records)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
