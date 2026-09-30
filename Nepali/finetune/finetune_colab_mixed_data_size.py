"""
===============================================================================
NEPALI — PHASE 3 FINE-TUNING — MIXED DATA-SIZE EXPERIMENT
===============================================================================

Each experiment uses EXACTLY 20,000 fine-tuning examples.

Mixtures:
    2,000 reasoning  + 18,000 Phase 2 pretraining = 20,000
    5,000 reasoning  + 15,000 Phase 2 pretraining = 20,000
    8,000 reasoning  + 12,000 Phase 2 pretraining = 20,000
   12,000 reasoning  +  8,000 Phase 2 pretraining = 20,000
   16,000 reasoning  +  4,000 Phase 2 pretraining = 20,000
   20,000 reasoning  +      0 Phase 2 pretraining = 20,000

Experiment controls:
    - Same Phase 2 pretrained checkpoint for every run.
    - Same fixed 8000-token Nepali tokenizer.
    - Same reasoning validation set for every run.
    - Exactly 20,000 training examples per run.
    - Only the training-data composition changes.
    - No Phase 3 test data is used.
    - Nepali data only.

The real-data portion is sampled from the SAME Phase 2 Nepali pretraining
training corpus. For tokenized .bin shards, uint16 token IDs are sampled and
decoded using the existing Nepali tokenizer. Each real example is converted
to a prompt/answer continuation pair so the existing finetune.py can consume
the mixed dataset without modification.

Default paths:
    Reasoning:
        /content/LMA_IP_DATA/reasoning/

    Phase 2 training corpus:
        /content/LMA_IP_DATA/train/

    Output:
        /content/drive/MyDrive/LMA_IP/Nepali/finetune/mixed_data_size/

This launcher reuses the existing Nepali finetune.py for model training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_LOCAL_REASONING_DIR = Path("/content/LMA_IP_DATA/reasoning")
DEFAULT_LOCAL_PRETRAIN_DIR = Path("/content/LMA_IP_DATA/train")

DEFAULT_OUTPUT_BASE = Path(
    "/content/drive/MyDrive/LMA_IP/Nepali/finetune/mixed_data_size"
)

TOTAL_TRAINING_EXAMPLES = 20_000

REASONING_SIZES = [2_000, 5_000, 8_000, 12_000, 16_000, 20_000]

DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_SEED = 42

VOCAB_SIZE = 8_000
MAX_SEQ_LEN = 512

# The real-data example is 256 prompt + 255 answer = 511 tokens.
# The existing finetune.py adds EOS during tokenization, so the complete
# causal-LM sequence fits within the 512-token context.
DEFAULT_REAL_PROMPT_TOKENS = 256
DEFAULT_REAL_ANSWER_TOKENS = 255


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nepali Phase 3 fine-tuning with exactly 20k examples, "
            "varying the fraction of synthetic reasoning data and real "
            "Phase 2 pretraining data."
        )
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the Phase 2 pretrained Nepali checkpoint.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the existing 8000-token Nepali tokenizer JSON.",
    )

    parser.add_argument(
        "--reasoning-dir",
        type=str,
        default=str(DEFAULT_LOCAL_REASONING_DIR),
        help="Directory containing reasoning train.jsonl and val.jsonl.",
    )

    parser.add_argument(
        "--pretrain-data-dir",
        type=str,
        default=str(DEFAULT_LOCAL_PRETRAIN_DIR),
        help="Directory containing the SAME Phase 2 Nepali training corpus.",
    )

    parser.add_argument(
        "--output-base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Drive directory for all mixed-data experiment outputs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Fine-tuning batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Fine-tuning learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for reproducible sampling.",
    )

    parser.add_argument(
        "--reasoning-sizes",
        type=int,
        nargs="+",
        default=REASONING_SIZES,
        help=(
            "Number of synthetic reasoning examples in each 20k run. "
            "Default: 2000 5000 8000 12000 16000 20000"
        ),
    )

    parser.add_argument(
        "--real-chunk-tokens",
        type=int,
        default=MAX_SEQ_LEN,
        help="Token IDs read from each real-data candidate chunk.",
    )

    parser.add_argument(
        "--real-prompt-tokens",
        type=int,
        default=DEFAULT_REAL_PROMPT_TOKENS,
        help="Number of tokens used as the real-data prompt.",
    )

    parser.add_argument(
        "--real-answer-tokens",
        type=int,
        default=DEFAULT_REAL_ANSWER_TOKENS,
        help="Number of tokens used as the real-data answer/target.",
    )

    parser.add_argument(
        "--max-real-examples-scan",
        type=int,
        default=2_000_000,
        help="Maximum JSONL/TXT records/lines inspected.",
    )

    parser.add_argument(
        "--keep-local-subsets",
        action="store_true",
        help="Compatibility flag; generated subsets are already saved in output.",
    )

    return parser.parse_args()


# ============================================================================
# JSONL HELPERS
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)

            if "prompt" not in record or "answer" not in record:
                raise ValueError(
                    f"{path}: line {line_no} must contain prompt and answer."
                )

            records.append(record)

    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_task_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    for record in records:
        counts[str(record.get("task_type", "unknown"))] += 1

    return dict(sorted(counts.items()))


# ============================================================================
# REASONING DATA SAMPLING
# ============================================================================

def sample_reasoning_records(
    records: list[dict],
    target_size: int,
    seed: int,
) -> list[dict]:
    """Sample a reproducible approximately task-balanced subset."""

    if target_size < 0:
        raise ValueError("Reasoning target size cannot be negative.")

    if target_size > len(records):
        raise ValueError(
            f"Requested {target_size:,} reasoning examples, but only "
            f"{len(records):,} are available."
        )

    if target_size == 0:
        return []

    if target_size == len(records):
        selected = list(records)
        random.Random(seed).shuffle(selected)
        return selected

    groups: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        groups[str(record.get("task_type", "unknown"))].append(record)

    rng = random.Random(seed)

    for group in groups.values():
        rng.shuffle(group)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []

    total = len(records)

    for task_type in sorted(groups):
        exact = target_size * len(groups[task_type]) / total
        count = min(int(math.floor(exact)), len(groups[task_type]))

        allocations[task_type] = count
        remainders.append((exact - count, task_type))

    remaining = target_size - sum(allocations.values())
    remainders.sort(reverse=True)

    while remaining > 0:
        added = False

        for _, task_type in remainders:
            if allocations[task_type] < len(groups[task_type]):
                allocations[task_type] += 1
                remaining -= 1
                added = True

                if remaining == 0:
                    break

        if not added:
            raise RuntimeError(
                "Could not complete reasoning subset allocation."
            )

    selected: list[dict] = []

    for task_type in sorted(groups):
        selected.extend(groups[task_type][:allocations[task_type]])

    rng.shuffle(selected)

    if len(selected) != target_size:
        raise AssertionError(
            f"Expected {target_size} reasoning examples, got {len(selected)}."
        )

    return selected


# ============================================================================
# REAL PRETRAINING DATA
# ============================================================================

def find_data_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Pretraining data directory does not exist:\n{data_dir}"
        )

    files: list[Path] = []

    for suffix in ("*.bin", "*.jsonl", "*.txt"):
        files.extend(data_dir.rglob(suffix))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            "No .bin, .jsonl, or .txt pretraining files were found under:\n"
            f"{data_dir}"
        )

    return files


def load_nepali_tokenizer(tokenizer_path: Path):
    """
    Load the tokenizer used by the supplied Nepali finetune.py.

    The uploaded Nepali finetune.py uses tokenizers.Tokenizer.from_file(),
    so this launcher uses the exact same API.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Expected {VOCAB_SIZE:,}-token Nepali tokenizer, "
            f"got {vocab_size:,}."
        )

    return tokenizer


def real_record_from_text(
    text: str,
    tokenizer,
    record_id: str,
    prompt_tokens: int,
    answer_tokens: int,
) -> dict | None:
    text = " ".join(text.split())

    if not text:
        return None

    ids = tokenizer.encode(text).ids

    required = prompt_tokens + answer_tokens

    if len(ids) < required:
        return None

    ids = ids[:required]

    prompt_ids = ids[:prompt_tokens]
    answer_ids = ids[prompt_tokens:required]

    prompt = tokenizer.decode(prompt_ids).strip()
    answer = tokenizer.decode(answer_ids).strip()

    if not prompt or not answer:
        return None

    return {
        "id": record_id,
        "language": "ne",
        "language_name": "Nepali",
        "split": "finetune_real_pretraining",
        "task_type": "pretraining_continuation",
        "difficulty": "language_modeling",
        "prompt": prompt,
        "answer": answer,
        "source": "phase2_pretraining_train_data",
        "prompt_token_count": len(prompt_ids),
        "answer_token_count": len(answer_ids),
    }


def sample_real_from_bin(
    files: list[Path],
    target_size: int,
    tokenizer,
    seed: int,
    prompt_tokens: int,
    answer_tokens: int,
    chunk_tokens: int,
) -> list[dict]:
    """
    Sample real examples from tokenized uint16 .bin shards.

    The Nepali vocabulary is 8000, so uint16 is sufficient for token IDs.
    Shards are memory-mapped and never loaded entirely into RAM.
    """

    if target_size == 0:
        return []

    required = prompt_tokens + answer_tokens

    if chunk_tokens < required:
        raise ValueError(
            "--real-chunk-tokens must be >= "
            "prompt_tokens + answer_tokens."
        )

    shard_info: list[tuple[Path, int]] = []

    for path in files:
        if path.suffix.lower() != ".bin":
            continue

        size_bytes = path.stat().st_size

        if size_bytes < 2:
            continue

        num_tokens = size_bytes // np.dtype(np.uint16).itemsize

        if num_tokens >= required:
            shard_info.append((path, int(num_tokens)))

    if not shard_info:
        raise RuntimeError(
            "No usable .bin shards containing enough Nepali token IDs were found."
        )

    rng = random.Random(seed)

    selected: list[dict] = []
    used_windows: set[tuple[str, int]] = set()

    max_attempts = max(target_size * 20, 1000)

    weights = [
        max(1, n - required + 1)
        for _, n in shard_info
    ]

    # Open one mmap per shard and reuse it. This is substantially faster than
    # opening/mapping a shard for every sampled example.
    mmaps: list[tuple[Path, int, np.memmap]] = []

    try:
        for path, num_tokens in shard_info:
            mmap = np.memmap(
                path,
                dtype=np.uint16,
                mode="r",
                shape=(num_tokens,),
            )
            mmaps.append((path, num_tokens, mmap))

        for _ in range(max_attempts):
            if len(selected) >= target_size:
                break

            shard_index = rng.choices(
                range(len(mmaps)),
                weights=weights,
                k=1,
            )[0]

            path, num_tokens, mmap = mmaps[shard_index]

            max_start = num_tokens - required

            if max_start < 0:
                continue

            start = rng.randint(0, max_start)

            key = (str(path), start)

            if key in used_windows:
                continue

            used_windows.add(key)

            token_ids = mmap[
                start:start + required
            ].astype(np.int64).tolist()

            prompt_ids = token_ids[:prompt_tokens]
            answer_ids = token_ids[prompt_tokens:required]

            prompt = tokenizer.decode(prompt_ids).strip()
            answer = tokenizer.decode(answer_ids).strip()

            if not prompt or not answer:
                continue

            selected.append(
                {
                    "id": f"real_bin_{len(selected):06d}",
                    "language": "ne",
                    "language_name": "Nepali",
                    "split": "finetune_real_pretraining",
                    "task_type": "pretraining_continuation",
                    "difficulty": "language_modeling",
                    "prompt": prompt,
                    "answer": answer,
                    "source": "phase2_pretraining_train_data",
                    "source_file": str(path),
                    "token_start": start,
                    "prompt_token_count": len(prompt_ids),
                    "answer_token_count": len(answer_ids),
                }
            )

    finally:
        for _, _, mmap in mmaps:
            del mmap

    if len(selected) < target_size:
        raise RuntimeError(
            f"Could generate only {len(selected):,} usable real examples "
            f"out of the requested {target_size:,}. "
            "Check the .bin format and tokenizer."
        )

    rng.shuffle(selected)

    return selected[:target_size]


def sample_real_from_jsonl(
    files: list[Path],
    target_size: int,
    tokenizer,
    seed: int,
    prompt_tokens: int,
    answer_tokens: int,
    max_scan: int,
) -> list[dict]:
    candidates: list[str] = []

    for path in files:
        if path.suffix.lower() != ".jsonl":
            continue

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(candidates) >= max_scan:
                    break

                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = record.get("text")

                if isinstance(text, str) and text.strip():
                    candidates.append(text)

        if len(candidates) >= max_scan:
            break

    if not candidates:
        raise RuntimeError(
            "No JSONL records with a non-empty 'text' field were found."
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected: list[dict] = []

    for text in candidates:
        record = real_record_from_text(
            text=text,
            tokenizer=tokenizer,
            record_id=f"real_jsonl_{len(selected):06d}",
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
        )

        if record is None:
            continue

        selected.append(record)

        if len(selected) == target_size:
            break

    if len(selected) < target_size:
        raise RuntimeError(
            f"Only {len(selected):,} usable real JSONL examples were found; "
            f"{target_size:,} were requested."
        )

    return selected


def sample_real_from_txt(
    files: list[Path],
    target_size: int,
    tokenizer,
    seed: int,
    prompt_tokens: int,
    answer_tokens: int,
    max_scan: int,
) -> list[dict]:
    lines: list[str] = []

    for path in files:
        if path.suffix.lower() != ".txt":
            continue

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            for line in f:
                if len(lines) >= max_scan:
                    break

                if line.strip():
                    lines.append(line.strip())

        if len(lines) >= max_scan:
            break

    if not lines:
        raise RuntimeError("No usable text lines were found.")

    rng = random.Random(seed)
    rng.shuffle(lines)

    selected: list[dict] = []

    for text in lines:
        record = real_record_from_text(
            text=text,
            tokenizer=tokenizer,
            record_id=f"real_txt_{len(selected):06d}",
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
        )

        if record is None:
            continue

        selected.append(record)

        if len(selected) == target_size:
            break

    if len(selected) < target_size:
        raise RuntimeError(
            f"Only {len(selected):,} usable real text examples were found; "
            f"{target_size:,} were requested."
        )

    return selected


def sample_real_pretraining_records(
    data_dir: Path,
    target_size: int,
    tokenizer,
    seed: int,
    prompt_tokens: int,
    answer_tokens: int,
    chunk_tokens: int,
    max_scan: int,
) -> list[dict]:
    if target_size == 0:
        return []

    files = find_data_files(data_dir)

    bin_files = [
        p for p in files
        if p.suffix.lower() == ".bin"
    ]
    jsonl_files = [
        p for p in files
        if p.suffix.lower() == ".jsonl"
    ]
    txt_files = [
        p for p in files
        if p.suffix.lower() == ".txt"
    ]

    # The .bin files are the actual tokenized Phase 2 training data, so they
    # are always preferred when available.
    if bin_files:
        return sample_real_from_bin(
            files=bin_files,
            target_size=target_size,
            tokenizer=tokenizer,
            seed=seed,
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            chunk_tokens=chunk_tokens,
        )

    if jsonl_files:
        return sample_real_from_jsonl(
            files=jsonl_files,
            target_size=target_size,
            tokenizer=tokenizer,
            seed=seed,
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            max_scan=max_scan,
        )

    if txt_files:
        return sample_real_from_txt(
            files=txt_files,
            target_size=target_size,
            tokenizer=tokenizer,
            seed=seed,
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            max_scan=max_scan,
        )

    raise RuntimeError(
        "No supported Nepali pretraining-data format was found."
    )


# ============================================================================
# MIXTURE CONSTRUCTION
# ============================================================================

def build_mixed_dataset(
    reasoning_records: list[dict],
    pretrain_dir: Path,
    tokenizer,
    reasoning_size: int,
    seed: int,
    prompt_tokens: int,
    answer_tokens: int,
    chunk_tokens: int,
    max_scan: int,
) -> tuple[list[dict], list[dict], list[dict]]:

    real_size = TOTAL_TRAINING_EXAMPLES - reasoning_size

    if reasoning_size < 0 or reasoning_size > TOTAL_TRAINING_EXAMPLES:
        raise ValueError(
            f"Reasoning size must be between 0 and "
            f"{TOTAL_TRAINING_EXAMPLES:,}."
        )

    reasoning = sample_reasoning_records(
        records=reasoning_records,
        target_size=reasoning_size,
        seed=seed + 101,
    )

    real = sample_real_pretraining_records(
        data_dir=pretrain_dir,
        target_size=real_size,
        tokenizer=tokenizer,
        seed=seed + 202,
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        chunk_tokens=chunk_tokens,
        max_scan=max_scan,
    )

    mixed = reasoning + real

    # IMPORTANT: mix the two sources rather than putting one source before
    # the other. The existing finetune.py then shuffles the DataLoader again.
    rng = random.Random(seed + 303)
    rng.shuffle(mixed)

    if len(mixed) != TOTAL_TRAINING_EXAMPLES:
        raise AssertionError(
            f"Expected exactly {TOTAL_TRAINING_EXAMPLES:,} examples, "
            f"got {len(mixed):,}."
        )

    return mixed, reasoning, real


# ============================================================================
# ONE-PASS STEP CALCULATION
# ============================================================================

def steps_for_one_pass(
    num_examples: int,
    batch_size: int,
) -> int:
    # finetune.py uses drop_last=False, so the last partial batch counts.
    return math.ceil(num_examples / batch_size)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.real_prompt_tokens <= 0:
        raise ValueError("--real-prompt-tokens must be positive.")

    if args.real_answer_tokens <= 0:
        raise ValueError("--real-answer-tokens must be positive.")

    if args.real_chunk_tokens <= 0:
        raise ValueError("--real-chunk-tokens must be positive.")

    required_real_tokens = (
        args.real_prompt_tokens +
        args.real_answer_tokens
    )

    if required_real_tokens > args.real_chunk_tokens:
        raise ValueError(
            "real_prompt_tokens + real_answer_tokens must be <= "
            "real_chunk_tokens."
        )

    if required_real_tokens > MAX_SEQ_LEN - 1:
        raise ValueError(
            "Real prompt + answer must leave room for EOS inside "
            "the 512-token context."
        )

    reasoning_sizes = sorted(set(args.reasoning_sizes))

    if not reasoning_sizes:
        raise ValueError("At least one reasoning size must be supplied.")

    if any(
        size < 0 or size > TOTAL_TRAINING_EXAMPLES
        for size in reasoning_sizes
    ):
        raise ValueError(
            f"All reasoning sizes must be between 0 and "
            f"{TOTAL_TRAINING_EXAMPLES:,}."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This script is intended for "
            "Colab GPU fine-tuning."
        )

    print("=" * 90)
    print("NEPALI — PHASE 3 MIXED DATA-SIZE FINE-TUNING EXPERIMENT")
    print("=" * 90)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    memory_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU memory: {memory_gb:.2f} GB")

    # ------------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------------

    reasoning_dir = Path(args.reasoning_dir)
    pretrain_dir = Path(args.pretrain_data_dir)
    output_base = Path(args.output_base)

    reasoning_train_path = reasoning_dir / "train.jsonl"
    reasoning_val_path = reasoning_dir / "val.jsonl"

    if not reasoning_train_path.exists():
        raise FileNotFoundError(
            f"Missing synthetic reasoning training data:\n"
            f"{reasoning_train_path}"
        )

    if not reasoning_val_path.exists():
        raise FileNotFoundError(
            f"Missing synthetic reasoning validation data:\n"
            f"{reasoning_val_path}"
        )

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing Phase 2 pretrained Nepali checkpoint:\n{model_path}"
        )

    tokenizer_path = Path(args.tokenizer)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Missing Nepali tokenizer:\n{tokenizer_path}"
        )

    if not pretrain_dir.exists():
        raise FileNotFoundError(
            f"Missing Phase 2 Nepali pretraining data directory:\n"
            f"{pretrain_dir}"
        )

    output_base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Load reasoning data and fixed validation.
    # ------------------------------------------------------------------------

    print("\nLoading synthetic reasoning data...")

    all_reasoning = load_jsonl(reasoning_train_path)
    val_records = load_jsonl(reasoning_val_path)

    print(
        f"Synthetic reasoning train examples : "
        f"{len(all_reasoning):,}"
    )
    print(
        f"Synthetic reasoning val examples   : "
        f"{len(val_records):,}"
    )

    if max(reasoning_sizes) > len(all_reasoning):
        raise ValueError(
            f"Largest requested reasoning size is "
            f"{max(reasoning_sizes):,}, but only "
            f"{len(all_reasoning):,} reasoning examples are available."
        )

    print("\nSynthetic reasoning task distribution:")

    for task_type, count in get_task_counts(all_reasoning).items():
        print(
            f"  {task_type:24s}: "
            f"{count:7,d} "
            f"({100.0 * count / len(all_reasoning):6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Load existing Nepali tokenizer.
    # ------------------------------------------------------------------------

    print("\nLoading existing Phase 2 Nepali tokenizer...")

    tokenizer = load_nepali_tokenizer(tokenizer_path)

    print(
        f"Tokenizer vocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")

    if pad_id is None:
        raise RuntimeError("Nepali tokenizer has no <pad> token.")

    if eos_id is None:
        raise RuntimeError("Nepali tokenizer has no <eos> token.")

    print(f"PAD token ID: {pad_id}")
    print(f"EOS token ID: {eos_id}")

    # ------------------------------------------------------------------------
    # Locate the existing finetune.py.
    # ------------------------------------------------------------------------

    script_dir = Path(__file__).resolve().parent

    finetune_path = script_dir / "finetune.py"

    if not finetune_path.exists():
        raise FileNotFoundError(
            "Could not find finetune.py next to this launcher.\n"
            f"Expected: {finetune_path}"
        )

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    project_root = script_dir.parents[1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ------------------------------------------------------------------------
    # Global manifest.
    # ------------------------------------------------------------------------

    manifest = {
        "experiment": "nepali_phase3_mixed_data_size",
        "language": "Nepali",
        "total_training_examples_per_run": TOTAL_TRAINING_EXAMPLES,
        "base_pretrained_checkpoint": str(model_path),
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "synthetic_reasoning_dir": str(reasoning_dir),
        "phase2_pretraining_data_dir": str(pretrain_dir),
        "output_base": str(output_base),
        "reasoning_sizes": reasoning_sizes,
        "real_pretraining_sizes": {
            str(reasoning_size):
                TOTAL_TRAINING_EXAMPLES - reasoning_size
            for reasoning_size in reasoning_sizes
        },
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "one_pass_only": True,
        "validation_is_fixed": True,
        "total_examples_fixed": True,
        "real_data_objective": "causal_continuation",
        "real_prompt_tokens": args.real_prompt_tokens,
        "real_answer_tokens": args.real_answer_tokens,
        "real_chunk_tokens": args.real_chunk_tokens,
        "steps_per_run": steps_for_one_pass(
            TOTAL_TRAINING_EXAMPLES,
            args.batch_size,
        ),
        "synthetic_task_distribution": get_task_counts(
            all_reasoning
        ),
        "validation_examples": len(val_records),
    }

    manifest_path = output_base / "experiment_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nExperiment manifest: {manifest_path}")

    # ------------------------------------------------------------------------
    # Run each mixture.
    # ------------------------------------------------------------------------

    for run_index, reasoning_size in enumerate(reasoning_sizes, 1):

        real_size = TOTAL_TRAINING_EXAMPLES - reasoning_size

        print("\n")
        print("#" * 90)
        print(
            f"EXPERIMENT {run_index}/{len(reasoning_sizes)}"
        )
        print(
            f"Reasoning: {reasoning_size:,} "
            f"({100.0 * reasoning_size / TOTAL_TRAINING_EXAMPLES:.1f}%)"
        )
        print(
            f"Real pretraining data: {real_size:,} "
            f"({100.0 * real_size / TOTAL_TRAINING_EXAMPLES:.1f}%)"
        )
        print(f"TOTAL: {TOTAL_TRAINING_EXAMPLES:,}")
        print("#" * 90)

        run_seed = args.seed + reasoning_size

        mixed, selected_reasoning, selected_real = build_mixed_dataset(
            reasoning_records=all_reasoning,
            pretrain_dir=pretrain_dir,
            tokenizer=tokenizer,
            reasoning_size=reasoning_size,
            seed=run_seed,
            prompt_tokens=args.real_prompt_tokens,
            answer_tokens=args.real_answer_tokens,
            chunk_tokens=args.real_chunk_tokens,
            max_scan=args.max_real_examples_scan,
        )

        steps = steps_for_one_pass(
            TOTAL_TRAINING_EXAMPLES,
            args.batch_size,
        )

        run_name = (
            f"reasoning_{reasoning_size // 1000}k"
            f"_real_{real_size // 1000}k"
        )

        output_dir = output_base / run_name
        run_data_dir = output_dir / "data"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------------
        # Save mixed training and fixed validation data.
        # --------------------------------------------------------------------

        mixed_train_path = run_data_dir / "train.jsonl"
        mixed_val_path = run_data_dir / "val.jsonl"

        write_jsonl(
            mixed_train_path,
            mixed,
        )

        # Every run receives the exact same validation file.
        shutil.copy2(
            reasoning_val_path,
            mixed_val_path,
        )

        # --------------------------------------------------------------------
        # Save per-run metadata.
        # --------------------------------------------------------------------

        source_counts = {
            "synthetic_reasoning": len(selected_reasoning),
            "phase2_pretraining": len(selected_real),
        }

        run_metadata = {
            "experiment": "nepali_phase3_mixed_data_size",
            "language": "Nepali",
            "run_name": run_name,
            "total_training_examples": len(mixed),
            "synthetic_reasoning_examples": len(selected_reasoning),
            "real_pretraining_examples": len(selected_real),
            "synthetic_reasoning_fraction":
                len(selected_reasoning) / len(mixed),
            "real_pretraining_fraction":
                len(selected_real) / len(mixed),
            "validation_examples": len(val_records),
            "batch_size": args.batch_size,
            "one_pass": True,
            "max_steps": steps,
            "run_seed": run_seed,
            "base_pretrained_checkpoint": str(model_path),
            "tokenizer": str(tokenizer_path),
            "synthetic_reasoning_source": str(
                reasoning_train_path
            ),
            "real_pretraining_source": str(pretrain_dir),
            "validation_source": str(reasoning_val_path),
            "mixed_train_path": str(mixed_train_path),
            "mixed_validation_path": str(mixed_val_path),
            "source_counts": source_counts,
            "reasoning_task_distribution":
                get_task_counts(selected_reasoning),
            "real_data_task_type":
                "pretraining_continuation",
            "real_prompt_tokens":
                args.real_prompt_tokens,
            "real_answer_tokens":
                args.real_answer_tokens,
            "total_examples_check":
                len(selected_reasoning) + len(selected_real)
                == TOTAL_TRAINING_EXAMPLES,
        }

        with (
            output_dir / "data_size_config.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                run_metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\nFinal mixture:")
        print(
            f"  Synthetic reasoning      : "
            f"{len(selected_reasoning):,} "
            f"({100.0 * len(selected_reasoning) / len(mixed):.1f}%)"
        )
        print(
            f"  Real Phase 2 train data  : "
            f"{len(selected_real):,} "
            f"({100.0 * len(selected_real) / len(mixed):.1f}%)"
        )
        print(
            f"  TOTAL                    : "
            f"{len(mixed):,}"
        )

        print("\nReasoning task distribution in this run:")

        if selected_reasoning:
            for task_type, count in get_task_counts(
                selected_reasoning
            ).items():
                print(
                    f"  {task_type:24s}: "
                    f"{count:7,d} "
                    f"({100.0 * count / len(selected_reasoning):6.2f}%)"
                )

        print(f"\nOne-pass optimizer steps: {steps:,}")
        print(f"Output directory: {output_dir}")

        # --------------------------------------------------------------------
        # Run the existing Nepali fine-tuner.
        # --------------------------------------------------------------------

        from finetune import main as finetune_main

        sys.argv = [
            "finetune.py",
            "--model-path",
            str(model_path),
            "--tokenizer",
            str(tokenizer_path),
            "--data-dir",
            str(run_data_dir),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(steps),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
        ]

        print(
            "\nStarting fine-tuning from the SAME Phase 2 checkpoint..."
        )
        print(
            f"Training composition: "
            f"{reasoning_size:,} reasoning + "
            f"{real_size:,} real"
        )
        print(f"max_steps = {steps:,}")
        print(
            "ONE PASS over the 20,000-example mixed dataset.\n"
        )

        finetune_main()

        print("\n" + "-" * 90)
        print(f"COMPLETED: {run_name}")
        print(f"Checkpoint: {output_dir / 'final.pt'}")
        print("-" * 90)

    # ------------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("ALL NEPALI MIXED DATA-SIZE EXPERIMENTS COMPLETE")
    print("=" * 90)

    print("\nEach run used exactly 20,000 training examples:")

    for reasoning_size in reasoning_sizes:
        real_size = TOTAL_TRAINING_EXAMPLES - reasoning_size

        run_name = (
            f"reasoning_{reasoning_size // 1000}k"
            f"_real_{real_size // 1000}k"
        )

        print(
            f"  {reasoning_size:6,d} reasoning + "
            f"{real_size:6,d} real -> "
            f"{output_base / run_name / 'final.pt'}"
        )

    print(
        "\nThe validation set was kept fixed, and every run used "
        "one pass over exactly 20,000 examples."
    )


if __name__ == "__main__":
    main()
