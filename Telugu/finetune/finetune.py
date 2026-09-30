"""
===============================================================================
TELUGU — PHASE 3 REASONING FINE-TUNING
===============================================================================

Fine-tunes the already-pretrained Telugu Model H on the synthetic Telugu
reasoning dataset.

IMPORTANT
---------
- This script does NOT train a new model from scratch.
- --model-path must point to the Phase 2 PRETRAINED Telugu checkpoint.
- The existing Telugu model architecture and 6000-token Telugu vocabulary are
  kept unchanged.
- --resume is only for resuming a Phase 3 fine-tuning checkpoint.
- The tokenizer is supplied explicitly and is never retrained.
- Checkpoints contain model, optimizer, scheduler, AMP scaler, RNG states,
  step, and configuration so the fine-tuning run is resumable.

Training examples are formatted as:

    prompt
    answer

The causal-LM loss is applied to the answer tokens (including EOS), while
prompt tokens are ignored in the loss. This makes the objective directly
target answer generation rather than merely memorizing the prompt text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from functools import partial
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset

from Telugu.config.model_config import ModelConfig
from Telugu.model.lm import TeluguLanguageModel


# =============================================================================
# CONFIGURATION
# =============================================================================

VOCAB_SIZE = 6000
MAX_SEQ_LEN = 512

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_STEPS = 2000
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.95
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WARMUP_STEPS = 100
DEFAULT_MIN_LR_RATIO = 0.1

DEFAULT_LOG_EVERY = 50
DEFAULT_EVAL_EVERY = 250
DEFAULT_CHECKPOINT_EVERY = 250
DEFAULT_VALIDATION_STEPS = 100
DEFAULT_SEED = 42


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse Phase 3 fine-tuning arguments."""

    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained Telugu LM on synthetic reasoning data."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to the Phase 2 PRETRAINED Telugu checkpoint. "
            "Required when starting a new fine-tuning run."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a Phase 3 fine-tuning checkpoint to resume from. "
            "When supplied, --model-path is not required."
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the existing 6000-token Telugu SentencePiece model.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl and val.jsonl.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory in which Phase 3 checkpoints and logs are saved.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# LEARNING-RATE SCHEDULE
# =============================================================================

def get_learning_rate(step: int, max_steps: int, learning_rate: float) -> float:
    """Return warmup + cosine-decay learning rate."""

    if step < DEFAULT_WARMUP_STEPS:
        return learning_rate * step / max(1, DEFAULT_WARMUP_STEPS)

    decay_steps = max_steps - DEFAULT_WARMUP_STEPS

    if decay_steps <= 0:
        return learning_rate

    progress = (
        (step - DEFAULT_WARMUP_STEPS) / decay_steps
    )
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    min_lr = learning_rate * DEFAULT_MIN_LR_RATIO

    return min_lr + (
        learning_rate - min_lr
    ) * cosine


# =============================================================================
# REASONING DATASET
# =============================================================================

class ReasoningDataset(Dataset):
    """
    Tokenize synthetic reasoning examples using the EXISTING Telugu tokenizer.

    Each example becomes:

        prompt + newline + answer + EOS

    Input/target are shifted for causal LM training. Target positions belonging
    to the prompt are replaced with pad_token_id, so loss is computed only on
    answer tokens and EOS.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int,
        pad_token_id: int,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        self.examples: list[dict[str, Any]] = []

        with self.jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue

                record = json.loads(line)

                if "prompt" not in record:
                    raise KeyError(
                        f"{self.jsonl_path}: line {line_no} "
                        "has no 'prompt' field."
                    )

                if "answer" not in record:
                    raise KeyError(
                        f"{self.jsonl_path}: line {line_no} "
                        "has no 'answer' field."
                    )

                prompt = str(record["prompt"]).strip()
                answer = str(record["answer"]).strip()

                if not prompt or not answer:
                    raise ValueError(
                        f"{self.jsonl_path}: line {line_no} "
                        "has an empty prompt or answer."
                    )

                prompt_ids = tokenizer.encode(
                    prompt + "\n",
                    out_type=int,
                )
                answer_ids = tokenizer.encode(
                    answer,
                    out_type=int,
                )

                eos_id = tokenizer.eos_id()

                if eos_id < 0:
                    raise RuntimeError(
                        "Telugu tokenizer has no EOS token."
                    )

                full_ids = (
                    prompt_ids
                    + answer_ids
                    + [eos_id]
                )

                # Need at least one input and one target token.
                if len(full_ids) < 2:
                    continue

                # Keep the COMPLETE answer whenever possible. If the example
                # is too long, truncate from the prompt side while preserving
                # the answer and EOS. No reasoning answer is truncated.
                if len(full_ids) > max_seq_len + 1:
                    answer_and_eos = (
                        answer_ids + [eos_id]
                    )

                    if len(answer_and_eos) > max_seq_len:
                        raise ValueError(
                            f"{self.jsonl_path}: line {line_no} answer alone "
                            f"requires {len(answer_and_eos)} tokens, exceeding "
                            f"max sequence length {max_seq_len}."
                        )

                    allowed_prompt = (
                        max_seq_len
                        + 1
                        - len(answer_and_eos)
                    )

                    if allowed_prompt > 0:
                        prompt_ids = prompt_ids[-allowed_prompt:]
                    else:
                        prompt_ids = []

                    full_ids = (
                        prompt_ids
                        + answer_and_eos
                    )

                input_ids = full_ids[:-1]
                target_ids = full_ids[1:]

                # The answer starts at sequence index len(prompt_ids).
                # Target index j predicts full_ids[j+1]. Therefore the first
                # answer token is predicted at target index len(prompt_ids)-1.
                first_answer_target = max(
                    0,
                    len(prompt_ids) - 1,
                )

                for j in range(first_answer_target):
                    target_ids[j] = pad_token_id

                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                        "id": record.get("id"),
                        "task_type": record.get("task_type"),
                    }
                )

        if not self.examples:
            raise RuntimeError(
                f"No usable reasoning examples found in {self.jsonl_path}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_reasoning(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad variable-length reasoning examples into a batch."""

    max_len = max(
        len(item["input_ids"])
        for item in batch
    )

    input_ids = torch.full(
        (len(batch), max_len),
        pad_token_id,
        dtype=torch.long,
    )

    targets = torch.full(
        (len(batch), max_len),
        pad_token_id,
        dtype=torch.long,
    )

    for row, item in enumerate(batch):
        length = len(item["input_ids"])

        input_ids[row, :length] = torch.tensor(
            item["input_ids"],
            dtype=torch.long,
        )

        targets[row, :length] = torch.tensor(
            item["target_ids"],
            dtype=torch.long,
        )

    return {
        "input_ids": input_ids,
        "targets": targets,
    }


# =============================================================================
# CHECKPOINTS
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    loss: float,
    val_loss: float | None,
    config: dict[str, Any],
    path: Path,
) -> None:
    """Save a resumable Phase 3 fine-tuning checkpoint."""

    checkpoint = {
        "phase": "phase3_reasoning_finetuning",
        "language": "Telugu",
        "step": step,
        "loss": loss,
        "validation_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": {
            "type": "warmup_cosine_manual",
            "step": step,
            "max_steps": config["max_steps"],
            "base_learning_rate": config["learning_rate"],
            "warmup_steps": config["warmup_steps"],
            "min_learning_rate_ratio": config["min_learning_rate_ratio"],
        },
        "scaler_state_dict": scaler.state_dict(),
        "finetune_config": config,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = (
            torch.cuda.get_rng_state_all()
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        path,
    )


def load_finetune_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float | None]:
    """Load a Phase 3 checkpoint and restore all resumable state."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("phase") != "phase3_reasoning_finetuning":
        raise RuntimeError(
            f"{path} is not a Phase 3 fine-tuning checkpoint."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if "scheduler_state_dict" not in checkpoint:
        raise RuntimeError(
            "Phase 3 checkpoint does not contain scheduler state."
        )

    scheduler_state = checkpoint["scheduler_state_dict"]
    if int(scheduler_state["step"]) != int(checkpoint["step"]):
        raise RuntimeError(
            "Checkpoint scheduler step does not match training step."
        )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    if "python_rng_state" in checkpoint:
        random.setstate(
            checkpoint["python_rng_state"]
        )

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(
            checkpoint["numpy_rng_state"]
        )

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )

    step = int(checkpoint["step"])
    loss = checkpoint.get("loss")
    val_loss = checkpoint.get("validation_loss")

    print(
        f"Resumed Phase 3 checkpoint from step {step:,}"
    )

    if loss is not None:
        print(f"Checkpoint training loss: {loss:.6f}")

    if val_loss is not None:
        print(f"Checkpoint validation loss: {val_loss:.6f}")

    return step, val_loss


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: TeluguLanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    """Evaluate causal-LM loss on the reasoning validation set."""

    model.eval()

    total_loss = 0.0
    batches = 0

    for batch in dataloader:
        if batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )
        targets = batch["targets"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            _, loss, _ = model(
                input_ids=input_ids,
                targets=targets,
            )

        if loss is None:
            raise RuntimeError(
                "Model returned no validation loss."
            )

        total_loss += loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError(
            "Reasoning validation DataLoader returned no batches."
        )

    average_loss = total_loss / batches
    perplexity = math.exp(
        min(average_loss, 20)
    )

    model.train()

    return average_loss, perplexity


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run Telugu Phase 3 reasoning fine-tuning."""

    args = parse_args()

    if args.model_path is None and args.resume is None:
        raise ValueError(
            "Provide --model-path for the pretrained checkpoint, "
            "or --resume for a Phase 3 checkpoint."
        )

    if args.model_path is not None and args.resume is not None:
        print(
            "Note: --resume takes precedence over --model-path."
        )

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("TELUGU MODEL H — PHASE 3 REASONING FINE-TUNING")
    print("=" * 80)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------

    print("\nLoading existing Telugu tokenizer...")

    tokenizer = spm.SentencePieceProcessor()

    if not tokenizer.load(str(tokenizer_path)):
        raise RuntimeError(
            f"Could not load tokenizer: {tokenizer_path}"
        )

    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise RuntimeError(
            "Expected 6000-token Telugu tokenizer, "
            f"got {tokenizer.vocab_size()}."
        )

    pad_token_id = tokenizer.pad_id()

    if pad_token_id < 0:
        raise RuntimeError(
            "Telugu tokenizer has no PAD token."
        )

    print(f"Tokenizer vocabulary: {tokenizer.vocab_size():,}")
    print(f"PAD token ID: {pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_id()}")

    # The existing Telugu SentencePiece tokenizer is authoritative for
    # special-token IDs. This tokenizer uses PAD=1; ID 0 is <unk>.
    # Set the runtime model config to the tokenizer PAD ID so that the
    # model loss uses the same ignore_index as dataset padding/masking.
    # No pretrained weights or tokenizer files are modified.
    model_config = ModelConfig()
    model_config.pad_token_id = pad_token_id

    assert model_config.vocab_size == 6000
    assert model_config.max_seq_len == 512
    assert model_config.d_model == 448
    assert model_config.n_layers == 10
    assert model_config.n_heads == 8
    assert model_config.d_ff == 1344

    model = TeluguLanguageModel(
        model_config
    ).to(device)

    print(
        f"\nModel parameters: "
        f"{model.count_parameters():,}"
    )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(
            DEFAULT_BETA1,
            DEFAULT_BETA2,
        ),
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(
        f"Mixed precision: "
        f"{'FP16' if use_amp else 'disabled'}"
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing reasoning training data: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing reasoning validation data: {val_path}"
        )

    print("\nTokenizing reasoning dataset with the existing tokenizer...")

    train_dataset = ReasoningDataset(
        train_path,
        tokenizer,
        MAX_SEQ_LEN,
        pad_token_id,
    )

    val_dataset = ReasoningDataset(
        val_path,
        tokenizer,
        MAX_SEQ_LEN,
        pad_token_id,
    )

    print(
        f"Training examples  : {len(train_dataset):,}"
    )
    print(
        f"Validation examples: {len(val_dataset):,}"
    )

    # Reasoning examples have variable tokenized lengths. Use the custom
    # collate function to pad each batch to its longest example. The default
    # PyTorch collate_fn cannot stack variable-length input/target lists.
    collate_fn = partial(
        collate_reasoning,
        pad_token_id=pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=use_amp,
        collate_fn=collate_fn,
    )

    # -------------------------------------------------------------------------
    # Start from pretrained model OR resume Phase 3
    # -------------------------------------------------------------------------

    start_step = 0
    last_val_loss: float | None = None

    if args.resume is not None:
        print("\nLoading Phase 3 resume checkpoint...")

        start_step, last_val_loss = load_finetune_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scaler,
            device,
        )

    else:
        pretrained_path = Path(args.model_path)

        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained model checkpoint not found: "
                f"{pretrained_path}"
            )

        print(
            "\nLoading Phase 2 PRETRAINED Telugu model:"
        )
        print(pretrained_path)

        checkpoint = torch.load(
            pretrained_path,
            map_location=device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Pretrained checkpoint does not contain "
                "'model_state_dict'."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "✓ Pretrained model weights loaded."
        )

    # -------------------------------------------------------------------------
    # Configuration log
    # -------------------------------------------------------------------------

    run_config = {
        "phase": "phase3_reasoning_finetuning",
        "language": "Telugu",
        "base_pretrained_checkpoint": (
            args.model_path
            if args.resume is None
            else "resumed_phase3_checkpoint"
        ),
        "tokenizer": str(tokenizer_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "vocab_size": VOCAB_SIZE,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_id(),
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "beta1": DEFAULT_BETA1,
        "beta2": DEFAULT_BETA2,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "min_learning_rate_ratio": DEFAULT_MIN_LR_RATIO,
        "log_every": DEFAULT_LOG_EVERY,
        "eval_every": DEFAULT_EVAL_EVERY,
        "checkpoint_every": DEFAULT_CHECKPOINT_EVERY,
        "validation_steps": DEFAULT_VALIDATION_STEPS,
        "seed": args.seed,
        "loss_masking": "answer_tokens_and_eos_only",
        "scheduler": {
            "type": "warmup_cosine_manual",
            "warmup_steps": DEFAULT_WARMUP_STEPS,
            "min_learning_rate_ratio": DEFAULT_MIN_LR_RATIO,
        },
    }

    with (
        output_dir / "finetune_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    model.train()

    step = start_step
    training_start = time.perf_counter()

    print("\nFine-tuning:")
    print("-" * 80)

    while step < args.max_steps:
        for batch in train_loader:

            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            targets = batch["targets"].to(
                device,
                non_blocking=True,
            )

            lr = get_learning_rate(
                step,
                args.max_steps,
                args.learning_rate,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                _, loss, _ = model(
                    input_ids=input_ids,
                    targets=targets,
                )

            if loss is None:
                raise RuntimeError(
                    "Model returned no training loss."
                )

            if use_amp:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    DEFAULT_GRAD_CLIP,
                )

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    DEFAULT_GRAD_CLIP,
                )

                optimizer.step()

            step += 1

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------

            if (
                step == 1
                or step % DEFAULT_LOG_EVERY == 0
            ):
                loss_value = loss.item()
                ppl = math.exp(
                    min(loss_value, 20)
                )

                print(
                    f"Step {step:5d} | "
                    f"LR {lr:.6e} | "
                    f"Loss {loss_value:.5f} | "
                    f"PPL {ppl:.2f}"
                )

            # -----------------------------------------------------------------
            # Validation
            # -----------------------------------------------------------------

            if (
                step % DEFAULT_EVAL_EVERY == 0
            ):
                val_loss, val_ppl = evaluate(
                    model,
                    val_loader,
                    device,
                    DEFAULT_VALIDATION_STEPS,
                )

                last_val_loss = val_loss

                print(
                    f"Validation | "
                    f"Loss {val_loss:.5f} | "
                    f"PPL {val_ppl:.2f}"
                )

            # -----------------------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------------------

            if (
                step % DEFAULT_CHECKPOINT_EVERY == 0
            ):
                checkpoint_path = (
                    output_dir
                    / f"step_{step:06d}.pt"
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    loss=loss.item(),
                    val_loss=last_val_loss,
                    config=run_config,
                    path=checkpoint_path,
                )

                print(
                    f"Saved checkpoint: "
                    f"{checkpoint_path}"
                )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    print("\nRunning final validation...")

    final_val_loss, final_val_ppl = evaluate(
        model,
        val_loader,
        device,
        DEFAULT_VALIDATION_STEPS,
    )

    # -------------------------------------------------------------------------
    # Final checkpoint
    # -------------------------------------------------------------------------

    final_path = output_dir / "final.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        loss=loss.item(),
        val_loss=final_val_loss,
        config=run_config,
        path=final_path,
    )

    elapsed_hours = (
        time.perf_counter() - training_start
    ) / 3600.0

    print(
        f"\nFinal validation loss : "
        f"{final_val_loss:.5f}"
    )
    print(
        f"Final validation PPL  : "
        f"{final_val_ppl:.2f}"
    )
    print(
        f"Training time         : "
        f"{elapsed_hours:.2f} hours"
    )
    print(
        f"Final checkpoint:"
        f"\n{final_path}"
    )

    print("\n" + "=" * 80)
    print("TELUGU PHASE 3 FINE-TUNING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
