import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

from .common import load_model, assert_nepali_config, load_jsonl_texts


def entropy_per_head(attention):
    safe = attention.float().clamp_min(1e-12)
    return -(safe * safe.log()).sum(dim=-1).mean(dim=-1)


def distance_per_head(attention):
    h, q, k = attention.shape
    pos_k = torch.arange(
        k, device=attention.device, dtype=attention.dtype
    )
    pos_q = torch.arange(
        q, device=attention.device, dtype=attention.dtype
    )
    dist = (pos_q[:, None] - pos_k[None, :]).abs()
    return (attention * dist[None, :, :]).sum(dim=-1).mean(dim=-1)


def save_heatmap(matrix, path, title):
    plt.figure(figsize=(9, 7))
    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--document-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = load_nepali_tokenizer(args.tokenizer)

    model, config, checkpoint_path = load_model(
        args.checkpoint_dir, device
    )
    assert_nepali_config(config)

    texts = load_jsonl_texts(args.test_text)
    if args.document_index >= len(texts):
        raise IndexError(
            f"document-index {args.document_index} >= "
            f"{len(texts)} test documents"
        )

    ids = tokenizer.encode(
        texts[args.document_index]
    ).ids
    ids = ids[:args.sequence_length]

    if len(ids) < 2:
        raise RuntimeError("Selected test document is too short.")

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            _, _, attentions = model(
                input_ids=x,
                targets=None,
                return_attention=True,
            )

    if attentions is None:
        raise RuntimeError(
            "Model did not return attention matrices. "
            "Check Nepali/model/attention.py and Transformer forward."
        )

    layers = []
    for a in attentions:
        if a.dim() == 4:
            layers.append(a[0])
        elif a.dim() == 3:
            layers.append(a)
        else:
            raise RuntimeError(
                f"Unexpected attention shape: {tuple(a.shape)}"
            )

    attention = torch.stack(layers, dim=0).float()
    n_layers, n_heads = attention.shape[:2]

    entropy = torch.stack([
        entropy_per_head(attention[i])
        for i in range(n_layers)
    ]).cpu().numpy()

    distance = torch.stack([
        distance_per_head(attention[i])
        for i in range(n_layers)
    ]).cpu().numpy()

    for layer, name in [
        (0, "early"),
        (n_layers - 1, "late"),
    ]:
        for head in args.heads:
            if head < 0 or head >= n_heads:
                continue

            save_heatmap(
                attention[layer, head].cpu().numpy(),
                out / (
                    f"attention_{name}_layer_"
                    f"{layer:02d}_head_{head:02d}.png"
                ),
                (
                    f"Nepali attention — {name} layer "
                    f"{layer}, head {head}"
                ),
            )

    entropy_rows = []
    distance_rows = []

    for layer in range(n_layers):
        for head in range(n_heads):
            entropy_rows.append({
                "layer": layer,
                "head": head,
                "attention_entropy": float(
                    entropy[layer, head]
                ),
            })

            distance_rows.append({
                "layer": layer,
                "head": head,
                "mean_attention_distance": float(
                    distance[layer, head]
                ),
            })

    (out / "attention_entropy.json").write_text(
        json.dumps(entropy_rows, indent=2),
        encoding="utf-8",
    )

    (out / "attention_distance.json").write_text(
        json.dumps(distance_rows, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "language": "Nepali",
        "phase": 2,
        "checkpoint": str(checkpoint_path),
        "document_index": args.document_index,
        "sequence_length": len(ids),
        "layers": n_layers,
        "heads": n_heads,
        "example_text": tokenizer.decode(ids),
    }

    (out / "attention_metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("NEPALI — PHASE 2 ATTENTION ANALYSIS")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Mixed precision: FP16 autocast")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Attention tensor: {tuple(attention.shape)}")
    print(f"Example: {metadata['example_text'][:300]}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
