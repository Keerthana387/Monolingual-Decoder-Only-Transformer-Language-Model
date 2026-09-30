"""
NEPALI — PHASE 3 PRETRAINED vs FINE-TUNED ATTENTION COMPARISON

Uses comparative-reasoning prompts from the held-out TEST set.

Outputs:
  - representative early/late heatmaps for pretrained and fine-tuned models
  - side-by-side difference heatmaps
  - aggregate attention entropy and mean-distance CSV/JSON
  - entropy/distance comparison plots

This reuses the Phase 2 model attention interface:
    model(..., return_attention=True)

No training or validation examples are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Nepali.config.model_config import ModelConfig
from Nepali.model.lm import LanguageModel


def load_model(path, device):
    config = ModelConfig()
    model = LanguageModel(config).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def entropy(att):
    safe = att.float().clamp_min(1e-12)
    return -(safe * safe.log()).sum(dim=-1).mean(dim=-1)


def distance(att):
    h, q, k = att.shape
    pk = torch.arange(k, device=att.device, dtype=att.dtype)
    pq = torch.arange(q, device=att.device, dtype=att.dtype)
    d = (pq[:, None] - pk[None, :]).abs()
    return (att * d[None, :, :]).sum(dim=-1).mean(dim=-1)


def extract_attention(model, ids, device):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        use_amp = device.type == "cuda"
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, attentions = model(input_ids=x, targets=None, return_attention=True)
        else:
            _, _, attentions = model(input_ids=x, targets=None, return_attention=True)
    layers = []
    for a in attentions:
        layers.append(a[0] if a.dim() == 4 else a)
    return torch.stack(layers).float()


def heatmap(matrix, path, title, xlabel="Key token", ylabel="Query token"):
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Attention weight")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-prompts", type=int, default=50)
    ap.add_argument("--sequence-length", type=int, default=96)
    ap.add_argument("--representative-index", type=int, default=-1,
                    help="Index within selected reasoning prompts; -1 chooses the first transitive/ranking prompt")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != 8000:
        raise RuntimeError("Expected 8000-token Nepali tokenizer")

    with Path(args.test).open("r", encoding="utf-8") as f:
        records = [json.loads(x) for x in f if x.strip()]

    # Focus on exactly the kind of prompts named by the Phase 3 requirement.
    preferred = [
        r for r in records
        if r.get("task_type") in {"transitive_2", "transitive_3", "ordering", "reverse_comparison", "direct_comparison"}
    ]
    if not preferred:
        preferred = records
    selected = preferred[:args.num_prompts]

    if args.representative_index >= 0:
        rep = selected[args.representative_index]
    else:
        rep = next((r for r in selected if r.get("task_type") in {"transitive_2", "transitive_3", "ordering"}), selected[0])

    models = {
        "pretrained": load_model(args.pretrained, device),
        "finetuned": load_model(args.finetuned, device),
    }

    # Per-model aggregate statistics.
    aggregate = {name: [] for name in models}
    representative_meta = {
        "id": rep.get("id"),
        "task_type": rep.get("task_type"),
        "difficulty": rep.get("difficulty"),
        "prompt": rep.get("prompt"),
        "answer": rep.get("answer"),
    }
    (out / "representative_prompt.json").write_text(
        json.dumps(representative_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name, model in models.items():
        att = extract_attention(model, tok.encode(rep["prompt"]).ids[:args.sequence_length], device)
        n_layers = att.shape[0]
        for layer, label in [(0, "early"), (n_layers - 1, "late")]:
            # Average heads for the representative heatmap; head-specific heatmaps
            # are generated below for the most long-range head.
            mean_map = att[layer].mean(dim=0).cpu().numpy()
            heatmap(mean_map, out / f"attention_{name}_{label}_layer_{layer:02d}.png",
                    f"Nepali {name} — {label} layer {layer} — mean over heads")

            # Most long-range head, using mean attention distance.
            head_dist = distance(att[layer]).cpu().numpy()
            head = int(np.argmax(head_dist))
            heatmap(att[layer, head].cpu().numpy(),
                    out / f"attention_{name}_{label}_layer_{layer:02d}_head_{head:02d}.png",
                    f"Nepali {name} — {label} layer {layer}, head {head}")

        ent = torch.stack([entropy(att[l]) for l in range(n_layers)]).cpu().numpy()
        dist = torch.stack([distance(att[l]) for l in range(n_layers)]).cpu().numpy()
        aggregate[name].append({"entropy": ent, "distance": dist})

        # Aggregate over additional prompts.
        for idx, r in enumerate(selected):
            ids = tok.encode(r["prompt"]).ids[:args.sequence_length]
            if len(ids) < 2:
                continue
            a = extract_attention(model, ids, device)
            aggregate[name].append({
                "entropy": torch.stack([entropy(a[l]) for l in range(a.shape[0])]).cpu().numpy(),
                "distance": torch.stack([distance(a[l]) for l in range(a.shape[0])]).cpu().numpy(),
            })
            if (idx + 1) % 10 == 0:
                print(f"{name}: attention prompts {idx + 1}/{len(selected)}")

    n_layers = aggregate["pretrained"][0]["entropy"].shape[0]
    n_heads = aggregate["pretrained"][0]["entropy"].shape[1]
    rows = []
    summary = {}
    for name in aggregate:
        ent_stack = np.stack([x["entropy"] for x in aggregate[name]], axis=0)
        dist_stack = np.stack([x["distance"] for x in aggregate[name]], axis=0)
        summary[name] = {
            "num_prompts": int(ent_stack.shape[0]),
            "entropy_mean": float(ent_stack.mean()),
            "distance_mean": float(dist_stack.mean()),
        }
        for layer in range(n_layers):
            for head in range(n_heads):
                rows.append({
                    "model": name,
                    "layer": layer,
                    "head": head,
                    "mean_entropy": float(ent_stack[:, layer, head].mean()),
                    "mean_attention_distance": float(dist_stack[:, layer, head].mean()),
                })

    with (out / "attention_comparison.json").open("w", encoding="utf-8") as f:
        json.dump({"representative": representative_meta, "summary": summary, "head_rows": rows}, f, ensure_ascii=False, indent=2)

    import csv
    with (out / "attention_head_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Layer-level plots.
    layer_summary = {}
    for name in aggregate:
        e = np.stack([x["entropy"] for x in aggregate[name]], axis=0).mean(axis=(0, 2))
        d = np.stack([x["distance"] for x in aggregate[name]], axis=0).mean(axis=(0, 2))
        layer_summary[name] = {"entropy": e.tolist(), "distance": d.tolist()}

    x = np.arange(n_layers)
    plt.figure(figsize=(10, 5))
    plt.plot(x, layer_summary["pretrained"]["entropy"], marker="o", label="Pretrained")
    plt.plot(x, layer_summary["finetuned"]["entropy"], marker="o", label="Fine-tuned")
    plt.xlabel("Layer")
    plt.ylabel("Mean attention entropy")
    plt.title("Nepali attention entropy before vs after reasoning fine-tuning")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "attention_entropy_comparison.png", dpi=300, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(x, layer_summary["pretrained"]["distance"], marker="o", label="Pretrained")
    plt.plot(x, layer_summary["finetuned"]["distance"], marker="o", label="Fine-tuned")
    plt.xlabel("Layer")
    plt.ylabel("Mean attention distance")
    plt.title("Nepali attention distance before vs after reasoning fine-tuning")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "attention_distance_comparison.png", dpi=300, bbox_inches="tight"); plt.close()

    # Difference heatmaps for representative early and late layers.
    for name in models:
        pass
    rep_att = {}
    rep_ids = tok.encode(rep["prompt"]).ids[:args.sequence_length]
    for name, model in models.items():
        rep_att[name] = extract_attention(model, rep_ids, device).cpu().numpy()
    for layer, label in [(0, "early"), (n_layers - 1, "late")]:
        diff = rep_att["finetuned"][layer].mean(axis=0) - rep_att["pretrained"][layer].mean(axis=0)
        heatmap(diff, out / f"attention_difference_{label}_layer_{layer:02d}.png",
                f"Nepali fine-tuned − pretrained attention — {label} layer {layer}")

    (out / "layer_summary.json").write_text(json.dumps(layer_summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("NEPALI — PHASE 3 ATTENTION COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Representative task: {rep.get('task_type')} | id={rep.get('id')}")
    print(f"Prompts analyzed: {len(selected)}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
