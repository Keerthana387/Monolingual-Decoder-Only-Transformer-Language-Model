import re
from pathlib import Path
import torch


def resolve_checkpoint(checkpoint_dir: str) -> Path:
    d = Path(checkpoint_dir)
    if not d.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {d}")

    final = d / "final.pt"
    if final.exists():
        print(f"Using final checkpoint: {final}")
        return final

    numbered = sorted(
        d.glob("step_*.pt"),
        key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1))
        if re.search(r"step_(\d+)", p.stem) else -1,
    )
    if numbered:
        print(
            "WARNING: final.pt was not found. "
            f"Falling back to latest checkpoint: {numbered[-1]}"
        )
        return numbered[-1]

    raise FileNotFoundError(
        f"No final.pt or step_*.pt checkpoint found in {d}"
    )


def load_model(checkpoint_dir: str, device):
    from Telugu.config.model_config import ModelConfig
    from Telugu.model.lm import TeluguLanguageModel

    checkpoint_path = resolve_checkpoint(checkpoint_dir)
    config = ModelConfig()
    model = TeluguLanguageModel(config).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, config, checkpoint_path


def assert_telugu_config(config):
    expected = {
        "vocab_size": 6000,
        "max_seq_len": 512,
        "d_model": 448,
        "n_layers": 10,
        "n_heads": 8,
        "d_ff": 1344,
    }
    for name, value in expected.items():
        actual = getattr(config, name)
        if actual != value:
            raise RuntimeError(
                f"Unexpected Telugu {name}: expected {value}, got {actual}"
            )


def load_jsonl_texts(path: str):
    import json
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "text" not in record:
                raise KeyError(
                    f"{path}: line {line_no} has no 'text' field"
                )
            text = str(record["text"]).strip()
            if text:
                texts.append(text)
    return texts
