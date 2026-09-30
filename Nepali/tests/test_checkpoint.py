from pathlib import Path

import torch

from Nepali.config.model_config import ModelConfig
from Nepali.config.train_config import TrainConfig
from Nepali.model.lm import LanguageModel
from Nepali.train import (
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path):
    model_config = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=32,
    )

    train_config = TrainConfig()

    model = LanguageModel(model_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=7,
        loss=1.23,
        config=train_config,
        path=path,
        scheduler_state={
            "step": 7,
            "current_lr": 1e-3,
            "schedule": "warmup_cosine",
        },
    )

    assert path.exists()

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["step"] == 7
    assert checkpoint["loss"] == 1.23
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert "train_config" in checkpoint
    assert "model_config" in checkpoint

    restored_model = LanguageModel(model_config)

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
    )

    step, loss, scheduler = load_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
    )

    assert step == 7
    assert loss == 1.23
    assert scheduler["step"] == 7
