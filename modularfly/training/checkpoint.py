"""Save and load a model + optimizer checkpoint. File I/O only.

The file holds {"model": state_dict, "optimizer": state_dict, "epoch": int}: tensors,
numbers and containers only, so it loads with torch.load(weights_only=True). Configs are
not stored here (Path fields fail weights_only); they belong in a separate JSON file.
"""
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int) -> Path:
    """Writes the checkpoint to `path` (overwriting), creating parent directories; returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
               path)
    return path


def load_checkpoint(path: Path) -> dict:
    """The saved dict, tensors on the CPU, read with weights_only=True."""
    return torch.load(path, weights_only=True, map_location="cpu")
