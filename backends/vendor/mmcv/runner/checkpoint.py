"""Checkpoint loading compatible with E2FGVI SPyNet pretrained weights."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format")


def load_checkpoint(
    model: nn.Module,
    filename: str,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
    logger: Any = None,
) -> dict[str, torch.Tensor]:
    """Load weights from a local path or URL into a module."""
    del logger

    if filename.startswith(("http://", "https://")):
        checkpoint = torch.hub.load_state_dict_from_url(
            filename,
            map_location=map_location,
            progress=True,
        )
    else:
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

    state_dict = _unwrap_state_dict(checkpoint)
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value

    model.load_state_dict(cleaned, strict=strict)
    return cleaned
