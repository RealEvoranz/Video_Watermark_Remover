"""Minimal mmcv CNN utilities."""

from __future__ import annotations

import torch.nn as nn


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    """Initialize module weights/bias to constants."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)
