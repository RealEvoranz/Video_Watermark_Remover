"""Install mmcv compatibility shims when mmcv-full is unavailable."""

from __future__ import annotations

import sys
import types
from importlib import import_module


def ensure_mmcv_compat() -> None:
    """
    Register a lightweight mmcv substitute for E2FGVI imports.

    E2FGVI expects mmcv-full, which does not provide wheels for modern
    Python versions on Windows. This shim uses torchvision deform_conv2d.
    """
    try:
        import mmcv  # noqa: F401

        return
    except ImportError:
        pass

    if "mmcv" in sys.modules:
        return

    root = import_module("backends.vendor.mmcv")
    cnn = import_module("backends.vendor.mmcv.cnn")
    ops = import_module("backends.vendor.mmcv.ops")
    runner = import_module("backends.vendor.mmcv.runner")

    mmcv_module = types.ModuleType("mmcv")
    mmcv_module.__version__ = getattr(root, "__version__", "compat")
    mmcv_module.cnn = cnn
    mmcv_module.ops = ops
    mmcv_module.runner = runner

    sys.modules["mmcv"] = mmcv_module
    sys.modules["mmcv.cnn"] = cnn
    sys.modules["mmcv.ops"] = ops
    sys.modules["mmcv.runner"] = runner
