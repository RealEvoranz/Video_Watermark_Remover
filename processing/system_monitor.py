"""System resource monitoring for processing UI."""

from __future__ import annotations

import psutil


def get_ram_usage_mb() -> float:
    """Return current process RAM usage in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_system_ram_usage_mb() -> float:
    """Return system-wide used RAM in megabytes."""
    return psutil.virtual_memory().used / (1024 * 1024)


def get_vram_usage_mb() -> float | None:
    """Return GPU VRAM usage in megabytes, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 * 1024)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    return None


def get_available_vram_mb() -> int | None:
    """Return available GPU VRAM in megabytes, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(info.free / (1024 * 1024))
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return int(free / (1024 * 1024))
    except Exception:
        pass

    return None
