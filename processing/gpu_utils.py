"""GPU detection and CUDA troubleshooting helpers."""

from __future__ import annotations


def get_nvidia_gpu_name() -> str | None:
    """Return the first NVIDIA GPU name via NVML, if available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace")
        return str(name)
    except Exception:
        return None


def cuda_diagnostics() -> dict[str, str | bool | None]:
    """Collect CUDA/PyTorch diagnostic information."""
    import torch

    gpu_name = get_nvidia_gpu_name()
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name_nvml": gpu_name,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "is_cpu_only_torch": "+cpu" in torch.__version__,
    }


def require_cuda(backend_name: str) -> None:
    """
    Raise a helpful error when CUDA is required but unavailable.

    Detects the common case: NVIDIA GPU present but CPU-only PyTorch installed.
    """
    import torch

    if torch.cuda.is_available():
        return

    diag = cuda_diagnostics()
    gpu_name = diag.get("gpu_name_nvml")
    is_cpu_torch = diag.get("is_cpu_only_torch")

    lines = [
        f"{backend_name} requires a CUDA-capable GPU with CUDA-enabled PyTorch.",
        "",
        f"PyTorch version: {diag['torch_version']}",
        f"torch.cuda.is_available(): False",
    ]

    if gpu_name:
        lines.append(f"NVIDIA GPU detected: {gpu_name}")
        lines.append(
            "Your GPU is present, but PyTorch cannot use it."
        )

    if is_cpu_torch or gpu_name:
        lines.extend(
            [
                "",
                "Fix: reinstall PyTorch with CUDA support:",
                "",
                "  pip uninstall torch torchvision -y",
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
                "",
                "Then verify:",
                "  python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No NVIDIA GPU was detected. AI inpainting backends need an NVIDIA GPU.",
                "You can use the Passthrough backend to test the pipeline without AI.",
            ]
        )

    raise RuntimeError("\n".join(lines))
