"""Verify all Python dependencies for the app and AI backends."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (import_name, display_name, required_for)
REQUIRED_PACKAGES = [
    ("PySide6", "PySide6", "GUI"),
    ("cv2", "opencv-python", "core"),
    ("numpy", "numpy", "core"),
    ("PIL", "Pillow", "core"),
    ("torch", "torch", "AI backends"),
    ("torchvision", "torchvision", "AI backends"),
    ("tqdm", "tqdm", "core"),
    ("psutil", "psutil", "monitoring"),
    ("pynvml", "nvidia-ml-py", "GPU monitoring"),
    ("imageio", "imageio", "core"),
    ("imageio_ffmpeg", "imageio-ffmpeg", "FFmpeg"),
    ("scipy", "scipy", "AI backends"),
    ("matplotlib", "matplotlib", "E2FGVI"),
    ("yaml", "PyYAML", "ProPainter"),
    ("addict", "addict", "E2FGVI/ProPainter"),
    ("skimage", "scikit-image", "E2FGVI"),
    ("einops", "einops", "ProPainter"),
    ("timm", "timm", "ProPainter"),
    ("av", "av", "ProPainter"),
    ("requests", "requests", "downloads"),
]


def main() -> int:
    print("Dependency check")
    print("=" * 50)
    missing: list[str] = []
    ok = 0

    for import_name, pip_name, used_by in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"  OK  {pip_name:22} ({used_by})")
            ok += 1
        except ImportError:
            print(f"  MISS {pip_name:22} ({used_by})")
            missing.append(pip_name)

    print("=" * 50)

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from backends.mmcv_compat import ensure_mmcv_compat

    ensure_mmcv_compat()
    import mmcv

    print(f"mmcv: {getattr(mmcv, '__version__', 'compat shim')}")

    if missing:
        print()
        print("Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return 1

    print()
    print(f"All {ok} packages available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
