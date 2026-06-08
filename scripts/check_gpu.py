"""Verify GPU and CUDA PyTorch setup."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processing.gpu_utils import cuda_diagnostics, require_cuda


def main() -> int:
    diag = cuda_diagnostics()
    print("GPU / CUDA diagnostics")
    print("-" * 40)
    for key, value in diag.items():
        print(f"  {key}: {value}")
    print("-" * 40)

    try:
        require_cuda("E2FGVI")
        print("CUDA check passed. AI backends should work.")
        return 0
    except RuntimeError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
