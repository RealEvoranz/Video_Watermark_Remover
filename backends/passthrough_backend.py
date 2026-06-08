"""Passthrough backend for testing the chunk processing pipeline."""

from __future__ import annotations

import numpy as np

from backends.base_backend import BaseBackend, ProgressCallback


class PassthroughBackend(BaseBackend):
    """Returns frames unchanged; used to verify chunking without GPU models."""

    name = "passthrough"
    display_name = "Passthrough (Test)"
    description = "No AI processing; verifies chunk pipeline"
    requires_gpu = False
    min_vram_mb = 0

    def initialize(self, progress_callback: ProgressCallback | None = None) -> None:
        if progress_callback:
            progress_callback("Passthrough backend ready")
        self._initialized = True

    def process_chunk(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray,
    ) -> list[np.ndarray]:
        self.ensure_initialized()
        if self.is_cancelled:
            return frames
        return [frame.copy() for frame in frames]

    def estimate_chunk_size(
        self,
        width: int,
        height: int,
        available_vram_mb: int | None = None,
    ) -> int:
        return 120
