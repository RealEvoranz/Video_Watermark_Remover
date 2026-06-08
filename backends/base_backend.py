"""Abstract base class for AI inpainting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BackendInfo:
    """Metadata about a backend implementation."""

    name: str
    display_name: str
    description: str
    requires_gpu: bool
    min_vram_mb: int
    supports_chunking: bool


ProgressCallback = Callable[[str], None]


class BaseBackend(ABC):
    """Common interface for all inpainting backends."""

    name: str = "base"
    display_name: str = "Base"
    description: str = ""
    requires_gpu: bool = False
    min_vram_mb: int = 0
    supports_chunking: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._cancel_requested = False

    @property
    def info(self) -> BackendInfo:
        """Return backend metadata."""
        return BackendInfo(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            requires_gpu=self.requires_gpu,
            min_vram_mb=self.min_vram_mb,
            supports_chunking=self.supports_chunking,
        )

    @abstractmethod
    def initialize(self, progress_callback: ProgressCallback | None = None) -> None:
        """Load models and prepare the backend for inference."""

    @abstractmethod
    def process_chunk(
        self,
        frames: list[np.ndarray],
        mask: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Process a chunk of BGR frames with a static mask.

        Args:
            frames: List of BGR uint8 frames.
            mask: Grayscale mask (255=inpaint, 0=preserve), same size as frames.

        Returns:
            Processed BGR frames matching input length.
        """

    @abstractmethod
    def estimate_chunk_size(
        self,
        width: int,
        height: int,
        available_vram_mb: int | None = None,
    ) -> int:
        """Estimate optimal chunk size for given resolution and VRAM."""

    def request_cancel(self) -> None:
        """Request cancellation of ongoing processing."""
        self._cancel_requested = True

    def reset_cancel(self) -> None:
        """Clear cancellation flag."""
        self._cancel_requested = False

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        return self._cancel_requested

    def cleanup(self) -> None:
        """Release GPU resources."""
        self._initialized = False

    def ensure_initialized(self) -> None:
        """Raise if backend has not been initialized."""
        if not self._initialized:
            raise RuntimeError(f"Backend '{self.name}' is not initialized")
