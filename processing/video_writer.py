"""Streaming video writer for processed frame chunks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Write video frames sequentially to disk."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self.path = Path(output_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._open()

    def _open(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*self.codec[:4].ljust(4, " "))
        self._writer = cv2.VideoWriter(
            str(self.path),
            fourcc,
            self.fps,
            self.frame_size,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame."""
        if self._writer is None:
            raise RuntimeError("VideoWriter is closed")
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)
        self._frame_count += 1

    def write_many(self, frames: list[np.ndarray]) -> None:
        """Write multiple frames."""
        for frame in frames:
            self.write(frame)

    @property
    def frame_count(self) -> int:
        """Return number of frames written."""
        return self._frame_count

    def close(self) -> None:
        """Release the writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
