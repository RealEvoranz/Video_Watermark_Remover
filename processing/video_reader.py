"""Streaming video reader that avoids loading entire videos into memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SUPPORTED_INPUT_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: str
    duration_seconds: float

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height)."""
        return self.width, self.height


class VideoReader:
    """Read video frames sequentially without buffering the full video."""

    def __init__(self, video_path: str | Path) -> None:
        self.path = Path(video_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{self.path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_INPUT_EXTENSIONS)}"
            )

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        self._metadata = self._read_metadata()
        self._frame_index = 0

    def _read_metadata(self) -> VideoMetadata:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            fourcc=fourcc,
            duration_seconds=duration,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Return video metadata."""
        return self._metadata

    @property
    def frame_index(self) -> int:
        """Return the index of the next frame to read."""
        return self._frame_index

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame.

        Returns:
            (success, frame) where frame is BGR uint8 ndarray or None on EOF.
        """
        success, frame = self._cap.read()
        if success:
            self._frame_index += 1
            return True, frame
        return False, None

    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame index."""
        if frame_index < 0:
            frame_index = 0
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if success:
            self._frame_index = frame_index
        return success

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames until end of video."""
        while True:
            success, frame = self.read()
            if not success or frame is None:
                break
            yield frame

    def read_chunk(self, count: int) -> list[np.ndarray]:
        """Read up to `count` frames."""
        frames: list[np.ndarray] = []
        for _ in range(count):
            success, frame = self.read()
            if not success or frame is None:
                break
            frames.append(frame)
        return frames

    def close(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self._metadata.frame_count
