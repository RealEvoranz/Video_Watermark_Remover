"""Chunk-based video processing orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from backends.base_backend import BaseBackend
from processing.system_monitor import get_available_vram_mb, get_ram_usage_mb, get_vram_usage_mb
from processing.video_reader import VideoReader
from processing.video_writer import VideoWriter


@dataclass
class ProcessingProgress:
    """Progress snapshot for UI updates."""

    current_chunk: int = 0
    total_chunks: int = 0
    frames_processed: int = 0
    total_frames: int = 0
    fps: float = 0.0
    vram_mb: float | None = None
    ram_mb: float = 0.0
    eta_seconds: float | None = None
    message: str = ""
    percent: float = 0.0
    cancelled: bool = False
    finished: bool = False
    error: str | None = None


ProgressCallback = Callable[[ProcessingProgress], None]


@dataclass
class ChunkProcessor:
    """
    Process videos in fixed-size frame chunks to keep memory usage constant.

    Supports temporal overlap between chunks for backends that need context.
    """

    backend: BaseBackend
    chunk_size: int | str = "auto"
    chunk_overlap: int = 5
    skip_start_seconds: int = 0
    progress_callback: ProgressCallback | None = None

    _cancel_requested: bool = field(default=False, init=False)

    def request_cancel(self) -> None:
        """Request processing cancellation."""
        self._cancel_requested = True
        self.backend.request_cancel()

    def _resolve_chunk_size(self, reader: VideoReader) -> int:
        if isinstance(self.chunk_size, int) and self.chunk_size > 0:
            return self.chunk_size

        meta = reader.metadata
        vram = get_available_vram_mb()
        return max(
            4,
            self.backend.estimate_chunk_size(meta.width, meta.height, vram),
        )

    def _emit(self, progress: ProcessingProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mask: np.ndarray,
    ) -> ProcessingProgress:
        """
        Process a video file with the given static mask.

        Args:
            input_path: Source video path.
            output_path: Destination video path (video only, no audio).
            mask: Grayscale mask aligned to video resolution.

        Returns:
            Final progress state.
        """
        self._cancel_requested = False
        self.backend.reset_cancel()
        progress = ProcessingProgress()
        start_time = time.perf_counter()
        frames_done = 0

        try:
            with VideoReader(input_path) as reader:
                meta = reader.metadata
                chunk_size = self._resolve_chunk_size(reader)
                overlap = max(0, min(self.chunk_overlap, chunk_size - 1))
                step = max(1, chunk_size - overlap)

                total_frames = meta.frame_count or 0
                if total_frames > 0:
                    total_chunks = max(
                        1,
                        (total_frames + step - 1) // step,
                    )
                else:
                    total_chunks = 1

                progress.total_frames = total_frames
                progress.total_chunks = total_chunks

                if mask.shape[:2] != (meta.height, meta.width):
                    raise ValueError(
                        f"Mask size {mask.shape[:2]} does not match "
                        f"video size {(meta.height, meta.width)}"
                    )

                with VideoWriter(
                    output_path,
                    fps=meta.fps,
                    frame_size=(meta.width, meta.height),
                ) as writer:
                    carry_over: list[np.ndarray] = []
                    # Optionally skip processing of the initial N seconds
                    skip_frames = min(int(self.skip_start_seconds * meta.fps), total_frames)
                    if skip_frames > 0:
                        initial = reader.read_chunk(skip_frames)
                        if initial:
                            writer.write_many(initial)
                            frames_done += len(initial)

                        remaining = max(0, total_frames - frames_done)
                        if remaining > 0:
                            total_chunks = max(1, (remaining + step - 1) // step)
                        else:
                            total_chunks = 0

                        progress.frames_processed = frames_done
                        progress.total_chunks = total_chunks
                        progress.percent = (
                            (frames_done / total_frames) * 100.0
                            if total_frames > 0
                            else 0.0
                        )
                        progress.message = (
                            f"Skipping {frames_done} initial frames ({self.skip_start_seconds}s)"
                        )
                        self._emit(progress)
                    chunk_index = 0

                    while True:
                        if self._cancel_requested or self.backend.is_cancelled:
                            progress.cancelled = True
                            progress.message = "Processing cancelled"
                            self._emit(progress)
                            return progress

                        buffer = list(carry_over)
                        needed = chunk_size - len(buffer)
                        new_frames = reader.read_chunk(needed)
                        buffer.extend(new_frames)

                        if not buffer:
                            break

                        is_last = len(new_frames) < needed
                        chunk_index += 1

                        result = self.backend.process_chunk(buffer, mask)

                        if len(result) != len(buffer):
                            raise RuntimeError(
                                f"Backend returned {len(result)} frames "
                                f"for {len(buffer)} input frames"
                            )

                        if is_last:
                            write_frames = result
                            carry_over = []
                        else:
                            write_count = len(result) - overlap
                            write_frames = result[:write_count]
                            carry_over = result[write_count:]

                        writer.write_many(write_frames)
                        frames_done += len(write_frames)

                        elapsed = time.perf_counter() - start_time
                        fps = frames_done / elapsed if elapsed > 0 else 0.0
                        remaining = max(0, total_frames - frames_done)
                        eta = remaining / fps if fps > 0 and total_frames else None

                        progress.current_chunk = chunk_index
                        progress.frames_processed = frames_done
                        progress.fps = fps
                        progress.vram_mb = get_vram_usage_mb()
                        progress.ram_mb = get_ram_usage_mb()
                        progress.eta_seconds = eta
                        progress.percent = (
                            (frames_done / total_frames) * 100.0
                            if total_frames > 0
                            else 0.0
                        )
                        progress.message = (
                            f"Chunk {chunk_index}/{total_chunks} "
                            f"({frames_done}/{total_frames} frames)"
                        )
                        self._emit(progress)

                        if is_last:
                            break

            progress.finished = True
            progress.percent = 100.0
            progress.message = "Processing complete"
            self._emit(progress)
            return progress

        except Exception as exc:
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            self._emit(progress)
            return progress
