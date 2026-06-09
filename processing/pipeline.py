"""End-to-end video watermark removal pipeline with audio preservation."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from backends import get_backend
from backends.base_backend import BaseBackend
from processing.chunk_processor import ChunkProcessor, ProcessingProgress
from processing.config_loader import get_cache_dir, get_output_dir, load_config
from processing.ffmpeg_utils import extract_audio, has_audio_stream, mux_audio, reencode_h264


@dataclass
class PipelineConfig:
    """Configuration for a processing run."""

    backend_name: str = "e2fgvi"
    chunk_size: int | str = "auto"
    chunk_overlap: int = 5
    preserve_audio: bool = True
    reencode: bool = True
    output_crf: int = 18
    skip_start_seconds: int = 0


ProgressCallback = Callable[[ProcessingProgress], None]


class ProcessingPipeline:
    """Orchestrate mask-based inpainting with optional audio muxing."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback
        self._backend: BaseBackend | None = None
        self._processor: ChunkProcessor | None = None

    def _emit(self, progress: ProcessingProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _get_backend(self) -> BaseBackend:
        if self._backend is None:
            self._backend = get_backend(self.config.backend_name)
            self._backend.initialize()
        return self._backend

    def cancel(self) -> None:
        """Cancel ongoing processing."""
        if self._processor:
            self._processor.request_cancel()
        if self._backend:
            self._backend.request_cancel()

    def process(
        self,
        input_path: str | Path,
        mask: np.ndarray,
        output_path: str | Path | None = None,
    ) -> ProcessingProgress:
        """
        Run the full pipeline: extract audio, process video, restore audio.

        Args:
            input_path: Source video.
            mask: Grayscale inpainting mask.
            output_path: Optional final output path.

        Returns:
            Final processing progress.
        """
        input_path = Path(input_path).resolve()
        app_config = load_config()

        if output_path is None:
            output_dir = get_output_dir()
            output_path = output_dir / f"{input_path.stem}_clean.mp4"
        else:
            output_path = Path(output_path).resolve()

        cache_dir = get_cache_dir() / input_path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        video_only_path = cache_dir / "processed_video.mp4"
        audio_path = cache_dir / "audio.aac"
        temp_muxed = cache_dir / "muxed.mp4"

        progress = ProcessingProgress(message="Initializing backend...")
        self._emit(progress)

        try:
            backend = self._get_backend()
            overlap = self.config.chunk_overlap
            if overlap == 5:
                overlap = int(app_config["processing"].get("chunk_overlap", 5))

            self._processor = ChunkProcessor(
                backend=backend,
                chunk_size=self.config.chunk_size,
                chunk_overlap=overlap,
                skip_start_seconds=self.config.skip_start_seconds,
                progress_callback=self.progress_callback,
            )

            progress.message = "Processing video frames..."
            self._emit(progress)

            result = self._processor.process(
                input_path=input_path,
                output_path=video_only_path,
                mask=mask,
            )

            if result.cancelled or result.error:
                return result

            final_path = output_path
            has_audio = self.config.preserve_audio and has_audio_stream(input_path)

            if has_audio:
                progress.message = "Extracting audio..."
                self._emit(progress)

                if extract_audio(input_path, audio_path):
                    progress.message = "Muxing audio..."
                    self._emit(progress)
                    mux_audio(video_only_path, audio_path, temp_muxed)
                    source_for_encode = temp_muxed
                else:
                    source_for_encode = video_only_path
            else:
                source_for_encode = video_only_path

            if self.config.reencode:
                progress.message = "Encoding H.264 output..."
                self._emit(progress)
                crf = self.config.output_crf or int(
                    app_config["processing"].get("output_crf", 18)
                )
                reencode_h264(source_for_encode, final_path, crf=crf)
            else:
                shutil.copy2(source_for_encode, final_path)

            result.message = f"Saved to {final_path}"
            result.finished = True
            result.percent = 100.0
            self._emit(result)
            return result

        except Exception as exc:
            progress.error = str(exc)
            progress.message = f"Pipeline error: {exc}"
            self._emit(progress)
            return progress

        finally:
            if self._backend:
                self._backend.cleanup()
