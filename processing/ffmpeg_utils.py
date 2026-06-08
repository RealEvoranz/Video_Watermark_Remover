"""FFmpeg utilities for audio extraction and muxing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str], None]


def find_ffmpeg() -> str:
    """Locate ffmpeg executable."""
    path = shutil.which("ffmpeg")
    if path:
        return path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or imageio-ffmpeg."
        ) from exc


def find_ffprobe() -> str | None:
    """Locate ffprobe executable if available."""
    path = shutil.which("ffprobe")
    if path:
        return path
    return None


def _run_ffmpeg(
    args: list[str],
    progress_callback: ProgressCallback | None = None,
) -> None:
    ffmpeg = find_ffmpeg()
    command = [ffmpeg, "-y", *args]
    if progress_callback:
        progress_callback(f"Running: {' '.join(command)}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"FFmpeg failed: {stderr}")


def extract_audio(
    video_path: str | Path,
    audio_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> bool:
    """
    Extract audio from a video file.

    Returns:
        True if audio was extracted, False if no audio stream exists.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    probe = find_ffprobe()
    if probe:
        check = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0 or not check.stdout.strip():
            if progress_callback:
                progress_callback("No audio stream found in source video")
            return False

    _run_ffmpeg(
        ["-i", str(video_path), "-vn", "-acodec", "copy", str(audio_path)],
        progress_callback,
    )
    return audio_path.exists() and audio_path.stat().st_size > 0


def mux_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Mux processed video with extracted audio."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(output_path),
        ],
        progress_callback,
    )


def reencode_h264(
    input_path: str | Path,
    output_path: str | Path,
    crf: int = 18,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Re-encode video to H.264 MP4."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-c:a",
            "copy",
            str(output_path),
        ],
        progress_callback,
    )


def has_audio_stream(video_path: str | Path) -> bool:
    """Check whether a video file contains an audio stream."""
    probe = find_ffprobe()
    if not probe:
        return True

    result = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())
