"""Resumable model and source code downloader with verification."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from models.registry import ModelEntry, ModelFile, ModelRegistry
from processing.config_loader import get_models_dir, load_config

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "VideoWatermarkRemover/1.0"
    ),
}


@dataclass
class DownloadProgress:
    """Progress snapshot for a download operation."""

    filename: str
    bytes_downloaded: int
    total_bytes: int | None
    percent: float
    message: str
    completed: bool = False
    error: str | None = None


ProgressCallback = Callable[[DownloadProgress], None]


class ModelDownloader:
    """Download and verify model weights and source archives."""

    def __init__(self, models_root: Path | None = None) -> None:
        self.models_root = models_root or get_models_dir()
        self.models_root.mkdir(parents=True, exist_ok=True)
        config = load_config()
        self.chunk_size = int(config["download"]["chunk_size_bytes"])
        self.timeout = int(config["download"]["timeout_seconds"])
        self.max_retries = int(config["download"]["max_retries"])

    def ensure_backend(
        self,
        backend_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ModelEntry:
        """Ensure source code and weights exist for a backend."""
        entry = ModelRegistry.get(backend_id)
        self._ensure_source(entry, progress_callback)
        self._ensure_weights(entry, progress_callback)
        return entry

    def _emit(
        self,
        callback: ProgressCallback | None,
        progress: DownloadProgress,
    ) -> None:
        if callback:
            callback(progress)

    def _download_file(
        self,
        urls: tuple[str, ...],
        destination: Path,
        expected_sha256: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        for url_index, url in enumerate(urls):
            partial = destination.with_suffix(
                destination.suffix + f".part{url_index}"
            )
            for stale in destination.parent.glob(f"{destination.name}.part*"):
                if stale != partial:
                    stale.unlink(missing_ok=True)

            try:
                self._download_single_url(
                    url,
                    partial,
                    destination,
                    expected_sha256,
                    progress_callback,
                )
                return
            except Exception as exc:
                partial.unlink(missing_ok=True)
                errors.append(f"{url}: {exc}")

        raise RuntimeError(
            "All download sources failed:\n"
            + "\n".join(f"  - {item}" for item in errors)
        )

    def _download_single_url(
        self,
        url: str,
        partial: Path,
        destination: Path,
        expected_sha256: str | None,
        progress_callback: ProgressCallback | None,
    ) -> None:
        downloaded = partial.stat().st_size if partial.exists() else 0
        headers = dict(_DEFAULT_HEADERS)
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

        for attempt in range(1, self.max_retries + 1):
            try:
                request = Request(url, headers=headers)
                with urlopen(request, timeout=self.timeout) as response:
                    if response.status == 404:
                        raise HTTPError(
                            url, 404, "Not Found", response.headers, None
                        )

                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type and downloaded == 0:
                        raise RuntimeError(
                            "Server returned HTML instead of a model file "
                            "(link may require a browser download)"
                        )

                    total_header = response.headers.get("Content-Length")
                    total_bytes = (
                        int(total_header) + downloaded if total_header else None
                    )

                    mode = "ab" if downloaded > 0 else "wb"
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(self.chunk_size)
                            if not chunk:
                                break
                            handle.write(chunk)
                            downloaded += len(chunk)
                            percent = (
                                (downloaded / total_bytes) * 100.0
                                if total_bytes
                                else 0.0
                            )
                            self._emit(
                                progress_callback,
                                DownloadProgress(
                                    filename=destination.name,
                                    bytes_downloaded=downloaded,
                                    total_bytes=total_bytes,
                                    percent=percent,
                                    message=f"Downloading {destination.name}",
                                ),
                            )

                if partial.stat().st_size < 1_000_000:
                    raise RuntimeError(
                        f"Downloaded file too small ({partial.stat().st_size} bytes)"
                    )

                if expected_sha256:
                    actual = self._sha256(partial)
                    if actual.lower() != expected_sha256.lower():
                        raise RuntimeError(
                            f"Checksum mismatch for {destination.name}"
                        )

                partial.replace(destination)
                self._emit(
                    progress_callback,
                    DownloadProgress(
                        filename=destination.name,
                        bytes_downloaded=downloaded,
                        total_bytes=downloaded,
                        percent=100.0,
                        message=f"Downloaded {destination.name}",
                        completed=True,
                    ),
                )
                return

            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(str(exc)) from exc
                time.sleep(min(2 ** attempt, 30))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _ensure_weights(
        self,
        entry: ModelEntry,
        progress_callback: ProgressCallback | None,
    ) -> None:
        weights_dir = entry.weights_dir(self.models_root)
        weights_dir.mkdir(parents=True, exist_ok=True)

        for model_file in entry.weights:
            destination = weights_dir / model_file.filename
            if destination.exists():
                if model_file.sha256:
                    if self._sha256(destination).lower() == model_file.sha256.lower():
                        continue
                    destination.unlink()
                elif model_file.size_bytes:
                    if destination.stat().st_size >= model_file.size_bytes * 0.95:
                        continue
                    destination.unlink()
                else:
                    continue

            try:
                self._download_file(
                    model_file.all_urls,
                    destination,
                    model_file.sha256,
                    progress_callback,
                )
            except RuntimeError as exc:
                manual_hint = self._manual_download_hint(entry, model_file)
                raise RuntimeError(f"{exc}\n\n{manual_hint}") from exc

    @staticmethod
    def _manual_download_hint(entry: ModelEntry, model_file: ModelFile) -> str:
        lines = [
            "Manual download instructions:",
            f"  1. Download {model_file.filename}",
        ]
        if entry.manual_download_url:
            lines.append(f"     {entry.manual_download_url}")
        for url in model_file.all_urls:
            lines.append(f"     or {url}")
        lines.append(
            f"  2. Place the file in: {entry.weights_dir(get_models_dir())}"
        )
        if entry.manual_download_note:
            lines.append(f"  Note: {entry.manual_download_note}")
        return "\n".join(lines)

    def _ensure_source(
        self,
        entry: ModelEntry,
        progress_callback: ProgressCallback | None,
    ) -> None:
        source_dir = entry.source_dir(self.models_root)
        if source_dir.exists() and any(source_dir.iterdir()):
            return

        archive_path = self.models_root / entry.backend_id / entry.source_archive_name
        if not archive_path.exists():
            self._download_file(
                (entry.source_url,),
                archive_path,
                progress_callback=progress_callback,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(temp_dir)

            extracted_roots = list(Path(temp_dir).iterdir())
            if not extracted_roots:
                raise RuntimeError(f"Empty archive: {archive_path}")

            root = extracted_roots[0]
            source_dir.parent.mkdir(parents=True, exist_ok=True)
            if source_dir.exists():
                shutil.rmtree(source_dir)
            shutil.move(str(root), str(source_dir))

        self._emit(
            progress_callback,
            DownloadProgress(
                filename=entry.source_archive_name,
                bytes_downloaded=archive_path.stat().st_size,
                total_bytes=archive_path.stat().st_size,
                percent=100.0,
                message=f"Extracted source for {entry.backend_id}",
                completed=True,
            ),
        )
