"""Model registry with download URLs and verification metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelFile:
    """A single downloadable model artifact."""

    filename: str
    url: str
    fallback_urls: tuple[str, ...] = ()
    sha256: str | None = None
    size_bytes: int | None = None
    required: bool = True

    @property
    def all_urls(self) -> tuple[str, ...]:
        """Primary URL followed by fallbacks."""
        return (self.url, *self.fallback_urls)


@dataclass(frozen=True)
class ModelEntry:
    """Registry entry for an AI backend model package."""

    backend_id: str
    display_name: str
    description: str
    source_url: str
    source_archive_name: str
    weights: list[ModelFile] = field(default_factory=list)
    source_subdir: str = "source"
    manual_download_url: str | None = None
    manual_download_note: str | None = None

    def weights_dir(self, models_root: Path) -> Path:
        """Directory for backend weight files."""
        return models_root / self.backend_id / "weights"

    def source_dir(self, models_root: Path) -> Path:
        """Directory for backend source code."""
        return models_root / self.backend_id / self.source_subdir

    def is_ready(self, models_root: Path) -> bool:
        """Check whether required weights are present."""
        weights_dir = self.weights_dir(models_root)
        for item in self.weights:
            if not item.required:
                continue
            if not (weights_dir / item.filename).exists():
                return False
        return self.source_dir(models_root).exists()


class ModelRegistry:
    """Central registry of downloadable models."""

    ENTRIES: dict[str, ModelEntry] = {
        "e2fgvi": ModelEntry(
            backend_id="e2fgvi",
            display_name="E2FGVI-HQ",
            description="Flow-guided video inpainting (CVPR 2022)",
            source_url=(
                "https://github.com/MCG-NKU/E2FGVI/archive/refs/heads/master.zip"
            ),
            source_archive_name="E2FGVI-master.zip",
            manual_download_url=(
                "https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view"
            ),
            manual_download_note=(
                "E2FGVI weights are not hosted on GitHub releases. "
                "If automatic download fails, download E2FGVI-HQ-CVPR22.pth "
                "from Google Drive and place it in models/e2fgvi/weights/"
            ),
            weights=[
                ModelFile(
                    filename="E2FGVI-HQ-CVPR22.pth",
                    url=(
                        "https://huggingface.co/spaces/VIPLab/Track-Anything/"
                        "resolve/main/checkpoints/E2FGVI-HQ-CVPR22.pth"
                    ),
                    fallback_urls=(
                        (
                            "https://huggingface.co/spaces/LionelM10/"
                            "Track-Anything2/resolve/main/checkpoints/"
                            "E2FGVI-HQ-CVPR22.pth"
                        ),
                    ),
                    size_bytes=164_535_938,
                ),
            ],
        ),
        "propainter": ModelEntry(
            backend_id="propainter",
            display_name="ProPainter",
            description="Propagation and Transformer video inpainting (ICCV 2023)",
            source_url=(
                "https://github.com/sczhou/ProPainter/archive/refs/heads/main.zip"
            ),
            source_archive_name="ProPainter-main.zip",
            weights=[
                ModelFile(
                    filename="ProPainter.pth",
                    url=(
                        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
                        "ProPainter.pth"
                    ),
                ),
                ModelFile(
                    filename="raft-things.pth",
                    url=(
                        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
                        "raft-things.pth"
                    ),
                ),
                ModelFile(
                    filename="recurrent_flow_completion.pth",
                    url=(
                        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
                        "recurrent_flow_completion.pth"
                    ),
                ),
            ],
        ),
    }

    @classmethod
    def get(cls, backend_id: str) -> ModelEntry:
        """Return registry entry for a backend."""
        if backend_id not in cls.ENTRIES:
            raise KeyError(f"Unknown model backend: {backend_id}")
        return cls.ENTRIES[backend_id]

    @classmethod
    def list_entries(cls) -> list[ModelEntry]:
        """Return all registry entries."""
        return list(cls.ENTRIES.values())

    @classmethod
    def status(cls, models_root: Path) -> dict[str, Any]:
        """Return readiness status for all backends."""
        return {
            entry.backend_id: {
                "display_name": entry.display_name,
                "ready": entry.is_ready(models_root),
                "weights_dir": str(entry.weights_dir(models_root)),
                "source_dir": str(entry.source_dir(models_root)),
            }
            for entry in cls.ENTRIES.values()
        }
