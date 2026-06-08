"""Application configuration loading and path resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from config.json."""
    path = config_path or (get_project_root() / "config.json")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(relative: str, config: dict[str, Any] | None = None) -> Path:
    """Resolve a config-relative path against the project root."""
    root = get_project_root()
    return (root / relative).resolve()


def get_models_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the models storage directory."""
    cfg = config or load_config()
    return resolve_path(cfg["paths"]["models_dir"], cfg)


def get_cache_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the cache directory."""
    cfg = config or load_config()
    path = resolve_path(cfg["paths"]["cache_dir"], cfg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the default output directory."""
    cfg = config or load_config()
    path = resolve_path(cfg["paths"]["output_dir"], cfg)
    path.mkdir(parents=True, exist_ok=True)
    return path
