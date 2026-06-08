"""AI inpainting backend implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from backends.base_backend import BaseBackend, BackendInfo
from backends.passthrough_backend import PassthroughBackend

if TYPE_CHECKING:
    from backends.e2fgvi_backend import E2FGVIBackend
    from backends.propainter_backend import ProPainterBackend

BACKEND_REGISTRY: dict[str, str] = {
    "passthrough": "backends.passthrough_backend.PassthroughBackend",
    "e2fgvi": "backends.e2fgvi_backend.E2FGVIBackend",
    "propainter": "backends.propainter_backend.ProPainterBackend",
}


def _load_class(dotted_path: str) -> type[BaseBackend]:
    module_name, class_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_backend(name: str) -> BaseBackend:
    """Instantiate a backend by registry key."""
    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(BACKEND_REGISTRY)}"
        )
    backend_cls = _load_class(BACKEND_REGISTRY[name])
    return backend_cls()


def list_backends() -> list[str]:
    """Return registered backend keys."""
    return list(BACKEND_REGISTRY)


__all__ = [
    "BACKEND_REGISTRY",
    "BackendInfo",
    "BaseBackend",
    "PassthroughBackend",
    "get_backend",
    "list_backends",
]
