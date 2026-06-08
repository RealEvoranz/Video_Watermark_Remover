"""Mask creation, loading, saving, and manipulation utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_empty_mask(width: int, height: int) -> np.ndarray:
    """Create a black (preserve) mask."""
    return np.zeros((height, width), dtype=np.uint8)


def load_mask(path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load a grayscale mask from PNG.

    Args:
        path: Path to mask image.
        target_size: Optional (width, height) to resize mask.

    Returns:
        Grayscale uint8 mask (255=remove, 0=preserve).
    """
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to load mask: {mask_path}")

    if target_size is not None:
        width, height = target_size
        if mask.shape[1] != width or mask.shape[0] != height:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return mask


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save mask as grayscale PNG."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output), mask)


def apply_rectangle(
    mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    value: int = 255,
) -> np.ndarray:
    """Draw a filled rectangle on the mask."""
    result = mask.copy()
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = max(0, left)
    top = max(0, top)
    right = min(result.shape[1], right)
    bottom = min(result.shape[0], bottom)
    result[top:bottom, left:right] = value
    return result


def apply_brush(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    radius: int,
    value: int = 255,
) -> np.ndarray:
    """Paint brush strokes onto the mask."""
    result = mask.copy()
    for x, y in points:
        cv2.circle(result, (x, y), radius, value, thickness=-1)
    return result


def dilate_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Dilate mask regions for better inpainting coverage."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=iterations)


def create_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a colored mask overlay onto a BGR frame."""
    overlay = frame.copy()
    tinted = np.zeros_like(frame)
    tinted[:] = color
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return frame

    blended = cv2.addWeighted(frame, 1.0 - alpha, tinted, alpha, 0)
    overlay[mask_bool] = blended[mask_bool]
    return overlay


def mask_coverage_percent(mask: np.ndarray) -> float:
    """Return percentage of mask marked for removal."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size) * 100.0
