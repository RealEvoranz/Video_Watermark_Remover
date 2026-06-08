"""Mask editor with undo/redo history."""

from __future__ import annotations

import numpy as np

from processing.mask_utils import create_empty_mask


class MaskEditor:
    """Manage mask state with undo/redo stacks."""

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self._width = width
        self._height = height
        self._mask = (
            create_empty_mask(width, height) if width > 0 and height > 0 else None
        )
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    @property
    def mask(self) -> np.ndarray | None:
        """Return current mask."""
        return self._mask

    @property
    def can_undo(self) -> bool:
        """Whether undo is available."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Whether redo is available."""
        return len(self._redo_stack) > 0

    def resize(self, width: int, height: int) -> None:
        """Create or resize mask for a new video resolution."""
        self._width = width
        self._height = height
        self._mask = create_empty_mask(width, height)
        self._undo_stack.clear()
        self._redo_stack.clear()

    def load(self, mask: np.ndarray) -> None:
        """Load a mask, resizing if needed."""
        if self._width > 0 and self._height > 0:
            if mask.shape[1] != self._width or mask.shape[0] != self._height:
                import cv2

                mask = cv2.resize(
                    mask,
                    (self._width, self._height),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            self._height, self._width = mask.shape[:2]

        self._mask = mask.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()

    def snapshot(self) -> None:
        """Save current mask state for undo."""
        if self._mask is None:
            return
        self._undo_stack.append(self._mask.copy())
        self._redo_stack.clear()
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def undo(self) -> np.ndarray | None:
        """Revert to previous mask state."""
        if not self.can_undo or self._mask is None:
            return self._mask
        self._redo_stack.append(self._mask.copy())
        self._mask = self._undo_stack.pop()
        return self._mask

    def redo(self) -> np.ndarray | None:
        """Reapply undone mask state."""
        if not self.can_redo or self._mask is None:
            return self._mask
        self._undo_stack.append(self._mask.copy())
        self._mask = self._redo_stack.pop()
        return self._mask

    def clear(self) -> None:
        """Clear all mask regions."""
        if self._mask is None:
            return
        self.snapshot()
        self._mask[:] = 0

    def update_mask(self, mask: np.ndarray) -> None:
        """Replace mask pixels without clearing undo history."""
        if self._mask is None:
            self.load(mask)
            return
        self._mask[:] = mask
