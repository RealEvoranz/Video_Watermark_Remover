"""Video frame viewer with zoom, pan, and mask overlay."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from processing.mask_utils import create_overlay


class ToolMode(Enum):
    """Active mask editing tool."""

    NONE = "none"
    RECTANGLE = "rectangle"
    BRUSH = "brush"
    ERASER = "eraser"


class VideoWidget(QGraphicsView):
    """Interactive video frame display with drawing support."""

    mask_edit_began = Signal()
    mask_changed = Signal()
    frame_clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._frame: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._overlay_color = (0, 0, 255)
        self._overlay_alpha = 0.45

        self._tool = ToolMode.NONE
        self._brush_size = 20
        self._drawing = False
        self._start_point: QPoint | None = None
        self._last_point: QPoint | None = None
        self._panning = False
        self._pan_start = QPointF()
        self._zoom_factor = 1.0

    def set_overlay_style(
        self,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Configure mask overlay appearance."""
        self._overlay_color = color
        self._overlay_alpha = alpha
        self._refresh_display()

    def set_frame(self, frame: np.ndarray | None) -> None:
        """Set the current BGR frame."""
        self._frame = frame.copy() if frame is not None else None
        self._refresh_display()

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Set the current mask."""
        self._mask = mask
        self._refresh_display()

    def get_mask(self) -> np.ndarray | None:
        """Return the current mask."""
        return self._mask

    def set_tool(self, tool: ToolMode) -> None:
        """Set active drawing tool."""
        self._tool = tool

    def set_brush_size(self, size: int) -> None:
        """Set brush/eraser radius."""
        self._brush_size = max(1, size)

    def reset_view(self) -> None:
        """Reset zoom and center the frame."""
        self.resetTransform()
        self._zoom_factor = 1.0
        if self._pixmap_item.pixmap().isNull():
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def dragEnterEvent(self, event: QMouseEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in [
                    ".mp4",
                    ".mov",
                    ".mkv",
                    ".avi",
                ]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QMouseEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]:
                        main_window = self.window()
                        if hasattr(main_window, "_open_video_path"):
                            main_window._open_video_path(path)
                            event.acceptProposedAction()
                            return
        event.ignore()

    def _refresh_display(self) -> None:
        if self._frame is None:
            self._pixmap_item.setPixmap(QPixmap())
            return

        display = self._frame
        if self._mask is not None:
            display = create_overlay(
                self._frame,
                self._mask,
                color=self._overlay_color,
                alpha=self._overlay_alpha,
            )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))

    def _scene_pos(self, event: QMouseEvent) -> tuple[int, int] | None:
        point = self.mapToScene(event.position().toPoint())
        x, y = int(point.x()), int(point.y())
        if self._frame is None:
            return None
        height, width = self._frame.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom_factor *= factor
        self._zoom_factor = max(0.1, min(self._zoom_factor, 10.0))
        self.setTransform(self.transform().scale(factor, factor))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        pos = self._scene_pos(event)
        if pos is None or self._mask is None:
            return

        self._drawing = True
        self._start_point = QPoint(*pos)
        self._last_point = QPoint(*pos)
        self.mask_edit_began.emit()

        if self._tool in (ToolMode.BRUSH, ToolMode.ERASER):
            value = 255 if self._tool == ToolMode.BRUSH else 0
            self._paint_stroke(pos, pos, value)
            self.mask_changed.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            return

        if not self._drawing or self._mask is None:
            super().mouseMoveEvent(event)
            return

        pos = self._scene_pos(event)
        if pos is None:
            return

        if self._tool in (ToolMode.BRUSH, ToolMode.ERASER):
            value = 255 if self._tool == ToolMode.BRUSH else 0
            if self._last_point is not None:
                self._paint_stroke(
                    (self._last_point.x(), self._last_point.y()),
                    pos,
                    value,
                )
            self._last_point = QPoint(*pos)
            self.mask_changed.emit()
        elif self._tool == ToolMode.RECTANGLE and self._start_point is not None:
            self._refresh_display()
            preview = self._frame.copy()
            if self._mask is not None:
                preview = create_overlay(
                    preview,
                    self._mask,
                    color=self._overlay_color,
                    alpha=self._overlay_alpha,
                )
            x1, y1 = self._start_point.x(), self._start_point.y()
            x2, y2 = pos
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            image = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888).copy()
            self._pixmap_item.setPixmap(QPixmap.fromImage(image))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if not self._drawing:
            super().mouseReleaseEvent(event)
            return

        pos = self._scene_pos(event)
        if pos and self._tool == ToolMode.RECTANGLE and self._start_point is not None:
            x1, y1 = self._start_point.x(), self._start_point.y()
            x2, y2 = pos
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            self._mask[top:bottom, left:right] = 255
            self.mask_changed.emit()

        self._drawing = False
        self._start_point = None
        self._last_point = None
        self._refresh_display()

    def _paint_stroke(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        value: int,
    ) -> None:
        if self._mask is None:
            return
        cv2.line(
            self._mask,
            start,
            end,
            int(value),
            thickness=self._brush_size * 2,
        )
        self._refresh_display()
