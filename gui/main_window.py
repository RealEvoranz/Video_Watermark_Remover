"""Main application window."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QUrl, Signal, Qt
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStackedLayout,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gui.mask_editor import MaskEditor
from gui.processing_dialog import ProcessingDialog
from gui.settings_dialog import SettingsDialog
from gui.video_widget import ToolMode, VideoWidget
from processing.config_loader import get_output_dir, get_masks_dir, load_config
from processing.mask_utils import load_mask, save_mask
from processing.pipeline import PipelineConfig, ProcessingPipeline
from processing.video_reader import VideoReader


class ProcessingWorker(QObject):
    """Background worker for video processing."""

    progress = Signal(object)
    finished = Signal(object)

    def __init__(
        self,
        input_path: Path,
        mask: np.ndarray,
        output_path: Path,
        pipeline_config: PipelineConfig,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.mask = mask
        self.output_path = output_path
        self.pipeline_config = pipeline_config
        self._pipeline: ProcessingPipeline | None = None

    def run(self) -> None:
        self._pipeline = ProcessingPipeline(
            config=self.pipeline_config,
            progress_callback=lambda p: self.progress.emit(p),
        )
        result = self._pipeline.process(
            self.input_path,
            self.mask,
            self.output_path,
        )
        self.finished.emit(result)

    def cancel(self) -> None:
        if self._pipeline:
            self._pipeline.cancel()


class OpenHintLabel(QLabel):
    clicked = Signal()
    fileDropped = Signal(Path)

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "color: gray; font-size: 14px; background: rgba(255,255,255,0.85);"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptDrops(True)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:
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

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]:
                        self.fileDropped.emit(path)
                        event.acceptProposedAction()
                        return
        event.ignore()


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Video Watermark Remover")
        self.resize(1280, 800)

        config = load_config()
        gui_cfg = config.get("gui", {})
        proc_cfg = config.get("processing", {})

        self._video_path: Path | None = None
        self._reader: VideoReader | None = None
        self._current_frame_index = 0
        self._brush_size = int(gui_cfg.get("default_brush_size", 20))
        self._overlay_alpha = float(gui_cfg.get("mask_overlay_alpha", 0.45))
        self._overlay_color = tuple(gui_cfg.get("mask_overlay_color", [255, 0, 0]))
        self._chunk_overlap = int(proc_cfg.get("chunk_overlap", 5))
        self._output_crf = int(proc_cfg.get("output_crf", 18))
        self._default_backend = proc_cfg.get("default_backend", "e2fgvi")
        if self._default_backend == "propainter":
            self._default_backend = "e2fgvi"
        self._default_chunk_size = proc_cfg.get("default_chunk_size", "auto")
        self._skip_start_seconds = int(proc_cfg.get("skip_start_seconds", 0))

        self._preserve_audio = True
        self._reencode = True
        self._output_path: Path | None = None

        self._mask_editor = MaskEditor()
        self._worker_thread: QThread | None = None
        self._worker: ProcessingWorker | None = None
        self._processing_dialog: ProcessingDialog | None = None

        self._build_toolbar()
        self._build_ui()
        self._connect_signals()
        self.setAcceptDrops(True)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        self.open_action = QAction("Open Video", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.save_mask_action = QAction("Save Mask", self)
        self.load_mask_action = QAction("Load Mask", self)
        self.settings_action = QAction("Settings", self)

        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_mask_action)
        toolbar.addAction(self.load_mask_action)
        toolbar.addSeparator()
        toolbar.addAction(self.settings_action)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # Left panel - tools
        left_panel = QGroupBox("Tools")
        left_layout = QVBoxLayout(left_panel)

        self.rect_button = QPushButton("Rectangle")
        self.brush_button = QPushButton("Brush")
        self.eraser_button = QPushButton("Eraser")
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.clear_mask_button = QPushButton("Clear Mask")
        self.reset_view_button = QPushButton("Reset View")

        for button in (self.rect_button, self.brush_button, self.eraser_button):
            button.setCheckable(True)

        left_layout.addWidget(self.rect_button)
        left_layout.addWidget(self.brush_button)
        left_layout.addWidget(self.eraser_button)
        left_layout.addWidget(QLabel("Brush Size"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 100)
        self.brush_slider.setValue(self._brush_size)
        left_layout.addWidget(self.brush_slider)
        self.brush_label = QLabel(str(self._brush_size))
        left_layout.addWidget(self.brush_label)
        left_layout.addWidget(self.undo_button)
        left_layout.addWidget(self.redo_button)
        left_layout.addWidget(self.clear_mask_button)
        left_layout.addWidget(self.reset_view_button)
        left_layout.addStretch()

        # Center - video viewer
        center_panel = QGroupBox("Video")
        center_layout = QVBoxLayout(center_panel)

        self.open_hint_label = OpenHintLabel(
            "Click To Open a Video or Drag and Drop One Here."
        )

        self.video_widget = VideoWidget()
        self.video_widget.setAcceptDrops(True)
        self.video_widget.set_overlay_style(
            (self._overlay_color[2], self._overlay_color[1], self._overlay_color[0]),
            self._overlay_alpha,
        )

        video_holder = QWidget()
        self.video_holder_layout = QStackedLayout(video_holder)
        self.video_holder_layout.addWidget(self.video_widget)
        self.video_holder_layout.addWidget(self.open_hint_label)
        self.video_holder_layout.setCurrentWidget(self.open_hint_label)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.video_metadata_label = QLabel("Resolution: -- | FPS: -- | Duration: -- | Current: --")

        nav_row = QHBoxLayout()
        self.prev_frame_button = QPushButton("◀")
        self.next_frame_button = QPushButton("▶")
        nav_row.addWidget(self.prev_frame_button)
        nav_row.addWidget(self.frame_slider, stretch=1)
        nav_row.addWidget(self.next_frame_button)

        center_layout.addWidget(video_holder, stretch=1)
        center_layout.addLayout(nav_row)
        center_layout.addWidget(self.frame_label)
        center_layout.addWidget(self.video_metadata_label)

        # Right panel - processing
        right_panel = QGroupBox("Processing")
        right_layout = QVBoxLayout(right_panel)

        form_layout = QFormLayout()

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["E2FGVI", "ProPainter (Coming Soon)", "Passthrough (Test)"])
        backend_map = {"e2fgvi": 0, "propainter": 1, "passthrough": 2}
        self.backend_combo.setCurrentIndex(backend_map.get(self._default_backend, 0))
        if self.backend_combo.model().item(1) is not None:
            self.backend_combo.model().item(1).setEnabled(False)
        self.backend_combo.setItemData(
            1,
            "ProPainter is coming soon in a future release.",
            Qt.ToolTipRole,
        )
        form_layout.addRow("Backend:", self.backend_combo)

        self.backend_info_label = QLabel(self._backend_description(self._default_backend))
        self.backend_info_label.setWordWrap(True)
        self.backend_info_label.setStyleSheet("color: #555;")
        right_layout.addLayout(form_layout)
        right_layout.addWidget(self.backend_info_label)

        form_layout = QFormLayout()
        self.chunk_combo = QComboBox()
        self.chunk_combo.addItems(["Auto", "4", "8", "12", "16", "24", "32", "48", "64"])
        default_chunk_text = str(self._default_chunk_size).capitalize()
        if default_chunk_text not in [self.chunk_combo.itemText(i) for i in range(self.chunk_combo.count())]:
            default_chunk_text = "Auto"
        self.chunk_combo.setCurrentText(default_chunk_text)
        form_layout.addRow("Chunk Size:", self.chunk_combo)

        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 600)
        self.skip_spin.setValue(self._skip_start_seconds)
        self.skip_spin.setSuffix(" s")
        form_layout.addRow("Skip Start:", self.skip_spin)

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(self._output_crf)
        self.crf_spin.setToolTip(
            "Lower CRF gives higher video quality and larger output size. "
            "Higher CRF gives smaller output size with lower quality."
        )
        form_layout.addRow("Output CRF:", self.crf_spin)

        self.crf_help_label = QLabel(
            "Lower CRF = higher quality / larger file. Higher CRF = smaller file / lower quality."
        )
        self.crf_help_label.setWordWrap(True)
        self.crf_help_label.setStyleSheet("color: #777; font-size: 11px;")

        self.preserve_audio_checkbox = QCheckBox("Preserve audio")
        self.preserve_audio_checkbox.setChecked(self._preserve_audio)
        self.reencode_checkbox = QCheckBox("Re-encode output")
        self.reencode_checkbox.setChecked(self._reencode)

        right_layout.addLayout(form_layout)
        right_layout.addWidget(self.crf_help_label)
        right_layout.addWidget(self.preserve_audio_checkbox)
        right_layout.addWidget(self.reencode_checkbox)
        right_layout.addSpacing(10)

        self.output_path_label = QLabel("Output: -")
        self.output_path_label.setWordWrap(True)
        self.open_output_folder_button = QPushButton("Open output folder")
        self.open_output_folder_button.setEnabled(False)

        right_layout.addWidget(self.output_path_label)
        right_layout.addWidget(self.open_output_folder_button)
        right_layout.addSpacing(10)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.cancel_button = QPushButton("Cancel")
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

        self.status_label = QLabel("Open a video to begin.")
        self.coverage_label = QLabel("Mask coverage: 0%")

        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.pause_button)
        right_layout.addWidget(self.cancel_button)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.coverage_label)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 1)

        root_layout.addWidget(splitter)

        self.video_status_label = QLabel("No video loaded. Open or drop one.")
        self.statusBar().addPermanentWidget(self.video_status_label)

    def _update_backend_info(self, _: int | None = None) -> None:
        backend = self._backend_name()
        self.backend_info_label.setText(self._backend_description(backend))

    def _description_for_backend(self, backend: str) -> str:
        descriptions = {
            "e2fgvi": "E2FGVI: best for high-quality watermark removal on general video. Use this backend for most edits.",
            "propainter": "ProPainter is coming soon in a future release.",
            "passthrough": "Passthrough: test mode only — it copies video without processing and is useful for debugging.",
        }
        return descriptions.get(backend, "Select a backend for processing.")

    def _backend_description(self, backend: str) -> str:
        return self._description_for_backend(backend)

    def _open_output_folder(self) -> None:
        if not self._output_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_path.parent)))

    def _update_output_path_label(self) -> None:
        if self._output_path:
            self.output_path_label.setText(f"Output: {self._output_path}")
            self.open_output_folder_button.setEnabled(True)
        else:
            self.output_path_label.setText("Output: -")
            self.open_output_folder_button.setEnabled(False)

    def _update_tool_buttons(self) -> None:
        active = self.video_widget._tool
        self.rect_button.setChecked(active == ToolMode.RECTANGLE)
        self.brush_button.setChecked(active == ToolMode.BRUSH)
        self.eraser_button.setChecked(active == ToolMode.ERASER)

    def _update_undo_redo_buttons(self) -> None:
        self.undo_button.setEnabled(self._mask_editor.can_undo)
        self.redo_button.setEnabled(self._mask_editor.can_redo)

    def _connect_signals(self) -> None:
        self.open_action.triggered.connect(self._open_video)
        self.save_mask_action.triggered.connect(self._save_mask)
        self.load_mask_action.triggered.connect(self._load_mask)
        self.settings_action.triggered.connect(self._open_settings)

        self.rect_button.clicked.connect(lambda: self._set_tool(ToolMode.RECTANGLE))
        self.brush_button.clicked.connect(lambda: self._set_tool(ToolMode.BRUSH))
        self.eraser_button.clicked.connect(lambda: self._set_tool(ToolMode.ERASER))
        self.undo_button.clicked.connect(self._undo_mask)
        self.redo_button.clicked.connect(self._redo_mask)
        self.clear_mask_button.clicked.connect(self._clear_mask)
        self.reset_view_button.clicked.connect(self.video_widget.reset_view)

        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.prev_frame_button.clicked.connect(self._prev_frame)
        self.next_frame_button.clicked.connect(self._next_frame)

        self.start_button.clicked.connect(self._start_processing)
        self.cancel_button.clicked.connect(self._cancel_processing)

        self.video_widget.mask_edit_began.connect(self._on_mask_edit_began)
        self.video_widget.mask_changed.connect(self._on_mask_edited)
        self.open_hint_label.clicked.connect(self._open_video)
        self.open_hint_label.fileDropped.connect(self._open_video_path)

        self.backend_combo.currentIndexChanged.connect(self._update_backend_info)
        self.open_output_folder_button.clicked.connect(self._open_output_folder)

    def _set_tool(self, tool: ToolMode) -> None:
        self.video_widget.set_tool(tool)
        self._update_tool_buttons()

    def _on_brush_size_changed(self, value: int) -> None:
        self._brush_size = value
        self.brush_label.setText(str(value))
        self.video_widget.set_brush_size(value)

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi);;All Files (*)",
        )
        if not path:
            return

        try:
            self._open_video_path(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open video:\n{exc}")

    def _on_hint_clicked(self, event) -> None:
        self._open_video()

    def _show_frame(self, index: int) -> None:
        if not self._reader:
            return

        self._reader.seek(index)
        success, frame = self._reader.read()
        if success and frame is not None:
            self._current_frame_index = index
            self.video_widget.set_frame(frame)
            self.frame_label.setText(
                f"Frame: {index + 1} / {self._reader.metadata.frame_count}"
            )
            self._update_video_metadata()

    def _on_frame_changed(self, index: int) -> None:
        if index != self._current_frame_index:
            self._show_frame(index)

    def _update_video_metadata(self) -> None:
        if not self._reader:
            self.video_metadata_label.setText(
                "Resolution: -- | FPS: -- | Duration: -- | Current: --"
            )
            self.video_status_label.setText("No video loaded. Open or drop one.")
            return

        meta = self._reader.metadata
        duration_text = self._format_timestamp(meta.duration_seconds)
        current_text = self._format_timestamp(self._current_frame_index / max(1.0, meta.fps))
        self.video_metadata_label.setText(
            f"Resolution: {meta.width}x{meta.height} | FPS: {meta.fps:.2f} | Duration: {duration_text} | Current: {current_text}"
        )
        self.video_status_label.setText(
            f"Loaded: {self._video_path.name} | {meta.width}x{meta.height} @ {meta.fps:.2f}"
        )

    def _format_timestamp(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    def _prev_frame(self) -> None:
        if self.frame_slider.value() > 0:
            self.frame_slider.setValue(self.frame_slider.value() - 1)

    def _next_frame(self) -> None:
        if self.frame_slider.value() < self.frame_slider.maximum():
            self.frame_slider.setValue(self.frame_slider.value() + 1)

    def _on_mask_edit_began(self) -> None:
        self._mask_editor.snapshot()

    def _on_mask_edited(self) -> None:
        mask = self.video_widget.get_mask()
        if mask is not None:
            self._mask_editor.update_mask(mask)
        self._update_coverage()
        self._update_undo_redo_buttons()

    def _undo_mask(self) -> None:
        mask = self._mask_editor.undo()
        if mask is not None:
            self.video_widget.set_mask(mask)
            self._update_coverage()
        self._update_undo_redo_buttons()

    def _redo_mask(self) -> None:
        mask = self._mask_editor.redo()
        if mask is not None:
            self.video_widget.set_mask(mask)
            self._update_coverage()
        self._update_undo_redo_buttons()

    def _clear_mask(self) -> None:
        self._mask_editor.clear()
        if self._mask_editor.mask is not None:
            self.video_widget.set_mask(self._mask_editor.mask)
            self._update_coverage()
        self._update_undo_redo_buttons()

    def _save_mask(self) -> None:
        if self._mask_editor.mask is None:
            QMessageBox.warning(self, "No Mask", "Open a video and create a mask first.")
            return

        masks_dir = get_masks_dir()
        default_name = f"{self._video_path.stem}_mask.png" if self._video_path else "mask.png"
        default_path = str(masks_dir / default_name)
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask",
            default_path,
            "PNG Images (*.png)",
        )
        if path:
            save_mask(self._mask_editor.mask, path)
            self.status_label.setText(f"Mask saved: {Path(path).name}")

    def _load_mask(self) -> None:
        if not self._reader:
            QMessageBox.warning(self, "No Video", "Open a video first.")
            return

        masks_dir = str(get_masks_dir())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Mask",
            masks_dir,
            "PNG Images (*.png)",
        )
        if not path:
            return

        try:
            meta = self._reader.metadata
            mask = load_mask(path, (meta.width, meta.height))
            self._mask_editor.load(mask)
            self.video_widget.set_mask(mask)
            self._update_coverage()
            self._update_undo_redo_buttons()
            self.status_label.setText(f"Mask loaded: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load mask:\n{exc}")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec():
            self._brush_size = dialog.brush_size
            self._overlay_alpha = dialog.overlay_alpha
            self._output_crf = dialog.output_crf
            self._chunk_overlap = dialog.chunk_overlap
            self._default_backend = dialog.default_backend
            self._default_chunk_size = dialog.default_chunk_size
            self._skip_start_seconds = dialog.skip_start_seconds

            self.brush_slider.setValue(self._brush_size)
            backend_map = {"e2fgvi": 0, "propainter": 1, "passthrough": 2}
            self.backend_combo.setCurrentIndex(
                backend_map.get(self._default_backend, 0)
            )
            self.chunk_combo.setCurrentText(str(self._default_chunk_size).capitalize())
            self.video_widget.set_overlay_style(
                (self._overlay_color[2], self._overlay_color[1], self._overlay_color[0]),
                self._overlay_alpha,
            )

    def _update_coverage(self) -> None:
        from processing.mask_utils import mask_coverage_percent

        if self._mask_editor.mask is None:
            self.coverage_label.setText("Mask coverage: 0%")
            return
        pct = mask_coverage_percent(self._mask_editor.mask)
        self.coverage_label.setText(f"Mask coverage: {pct:.1f}%")

    def _backend_name(self) -> str:
        mapping = {0: "e2fgvi", 1: "propainter", 2: "passthrough"}
        return mapping[self.backend_combo.currentIndex()]

    def _chunk_size(self) -> int | str:
        text = self.chunk_combo.currentText()
        if text == "Auto":
            return "auto"
        return int(text)

    def _start_processing(self) -> None:
        if not self._video_path or self._mask_editor.mask is None:
            QMessageBox.warning(
                self,
                "Not Ready",
                "Open a video and draw a watermark mask before processing.",
            )
            return

        if np.count_nonzero(self._mask_editor.mask) == 0:
            QMessageBox.warning(
                self,
                "Empty Mask",
                "The mask is empty. Draw a region to remove first.",
            )
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Video",
            str(get_output_dir() / f"{self._video_path.stem}_clean.mp4"),
            "MP4 Video (*.mp4)",
        )
        if not output_path:
            return

        self._output_path = Path(output_path)
        self._update_output_path_label()
        self._skip_start_seconds = self.skip_spin.value()
        self._preserve_audio = self.preserve_audio_checkbox.isChecked()
        self._reencode = self.reencode_checkbox.isChecked()
        self._output_crf = self.crf_spin.value()

        pipeline_config = PipelineConfig(
            backend_name=self._backend_name(),
            chunk_size=self._chunk_size(),
            chunk_overlap=self._chunk_overlap,
            preserve_audio=self._preserve_audio,
            reencode=self._reencode,
            output_crf=self._output_crf,
            skip_start_seconds=self._skip_start_seconds,
        )

        self._processing_dialog = ProcessingDialog(self)
        self.cancel_button.setEnabled(True)
        self.start_button.setEnabled(False)

        self._worker = ProcessingWorker(
            self._video_path,
            self._mask_editor.mask.copy(),
            self._output_path,
            pipeline_config,
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_processing_progress)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self._processing_dialog.cancel_button.clicked.connect(self._cancel_processing)
        self._worker_thread.start()
        self._processing_dialog.show()

    def _on_processing_progress(self, progress) -> None:
        if self._processing_dialog:
            self._processing_dialog.update_progress(progress)

    def _on_processing_finished(self, progress) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        if self._processing_dialog:
            self._processing_dialog.update_progress(progress)
            if progress.finished or progress.error:
                self._processing_dialog.close()

        if progress.finished and not progress.error:
            self.status_label.setText(progress.message)
            self._update_output_path_label()
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Complete")
            dialog.setText(progress.message)
            open_button = dialog.addButton(
                "Open output folder",
                QMessageBox.ButtonRole.ActionRole,
            )
            dialog.addButton(QMessageBox.StandardButton.Ok)
            dialog.exec()
            if dialog.clickedButton() == open_button:
                self._open_output_folder()
        elif progress.error:
            QMessageBox.critical(self, "Processing Error", progress.error)

    def _cancel_processing(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._processing_dialog and not self._processing_dialog.is_cancelled:
            self._processing_dialog._on_cancel()  # noqa: SLF001

    def dragEnterEvent(self, event) -> None:
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

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi"]:
                        self._open_video_path(path)
                        event.acceptProposedAction()
                        return
        event.ignore()

    def _open_video_path(self, path: Path) -> None:
        try:
            if self._reader:
                self._reader.close()

            self._reader = VideoReader(path)
            self._video_path = path
            meta = self._reader.metadata

            self._mask_editor.resize(meta.width, meta.height)
            self.video_widget.set_mask(self._mask_editor.mask)
            self.video_widget.set_brush_size(self._brush_size)
            self._set_tool(ToolMode.RECTANGLE)

            self.frame_slider.setRange(0, max(0, meta.frame_count - 1))
            self._current_frame_index = 0
            self._show_frame(0)
            self.open_hint_label.hide()
            self.video_holder_layout.setCurrentWidget(self.video_widget)

            self.status_label.setText(f"Loaded: {self._video_path.name}")
            self._update_coverage()
            self._update_undo_redo_buttons()
            self._update_video_metadata()
            self.video_widget.reset_view()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open video:\n{exc}")

    def closeEvent(self, event) -> None:
        if self._reader:
            self._reader.close()
        if self._worker_thread and self._worker_thread.isRunning():
            self._cancel_processing()
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        super().closeEvent(event)
