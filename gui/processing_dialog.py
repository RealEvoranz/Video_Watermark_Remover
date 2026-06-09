"""Processing progress dialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from processing.chunk_processor import ProcessingProgress


class ProcessingDialog(QDialog):
    """Non-modal dialog showing processing progress and resource usage."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Processing Video")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.resize(560, 320)

        self._cancelled = False
        self._last_message = ""

        self.status_label = QLabel("Preparing...")
        self.chunk_label = QLabel("Chunk: 0 / 0")
        self.fps_label = QLabel("FPS: 0.0")
        self.vram_label = QLabel("VRAM: --")
        self.ram_label = QLabel("RAM: --")
        self.eta_label = QLabel("ETA: --")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Processing log...")
        self.log_area.setMaximumBlockCount(500)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel)

        stats_row = QHBoxLayout()
        stats_row.addWidget(self.fps_label)
        stats_row.addWidget(self.vram_label)
        stats_row.addWidget(self.ram_label)
        stats_row.addWidget(self.eta_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.chunk_label)
        layout.addLayout(stats_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_area)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

    @property
    def is_cancelled(self) -> bool:
        """Return whether user requested cancellation."""
        return self._cancelled

    def _on_cancel(self) -> None:
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")

    def update_progress(self, progress: ProcessingProgress) -> None:
        """Update UI from a progress snapshot."""
        self.status_label.setText(progress.message)
        self.chunk_label.setText(
            f"Chunk: {progress.current_chunk} / {progress.total_chunks}"
        )
        self.fps_label.setText(f"FPS: {progress.fps:.1f}")
        self.vram_label.setText(
            f"VRAM: {progress.vram_mb:.0f} MB"
            if progress.vram_mb is not None
            else "VRAM: --"
        )
        self.ram_label.setText(f"RAM: {progress.ram_mb:.0f} MB")
        if progress.eta_seconds is not None:
            minutes, seconds = divmod(int(progress.eta_seconds), 60)
            self.eta_label.setText(f"ETA: {minutes:02d}:{seconds:02d}")
        else:
            self.eta_label.setText("ETA: --")

        self.progress_bar.setValue(int(progress.percent))

        if progress.message and progress.message != self._last_message:
            self.log_area.appendPlainText(progress.message)
            self._last_message = progress.message

        if progress.finished:
            self.cancel_button.setText("Close")
            self.cancel_button.setEnabled(True)
            try:
                self.cancel_button.clicked.disconnect()
            except Exception:
                pass
            self.cancel_button.clicked.connect(self.accept)
        elif progress.error:
            self.status_label.setText(f"Error: {progress.error}")
            self.cancel_button.setText("Close")
            try:
                self.cancel_button.clicked.disconnect()
            except Exception:
                pass
            self.cancel_button.clicked.connect(self.reject)

    def pump_events(self) -> None:
        """Process pending Qt events (call from worker thread via signal)."""
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
