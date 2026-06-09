"""Application settings dialog."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
)

import json
from processing.config_loader import load_config, get_project_root


class SettingsDialog(QDialog):
    """Edit user-facing processing and display settings."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(360, 200)

        config = load_config()
        gui_cfg = config.get("gui", {})
        proc_cfg = config.get("processing", {})

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(int(gui_cfg.get("default_brush_size", 20)))

        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(10, 90)
        self.alpha_spin.setValue(int(float(gui_cfg.get("mask_overlay_alpha", 0.45)) * 100))
        self.alpha_spin.setSuffix(" %")

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(int(proc_cfg.get("output_crf", 18)))

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 30)
        self.overlap_spin.setValue(int(proc_cfg.get("chunk_overlap", 5)))

        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 600)
        self.skip_spin.setValue(int(proc_cfg.get("skip_start_seconds", 0)))

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["e2fgvi", "propainter", "passthrough"])
        default_backend = proc_cfg.get("default_backend", "e2fgvi")
        index = self.backend_combo.findText(default_backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)

        form = QFormLayout(self)
        form.addRow("Default brush size:", self.brush_spin)
        form.addRow("Mask overlay opacity:", self.alpha_spin)
        form.addRow("Output quality (CRF):", self.crf_spin)
        form.addRow("Chunk overlap:", self.overlap_spin)
        form.addRow("Skip start (seconds):", self.skip_spin)
        form.addRow("Default backend:", self.backend_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @property
    def brush_size(self) -> int:
        return self.brush_spin.value()

    @property
    def overlay_alpha(self) -> float:
        return self.alpha_spin.value() / 100.0

    @property
    def output_crf(self) -> int:
        return self.crf_spin.value()

    @property
    def chunk_overlap(self) -> int:
        return self.overlap_spin.value()

    @property
    def skip_start_seconds(self) -> int:
        return self.skip_spin.value()

    @property
    def default_backend(self) -> str:
        return self.backend_combo.currentText()

    def _on_accept(self) -> None:
        """Persist updated settings to config.json then accept the dialog."""
        cfg = load_config()
        # Update gui settings
        gui_cfg = cfg.get("gui", {})
        gui_cfg["default_brush_size"] = int(self.brush_size)
        gui_cfg["mask_overlay_alpha"] = float(self.overlay_alpha)
        cfg["gui"] = gui_cfg

        # Update processing settings
        proc_cfg = cfg.get("processing", {})
        proc_cfg["output_crf"] = int(self.output_crf)
        proc_cfg["chunk_overlap"] = int(self.chunk_overlap)
        proc_cfg["default_backend"] = str(self.default_backend)
        proc_cfg["skip_start_seconds"] = int(self.skip_start_seconds)
        cfg["processing"] = proc_cfg

        path = get_project_root() / "config.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2)

        self.accept()
