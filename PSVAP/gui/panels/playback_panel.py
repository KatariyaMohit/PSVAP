"""
gui/panels/playback_panel.py
----------------------------
Minimalist playback control bar.
Design: flat, monochrome, no chrome — just controls.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QWidget,
)

from PSVAP.app.controller import ApplicationController

BG       = "#111111"
TEXT     = "#F0F0F0"
TEXT_DIM = "#555555"
BORDER   = "#2A2A2A"
ACCENT   = "#E8FF00"


class PlaybackPanel(QWidget):

    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._playing = False
        self._total = 0
        self._build()
        self._connect()

    def _build(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Play/Pause button
        self._play_btn = QPushButton("▶  PLAY")
        self._play_btn.setFixedWidth(90)
        self._play_btn.setFixedHeight(30)
        self._play_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_DIM}; "
            f"border: 1px solid {BORDER}; font-size: 9px; letter-spacing: 2px; }}"
            f"QPushButton:hover {{ color: {TEXT}; border-color: {TEXT_DIM}; }}"
            f"QPushButton:pressed {{ background: {ACCENT}; color: #0A0A0A; border-color: {ACCENT}; }}"
        )
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet(f"color: {BORDER}; background: transparent;")
        layout.addWidget(sep)

        # Frame slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, stretch=1)

        # Frame counter
        self._counter = QLabel("0 / 0")
        self._counter.setFixedWidth(80)
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._counter.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; "
            f"letter-spacing: 2px; background: transparent;"
        )
        layout.addWidget(self._counter)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet(f"color: {BORDER}; background: transparent;")
        layout.addWidget(sep2)

        # FPS
        fps_label = QLabel("FPS")
        fps_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; "
            f"letter-spacing: 2px; background: transparent;"
        )
        layout.addWidget(fps_label)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 120)
        self._fps_spin.setValue(15)
        self._fps_spin.setFixedWidth(56)
        self._fps_spin.setFixedHeight(28)
        self._fps_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self._fps_spin)

    def _connect(self) -> None:
        m = self.controller.model
        m.data_loaded.connect(self._on_data_loaded)
        m.frame_changed.connect(self._on_frame_changed)

    # ------------------------------------------------------------------ #

    @Slot()
    def _on_data_loaded(self) -> None:
        self._total = self.controller.model.n_frames()
        self._slider.setMaximum(max(0, self._total - 1))
        self._slider.setValue(0)
        self._counter.setText(f"1 / {self._total}")
        if self._playing:
            self._stop()

    @Slot(int)
    def _on_frame_changed(self, n: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(n)
        self._slider.blockSignals(False)
        self._counter.setText(f"{n + 1} / {max(self._total, 1)}")

    @Slot(int)
    def _on_slider(self, value: int) -> None:
        self.controller.update_frame(value)

    @Slot(int)
    def _on_fps_changed(self, fps: int) -> None:
        if self._playing:
            self._timer.setInterval(max(1, 1000 // fps))

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop()
        else:
            self._play()

    def _play(self) -> None:
        if self._total < 2:
            return
        self._playing = True
        self._play_btn.setText("⏸  PAUSE")
        self._play_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {ACCENT}; "
            f"border: 1px solid {ACCENT}; font-size: 9px; letter-spacing: 2px; }}"
            f"QPushButton:hover {{ background: {ACCENT}20; }}"
        )
        fps = self._fps_spin.value()
        self._timer.start(max(1, 1000 // fps))

    def _stop(self) -> None:
        self._playing = False
        self._timer.stop()
        self._play_btn.setText("▶  PLAY")
        self._play_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: #555555; "
            f"border: 1px solid #2A2A2A; font-size: 9px; letter-spacing: 2px; }}"
            f"QPushButton:hover {{ color: {TEXT}; border-color: #555555; }}"
        )

    def _advance_frame(self) -> None:
        if self._total < 2:
            self._stop()
            return
        cur = self._slider.value()
        nxt = (cur + 1) % self._total
        self.controller.update_frame(nxt)