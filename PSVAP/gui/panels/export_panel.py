"""
gui/panels/export_panel.py
---------------------------
Phase 7: Export Panel — PNG frame export, data export (CSV),
and MP4 trajectory video (requires imageio + ffmpeg).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget, QFileDialog,
    QCheckBox,
)

from PSVAP.app.controller import ApplicationController

BG        = "#111111"
PANEL_ALT = "#1A1A1A"
BORDER    = "#2A2A2A"
TEXT      = "#CCCCCC"
TEXT_DIM  = "#888888"
TEXT_HINT = "#555555"
ACCENT    = "#E8FF00"
MONO      = "Courier New, monospace"


def _lbl(text: str, dim: bool = False, hint: bool = False) -> QLabel:
    lbl = QLabel(text)
    color = TEXT_HINT if hint else (TEXT_DIM if dim else TEXT)
    lbl.setStyleSheet(
        f"color:{color}; font-size:9px; letter-spacing:2px; background:transparent;")
    lbl.setWordWrap(True)
    return lbl


def _divider() -> QFrame:
    f = QFrame(); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER}; margin:0;"); return f


def _result_box(height: int = 80) -> QTextEdit:
    tb = QTextEdit(); tb.setReadOnly(True); tb.setFixedHeight(height)
    tb.setStyleSheet(
        f"QTextEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; font-family:{MONO}; font-size:10px; padding:6px; }}")
    return tb


def _input_line(placeholder: str = "") -> QLineEdit:
    le = QLineEdit(); le.setPlaceholderText(placeholder)
    le.setStyleSheet(
        f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; padding:6px 10px; font-size:11px; }}"
        f"QLineEdit:focus {{ border-color:{TEXT_DIM}; }}")
    return le


def _btn(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
        f"border:1px solid {BORDER}; padding:7px 16px; font-size:9px; letter-spacing:2px; }}"
        f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}"
        f"QPushButton:pressed {{ background:{ACCENT}; color:#0A0A0A; border-color:{ACCENT}; }}")
    return b


class ExportPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self._build()

    def _get_atoms(self) -> list:
        return getattr(self.controller.model, 'atoms', [])

    def _get_trajectory(self) -> list:
        return getattr(self.controller.model, 'trajectory', [])

    def _get_positions(self) -> "np.ndarray | None":
        try:
            m = self.controller.model
            f = m.get_frame(getattr(m, '_current_frame', 0))
            return np.asarray(f, dtype=np.float64) if f is not None else None
        except Exception:
            return None

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        tabs = QTabWidget(); tabs.setDocumentMode(True)
        tabs.setStyleSheet(
            f"QTabWidget::pane {{ border:none; border-top:1px solid {BORDER}; background:{BG}; }}"
            f"QTabBar {{ background:{BG}; }}"
            f"QTabBar::tab {{ background:{BG}; color:{TEXT_HINT}; font-size:8px; "
            f"letter-spacing:2px; padding:8px 12px; border:none; "
            f"border-bottom:2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; "
            f"background:{PANEL_ALT}; }}"
            f"QTabBar::tab:hover:!selected {{ color:{TEXT_DIM}; background:{PANEL_ALT}; }}")
        tabs.addTab(self._build_image_tab(),  "IMAGE")
        tabs.addTab(self._build_data_tab(),   "DATA")
        tabs.addTab(self._build_video_tab(),  "VIDEO")
        layout.addWidget(tabs)

    def _build_image_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("EXPORT CURRENT FRAME AS IMAGE"))
        layout.addWidget(_divider())

        sg = QGridLayout(); sg.setSpacing(8)
        sg.addWidget(_lbl("WIDTH (px):", dim=True), 0, 0)
        self._img_w = QSpinBox()
        self._img_w.setRange(400, 4096); self._img_w.setValue(1920)
        self._img_w.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._img_w.setFixedWidth(90); sg.addWidget(self._img_w, 0, 1)

        sg.addWidget(_lbl("HEIGHT (px):", dim=True), 1, 0)
        self._img_h = QSpinBox()
        self._img_h.setRange(300, 4096); self._img_h.setValue(1080)
        self._img_h.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._img_h.setFixedWidth(90); sg.addWidget(self._img_h, 1, 1)
        layout.addLayout(sg)

        self._img_btn = _btn("SAVE SCREENSHOT")
        layout.addWidget(self._img_btn)
        self._img_result = _result_box(60)
        layout.addWidget(self._img_result)
        self._img_btn.clicked.connect(self._run_screenshot)

        layout.addStretch()
        w.setWidget(inner); return w

    def _build_data_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("EXPORT ATOM DATA AS CSV"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("EXPORT CURRENT FRAME COORDINATES", dim=True))
        self._csv_btn = _btn("SAVE COORDINATES CSV")
        layout.addWidget(self._csv_btn)
        self._csv_result = _result_box(60)
        layout.addWidget(self._csv_result)
        self._csv_btn.clicked.connect(self._run_export_csv)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("EXPORT STRUCTURE AS PDB", dim=True))
        self._pdb_btn = _btn("SAVE AS PDB")
        layout.addWidget(self._pdb_btn)
        self._pdb_result = _result_box(60)
        layout.addWidget(self._pdb_result)
        self._pdb_btn.clicked.connect(self._run_export_pdb)

        layout.addStretch()
        w.setWidget(inner); return w

    def _build_video_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("EXPORT TRAJECTORY AS VIDEO"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Requires imageio and ffmpeg. Install with:\n"
            "pip install imageio imageio-ffmpeg",
            hint=True))

        vg = QGridLayout(); vg.setSpacing(8)
        vg.addWidget(_lbl("FPS:", dim=True), 0, 0)
        self._vid_fps = QSpinBox()
        self._vid_fps.setRange(1, 60); self._vid_fps.setValue(15)
        self._vid_fps.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._vid_fps.setFixedWidth(70); vg.addWidget(self._vid_fps, 0, 1)

        vg.addWidget(_lbl("WIDTH (px):", dim=True), 1, 0)
        self._vid_w = QSpinBox()
        self._vid_w.setRange(400, 1920); self._vid_w.setValue(1280)
        self._vid_w.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._vid_w.setFixedWidth(90); vg.addWidget(self._vid_w, 1, 1)
        layout.addLayout(vg)

        self._vid_btn = _btn("EXPORT MP4 VIDEO")
        layout.addWidget(self._vid_btn)
        self._vid_result = _result_box(100)
        layout.addWidget(self._vid_result)
        self._vid_btn.clicked.connect(self._run_export_video)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Slots ──────────────────────────────────────────────────────────────

    @Slot()
    def _run_screenshot(self) -> None:
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Screenshot", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*)")
            if not path:
                return
            engine = self.controller._engine
            if engine and engine._plotter:
                engine._plotter.screenshot(
                    path,
                    window_size=[self._img_w.value(), self._img_h.value()],
                )
                self._img_result.setText(f"SAVED\n  {path}")
            else:
                self._img_result.setText("No viewport active.")
        except Exception as e:
            self._img_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_export_csv(self) -> None:
        try:
            import csv
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._csv_result.setText("NO DATA LOADED"); return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)")
            if not path:
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "element", "resname", "residue_id",
                                 "chain_id", "name", "x", "y", "z"])
                for i, atom in enumerate(atoms):
                    if i >= len(pos):
                        break
                    writer.writerow([
                        atom.id,
                        getattr(atom, "element",    None) or "",
                        getattr(atom, "resname",    None) or "",
                        getattr(atom, "residue_id", None) or "",
                        getattr(atom, "chain_id",   None) or "",
                        getattr(atom, "name",       None) or "",
                        f"{pos[i,0]:.4f}",
                        f"{pos[i,1]:.4f}",
                        f"{pos[i,2]:.4f}",
                    ])
            self._csv_result.setText(
                f"SAVED\n  {path}\n  {len(atoms):,} atoms")
        except Exception as e:
            self._csv_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_export_pdb(self) -> None:
        try:
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._pdb_result.setText("NO DATA LOADED"); return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save PDB", "", "PDB Files (*.pdb);;All Files (*)")
            if not path:
                return
            from PSVAP.modeling.mutation_engine import write_pdb
            write_pdb(atoms, pos, path)
            self._pdb_result.setText(
                f"SAVED\n  {path}\n  {len(atoms):,} atoms")
        except Exception as e:
            self._pdb_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_export_video(self) -> None:
        try:
            import imageio
        except ImportError:
            self._vid_result.setText(
                "imageio not installed.\n"
                "Install: pip install imageio imageio-ffmpeg")
            return
        try:
            traj   = self._get_trajectory()
            engine = self.controller._engine
            if not traj or engine is None or engine._plotter is None:
                self._vid_result.setText("No trajectory or viewport."); return

            path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "", "MP4 Video (*.mp4);;All Files (*)")
            if not path:
                return

            fps = self._vid_fps.value()
            w   = self._vid_w.value()
            h   = int(w * 9 / 16)   # 16:9 aspect

            self._vid_result.setText(
                f"Rendering {len(traj)} frames at {fps} FPS...")

            frames_list = []
            for frame_idx in range(len(traj)):
                self.controller.update_frame(frame_idx)
                img = engine._plotter.screenshot(
                    return_img=True,
                    window_size=[w, h],
                )
                frames_list.append(img)

            writer = imageio.get_writer(path, fps=fps)
            for frame in frames_list:
                writer.append_data(frame)
            writer.close()

            self._vid_result.setText(
                f"VIDEO SAVED\n  {path}\n"
                f"  {len(frames_list)} frames at {fps} FPS")
        except Exception as e:
            self._vid_result.setText(f"ERROR: {e}")