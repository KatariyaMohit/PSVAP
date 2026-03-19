"""
gui/dialogs/settings_dialog.py
--------------------------------
Phase 7: Application settings dialog.

Sections:
  DISPLAY   — background color, atom size, bond width
  RENDERING — render mode default, FPS target
  ANALYSIS  — H-bond cutoff, clash overlap threshold
  PATHS     — default output directory, external tool paths
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget, QComboBox, QFileDialog,
)

BG        = "#111111"
PANEL     = "#1A1A1A"
BORDER    = "#2A2A2A"
TEXT      = "#CCCCCC"
TEXT_DIM  = "#888888"
ACCENT    = "#E8FF00"


def _lbl(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; background:transparent;")
    return lbl


def _divider() -> QFrame:
    f = QFrame(); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER};"); return f


class SettingsDialog(QDialog):
    """
    Application settings dialog — display preferences, analysis cutoffs,
    and external tool paths.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PSVAP SETTINGS")
        self.setMinimumSize(520, 480)
        self.setStyleSheet(f"""
            QDialog {{ background:{BG}; color:{TEXT}; }}
            QTabWidget::pane {{ border:none; border-top:1px solid {BORDER};
                                background:{PANEL}; }}
            QTabBar {{ background:{BG}; }}
            QTabBar::tab {{ background:{BG}; color:{TEXT_DIM}; font-size:9px;
                            letter-spacing:2px; padding:8px 14px; border:none;
                            border-bottom:2px solid transparent; }}
            QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT};
                                     background:{PANEL}; }}
            QLineEdit {{ background:{PANEL}; border:1px solid {BORDER};
                        color:{TEXT}; padding:5px 8px; font-size:11px; }}
            QSpinBox, QDoubleSpinBox {{ background:{PANEL}; border:1px solid {BORDER};
                                        color:{TEXT}; padding:4px 8px; }}
            QComboBox {{ background:{PANEL}; border:1px solid {BORDER};
                        color:{TEXT}; padding:4px 8px; }}
            QPushButton {{ background:transparent; color:{TEXT_DIM};
                          border:1px solid {BORDER}; padding:6px 14px;
                          font-size:9px; letter-spacing:2px; }}
            QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}
            QPushButton:pressed {{ background:{ACCENT}; color:#0A0A0A;
                                   border-color:{ACCENT}; }}
            QLabel {{ background:transparent; color:{TEXT}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget(); tabs.setDocumentMode(True)
        tabs.addTab(self._build_display_tab(),   "DISPLAY")
        tabs.addTab(self._build_analysis_tab(),  "ANALYSIS")
        tabs.addTab(self._build_paths_tab(),     "PATHS")
        layout.addWidget(tabs, stretch=1)

        # Button box
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        btn_box.setStyleSheet(f"""
            QDialogButtonBox {{ background:{BG}; border-top:1px solid {BORDER};
                                padding:8px 16px; }}
        """)
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        btn_box.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(self._restore_defaults)
        layout.addWidget(btn_box)

    # ── Tab builders ───────────────────────────────────────────────────────

    def _build_display_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background:{PANEL};")
        form = QFormLayout(w)
        form.setContentsMargins(20, 16, 20, 16)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._bg_color = QComboBox()
        self._bg_color.addItems(["Black (#0A0A0A)", "Dark (#1A1A1A)",
                                  "White (#FFFFFF)", "Grey (#888888)"])
        form.addRow(_lbl("BACKGROUND COLOR:"), self._bg_color)

        self._atom_size = QDoubleSpinBox()
        self._atom_size.setRange(1.0, 30.0)
        self._atom_size.setValue(8.0)
        self._atom_size.setSingleStep(0.5)
        form.addRow(_lbl("ATOM POINT SIZE:"), self._atom_size)

        self._bond_width = QDoubleSpinBox()
        self._bond_width.setRange(0.5, 10.0)
        self._bond_width.setValue(1.5)
        self._bond_width.setSingleStep(0.5)
        form.addRow(_lbl("BOND LINE WIDTH:"), self._bond_width)

        self._render_mode = QComboBox()
        self._render_mode.addItems(["ATOMS + BONDS (default)", "ATOMS ONLY"])
        form.addRow(_lbl("DEFAULT RENDER MODE:"), self._render_mode)

        self._fps_target = QSpinBox()
        self._fps_target.setRange(5, 120)
        self._fps_target.setValue(30)
        form.addRow(_lbl("TARGET FPS:"), self._fps_target)

        return w

    def _build_analysis_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background:{PANEL};")
        form = QFormLayout(w)
        form.setContentsMargins(20, 16, 20, 16)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._hbond_dist = QDoubleSpinBox()
        self._hbond_dist.setRange(2.0, 6.0)
        self._hbond_dist.setValue(3.5)
        self._hbond_dist.setSingleStep(0.1)
        self._hbond_dist.setSuffix(" Å")
        form.addRow(_lbl("H-BOND DISTANCE CUTOFF:"), self._hbond_dist)

        self._hbond_angle = QDoubleSpinBox()
        self._hbond_angle.setRange(90.0, 180.0)
        self._hbond_angle.setValue(120.0)
        self._hbond_angle.setSingleStep(5.0)
        self._hbond_angle.setSuffix(" °")
        form.addRow(_lbl("H-BOND ANGLE CUTOFF:"), self._hbond_angle)

        self._clash_overlap = QDoubleSpinBox()
        self._clash_overlap.setRange(0.1, 2.0)
        self._clash_overlap.setValue(0.4)
        self._clash_overlap.setSingleStep(0.1)
        self._clash_overlap.setSuffix(" Å")
        form.addRow(_lbl("CLASH VDW OVERLAP:"), self._clash_overlap)

        self._salt_bridge = QDoubleSpinBox()
        self._salt_bridge.setRange(2.0, 8.0)
        self._salt_bridge.setValue(4.0)
        self._salt_bridge.setSingleStep(0.5)
        self._salt_bridge.setSuffix(" Å")
        form.addRow(_lbl("SALT BRIDGE CUTOFF:"), self._salt_bridge)

        self._water_probe = QDoubleSpinBox()
        self._water_probe.setRange(1.0, 2.0)
        self._water_probe.setValue(1.4)
        self._water_probe.setSingleStep(0.05)
        self._water_probe.setSuffix(" Å")
        form.addRow(_lbl("WATER PROBE RADIUS:"), self._water_probe)

        return w

    def _build_paths_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        def _path_row(label: str, default: str) -> QLineEdit:
            layout.addWidget(_lbl(label))
            row = QHBoxLayout(); row.setSpacing(8)
            le = QLineEdit(default)
            browse = QPushButton("BROWSE")
            browse.setFixedWidth(80)

            def _browse():
                p = QFileDialog.getExistingDirectory(w, f"Select {label}")
                if p:
                    le.setText(p)
            browse.clicked.connect(_browse)
            row.addWidget(le, stretch=1)
            row.addWidget(browse)
            layout.addLayout(row)
            return le

        self._output_dir  = _path_row("DEFAULT OUTPUT DIRECTORY:", "output")
        self._vina_path   = _path_row("AUTODOCK VINA EXECUTABLE:", "vina")
        self._fpocket_path = _path_row("FPOCKET EXECUTABLE:", "fpocket")

        # For executable paths, allow file selection too
        layout.addWidget(_divider())
        layout.addWidget(_lbl("MARTINIZE2 EXECUTABLE:"))
        self._martinize2_path = QLineEdit("martinize2")
        layout.addWidget(self._martinize2_path)

        layout.addWidget(_lbl("PROPKA EXECUTABLE:"))
        self._propka_path = QLineEdit("propka")
        layout.addWidget(self._propka_path)

        layout.addStretch()
        return w

    # ── Slots ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_ok(self) -> None:
        """Apply settings to core/constants.py values and close."""
        try:
            import PSVAP.core.constants as C
            C.HBOND_DISTANCE_CUTOFF = self._hbond_dist.value()
            C.HBOND_ANGLE_CUTOFF    = self._hbond_angle.value()
            C.CLASH_VDW_OVERLAP     = self._clash_overlap.value()
            C.SALT_BRIDGE_CUTOFF    = self._salt_bridge.value()
            C.WATER_PROBE_RADIUS    = self._water_probe.value()
            C.DEFAULT_ATOM_POINT_SIZE = self._atom_size.value()
            C.DEFAULT_BOND_RADIUS     = self._bond_width.value()
        except Exception:
            pass
        self.accept()

    @Slot()
    def _restore_defaults(self) -> None:
        """Reset all fields to master plan default values."""
        self._hbond_dist.setValue(3.5)
        self._hbond_angle.setValue(120.0)
        self._clash_overlap.setValue(0.4)
        self._salt_bridge.setValue(4.0)
        self._water_probe.setValue(1.4)
        self._atom_size.setValue(8.0)
        self._bond_width.setValue(1.5)
        self._fps_target.setValue(30)
        self._render_mode.setCurrentIndex(0)
        self._bg_color.setCurrentIndex(0)

    def get_render_mode(self) -> str:
        return "atoms" if "ONLY" in self._render_mode.currentText() else "atoms_bonds"

    def get_background_color(self) -> str:
        text = self._bg_color.currentText()
        if "White" in text:
            return "#FFFFFF"
        if "Grey" in text:
            return "#888888"
        if "Dark" in text:
            return "#1A1A1A"
        return "#0A0A0A"