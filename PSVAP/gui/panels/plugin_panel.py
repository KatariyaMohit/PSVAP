"""
gui/panels/plugin_panel.py
---------------------------
Phase 7: Plugin Panel — Python script sandbox execution.

Users can write or load Python scripts that access the loaded molecular
data via the PSVAP Plugin API. Scripts run in a restricted environment
(RestrictedPython) in a background thread so the GUI never freezes.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QVBoxLayout, QWidget, QFileDialog,
    QSplitter,
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


def _lbl(text: str, dim: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{TEXT_DIM if dim else TEXT}; font-size:9px; "
        f"letter-spacing:2px; background:transparent;")
    return lbl


def _divider() -> QFrame:
    f = QFrame(); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER};"); return f


def _btn(text: str, accent: bool = False) -> QPushButton:
    b = QPushButton(text)
    if accent:
        b.setStyleSheet(
            f"QPushButton {{ background:{ACCENT}; color:#0A0A0A; "
            f"border:1px solid {ACCENT}; padding:7px 16px; font-size:9px; "
            f"letter-spacing:2px; font-weight:700; }}"
            f"QPushButton:hover {{ background:#FFFF44; }}")
    else:
        b.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
            f"border:1px solid {BORDER}; padding:7px 16px; font-size:9px; "
            f"letter-spacing:2px; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}"
            f"QPushButton:pressed {{ background:{ACCENT}; color:#0A0A0A; "
            f"border-color:{ACCENT}; }}")
    return b


# ── Plugin runner thread ───────────────────────────────────────────────────

class PluginRunnerThread(QThread):
    """
    Runs a user plugin script in a background thread.
    Emits output lines as they are produced.
    """
    output_line = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(self, script: str, api) -> None:
        super().__init__()
        self._script = script
        self._api    = api

    def run(self) -> None:
        try:
            from PSVAP.plugins.sandbox import run_plugin_script
            stdout_lines: list[str] = []
            run_plugin_script(
                self._script,
                self._api,
                stdout_callback=self.output_line.emit,
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.finished_err.emit(str(exc))


class PluginPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller  = controller
        self._runner     = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        # Header
        header = QWidget(); header.setFixedHeight(36)
        header.setStyleSheet(
            f"background:{PANEL_ALT}; border-bottom:1px solid {BORDER};")
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(16, 0, 16, 0); hrow.setSpacing(8)
        hrow.addWidget(_lbl("PYTHON PLUGIN CONSOLE"))
        hrow.addStretch()

        self._load_btn = _btn("LOAD .py")
        self._run_btn  = _btn("RUN", accent=True)
        self._clear_btn = _btn("CLEAR OUTPUT")
        hrow.addWidget(self._load_btn)
        hrow.addWidget(self._run_btn)
        hrow.addWidget(self._clear_btn)
        layout.addWidget(header)

        # Split: editor top, output bottom
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background:{BORDER}; }}")

        # Script editor
        self._editor = QTextEdit()
        self._editor.setStyleSheet(
            f"QTextEdit {{ background:{BG}; border:none; color:{TEXT}; "
            f"font-family:{MONO}; font-size:11px; padding:8px; }}")
        self._editor.setPlaceholderText(
            "# Write your plugin script here.\n"
            "# Available API:\n"
            "#   atoms    = get_atoms()\n"
            "#   pos      = get_positions()     # (N,3) numpy array\n"
            "#   mask     = get_selection(query)\n"
            "#   log(msg)                       # print to console\n"
            "#   highlight(mask, color)\n\n"
            "atoms = get_atoms()\n"
            "log(f'Loaded {len(atoms)} atoms')\n"
        )
        splitter.addWidget(self._editor)

        # Output console
        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setStyleSheet(
            f"QTextEdit {{ background:#080808; border:none; "
            f"color:#00FF88; font-family:{MONO}; font-size:10px; padding:8px; }}")
        splitter.addWidget(self._console)

        splitter.setSizes([300, 200])
        layout.addWidget(splitter, stretch=1)

        # Bottom bar
        bar = QWidget(); bar.setFixedHeight(28)
        bar.setStyleSheet(
            f"background:{PANEL_ALT}; border-top:1px solid {BORDER};")
        brow = QHBoxLayout(bar)
        brow.setContentsMargins(16, 0, 16, 0); brow.setSpacing(0)
        self._status_lbl = QLabel("READY")
        self._status_lbl.setStyleSheet(
            f"color:{TEXT_HINT}; font-size:9px; letter-spacing:2px; background:transparent;")
        brow.addWidget(self._status_lbl)
        layout.addWidget(bar)

        # Wire buttons
        self._load_btn.clicked.connect(self._load_script)
        self._run_btn.clicked.connect(self._run_script)
        self._clear_btn.clicked.connect(self._clear_output)

        # Example scripts
        self._load_example()

    def _load_example(self) -> None:
        self._editor.setPlainText(
            "# PSVAP Plugin Example\n"
            "# The following variables are automatically available:\n"
            "#   get_atoms()        → list of Atom objects\n"
            "#   get_positions()    → (N,3) numpy array\n"
            "#   get_frame(n)       → (N,3) array for frame n\n"
            "#   get_selection(q)   → boolean mask\n"
            "#   log(msg)           → print to this console\n"
            "#   highlight(mask, color='yellow')\n\n"
            "import numpy as np\n\n"
            "atoms = get_atoms()\n"
            "pos   = get_positions()\n\n"
            "log(f'Structure has {len(atoms)} atoms')\n\n"
            "if pos is not None:\n"
            "    center = pos.mean(axis=0)\n"
            "    log(f'Center of mass: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) Å')\n"
            "    spread = pos.std(axis=0).mean()\n"
            "    log(f'Average spread: {spread:.2f} Å')\n"
        )

    @Slot()
    def _load_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Plugin Script", "",
            "Python Files (*.py);;All Files (*)")
        if not path:
            return
        try:
            from pathlib import Path
            text = Path(path).read_text(encoding="utf-8")
            self._editor.setPlainText(text)
            self._status_lbl.setText(f"LOADED: {path}")
        except Exception as e:
            self._console.append(f"ERROR loading file: {e}")

    @Slot()
    def _run_script(self) -> None:
        script = self._editor.toPlainText().strip()
        if not script:
            return

        self._console.append("─" * 40)
        self._console.append("▶ RUNNING SCRIPT...")
        self._status_lbl.setText("RUNNING...")
        self._run_btn.setEnabled(False)

        # Build plugin API
        try:
            from PSVAP.plugins.api import PluginAPI
            api = PluginAPI(model=self.controller.model,
                            engine=self.controller._engine)
        except Exception as exc:
            self._console.append(f"API ERROR: {exc}")
            self._run_btn.setEnabled(True)
            return

        # Run in thread
        self._runner = PluginRunnerThread(script, api)
        self._runner.output_line.connect(self._on_output)
        self._runner.finished_ok.connect(self._on_finished_ok)
        self._runner.finished_err.connect(self._on_finished_err)
        self._runner.start()

    @Slot(str)
    def _on_output(self, line: str) -> None:
        self._console.append(line)

    @Slot()
    def _on_finished_ok(self) -> None:
        self._console.append("✓ SCRIPT COMPLETED")
        self._status_lbl.setText("DONE")
        self._run_btn.setEnabled(True)

    @Slot(str)
    def _on_finished_err(self, error: str) -> None:
        self._console.append(f"✗ ERROR: {error}")
        self._status_lbl.setText("ERROR")
        self._run_btn.setEnabled(True)

    @Slot()
    def _clear_output(self) -> None:
        self._console.clear()
        self._status_lbl.setText("READY")