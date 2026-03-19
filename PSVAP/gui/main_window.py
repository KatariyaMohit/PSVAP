"""
gui/main_window.py
------------------
PSVAPMainWindow — minimalist monochrome UI.

v0.2 additions:
  - VIEW menu: RENDER MODE submenu (ATOMS ONLY / ATOMS + BONDS)
  - Atom legend strip below the info strip
  - Atom pick info shown in status bar
  - All original structure preserved exactly

Fix in this version
-------------------
_on_data_loaded had: `if bonds else ""`
`bonds` is a numpy array — using it directly in a boolean context raises:
  ValueError: The truth value of an array with more than one element is ambiguous.
Fixed to: `if (bonds is not None and len(bonds) > 0) else ""`

Render mode fix (v0.2.1)
------------------------
Previous version used plain checkable QActions which:
  1. Could all be checked simultaneously (no mutual exclusion)
  2. Toggled off when re-clicked (emitting triggered(False) → engine got wrong state)
  3. Had a "BONDS ONLY" mode that served no practical purpose

Fix: QActionGroup with setExclusive(True) enforces radio-button behaviour.
     _set_render_mode() no longer reads the triggered bool — it always uses
     the mode string passed from the lambda, so re-clicking a checked item
     still triggers a rebuild correctly.
     "BONDS ONLY" removed — only ATOMS ONLY and ATOMS + BONDS remain.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont, QColor, QPalette, QCursor, QActionGroup
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFrame,
    QSizePolicy,
)

from PSVAP.app.controller import ApplicationController
from PSVAP.gui.panels.analysis_panel import AnalysisPanel
from PSVAP.gui.panels.docking_panel import DockingPanel
from PSVAP.gui.panels.export_panel import ExportPanel
from PSVAP.gui.panels.modeling_panel import ModelingPanel
from PSVAP.gui.panels.playback_panel import PlaybackPanel
from PSVAP.gui.panels.plugin_panel import PluginPanel
from PSVAP.gui.panels.selection_panel import SelectionPanel
from PSVAP.gui.panels.viewport_panel import ViewportPanel

# ── Palette ────────────────────────────────────────────────────────────────
BG        = "#0A0A0A"
PANEL     = "#111111"
PANEL_ALT = "#1A1A1A"
BORDER    = "#2A2A2A"
TEXT      = "#CCCCCC"
TEXT_DIM  = "#888888"
TEXT_HINT = "#555555"
ACCENT    = "#E8FF00"
WHITE     = "#FFFFFF"
ERROR     = "#FF3B30"

# ── File dialog filters ────────────────────────────────────────────────────
_ALL_FILTER = (
    "All Supported ("
    "*.lammpstrj *.traj *.data *.lammps "
    "*.gro *.xtc *.trr "
    "*.pdb *.cif *.mmcif "
    "*.nc *.ncdf *.mdcrd *.crd *.rst7 *.rst "
    "*.dcd *.xyz *.mol2 *.sdf *.mol"
    ")"
)
_SINGLE_FILTERS = ";;".join([
    _ALL_FILTER,
    "LAMMPS Trajectory (*.lammpstrj *.traj)",
    "LAMMPS Data File (*.data *.lammps)",
    "GROMACS Structure (*.gro)",
    "GROMACS Trajectory (*.xtc *.trr)",
    "Protein Data Bank (*.pdb)",
    "mmCIF / PDBx (*.cif *.mmcif)",
    "AMBER Trajectory (*.nc *.ncdf *.mdcrd *.crd)",
    "AMBER Restart (*.rst7 *.rst)",
    "CHARMM/NAMD DCD (*.dcd)",
    "XYZ Format (*.xyz)",
    "MOL2 Format (*.mol2)",
    "SDF / MOL (*.sdf *.mol)",
    "All Files (*)",
])
_TOPO_FILTERS = ";;".join([
    "All Topology Files (*.data *.lammps *.lammpstrj *.gro *.pdb *.prmtop *.parm7 *.psf)",
    "LAMMPS Data (*.data *.lammps *.lammpstrj)",
    "GROMACS Structure (*.gro)",
    "PDB Topology (*.pdb)",
    "AMBER Topology (*.prmtop *.parm7)",
    "CHARMM PSF (*.psf)",
    "All Files (*)",
])
_TRAJ_FILTERS = ";;".join([
    "All Trajectory Files (*.lammpstrj *.traj *.xtc *.trr *.nc *.ncdf *.mdcrd *.dcd)",
    "LAMMPS Trajectory (*.lammpstrj *.traj)",
    "GROMACS Trajectory (*.xtc *.trr)",
    "AMBER Trajectory (*.nc *.ncdf *.mdcrd *.crd)",
    "CHARMM/NAMD DCD (*.dcd)",
    "All Files (*)",
])


# ── Global stylesheet ──────────────────────────────────────────────────────
GLOBAL_STYLE = f"""
/* ── Base ───────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Helvetica Neue", "Arial", sans-serif;
    font-size: 12px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QMainWindow {{ background-color: {BG}; }}

/* ── Menu bar ───────────────────────────────────────────────── */
QMenuBar {{
    background-color: {BG};
    color: {TEXT_DIM};
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid {BORDER};
    padding: 0px 8px;
}}
QMenuBar::item {{ background: transparent; padding: 6px 12px; color: {TEXT_DIM}; }}
QMenuBar::item:selected {{ background-color: {PANEL_ALT}; color: {TEXT}; }}
QMenu {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 0px;
    padding: 4px 0px;
}}
QMenu::item {{ padding: 8px 24px 8px 16px; color: {TEXT}; font-size: 11px; letter-spacing: 1px; }}
QMenu::item:selected {{ background-color: {PANEL_ALT}; color: {ACCENT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 0; }}

/* ── Status bar ─────────────────────────────────────────────── */
QStatusBar {{
    background-color: {BG};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
    font-size: 10px;
    letter-spacing: 1px;
    padding: 0px 12px;
}}

/* ── Splitter ───────────────────────────────────────────────── */
QSplitter::handle {{ background-color: {BORDER}; width: 1px; }}
QSplitter::handle:hover {{ background-color: {ACCENT}; }}

/* ── Tab widget ─────────────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {BORDER};
    background-color: {PANEL};
}}
QTabBar {{ background-color: {PANEL}; }}
QTabBar::tab {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    min-width: 70px;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; background-color: {PANEL_ALT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; background-color: {PANEL_ALT}; }}

/* ── Scroll bars ────────────────────────────────────────────── */
QScrollBar:vertical {{ background: {BG}; width: 4px; border: none; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 2px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_DIM}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{ background: {BG}; height: 4px; border: none; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 2px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}

/* ── Push buttons ───────────────────────────────────────────── */
QPushButton {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    padding: 7px 16px;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 0px;
}}
QPushButton:hover {{ background-color: {PANEL_ALT}; color: {TEXT}; border-color: {TEXT_DIM}; }}
QPushButton:pressed {{ background-color: {ACCENT}; color: {BG}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_HINT}; border-color: {TEXT_HINT}; }}

/* ── Line edits / inputs ────────────────────────────────────── */
QLineEdit {{
    background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 0px;
    color: {TEXT};
    padding: 6px 10px;
    font-size: 11px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}
QLineEdit:focus {{ border-color: {TEXT_DIM}; background-color: {PANEL}; }}
QLineEdit::placeholder {{ color: {TEXT_HINT}; }}

/* ── Spinboxes ──────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    color: {TEXT};
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {TEXT_DIM}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {PANEL_ALT};
    border: none;
    border-left: 1px solid {BORDER};
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {BORDER};
}}

/* ── Sliders ────────────────────────────────────────────────── */
QSlider::groove:horizontal {{ height: 1px; background: {BORDER}; border: none; }}
QSlider::handle:horizontal {{
    background: {TEXT};
    border: none;
    width: 10px; height: 10px;
    margin: -5px 0;
    border-radius: 5px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QSlider::sub-page:horizontal {{ background: {TEXT_DIM}; height: 1px; }}

/* ── Progress bar ───────────────────────────────────────────── */
QProgressBar {{ background-color: {PANEL_ALT}; border: none; height: 2px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background-color: {ACCENT}; }}

/* ── Group boxes ────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 0px;
    margin-top: 16px;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {TEXT_DIM};
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background-color: {PANEL};
}}

/* ── Labels ─────────────────────────────────────────────────── */
QLabel {{ background: transparent; color: {TEXT}; }}

/* ── Combo boxes ────────────────────────────────────────────── */
QComboBox {{
    background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    color: {TEXT};
    padding: 5px 10px;
    border-radius: 0px;
    font-size: 11px;
}}
QComboBox:hover {{ border-color: {TEXT_DIM}; }}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    color: {TEXT};
    selection-background-color: {PANEL_ALT};
    selection-color: {ACCENT};
}}

/* ── Check boxes ────────────────────────────────────────────── */
QCheckBox {{ spacing: 8px; color: {TEXT}; font-size: 11px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {TEXT_DIM};
    background: transparent;
    border-radius: 0px;
}}
QCheckBox::indicator:checked {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:hover {{ border-color: {TEXT}; }}

/* ── Tooltips ───────────────────────────────────────────────── */
QToolTip {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 8px; font-size: 10px; }}

/* ── Text edits ─────────────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    color: {TEXT};
    padding: 8px;
    font-family: "Courier New", monospace;
    font-size: 11px;
    selection-background-color: {ACCENT};
    selection-color: {BG};
}}

/* ── List / tree widgets ────────────────────────────────────── */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    color: {TEXT};
    outline: none;
    alternate-background-color: {PANEL};
}}
QListWidget::item, QTreeWidget::item {{ padding: 5px 8px; border: none; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background-color: {PANEL}; color: {ACCENT}; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background-color: {PANEL}; }}
QHeaderView::section {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 10px;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
}}
"""


class PSVAPMainWindow(QMainWindow):
    """
    Minimalist monochrome main window for PSVAP.

    Layout
    ------
    ┌─────────────────────────────────────────────────────┐
    │  MENU BAR  (near-black, spaced caps)                │
    ├─────────────────┬───────────────────────────────────┤
    │                 │  HEADER BAR (logo + status)        │
    │  3D VIEWPORT    ├───────────────────────────────────┤
    │  (full bleed)   │  INFO STRIP + LEGEND              │
    │                 ├───────────────────────────────────┤
    │                 │  TABBED PANELS                     │
    ├─────────────────┴───────────────────────────────────┤
    │  PLAYBACK BAR (full width, minimal)                 │
    ├─────────────────────────────────────────────────────┤
    │  STATUS BAR (1px top border, atom info on click)    │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("PSVAP")
        self.resize(1440, 900)

        QApplication.instance().setStyle("Fusion")
        self.setStyleSheet(GLOBAL_STYLE)

        self._build_menu()
        self._build_central()
        self._build_status_bar()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    #  Menu                                                                #
    # ------------------------------------------------------------------ #

    def _build_menu(self) -> None:
        mb = QMenuBar(self)
        self.setMenuBar(mb)

        file_menu = mb.addMenu("FILE")
        act_open = file_menu.addAction("OPEN FILE")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_single)

        act_dual = file_menu.addAction("OPEN TOPOLOGY + TRAJECTORY")
        act_dual.setShortcut("Ctrl+Shift+O")
        act_dual.triggered.connect(self._on_open_dual)

        file_menu.addSeparator()
        act_quit = file_menu.addAction("QUIT")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        # ── View menu ──────────────────────────────────────────────────
        view_menu = mb.addMenu("VIEW")

        act_reset = view_menu.addAction("RESET CAMERA")
        act_reset.setShortcut("Ctrl+R")
        act_reset.triggered.connect(self._on_reset_camera)

        view_menu.addSeparator()

        # ── Render mode submenu — radio-button behaviour via QActionGroup ──
        # QActionGroup with setExclusive(True) ensures only one action can
        # be checked at a time (like radio buttons).  Without this, each
        # QAction toggles independently and multiple items end up checked.
        rm_menu = view_menu.addMenu("RENDER MODE")

        self._render_mode_group = QActionGroup(self)
        self._render_mode_group.setExclusive(True)

        self._render_mode_actions: dict[str, object] = {}

        # Only two modes: atoms-only and the default atoms+bonds.
        # "Bonds only" was removed — not practically useful and confusing.
        for mode, label in [
            ("atoms_bonds", "ATOMS + BONDS  (default)"),
            ("atoms",       "ATOMS ONLY"),
        ]:
            act = rm_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(mode == "atoms_bonds")   # default selection
            # Add to group — group enforces mutual exclusion automatically
            self._render_mode_group.addAction(act)
            self._render_mode_actions[mode] = act
            # Connect: lambda captures mode string, ignores the `checked`
            # bool that triggered() emits.  This is critical — if we used
            # `checked` we'd get False when the user re-clicks an item,
            # which would still call _set_render_mode with the wrong intent.
            act.triggered.connect(
                lambda *args, m=mode: self._set_render_mode(m)
            )

        help_menu = mb.addMenu("HELP")
        act_about = help_menu.addAction("ABOUT")
        act_about.triggered.connect(self._on_about)
        help_menu.addSeparator()
        act_settings = help_menu.addAction("SETTINGS")
        act_settings.setShortcut("Ctrl+,")
        act_settings.triggered.connect(self._on_settings)

    # ------------------------------------------------------------------ #
    #  Central widget                                                      #
    # ------------------------------------------------------------------ #

    def _build_central(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header bar
        self._header = self._make_header()
        root_layout.addWidget(self._header)

        # Thin separator
        sep = QFrame(); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER};")
        root_layout.addWidget(sep)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        root_layout.addWidget(splitter, stretch=1)

        # Left: viewport
        self.viewport_panel = ViewportPanel(controller=self.controller)
        self.viewport_panel.setStyleSheet(f"background-color: {BG}; border: none;")
        splitter.addWidget(self.viewport_panel)

        # Right column
        right = QWidget()
        right.setStyleSheet(f"background-color: {PANEL};")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        splitter.addWidget(right)

        # Info strip
        self._info_strip = self._make_info_strip()
        right_layout.addWidget(self._info_strip)

        # Legend strip (atom type colours)
        self._legend_strip = self._make_legend_strip()
        right_layout.addWidget(self._legend_strip)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.selection_panel = SelectionPanel(controller=self.controller)
        self.analysis_panel  = AnalysisPanel(controller=self.controller)
        self.modeling_panel  = ModelingPanel(controller=self.controller)
        self.docking_panel   = DockingPanel(controller=self.controller)
        self.plugin_panel    = PluginPanel(controller=self.controller)
        self.export_panel    = ExportPanel(controller=self.controller)

        for panel, label in [
            (self.selection_panel, "SELECT"),
            (self.analysis_panel,  "ANALYSIS"),
            (self.modeling_panel,  "MODEL"),
            (self.docking_panel,   "DOCK"),
            (self.plugin_panel,    "PLUGINS"),
            (self.export_panel,    "EXPORT"),
        ]:
            panel.setStyleSheet(f"background-color: {PANEL}; border: none;")
            self.tabs.addTab(panel, label)

        right_layout.addWidget(self.tabs, stretch=1)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1000, 440])

        # Playback bar
        playback_container = QWidget()
        playback_container.setFixedHeight(52)
        playback_container.setStyleSheet(
            f"background-color: {PANEL}; border-top: 1px solid {BORDER};")
        pb_layout = QHBoxLayout(playback_container)
        pb_layout.setContentsMargins(16, 0, 16, 0)
        pb_layout.setSpacing(0)
        self.playback_panel = PlaybackPanel(controller=self.controller)
        self.playback_panel.setStyleSheet("background: transparent; border: none;")
        pb_layout.addWidget(self.playback_panel)
        root_layout.addWidget(playback_container)

    def _make_header(self) -> QWidget:
        header = QWidget(); header.setFixedHeight(44)
        header.setStyleSheet(f"background-color: {BG}; border-bottom: 1px solid {BORDER};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0); layout.setSpacing(0)

        logo = QLabel("PSVAP")
        logo.setStyleSheet(f"color:{WHITE}; font-size:13px; font-weight:700; letter-spacing:6px; background:transparent;")
        layout.addWidget(logo)

        dot = QLabel("·")
        dot.setStyleSheet(f"color:{ACCENT}; font-size:20px; font-weight:900; background:transparent; margin:0 8px;")
        layout.addWidget(dot)

        sub = QLabel("PARTICLE SIMULATION VISUALIZATION & ANALYSIS")
        sub.setStyleSheet(f"color:{TEXT_DIM}; font-size:8px; letter-spacing:3px; background:transparent;")
        layout.addWidget(sub)
        layout.addStretch()

        ver = QLabel("v0.2  PHASE 1")
        ver.setStyleSheet(f"color:{TEXT_HINT}; font-size:9px; letter-spacing:2px; background:transparent;")
        layout.addWidget(ver)

        self._header_progress = QProgressBar()
        self._header_progress.setFixedWidth(120); self._header_progress.setFixedHeight(2)
        self._header_progress.setRange(0, 100); self._header_progress.setVisible(False)
        self._header_progress.setTextVisible(False)
        layout.addWidget(self._header_progress)
        return header

    def _make_info_strip(self) -> QWidget:
        strip = QWidget(); strip.setFixedHeight(40)
        strip.setStyleSheet(f"background-color: {PANEL_ALT}; border-bottom: 1px solid {BORDER};")
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(16, 0, 16, 0); layout.setSpacing(16)

        self._atoms_label = QLabel("NO FILE LOADED")
        self._atoms_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; background:transparent;")
        layout.addWidget(self._atoms_label)
        layout.addStretch()

        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; background:transparent;")
        layout.addWidget(self._frame_label)
        return strip

    def _make_legend_strip(self) -> QWidget:
        """Colour legend for atom types / elements. Hidden until data loads."""
        self._legend_widget = QWidget()
        self._legend_widget.setFixedHeight(28)
        self._legend_widget.setStyleSheet(
            f"background-color: {BG}; border-bottom: 1px solid {BORDER};")
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(12, 0, 12, 0)
        self._legend_layout.setSpacing(12)
        self._legend_widget.setVisible(False)
        return self._legend_widget

    def _update_legend(self) -> None:
        """Rebuild the legend strip from the engine's current atom types."""
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            engine = self.controller._engine
            if engine is None:
                return
            items = engine.get_legend_items()
        except Exception:
            return

        if not items:
            self._legend_widget.setVisible(False)
            return

        lbl = QLabel("TYPES:")
        lbl.setStyleSheet(f"color:{TEXT_HINT}; font-size:8px; letter-spacing:2px; background:transparent;")
        self._legend_layout.addWidget(lbl)

        for label, (r, g, b) in items[:12]:
            hex_col = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{hex_col}; font-size:14px; background:transparent;")
            txt = QLabel(label)
            txt.setStyleSheet(f"color:{TEXT_DIM}; font-size:8px; letter-spacing:1px; background:transparent;")
            self._legend_layout.addWidget(dot)
            self._legend_layout.addWidget(txt)

        self._legend_layout.addStretch()
        self._legend_widget.setVisible(True)

    # ------------------------------------------------------------------ #
    #  Status bar                                                          #
    # ------------------------------------------------------------------ #

    def _build_status_bar(self) -> None:
        sb = QStatusBar(); sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)

        self._status_label = QLabel("READY")
        self._status_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; background:transparent;")
        sb.addWidget(self._status_label, 1)

        self._format_label = QLabel("")
        self._format_label.setStyleSheet(
            f"color:{TEXT_HINT}; font-size:9px; letter-spacing:2px; background:transparent;")
        sb.addPermanentWidget(self._format_label)

    # ------------------------------------------------------------------ #
    #  Signal wiring                                                       #
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        c = self.controller
        c.load_started.connect(self._on_load_started)
        c.load_progress.connect(self._on_load_progress)
        c.load_finished.connect(self._on_load_done)
        c.load_error.connect(self._on_load_error)
        c.status_message.connect(self._on_status)
        c.model.frame_changed.connect(self._on_frame_changed)
        c.model.data_loaded.connect(self._on_data_loaded)

    def _connect_engine_signals(self) -> None:
        try:
            engine = self.controller._engine
            if engine and hasattr(engine, 'atom_picked'):
                engine.atom_picked.connect(self._on_atom_picked)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Slots                                                               #
    # ------------------------------------------------------------------ #

    @Slot()
    def _on_open_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "OPEN FILE", "", _SINGLE_FILTERS)
        if path:
            self.controller.load_file(path)

    @Slot()
    def _on_open_dual(self) -> None:
        topo, _ = QFileDialog.getOpenFileName(self, "SELECT TOPOLOGY — STEP 1 OF 2", "", _TOPO_FILTERS)
        if not topo: return
        traj, _ = QFileDialog.getOpenFileName(self, "SELECT TRAJECTORY — STEP 2 OF 2", "", _TRAJ_FILTERS)
        if not traj: return
        self.controller.load_topology_and_trajectory(topo, traj)

    @Slot()
    def _on_reset_camera(self) -> None:
        try:
            engine = self.controller._engine
            if engine and engine._plotter:
                engine._plotter.reset_camera()
                engine._plotter.render()
        except Exception:
            pass

    def _set_render_mode(self, mode: str) -> None:
        """
        Switch render mode.

        The QActionGroup already handles unchecking the previously selected
        item — we only need to update the engine here.

        We do NOT read the QAction's checked state here.  We always trust
        the mode string passed from the lambda, because:
          - If the user clicks an already-checked item, Qt still emits
            triggered() and the group keeps it checked — so the engine
            should still process the mode (idempotent is fine).
          - Reading `act.isChecked()` after the fact could race with Qt's
            internal state update.
        """
        try:
            engine = self.controller._engine
            if engine:
                engine.set_render_mode(mode)
                mode_label = "ATOMS + BONDS" if mode == "atoms_bonds" else "ATOMS ONLY"
                self._status_label.setText(f"RENDER MODE: {mode_label}")
        except Exception as exc:
            self._status_label.setText(f"RENDER MODE ERROR: {exc}")

    @Slot()
    def _on_about(self) -> None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle("ABOUT PSVAP")
        dlg.setText(
            "<div style='font-family:Helvetica Neue,Arial;'>"
            f"<div style='font-size:18px;font-weight:700;letter-spacing:6px;color:{TEXT};'>PSVAP</div>"
            f"<div style='font-size:9px;letter-spacing:3px;color:{TEXT_DIM};margin-top:4px;'>"
            "PARTICLE SIMULATION VISUALIZATION &amp; ANALYSIS PACKAGE</div>"
            "<hr style='border:1px solid #2A2A2A;margin:12px 0;'>"
            f"<div style='font-size:11px;color:{TEXT_DIM};'>Version 0.2.0 — Phase 1 Multi-Format I/O</div>"
            f"<div style='font-size:11px;color:{TEXT_DIM};margin-top:8px;'>"
            "Group F · CS 310 Software Engineering · Spring 2026</div>"
            f"<div style='font-size:10px;color:{TEXT_HINT};margin-top:8px;'>"
            "Mohit Katariya · Sai Sashank · Sameer Choudhary<br>"
            "Srikanth · Annamareddi Suhitha</div>"
            "<hr style='border:1px solid #2A2A2A;margin:12px 0;'>"
            f"<div style='font-size:9px;color:{TEXT_HINT};letter-spacing:1px;'>"
            "Formats: LAMMPS · GROMACS · PDB · mmCIF<br>"
            "AMBER · DCD · XYZ · MOL2 · SDF</div>"
            "</div>"
        )
        dlg.setStyleSheet(
            f"QMessageBox {{ background-color: {PANEL}; color: {TEXT}; }}"
            f"QLabel {{ color: {TEXT}; font-size: 12px; }}"
            f"QPushButton {{ background: {PANEL_ALT}; color: {TEXT_DIM}; "
            f"border: 1px solid {BORDER}; padding: 6px 20px; font-size: 9px; letter-spacing: 2px; }}"
            f"QPushButton:hover {{ color: {TEXT}; border-color: {TEXT_DIM}; }}"
        )
        dlg.exec()


    @Slot()
    def _on_settings(self) -> None:
        from PSVAP.gui.dialogs.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec():
            # Apply display settings to engine if loaded
            try:
                engine = self.controller._engine
                if engine and engine._plotter:
                    bg = dlg.get_background_color()
                    engine._plotter.set_background(bg)
                    mode = dlg.get_render_mode()
                    engine.set_render_mode(mode)
            except Exception:
                pass


    @Slot(str)
    def _on_load_started(self, message: str) -> None:
        self._status_label.setText(message.upper())
        self._header_progress.setValue(0); self._header_progress.setVisible(True)
        self.setEnabled(False)

    @Slot(int)
    def _on_load_progress(self, pct: int) -> None:
        self._header_progress.setValue(pct)

    @Slot()
    def _on_load_done(self) -> None:
        self._header_progress.setVisible(False); self.setEnabled(True)
        self._status_label.setText("LOADED  ·  Click atom for details  ·  View → Render Mode to toggle bonds")
        self._connect_engine_signals()

    @Slot(str)
    def _on_load_error(self, message: str) -> None:
        self._header_progress.setVisible(False); self.setEnabled(True)
        self._status_label.setText("ERROR")
        short = message[:1200] + "\n…" if len(message) > 1200 else message
        dlg = QMessageBox(self)
        dlg.setWindowTitle("LOAD ERROR"); dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setText(f"<pre style='font-family:Courier New;font-size:11px;color:{TEXT};'>{short}</pre>")
        dlg.setStyleSheet(
            f"QMessageBox {{ background-color: {PANEL}; }}"
            f"QLabel {{ color: {TEXT}; }}"
            f"QPushButton {{ background: {PANEL_ALT}; color: {TEXT_DIM}; border: 1px solid {BORDER}; padding: 6px 20px; }}"
        )
        dlg.exec()

    @Slot(str)
    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg.upper())

    @Slot(int)
    def _on_frame_changed(self, n: int) -> None:
        total = self.controller.model.n_frames()
        ts_list = self.controller.model.metadata.timesteps
        ts = ts_list[n] if ts_list and n < len(ts_list) else n
        self._frame_label.setText(f"FRAME {n+1}/{total}  ·  STEP {ts}")

    @Slot()
    def _on_data_loaded(self) -> None:
        atoms = getattr(self.controller.model, 'atoms',
                getattr(self.controller.model, '_atoms', []))
        n = len(atoms)
        f = self.controller.model.n_frames()
        src = self.controller.model.metadata.source_path
        fmt = src.suffix.upper().lstrip(".") if src else "?"
        bonds = getattr(self.controller.model.metadata, 'bonds', None)
        bond_info = f"  ·  {len(bonds)//3:,} BONDS" if (bonds is not None and len(bonds) > 0) else ""
        self._atoms_label.setText(
            f"{n:,} ATOMS{bond_info}  ·  {f} FRAME{'S' if f != 1 else ''}  ·  {fmt}")
        self._format_label.setText(fmt)

        QTimer.singleShot(500, self._update_legend)
        self._connect_engine_signals()

    @Slot(str)
    def _on_atom_picked(self, info: str) -> None:
        """Show atom details in the status bar when user clicks an atom."""
        self._status_label.setText(info)