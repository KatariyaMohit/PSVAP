"""
gui/panels/docking_panel.py
----------------------------
Phase 5 Docking Panel — Feature 13: Molecular Docking via AutoDock Vina.

Workflow tabs:
  PREPARE  — receptor + ligand PDBQT preparation
  BOX      — docking box definition (manual or from selection)
  DOCK     — run Vina, show results
  RESULTS  — pose viewer and interaction analysis
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QVBoxLayout,
    QWidget, QFileDialog,
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
        f"color:{color}; font-size:9px; letter-spacing:2px; background:transparent;"
    )
    lbl.setWordWrap(True)
    return lbl


def _divider() -> QFrame:
    f = QFrame(); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER}; margin:0;")
    return f


def _result_box(height: int = 120) -> QTextEdit:
    tb = QTextEdit(); tb.setReadOnly(True); tb.setFixedHeight(height)
    tb.setStyleSheet(
        f"QTextEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; font-family:{MONO}; font-size:10px; padding:6px; }}"
    )
    return tb


def _input_line(placeholder: str = "") -> QLineEdit:
    le = QLineEdit(); le.setPlaceholderText(placeholder)
    le.setStyleSheet(
        f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; padding:6px 10px; font-size:11px; }}"
        f"QLineEdit:focus {{ border-color:{TEXT_DIM}; }}"
    )
    return le


def _btn(text: str, accent: bool = False) -> QPushButton:
    b = QPushButton(text)
    if accent:
        b.setStyleSheet(
            f"QPushButton {{ background:{ACCENT}; color:#0A0A0A; "
            f"border:1px solid {ACCENT}; padding:7px 16px; font-size:9px; "
            f"letter-spacing:2px; font-weight:700; }}"
            f"QPushButton:hover {{ background:#FFFF44; }}"
            f"QPushButton:disabled {{ background:{BORDER}; color:{TEXT_HINT}; "
            f"border-color:{BORDER}; }}"
        )
    else:
        b.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
            f"border:1px solid {BORDER}; padding:7px 16px; font-size:9px; "
            f"letter-spacing:2px; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}"
            f"QPushButton:pressed {{ background:{ACCENT}; color:#0A0A0A; "
            f"border-color:{ACCENT}; }}"
        )
    return b


def _dspin(val: float, lo: float = -999.0, hi: float = 999.0) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi); sp.setValue(val); sp.setDecimals(2)
    sp.setStyleSheet(
        f"QDoubleSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; padding:4px 6px; }}")
    sp.setFixedWidth(90)
    return sp


class DockingPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self._last_result = None     # last DockingResult
        self._receptor_pdbqt: str | None = None
        self._ligand_pdbqt:   str | None = None
        self._build()

    def _get_atoms(self) -> list:
        return getattr(self.controller.model, 'atoms', [])

    def _get_positions(self) -> "np.ndarray | None":
        try:
            m = self.controller.model
            f = m.get_frame(getattr(m, '_current_frame', 0))
            return np.asarray(f, dtype=np.float64) if f is not None else None
        except Exception:
            return None

    def _get_n(self) -> int:
        return len(self._get_atoms())

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
            f"QTabBar::tab:hover:!selected {{ color:{TEXT_DIM}; background:{PANEL_ALT}; }}"
        )
        tabs.addTab(self._build_prepare_tab(), "PREPARE")
        tabs.addTab(self._build_box_tab(),     "BOX")
        tabs.addTab(self._build_dock_tab(),    "DOCK")
        tabs.addTab(self._build_results_tab(), "RESULTS")
        layout.addWidget(tabs)

    # ── PREPARE tab ────────────────────────────────────────────────────────

    def _build_prepare_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("STEP 1: PREPARE RECEPTOR"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Load a PDB file as receptor. Waters are removed automatically. "
            "Uses Meeko if installed, otherwise manual PDBQT conversion.",
            hint=True))

        layout.addWidget(_lbl("RECEPTOR PDB FILE:", dim=True))
        rec_row = QHBoxLayout(); rec_row.setSpacing(8)
        self._receptor_path = _input_line("path/to/receptor.pdb")
        self._receptor_browse = _btn("BROWSE")
        rec_row.addWidget(self._receptor_path, stretch=1)
        rec_row.addWidget(self._receptor_browse)
        layout.addLayout(rec_row)

        self._prep_receptor_btn = _btn("PREPARE RECEPTOR → PDBQT")
        layout.addWidget(self._prep_receptor_btn)
        self._prep_receptor_result = _result_box(80)
        layout.addWidget(self._prep_receptor_result)

        self._receptor_browse.clicked.connect(self._browse_receptor)
        self._prep_receptor_btn.clicked.connect(self._run_prep_receptor)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("STEP 2: PREPARE LIGAND"))

        layout.addWidget(_lbl("LIGAND (SMILES or SDF path):", dim=True))
        self._ligand_input = _input_line("CCO  or  path/to/ligand.sdf")
        layout.addWidget(self._ligand_input)

        self._prep_ligand_btn = _btn("PREPARE LIGAND → PDBQT")
        layout.addWidget(self._prep_ligand_btn)
        self._prep_ligand_result = _result_box(80)
        layout.addWidget(self._prep_ligand_result)
        self._prep_ligand_btn.clicked.connect(self._run_prep_ligand)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("USE LOADED STRUCTURE AS RECEPTOR", dim=True))
        self._use_loaded_btn = _btn("USE CURRENT LOADED STRUCTURE")
        layout.addWidget(self._use_loaded_btn)
        self._use_loaded_result = _result_box(60)
        layout.addWidget(self._use_loaded_result)
        self._use_loaded_btn.clicked.connect(self._run_use_loaded_as_receptor)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── BOX tab ────────────────────────────────────────────────────────────

    def _build_box_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("STEP 3: DEFINE DOCKING BOX"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("MANUAL BOX DEFINITION", dim=True))
        cg = QGridLayout(); cg.setSpacing(8)
        cg.addWidget(_lbl("Center X:", dim=True), 0, 0)
        self._cx = _dspin(0.0); cg.addWidget(self._cx, 0, 1)
        cg.addWidget(_lbl("Size X (Å):", dim=True), 0, 2)
        self._sx = _dspin(20.0, 5.0, 200.0); cg.addWidget(self._sx, 0, 3)

        cg.addWidget(_lbl("Center Y:", dim=True), 1, 0)
        self._cy = _dspin(0.0); cg.addWidget(self._cy, 1, 1)
        cg.addWidget(_lbl("Size Y (Å):", dim=True), 1, 2)
        self._sy = _dspin(20.0, 5.0, 200.0); cg.addWidget(self._sy, 1, 3)

        cg.addWidget(_lbl("Center Z:", dim=True), 2, 0)
        self._cz = _dspin(0.0); cg.addWidget(self._cz, 2, 1)
        cg.addWidget(_lbl("Size Z (Å):", dim=True), 2, 2)
        self._sz = _dspin(20.0, 5.0, 200.0); cg.addWidget(self._sz, 2, 3)
        layout.addLayout(cg)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("BOX FROM ATOM SELECTION", dim=True))
        sel_row = QHBoxLayout(); sel_row.setSpacing(8)
        sel_row.addWidget(_lbl("ATOM INDICES:", dim=True))
        self._box_sel = _input_line("e.g. 100-250  or  all")
        self._box_from_sel_btn = _btn("SET BOX FROM SELECTION")
        sel_row.addWidget(self._box_sel, stretch=1)
        sel_row.addWidget(self._box_from_sel_btn)
        layout.addLayout(sel_row)

        layout.addWidget(_lbl("PADDING (Å):", dim=True))
        self._box_padding = _dspin(5.0, 0.0, 30.0)
        layout.addWidget(self._box_padding)

        self._box_result = _result_box(80)
        layout.addWidget(self._box_result)
        self._box_from_sel_btn.clicked.connect(self._run_box_from_selection)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── DOCK tab ───────────────────────────────────────────────────────────

    def _build_dock_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("STEP 4: RUN DOCKING"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("VINA PARAMETERS", dim=True))
        pg = QGridLayout(); pg.setSpacing(8)
        pg.addWidget(_lbl("Exhaustiveness:", dim=True), 0, 0)
        self._exhaustiveness = QSpinBox()
        self._exhaustiveness.setRange(1, 32); self._exhaustiveness.setValue(8)
        self._exhaustiveness.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._exhaustiveness.setFixedWidth(80); pg.addWidget(self._exhaustiveness, 0, 1)

        pg.addWidget(_lbl("Number of poses:", dim=True), 1, 0)
        self._n_poses = QSpinBox()
        self._n_poses.setRange(1, 20); self._n_poses.setValue(9)
        self._n_poses.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._n_poses.setFixedWidth(80); pg.addWidget(self._n_poses, 1, 1)

        pg.addWidget(_lbl("Energy range (kcal/mol):", dim=True), 2, 0)
        self._energy_range = _dspin(3.0, 0.5, 20.0)
        pg.addWidget(self._energy_range, 2, 1)
        layout.addLayout(pg)

        layout.addWidget(_lbl("VINA EXECUTABLE PATH (if not in PATH):", dim=True))
        self._vina_exe = _input_line("vina   (or full path)")
        layout.addWidget(self._vina_exe)

        self._dock_btn = _btn("RUN DOCKING", accent=True)
        layout.addWidget(self._dock_btn)
        self._dock_result = _result_box(200)
        layout.addWidget(self._dock_result)
        self._dock_btn.clicked.connect(self._run_docking)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── RESULTS tab ────────────────────────────────────────────────────────

    def _build_results_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("DOCKING RESULTS"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("POSE SELECTION", dim=True))
        pose_row = QHBoxLayout(); pose_row.setSpacing(8)
        pose_row.addWidget(_lbl("POSE RANK:", dim=True))
        self._pose_selector = QSpinBox()
        self._pose_selector.setRange(1, 9); self._pose_selector.setValue(1)
        self._pose_selector.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._pose_selector.setFixedWidth(70)
        pose_row.addWidget(self._pose_selector)
        self._show_pose_btn = _btn("SHOW POSE IN VIEWPORT")
        pose_row.addWidget(self._show_pose_btn)
        pose_row.addStretch()
        layout.addLayout(pose_row)
        self._show_pose_btn.clicked.connect(self._run_show_pose)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("INTERACTION ANALYSIS ON BEST POSE", dim=True))
        self._inter_btn = _btn("ANALYSE INTERACTIONS")
        layout.addWidget(self._inter_btn)
        self._inter_result = _result_box(200)
        layout.addWidget(self._inter_result)
        self._inter_btn.clicked.connect(self._run_pose_interactions)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Slots ──────────────────────────────────────────────────────────────

    @Slot()
    def _browse_receptor(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Receptor PDB", "",
            "PDB Files (*.pdb);;All Files (*)"
        )
        if path:
            self._receptor_path.setText(path)

    @Slot()
    def _run_prep_receptor(self) -> None:
        try:
            from PSVAP.modeling.docking_engine import prepare_receptor_pdbqt
            pdb = self._receptor_path.text().strip()
            if not pdb:
                self._prep_receptor_result.setText(
                    "Enter a receptor PDB path."); return
            out = prepare_receptor_pdbqt(pdb)
            self._receptor_pdbqt = str(out)
            self._prep_receptor_result.setText(
                f"RECEPTOR PREPARED\n  Input:  {pdb}\n  Output: {out}")
        except Exception as e:
            self._prep_receptor_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_prep_ligand(self) -> None:
        try:
            from PSVAP.modeling.docking_engine import prepare_ligand_pdbqt
            inp = self._ligand_input.text().strip()
            if not inp:
                self._prep_ligand_result.setText(
                    "Enter a SMILES string or SDF path."); return
            out = prepare_ligand_pdbqt(inp)
            self._ligand_pdbqt = str(out)
            self._prep_ligand_result.setText(
                f"LIGAND PREPARED\n  Input:  {inp[:60]}\n  Output: {out}")
        except Exception as e:
            self._prep_ligand_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_use_loaded_as_receptor(self) -> None:
        """Save current loaded structure as a temp PDB and prepare as receptor."""
        try:
            from PSVAP.modeling.mutation_engine import write_pdb
            from PSVAP.modeling.docking_engine import prepare_receptor_pdbqt
            import tempfile

            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._use_loaded_result.setText("NO STRUCTURE LOADED"); return

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_pdb = tmp.name
            write_pdb(atoms, pos, tmp_pdb)
            out = prepare_receptor_pdbqt(tmp_pdb)
            self._receptor_pdbqt = str(out)
            self._use_loaded_result.setText(
                f"LOADED STRUCTURE PREPARED AS RECEPTOR\n"
                f"  Atoms: {len(atoms):,}\n"
                f"  PDBQT: {out}")
        except Exception as e:
            self._use_loaded_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_box_from_selection(self) -> None:
        try:
            from PSVAP.modeling.docking_engine import docking_box_from_selection
            atoms = self._get_atoms()
            pos   = self._get_positions()
            n     = self._get_n()
            if not atoms or pos is None:
                self._box_result.setText("NO STRUCTURE LOADED"); return

            sel_text = self._box_sel.text().strip()
            if not sel_text or sel_text.lower() == "all":
                sel_idx = list(range(n))
            elif "-" in sel_text and "," not in sel_text:
                parts = sel_text.split("-")
                sel_idx = list(range(int(parts[0].strip()),
                                     int(parts[1].strip()) + 1))
            else:
                sel_idx = [int(x.strip()) for x in sel_text.split(",")]
            sel_idx = [i for i in sel_idx if i < n]

            padding = float(self._box_padding.value())
            cx, cy, cz, sx, sy, sz = docking_box_from_selection(
                atoms, pos, sel_idx, padding=padding
            )
            self._cx.setValue(cx); self._cy.setValue(cy); self._cz.setValue(cz)
            self._sx.setValue(sx); self._sy.setValue(sy); self._sz.setValue(sz)

            self._box_result.setText(
                f"BOX SET FROM SELECTION ({len(sel_idx)} atoms)\n"
                f"  Center: ({cx:.2f}, {cy:.2f}, {cz:.2f})\n"
                f"  Size:   ({sx:.2f}, {sy:.2f}, {sz:.2f}) Å"
            )
        except Exception as e:
            self._box_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_docking(self) -> None:
        try:
            from PSVAP.modeling.docking_engine import DockingConfig, run_vina

            if not self._receptor_pdbqt:
                self._dock_result.setText(
                    "No receptor PDBQT prepared.\n"
                    "Go to PREPARE tab and prepare a receptor first."); return
            if not self._ligand_pdbqt:
                self._dock_result.setText(
                    "No ligand PDBQT prepared.\n"
                    "Go to PREPARE tab and prepare a ligand first."); return

            vina_exe = self._vina_exe.text().strip() or "vina"

            config = DockingConfig(
                receptor_pdbqt=self._receptor_pdbqt,
                ligand_pdbqt=self._ligand_pdbqt,
                center_x=self._cx.value(),
                center_y=self._cy.value(),
                center_z=self._cz.value(),
                size_x=self._sx.value(),
                size_y=self._sy.value(),
                size_z=self._sz.value(),
                exhaustiveness=self._exhaustiveness.value(),
                n_poses=self._n_poses.value(),
                energy_range=self._energy_range.value(),
                vina_executable=vina_exe,
            )

            self._dock_result.setText("RUNNING VINA — please wait...")
            result = run_vina(config)
            self._last_result = result
            self._pose_selector.setMaximum(max(1, len(result.poses)))
            self._dock_result.setText(result.summary())

        except Exception as e:
            self._dock_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_show_pose(self) -> None:
        if self._last_result is None or not self._last_result.poses:
            return
        rank = self._pose_selector.value() - 1
        if rank >= len(self._last_result.poses):
            return
        pose = self._last_result.poses[rank]
        # Show pose positions in viewport as a separate point cloud
        try:
            import pyvista as pv
            engine = self.controller._engine
            if engine and engine._plotter:
                cloud = pv.PolyData(pose.positions.astype(np.float32))
                engine._plotter.add_mesh(
                    cloud,
                    color="lime",
                    render_points_as_spheres=True,
                    point_size=10.0,
                    show_scalar_bar=False,
                    name=f"docking_pose_{rank+1}",
                )
                engine._plotter.render()
        except Exception:
            pass

    @Slot()
    def _run_pose_interactions(self) -> None:
        try:
            if self._last_result is None or not self._last_result.poses:
                self._inter_result.setText(
                    "No docking results available.\nRun docking first."); return

            from PSVAP.analysis.interactions import detect_all_interactions

            atoms   = self._get_atoms()
            pos     = self._get_positions()
            pose    = self._last_result.poses[0]

            if not atoms or pos is None:
                self._inter_result.setText(
                    "Receptor structure not loaded."); return

            # Build a combined receptor + ligand system
            n_receptor = len(atoms)
            n_ligand   = len(pose.positions)

            combined_pos = np.vstack([pos, pose.positions])
            receptor_idx = list(range(n_receptor))
            ligand_idx   = list(range(n_receptor, n_receptor + n_ligand))

            # Use receptor atoms for group A, ligand indices for group B
            # (ligand atoms have no element info in Atom objects, so only
            # clash and hydrophobic detection will be meaningful)
            result = detect_all_interactions(
                atoms, combined_pos,
                receptor_idx, ligand_idx,
            )
            self._inter_result.setText(
                f"POSE 1 INTERACTIONS\n\n{result.summary()}\n\n"
                f"Note: Full H-bond/salt bridge analysis requires element\n"
                f"information on ligand atoms. Use a PDB-format ligand for\n"
                f"complete interaction profiling."
            )
        except Exception as e:
            self._inter_result.setText(f"ERROR: {e}")