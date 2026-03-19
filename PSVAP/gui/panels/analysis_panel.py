"""
gui/panels/analysis_panel.py
-----------------------------
Phase 2 Analysis Panel — MERGED FINAL version.
Phase 3 additions: INTERACT tab (Feature 12) and SURFACE tab (Feature 9/18).

[all existing docstring content preserved]
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget, QCheckBox,
)

from PSVAP.app.controller import ApplicationController

# ── Theme ──────────────────────────────────────────────────────────────────
BG        = "#111111"
PANEL_ALT = "#1A1A1A"
BORDER    = "#2A2A2A"
TEXT      = "#CCCCCC"
TEXT_DIM  = "#888888"
TEXT_HINT = "#555555"
ACCENT    = "#E8FF00"
ERROR_COL = "#FF6060"
MONO      = "Courier New, monospace"


def _lbl(text: str, dim: bool = False, hint: bool = False) -> QLabel:
    lbl = QLabel(text)
    color = TEXT_HINT if hint else (TEXT_DIM if dim else TEXT)
    lbl.setStyleSheet(
        f"color:{color}; font-size:9px; letter-spacing:2px; background:transparent;"
    )
    return lbl


def _divider() -> QFrame:
    f = QFrame(); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER}; margin:0;")
    return f


def _result_box(height: int = 120) -> QTextEdit:
    tb = QTextEdit()
    tb.setReadOnly(True)
    tb.setFixedHeight(height)
    tb.setStyleSheet(
        f"QTextEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; font-family:{MONO}; font-size:10px; padding:6px; }}"
    )
    return tb


def _input_line(placeholder: str = "") -> QLineEdit:
    le = QLineEdit()
    le.setPlaceholderText(placeholder)
    le.setStyleSheet(
        f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
        f"color:{TEXT}; padding:6px 10px; font-size:11px; }}"
        f"QLineEdit:focus {{ border-color:{TEXT_DIM}; }}"
    )
    return le


def _btn(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
        f"border:1px solid {BORDER}; padding:6px 14px; font-size:9px; letter-spacing:2px; }}"
        f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}"
        f"QPushButton:pressed {{ background:{ACCENT}; color:#0A0A0A; border-color:{ACCENT}; }}"
    )
    return b


def _checkbox(text: str, checked: bool = True) -> QCheckBox:
    cb = QCheckBox(text)
    cb.setChecked(checked)
    cb.setStyleSheet(
        f"QCheckBox {{ color:{TEXT_DIM}; font-size:9px; "
        f"letter-spacing:1px; background:transparent; spacing:6px; }}"
        f"QCheckBox::indicator {{ width:12px; height:12px; "
        f"border:1px solid {BORDER}; background:transparent; }}"
        f"QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}"
    )
    return cb


def _safe_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return None


class AnalysisPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self._build()

    # ── Safe SystemModel accessors ─────────────────────────────────────────

    def _get_atoms(self) -> list:
        m = self.controller.model
        return getattr(m, 'atoms', getattr(m, '_atoms', []))

    def _get_trajectory(self) -> list:
        m = self.controller.model
        return getattr(m, 'trajectory', getattr(m, '_trajectory', []))

    def _get_frame(self) -> "np.ndarray | None":
        m = self.controller.model
        cur = getattr(m, '_current_frame', 0)
        return m.get_frame(cur)

    def _get_positions(self) -> "np.ndarray | None":
        try:
            model = self.controller.model
            frame = model.get_frame(getattr(model, '_current_frame', 0))
            if frame is None:
                return None
            return np.asarray(frame, dtype=np.float64)
        except Exception:
            return None

    def _get_atom_count(self) -> int:
        try:
            return len(getattr(self.controller.model, 'atoms', []))
        except Exception:
            return 0

    def _get_raw_bonds(self) -> "np.ndarray | None":
        m = self.controller.model
        meta = getattr(m, 'metadata', None)
        bonds = getattr(meta, 'bonds', None)
        if bonds is None or len(bonds) == 0:
            return None
        return bonds

    def _get_bond_pairs(self) -> set:
        bonds = self._get_raw_bonds()
        if bonds is None or len(bonds) == 0:
            return set()
        pairs: set = set()
        for k in range(0, len(bonds) - 2, 3):
            if bonds[k] == 2:
                i = int(bonds[k + 1])
                j = int(bonds[k + 2])
                pairs.add((min(i, j), max(i, j)))
        return pairs

    def _bond_warning(self, *indices: int) -> str:
        bonds = self._get_raw_bonds()
        if bonds is None or len(bonds) == 0:
            return ""
        bond_set = self._get_bond_pairs()
        warnings = []
        atom_list = list(indices)
        for k in range(len(atom_list) - 1):
            a, b = atom_list[k], atom_list[k + 1]
            pair = (min(a, b), max(a, b))
            if pair not in bond_set:
                warnings.append(f"NO BOND BETWEEN ATOMS {a} AND {b}")
        return "\n".join(warnings)

    def _parse_index_range(self, text: str, n_atoms: int) -> list[int] | None:
        """
        Parse an atom index range string into a list of indices.

        Supported formats:
          "0-99"       → [0, 1, ..., 99]
          "0,1,2,5"    → [0, 1, 2, 5]
          "all"        → [0, 1, ..., n_atoms-1]
          ""           → [0, 1, ..., n_atoms-1]  (empty = all)

        Returns None on parse error.
        """
        text = text.strip()
        if not text or text.lower() == "all":
            return list(range(n_atoms))

        # Range: "0-99"
        if "-" in text and "," not in text:
            parts = text.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0].strip()), int(parts[1].strip())
                    return [i for i in range(start, end + 1) if i < n_atoms]
                except ValueError:
                    return None

        # Comma list: "0,1,2,5"
        if "," in text:
            try:
                return [int(x.strip()) for x in text.split(",")
                        if x.strip() and int(x.strip()) < n_atoms]
            except ValueError:
                return None

        # Single integer
        try:
            v = int(text)
            return [v] if v < n_atoms else None
        except ValueError:
            return None

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
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
        tabs.addTab(self._build_geometry_tab(),    "GEOMETRY")
        tabs.addTab(self._build_rmsd_tab(),        "RMSD")
        tabs.addTab(self._build_alignment_tab(),   "ALIGN")
        tabs.addTab(self._build_sequence_tab(),    "SEQUENCE")
        tabs.addTab(self._build_interact_tab(),    "INTERACT")   # Phase 3
        tabs.addTab(self._build_surface_tab(),     "SURFACE")    # Phase 3
        tabs.addTab(self._build_ligand_tab(),    "LIGAND")
        tabs.addTab(self._build_mmp_tab(),       "MMP")
        tabs.addTab(self._build_pharma_tab(),    "PHARMA")
        tabs.addTab(self._build_qsar_tab(),      "QSAR")
        tabs.addTab(self._build_pka_tab(),       "PKA")
        tabs.addTab(self._build_conform_tab(),   "CONFORM")
        tabs.addTab(self._build_sites_tab(),    "SITES")
        tabs.addTab(self._build_watermap_tab(), "WATER MAP")
        tabs.addTab(self._build_cluster_tab(),  "CLUSTER")
        layout.addWidget(tabs)

    # ── Geometry tab ───────────────────────────────────────────────────────

    def _build_geometry_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("GEOMETRY MEASUREMENTS"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("DISTANCE  ( atom i — atom j )", dim=True))
        gr = QGridLayout(); gr.setSpacing(8)
        self._dist_i = _input_line("atom index i"); gr.addWidget(self._dist_i, 0, 0)
        self._dist_j = _input_line("atom index j"); gr.addWidget(self._dist_j, 0, 1)
        self._dist_btn = _btn("MEASURE"); gr.addWidget(self._dist_btn, 0, 2)
        layout.addLayout(gr)
        self._dist_result = _result_box(80)
        layout.addWidget(self._dist_result)
        self._dist_btn.clicked.connect(self._run_distance)

        layout.addWidget(_divider())

        layout.addWidget(_lbl("BOND ANGLE  ( i — j — k )", dim=True))
        ag = QGridLayout(); ag.setSpacing(8)
        self._angle_i = _input_line("i"); ag.addWidget(self._angle_i, 0, 0)
        self._angle_j = _input_line("j"); ag.addWidget(self._angle_j, 0, 1)
        self._angle_k = _input_line("k"); ag.addWidget(self._angle_k, 0, 2)
        self._angle_btn = _btn("MEASURE"); ag.addWidget(self._angle_btn, 0, 3)
        layout.addLayout(ag)
        self._angle_result = _result_box(80)
        layout.addWidget(self._angle_result)
        self._angle_btn.clicked.connect(self._run_angle)

        layout.addWidget(_divider())

        layout.addWidget(_lbl("TORSION / DIHEDRAL  ( i — j — k — l )", dim=True))
        tg = QGridLayout(); tg.setSpacing(8)
        self._tors_i = _input_line("i"); tg.addWidget(self._tors_i, 0, 0)
        self._tors_j = _input_line("j"); tg.addWidget(self._tors_j, 0, 1)
        self._tors_k = _input_line("k"); tg.addWidget(self._tors_k, 0, 2)
        self._tors_l = _input_line("l"); tg.addWidget(self._tors_l, 0, 3)
        self._tors_btn = _btn("MEASURE"); tg.addWidget(self._tors_btn, 0, 4)
        layout.addLayout(tg)
        self._tors_result = _result_box(80)
        layout.addWidget(self._tors_result)
        self._tors_btn.clicked.connect(self._run_torsion)

        layout.addWidget(_divider())

        layout.addWidget(_lbl("RAMACHANDRAN PLOT", dim=True))
        self._rama_btn = _btn("COMPUTE PHI / PSI")
        layout.addWidget(self._rama_btn)
        self._rama_result = _result_box(100)
        layout.addWidget(self._rama_result)
        self._rama_btn.clicked.connect(self._run_ramachandran)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── RMSD tab ───────────────────────────────────────────────────────────

    def _build_rmsd_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("RMSD / RMSF ANALYSIS"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("RMSD VS TIME", dim=True))
        rg = QHBoxLayout(); rg.setSpacing(8)
        rg.addWidget(_lbl("REF FRAME:", dim=True))
        self._rmsd_ref = QSpinBox()
        self._rmsd_ref.setMinimum(0); self._rmsd_ref.setMaximum(999999)
        self._rmsd_ref.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._rmsd_ref.setFixedWidth(80); rg.addWidget(self._rmsd_ref)
        rg.addWidget(_lbl("ATOMS:", dim=True))
        self._rmsd_sel = QComboBox()
        self._rmsd_sel.addItems(["ALL HEAVY", "BACKBONE ONLY", "CA ONLY"])
        self._rmsd_sel.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        rg.addWidget(self._rmsd_sel, stretch=1); layout.addLayout(rg)

        self._rmsd_btn = _btn("RUN RMSD")
        layout.addWidget(self._rmsd_btn)
        self._rmsd_result = _result_box(160)
        layout.addWidget(self._rmsd_result)
        self._rmsd_btn.clicked.connect(self._run_rmsd)

        self._rmsd_all_btn = _btn("RMSD ALL FRAMES vs REF")
        layout.addWidget(self._rmsd_all_btn)
        self._rmsd_all_result = _result_box(160)
        layout.addWidget(self._rmsd_all_result)
        self._rmsd_all_btn.clicked.connect(self._compute_rmsd_all)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("RMSF PER RESIDUE / ATOM", dim=True))
        self._rmsf_btn = _btn("RUN RMSF")
        layout.addWidget(self._rmsf_btn)
        self._rmsf_result = _result_box(160)
        layout.addWidget(self._rmsf_result)
        self._rmsf_btn.clicked.connect(self._run_rmsf)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Alignment tab ──────────────────────────────────────────────────────

    def _build_alignment_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("STRUCTURE ALIGNMENT (KABSCH)"))
        layout.addWidget(_divider())

        hb = QHBoxLayout(); hb.setSpacing(8)
        hb.addWidget(_lbl("REF FRAME:", dim=True))
        self._align_ref = QSpinBox()
        self._align_ref.setMinimum(0); self._align_ref.setMaximum(999999)
        self._align_ref.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; color:{TEXT}; padding:4px 8px; }}")
        self._align_ref.setFixedWidth(80); hb.addWidget(self._align_ref)
        hb.addWidget(_lbl("MOBILE FRAME:", dim=True))
        self._align_mob = QSpinBox()
        self._align_mob.setMinimum(0); self._align_mob.setMaximum(999999); self._align_mob.setValue(1)
        self._align_mob.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; color:{TEXT}; padding:4px 8px; }}")
        self._align_mob.setFixedWidth(80); hb.addWidget(self._align_mob)
        hb.addStretch(); layout.addLayout(hb)

        layout.addWidget(_lbl("ATOM SUBSET (empty = all)", dim=True))
        self._align_atoms = _input_line("e.g.  backbone / ca / all")
        layout.addWidget(self._align_atoms)

        self._align_btn = _btn("SUPERIMPOSE")
        layout.addWidget(self._align_btn)
        self._align_result = _result_box(120)
        layout.addWidget(self._align_result)
        self._align_btn.clicked.connect(self._run_alignment)

        layout.addWidget(_lbl("RMSD MATRIX  (all frames vs all frames)", dim=True))
        self._matrix_btn = _btn("COMPUTE RMSD MATRIX")
        layout.addWidget(self._matrix_btn)
        self._matrix_result = _result_box(160)
        layout.addWidget(self._matrix_result)
        self._matrix_btn.clicked.connect(self._run_rmsd_matrix)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("ALIGN ENTIRE TRAJECTORY TO REF FRAME", dim=True))
        self._align_traj_btn = _btn("ALIGN TRAJECTORY")
        layout.addWidget(self._align_traj_btn)
        self._align_traj_result = _result_box(100)
        layout.addWidget(self._align_traj_result)
        self._align_traj_btn.clicked.connect(self._run_align_trajectory)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Sequence tab ───────────────────────────────────────────────────────

    def _build_sequence_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("SEQUENCE ALIGNMENT"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("EXTRACT SEQUENCES FROM LOADED STRUCTURE", dim=True))
        self._seq_extract_btn = _btn("EXTRACT SEQUENCES")
        layout.addWidget(self._seq_extract_btn)
        self._seq_extract_result = _result_box(100)
        layout.addWidget(self._seq_extract_result)
        self._seq_extract_btn.clicked.connect(self._run_extract_sequences)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("PAIRWISE ALIGNMENT", dim=True))

        mb = QHBoxLayout(); mb.setSpacing(8)
        mb.addWidget(_lbl("MODE:", dim=True))
        self._seq_mode = QComboBox()
        self._seq_mode.addItems(["GLOBAL (NEEDLEMAN-WUNSCH)", "LOCAL (SMITH-WATERMAN)"])
        self._seq_mode.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        mb.addWidget(self._seq_mode, stretch=1); layout.addLayout(mb)

        layout.addWidget(_lbl("SEQUENCE 1:", dim=True))
        self._seq1 = _input_line("ACDEFGHIKLMNPQRSTVWY...")
        layout.addWidget(self._seq1)

        layout.addWidget(_lbl("SEQUENCE 2:", dim=True))
        self._seq2 = _input_line("ACDEFGHIKLMNPQRSTVWY...")
        layout.addWidget(self._seq2)

        self._seq_align_btn = _btn("ALIGN")
        layout.addWidget(self._seq_align_btn)
        self._seq_result = _result_box(160)
        layout.addWidget(self._seq_result)
        self._seq_align_btn.clicked.connect(self._run_sequence_alignment)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Interactions tab (Phase 3) ─────────────────────────────────────────

    def _build_interact_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("NON-COVALENT INTERACTIONS"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("ATOM GROUPS  (index range: 0-99 / comma list / 'all')", dim=True))
        gg = QGridLayout(); gg.setSpacing(8)
        gg.addWidget(_lbl("GROUP A:", dim=True), 0, 0)
        self._inter_group_a = _input_line("e.g. 0-399 or all")
        gg.addWidget(self._inter_group_a, 0, 1)
        gg.addWidget(_lbl("GROUP B:", dim=True), 1, 0)
        self._inter_group_b = _input_line("e.g. 400-499")
        gg.addWidget(self._inter_group_b, 1, 1)
        layout.addLayout(gg)

        layout.addWidget(_lbl("INTERACTION TYPES", dim=True))
        cb_row1 = QHBoxLayout(); cb_row1.setSpacing(16)
        self._cb_hbond  = _checkbox("H-BONDS",    checked=True)
        self._cb_salt   = _checkbox("SALT BRIDGES", checked=True)
        self._cb_clash  = _checkbox("CLASHES",     checked=True)
        cb_row1.addWidget(self._cb_hbond)
        cb_row1.addWidget(self._cb_salt)
        cb_row1.addWidget(self._cb_clash)
        cb_row1.addStretch()
        layout.addLayout(cb_row1)

        cb_row2 = QHBoxLayout(); cb_row2.setSpacing(16)
        self._cb_halogen = _checkbox("HALOGEN",     checked=False)
        self._cb_hydro   = _checkbox("HYDROPHOBIC", checked=False)
        self._cb_pistack = _checkbox("PI-STACK",    checked=False)
        cb_row2.addWidget(self._cb_halogen)
        cb_row2.addWidget(self._cb_hydro)
        cb_row2.addWidget(self._cb_pistack)
        cb_row2.addStretch()
        layout.addLayout(cb_row2)

        self._inter_btn = _btn("DETECT INTERACTIONS")
        layout.addWidget(self._inter_btn)
        self._inter_result = _result_box(180)
        layout.addWidget(self._inter_result)
        self._inter_btn.clicked.connect(self._run_interactions)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("INTERACTION PERSISTENCE OVER TRAJECTORY", dim=True))
        self._inter_traj_btn = _btn("RUN OVER TRAJECTORY")
        layout.addWidget(self._inter_traj_btn)
        self._inter_traj_result = _result_box(160)
        layout.addWidget(self._inter_traj_result)
        self._inter_traj_btn.clicked.connect(self._run_interactions_trajectory)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Surface tab (Phase 3) ──────────────────────────────────────────────

    def _build_surface_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("SOLVENT ACCESSIBLE SURFACE AREA (SASA)"))
        layout.addWidget(_divider())

        pg = QHBoxLayout(); pg.setSpacing(8)
        pg.addWidget(_lbl("PROBE RADIUS (Å):", dim=True))
        self._sasa_probe = QLineEdit("1.4")
        self._sasa_probe.setFixedWidth(60)
        self._sasa_probe.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; font-size:11px; }}")
        pg.addWidget(self._sasa_probe)
        pg.addStretch()
        layout.addLayout(pg)

        self._sasa_btn = _btn("COMPUTE SASA")
        layout.addWidget(self._sasa_btn)
        self._sasa_result = _result_box(180)
        layout.addWidget(self._sasa_result)
        self._sasa_btn.clicked.connect(self._run_sasa)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("SURFACE PATCH CLASSIFICATION", dim=True))
        layout.addWidget(_lbl(
            "Classifies surface-exposed residues by character:\n"
            "hydrophobic / positive / negative / polar",
            hint=True))

        self._patch_btn = _btn("CLASSIFY PATCHES")
        layout.addWidget(self._patch_btn)
        self._patch_result = _result_box(160)
        layout.addWidget(self._patch_result)
        self._patch_btn.clicked.connect(self._run_patch_classification)

        layout.addStretch()
        w.setWidget(inner); return w

    # ──────────────────────────────────────────────────────────────────────
    #  Compute slots — Phase 2 (all preserved exactly)
    # ──────────────────────────────────────────────────────────────────────

    @Slot()
    def _run_distance(self) -> None:
        i = _safe_int(self._dist_i.text())
        j = _safe_int(self._dist_j.text())

        if i is None or j is None:
            self._dist_result.setText("Enter valid atom indices (0-based)."); return

        n = self._get_atom_count()
        if n == 0:
            self._dist_result.setText("NO DATA LOADED"); return
        if not (0 <= i < n and 0 <= j < n):
            self._dist_result.setText(
                f"Index out of range. Valid range: 0 – {n-1}.\nGot i={i}, j={j}.")
            return
        if i == j:
            self._dist_result.setText("Atoms i and j must be different."); return

        pos = self._get_positions()
        if pos is None:
            self._dist_result.setText("Could not read current frame positions."); return

        try:
            from PSVAP.analysis.geometry import distance, distance_trajectory
            d = distance(pos[i], pos[j])
            warning = self._bond_warning(i, j)
            traj = self._get_trajectory()
            if len(traj) > 1:
                ts = distance_trajectory(traj, i, j)
                txt = (f"DISTANCE  {i} — {j}\n"
                       f"CURRENT FRAME : {d:.4f} Å\n"
                       f"MIN / MEAN / MAX : {ts.min():.4f} / {ts.mean():.4f} / {ts.max():.4f} Å")
            else:
                txt = (f"DISTANCE  atom {i} — atom {j}\n"
                       f"  d = {d:.4f} Å\n"
                       f"  pos[{i}] = ({pos[i,0]:.3f}, {pos[i,1]:.3f}, {pos[i,2]:.3f})\n"
                       f"  pos[{j}] = ({pos[j,0]:.3f}, {pos[j,1]:.3f}, {pos[j,2]:.3f})")
            if warning:
                txt += f"\n\n⚠  {warning}"
            self._dist_result.setText(txt)
        except ImportError:
            d = float(np.linalg.norm(pos[j] - pos[i]))
            self._dist_result.setText(
                f"DISTANCE  atom {i} — atom {j}\n  d = {d:.4f} Å")
        except Exception as e:
            self._dist_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_angle(self) -> None:
        i = _safe_int(self._angle_i.text())
        j = _safe_int(self._angle_j.text())
        k = _safe_int(self._angle_k.text())

        if any(x is None for x in [i, j, k]):
            self._angle_result.setText("Enter valid atom indices (0-based)."); return

        n = self._get_atom_count()
        if n == 0:
            self._angle_result.setText("NO DATA LOADED"); return
        for label, idx in [("i", i), ("j", j), ("k", k)]:
            if not (0 <= idx < n):
                self._angle_result.setText(f"Index {label}={idx} out of range (0–{n-1}).")
                return
        if len({i, j, k}) < 3:
            self._angle_result.setText("Atoms i, j, k must all be different."); return

        pos = self._get_positions()
        if pos is None:
            self._angle_result.setText("Could not read current frame positions."); return

        try:
            from PSVAP.analysis.geometry import angle
            a = angle(pos[i], pos[j], pos[k])
            warning = self._bond_warning(i, j, k)
            txt = (f"BOND ANGLE  atom {i} — atom {j} — atom {k}\n"
                   f"  θ = {a:.4f}°")
            if warning:
                txt += f"\n\n⚠  {warning}"
            self._angle_result.setText(txt)
        except ImportError:
            v1 = pos[i] - pos[j]
            v2 = pos[k] - pos[j]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            a = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            self._angle_result.setText(
                f"BOND ANGLE  atom {i} — atom {j} — atom {k}\n  θ = {a:.4f}°")
        except Exception as e:
            self._angle_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_torsion(self) -> None:
        i = _safe_int(self._tors_i.text())
        j = _safe_int(self._tors_j.text())
        k = _safe_int(self._tors_k.text())
        l = _safe_int(self._tors_l.text())

        if any(x is None for x in [i, j, k, l]):
            self._tors_result.setText("Enter valid atom indices (0-based)."); return

        n = self._get_atom_count()
        if n == 0:
            self._tors_result.setText("NO DATA LOADED"); return
        for label, idx in [("i", i), ("j", j), ("k", k), ("l", l)]:
            if not (0 <= idx < n):
                self._tors_result.setText(f"Index {label}={idx} out of range (0–{n-1}).")
                return
        if len({i, j, k, l}) < 4:
            self._tors_result.setText("Atoms i, j, k, l must all be different."); return

        pos = self._get_positions()
        if pos is None:
            self._tors_result.setText("Could not read current frame positions."); return

        try:
            from PSVAP.analysis.geometry import torsion
            meta = getattr(self.controller.model, 'metadata', None)
            src  = getattr(meta, 'source_path', None)
            is_lammps = src and src.suffix.lower() in {'.lammpstrj', '.traj', '.data', '.lammps'}
            t = torsion(pos[i], pos[j], pos[k], pos[l])
            warning = self._bond_warning(i, j, k, l)
            txt = f"TORSION  {i}—{j}—{k}—{l}  :  {t:.4f}°"
            if is_lammps:
                txt += "\n\nNOTE: Loaded file has 0 explicit angles/dihedrals in topology."
                txt += "\nThis torsion is computed from coordinates only."
            if warning:
                txt += f"\n\n⚠  {warning}"
            self._tors_result.setText(txt)
        except ImportError:
            b0 = pos[i] - pos[j]; b1 = pos[k] - pos[j]; b2 = pos[l] - pos[k]
            b1_norm = b1 / (np.linalg.norm(b1) + 1e-12)
            v = b0 - np.dot(b0, b1_norm) * b1_norm
            ww = b2 - np.dot(b2, b1_norm) * b1_norm
            t = float(np.degrees(np.arctan2(np.dot(np.cross(b1_norm, v), ww), np.dot(v, ww))))
            self._tors_result.setText(
                f"TORSION  atom {i} — {j} — {k} — {l}\n  φ = {t:.4f}°")
        except Exception as e:
            self._tors_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_ramachandran(self) -> None:
        try:
            from PSVAP.analysis.geometry import ramachandran
            atoms = self._get_atoms()
            traj  = self._get_trajectory()
            if not atoms:
                self._rama_result.setText("NO DATA LOADED"); return
            data = ramachandran(atoms, traj)
            if not data:
                self._rama_result.setText(
                    "NO BACKBONE ATOMS FOUND\n"
                    "(Requires N, CA, C atom names — LAMMPS traj files\n"
                    " don't have element names; load a PDB file instead)")
                return
            lines = [f"{'RESID':>6}  {'RESNAME':<6}  {'PHI(°)':>8}  {'PSI(°)':>8}"]
            lines.append("-" * 38)
            for rid, v in list(data.items())[:40]:
                phi = f"{v['phi'][0]:.1f}" if not np.isnan(v['phi'][0]) else "  N/A "
                psi = f"{v['psi'][0]:.1f}" if not np.isnan(v['psi'][0]) else "  N/A "
                lines.append(f"{rid:>6}  {v['resname']:<6}  {phi:>8}  {psi:>8}")
            if len(data) > 40:
                lines.append(f"... {len(data)-40} more residues")
            self._rama_result.setText("\n".join(lines))
        except Exception as e:
            self._rama_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_rmsd(self) -> None:
        try:
            from PSVAP.analysis.rmsd import rmsd_trajectory
            traj = self._get_trajectory()
            if not traj:
                self._rmsd_result.setText("NO DATA LOADED"); return
            ref = self._rmsd_ref.value()
            values = rmsd_trajectory(traj, reference_frame=ref)
            lines = [f"RMSD VS FRAME {ref}  ({len(values)} frames)\n"]
            step = max(1, len(values) // 20)
            for idx in range(0, len(values), step):
                lines.append(f"  frame {idx:>5} :  {values[idx]:.4f} Å")
            lines.append(f"\nMIN  {values.min():.4f} Å")
            lines.append(f"MEAN {values.mean():.4f} Å")
            lines.append(f"MAX  {values.max():.4f} Å")
            self._rmsd_result.setText("\n".join(lines))
        except Exception as e:
            self._rmsd_result.setText(f"ERROR: {e}")

    @Slot()
    def _compute_rmsd_all(self) -> None:
        ref_idx = self._rmsd_ref.value()
        try:
            model = self.controller.model
            n_frames = model.n_frames()
            if n_frames == 0:
                self._rmsd_all_result.setText("No file loaded."); return
            ref_pos = model.get_frame(ref_idx)
            if ref_pos is None:
                self._rmsd_all_result.setText(f"Frame {ref_idx} not found."); return
            ref = np.asarray(ref_pos, dtype=np.float64)
            try:
                from PSVAP.analysis.rmsd import rmsd as compute_rmsd
            except ImportError:
                def compute_rmsd(a, b):
                    diff = b - a
                    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            results = []
            max_frames = min(n_frames, 50)
            indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
            for fi in indices:
                pos = model.get_frame(int(fi))
                if pos is None: continue
                cur = np.asarray(pos, dtype=np.float64)
                if cur.shape == ref.shape:
                    r = compute_rmsd(ref, cur)
                    results.append(f"  frame {fi:5d}  →  {r:.4f} Å")
            if results:
                self._rmsd_all_result.setText(
                    f"RMSD vs frame {ref_idx} (showing {len(results)} frames):\n" +
                    "\n".join(results))
            else:
                self._rmsd_all_result.setText("No valid frames found.")
        except Exception as exc:
            self._rmsd_all_result.setText(f"ERROR: {exc}")

    @Slot()
    def _run_rmsf(self) -> None:
        try:
            from PSVAP.analysis.rmsd import rmsf, rmsf_per_residue
            atoms = self._get_atoms()
            traj  = self._get_trajectory()
            if not traj:
                self._rmsf_result.setText("NO DATA LOADED"); return
            data = rmsf_per_residue(traj, atoms)
            if not data:
                vals = rmsf(traj)
                lines = [f"RMSF PER ATOM  ({len(vals)} atoms)\n"]
                step = max(1, len(vals) // 20)
                for idx in range(0, len(vals), step):
                    lines.append(f"  atom {idx:>5} :  {vals[idx]:.4f} Å")
                self._rmsf_result.setText("\n".join(lines))
                return
            lines = [f"RMSF PER RESIDUE  ({len(data)} residues)\n"]
            for rid, val in list(sorted(data.items(), key=lambda x: -x[1]))[:30]:
                lines.append(f"  residue {rid:>5} :  {val:.4f} Å")
            self._rmsf_result.setText("\n".join(lines))
        except Exception as e:
            self._rmsf_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_alignment(self) -> None:
        try:
            from PSVAP.analysis.alignment import superimpose
            traj = self._get_trajectory()
            if len(traj) < 2:
                self._align_result.setText("NEED >= 2 FRAMES"); return
            ref_idx = self._align_ref.value(); mob_idx = self._align_mob.value()
            ref = traj[min(ref_idx, len(traj)-1)]
            mob = traj[min(mob_idx, len(traj)-1)]
            atom_sel = self._align_atoms.text().strip().lower()
            atoms = self._get_atoms()
            if atom_sel in {"", "all"}:
                indices = None
            elif "backbone" in atom_sel:
                indices = [idx for idx, a in enumerate(atoms)
                           if (getattr(a,'name',None) or "").strip().upper() in {"N","CA","C","O"}]
            elif "ca" in atom_sel:
                indices = [idx for idx, a in enumerate(atoms)
                           if (getattr(a,'name',None) or "").strip().upper() == "CA"]
            else:
                indices = None
            _, R, t, rmsd_val = superimpose(mob, ref, atom_indices=indices)
            txt = (f"SUPERIMPOSITION  frame {mob_idx} → frame {ref_idx}\n\n"
                   f"RMSD AFTER ALIGNMENT : {rmsd_val:.4f} Å\n\n"
                   f"ROTATION MATRIX R:\n"
                   f"  {R[0,0]:7.4f}  {R[0,1]:7.4f}  {R[0,2]:7.4f}\n"
                   f"  {R[1,0]:7.4f}  {R[1,1]:7.4f}  {R[1,2]:7.4f}\n"
                   f"  {R[2,0]:7.4f}  {R[2,1]:7.4f}  {R[2,2]:7.4f}\n\n"
                   f"TRANSLATION t : [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] Å")
            self._align_result.setText(txt)
        except Exception as e:
            self._align_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_rmsd_matrix(self) -> None:
        try:
            from PSVAP.analysis.alignment import rmsd_matrix
            traj = self._get_trajectory()
            if len(traj) < 2:
                self._matrix_result.setText("NEED >= 2 FRAMES"); return
            n = min(len(traj), 50)
            mat = rmsd_matrix(traj[:n], align_first=True)
            lines = [f"RMSD MATRIX  {n}×{n} frames\n"]
            pairs = [(mat[i,j], i, j) for i in range(n) for j in range(i+1,n)]
            pairs.sort(reverse=True)
            lines.append("TOP 10 MOST DIVERGENT FRAME PAIRS:")
            for d, i, j in pairs[:10]:
                lines.append(f"  frame {i:>4} vs frame {j:>4}  :  {d:.4f} Å")
            lines.append(f"\nMEAN PAIRWISE RMSD : {mat[mat>0].mean():.4f} Å")
            self._matrix_result.setText("\n".join(lines))
        except Exception as e:
            self._matrix_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_align_trajectory(self) -> None:
        ref_idx = self._align_ref.value()
        try:
            model = self.controller.model
            if model.n_frames() == 0:
                self._align_traj_result.setText("No file loaded."); return
            try:
                from PSVAP.analysis.alignment import align_trajectory
                n_aligned = align_trajectory(model, ref_idx)
                self._align_traj_result.setText(
                    f"ALIGNMENT COMPLETE\n"
                    f"  Reference frame: {ref_idx}\n"
                    f"  Frames aligned:  {n_aligned}\n"
                    f"  Algorithm: Kabsch (least-squares superposition)")
            except ImportError:
                self._align_traj_result.setText(
                    "align_trajectory not available.\n"
                    "Use SUPERIMPOSE above for frame-pair alignment.")
        except Exception as exc:
            self._align_traj_result.setText(f"ERROR: {exc}")

    @Slot()
    def _run_extract_sequences(self) -> None:
        try:
            from PSVAP.analysis.sequence import extract_sequence
            atoms = self._get_atoms()
            if not atoms:
                self._seq_extract_result.setText("NO DATA LOADED"); return
            has_elements = any(getattr(a, 'element', None) for a in atoms[:20])
            has_residues = any(getattr(a, 'residue_id', None) is not None for a in atoms[:20])
            if not has_elements and not has_residues:
                self._seq_extract_result.setText(
                    "LAMMPS files use numeric type IDs — no residue sequence.\n"
                    "Sequence extraction is available for PDB / GRO / mmCIF files.")
                return
            seqs = extract_sequence(atoms)
            if not seqs:
                self._seq_extract_result.setText(
                    "NO SEQUENCES FOUND\n"
                    "(Requires protein/nucleic acid with named residues.\n"
                    " Load a PDB file to extract sequences.)")
                return
            lines = []
            for chain, seq in seqs.items():
                lines.append(f"CHAIN {chain}  ({len(seq)} residues)")
                for k in range(0, len(seq), 60):
                    lines.append(f"  {k+1:>4}  {seq[k:k+60]}")
                if not self._seq1.text():
                    self._seq1.setText(seq)
            self._seq_extract_result.setText("\n".join(lines))
        except Exception as e:
            self._seq_extract_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_sequence_alignment(self) -> None:
        try:
            from PSVAP.analysis.sequence import align_pairwise
            s1 = self._seq1.text().strip().upper()
            s2 = self._seq2.text().strip().upper()
            if not s1 or not s2:
                self._seq_result.setText("ENTER BOTH SEQUENCES FIRST"); return
            mode = "local" if "LOCAL" in self._seq_mode.currentText() else "global"
            result = align_pairwise(s1, s2, mode=mode)
            txt = (f"MODE : {mode.upper()}  |  SCORE : {result.score:.1f}\n"
                   f"IDENTITY : {result.identity*100:.1f}%  "
                   f"SIMILARITY : {result.similarity*100:.1f}%\n\n"
                   f"SEQ 1 : {result.aligned_seq1[:80]}\n"
                   f"SEQ 2 : {result.aligned_seq2[:80]}")
            if len(result.aligned_seq1) > 80:
                txt += f"\n... ({len(result.aligned_seq1)} total aligned positions)"
            self._seq_result.setText(txt)
        except Exception as e:
            self._seq_result.setText(f"ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────────
    #  Compute slots — Phase 3 NEW
    # ──────────────────────────────────────────────────────────────────────

    @Slot()
    def _run_interactions(self) -> None:
        """Detect all non-covalent interactions between group A and group B."""
        try:
            from PSVAP.analysis.interactions import detect_all_interactions

            atoms = self._get_atoms()
            pos   = self._get_positions()
            n     = self._get_atom_count()

            if n == 0 or pos is None:
                self._inter_result.setText("NO DATA LOADED"); return

            group_a = self._parse_index_range(self._inter_group_a.text(), n)
            group_b = self._parse_index_range(self._inter_group_b.text(), n)

            if group_a is None:
                self._inter_result.setText(
                    f"GROUP A: invalid range.\n"
                    f"Use: '0-99', '0,1,2', or 'all'"); return
            if group_b is None:
                self._inter_result.setText(
                    f"GROUP B: invalid range.\n"
                    f"Use: '400-499', 'all'"); return

            if not group_a or not group_b:
                self._inter_result.setText(
                    "One or both groups are empty. Check index ranges."); return

            # Guard against computing interactions on LAMMPS files
            # (no element info → H-bond/salt bridge detection is meaningless)
            has_elements = any(
                getattr(atoms[i], "element", None) for i in group_a[:10]
            )

            result = detect_all_interactions(atoms, pos, group_a, group_b)

            lines = [
                f"INTERACTIONS  GROUP A ({len(group_a)} atoms) vs "
                f"GROUP B ({len(group_b)} atoms)\n",
                result.summary(),
                "",
            ]

            if not has_elements:
                lines.append(
                    "NOTE: No element symbols found. H-bond and salt bridge\n"
                    "detection requires element-labelled files (PDB/GRO/mmCIF).\n"
                    "Clash detection works on any format.\n"
                )

            if result.hbonds and self._cb_hbond.isChecked():
                lines.append(f"\nH-BONDS ({len(result.hbonds)}):")
                for h in result.hbonds[:20]:
                    lines.append(
                        f"  donor {h.donor_idx} → acceptor {h.acceptor_idx} "
                        f"  d={h.distance:.2f} Å"
                        + (f"  θ={h.angle:.1f}°" if not np.isnan(h.angle) else "")
                    )

            if result.salt_bridges and self._cb_salt.isChecked():
                lines.append(f"\nSALT BRIDGES ({len(result.salt_bridges)}):")
                for s in result.salt_bridges[:10]:
                    lines.append(
                        f"  (+){s.pos_idx} ↔ (-){s.neg_idx}  d={s.distance:.2f} Å"
                    )

            if result.clashes and self._cb_clash.isChecked():
                lines.append(f"\nCLASHES ({len(result.clashes)}):")
                for c in sorted(result.clashes, key=lambda x: x.overlap, reverse=True)[:10]:
                    lines.append(
                        f"  atom {c.idx_a} ↔ atom {c.idx_b}"
                        f"  d={c.distance:.2f} Å  overlap={c.overlap:.2f} Å"
                    )

            if result.hydrophobic and self._cb_hydro.isChecked():
                lines.append(f"\nHYDROPHOBIC CONTACTS ({len(result.hydrophobic)}):")
                for h in result.hydrophobic[:10]:
                    lines.append(f"  atom {h.idx_a} ↔ atom {h.idx_b}  d={h.distance:.2f} Å")

            if result.halogen_bonds and self._cb_halogen.isChecked():
                lines.append(f"\nHALOGEN BONDS ({len(result.halogen_bonds)}):")
                for h in result.halogen_bonds[:10]:
                    lines.append(
                        f"  halogen {h.halogen_idx} → acceptor {h.acceptor_idx}"
                        f"  d={h.distance:.2f} Å"
                    )

            if result.pi_stacks and self._cb_pistack.isChecked():
                lines.append(f"\nPI-STACKS ({len(result.pi_stacks)}):")
                for p in result.pi_stacks[:10]:
                    lines.append(f"  ring centroid distance: {p.distance:.2f} Å")

            self._inter_result.setText("\n".join(lines))

        except Exception as e:
            self._inter_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_interactions_trajectory(self) -> None:
        """Run interaction detection over all trajectory frames."""
        try:
            from PSVAP.analysis.interactions import interactions_over_trajectory

            atoms = self._get_atoms()
            traj  = self._get_trajectory()
            n     = self._get_atom_count()

            if not traj or n == 0:
                self._inter_traj_result.setText("NO DATA LOADED"); return

            group_a = self._parse_index_range(self._inter_group_a.text(), n)
            group_b = self._parse_index_range(self._inter_group_b.text(), n)

            if group_a is None or group_b is None:
                self._inter_traj_result.setText(
                    "Set valid GROUP A and GROUP B ranges first."); return

            # Limit to 50 frames for performance
            max_frames = min(len(traj), 50)
            step = max(1, len(traj) // max_frames)
            sub_traj = traj[::step][:max_frames]

            data = interactions_over_trajectory(atoms, sub_traj, group_a, group_b)

            from PSVAP.visualization.plot_renderer import PlotRenderer
            self._inter_traj_result.setText(
                PlotRenderer.interactions_to_text(data)
            )

        except Exception as e:
            self._inter_traj_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_sasa(self) -> None:
        """Compute SASA for the current frame."""
        try:
            from PSVAP.analysis.surface import compute_sasa, sasa_per_residue

            atoms = self._get_atoms()
            pos   = self._get_positions()

            if not atoms or pos is None:
                self._sasa_result.setText("NO DATA LOADED"); return

            # Parse probe radius
            try:
                probe = float(self._sasa_probe.text().strip())
            except ValueError:
                probe = 1.4

            # Warn if too many atoms (SASA is O(N²))
            n = len(atoms)
            if n > 2000:
                self._sasa_result.setText(
                    f"WARNING: {n} atoms — SASA is O(N²) and will be slow.\n"
                    f"Computing on first 500 atoms only for speed.\n"
                    f"Use a selection or load a smaller structure.\n\n"
                    f"Computing..."
                )
                atoms = atoms[:500]
                pos   = pos[:500]

            per_atom = compute_sasa(atoms, pos, probe_radius=probe)
            per_res  = sasa_per_residue(atoms, pos, probe_radius=probe)

            total_sasa = sum(per_atom.values())

            lines = [
                f"SASA  (probe radius = {probe:.2f} Å)\n",
                f"TOTAL SASA : {total_sasa:.2f} Å²",
                f"ATOMS      : {len(per_atom)}",
                f"RESIDUES   : {len(per_res)}",
                "",
                "TOP 20 MOST EXPOSED RESIDUES:",
                f"{'RESID':>8}  {'SASA (Å²)':>12}",
                "-" * 24,
            ]

            sorted_res = sorted(per_res.items(), key=lambda x: -x[1])
            for rid, val in sorted_res[:20]:
                lines.append(f"{rid:>8}  {val:>12.2f}")

            if len(sorted_res) > 20:
                lines.append(f"  ... {len(sorted_res)-20} more residues")

            self._sasa_result.setText("\n".join(lines))

        except Exception as e:
            self._sasa_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_patch_classification(self) -> None:
        """Classify surface patches by residue character."""
        try:
            from PSVAP.analysis.surface import classify_surface_patches

            atoms = self._get_atoms()
            pos   = self._get_positions()

            if not atoms or pos is None:
                self._patch_result.setText("NO DATA LOADED"); return

            # Check if resnames are available
            has_resnames = any(
                getattr(a, "resname", None) for a in atoms[:20]
            )
            if not has_resnames:
                self._patch_result.setText(
                    "PATCH CLASSIFICATION REQUIRES RESIDUE NAMES\n\n"
                    "This feature works with PDB, GRO, and mmCIF files.\n"
                    "LAMMPS files use numeric type IDs — no residue names available.")
                return

            n = len(atoms)
            if n > 2000:
                atoms = atoms[:500]
                pos   = pos[:500]

            try:
                probe = float(self._sasa_probe.text().strip())
            except ValueError:
                probe = 1.4

            patches = classify_surface_patches(atoms, pos, probe_radius=probe)

            if not patches:
                self._patch_result.setText(
                    "NO SURFACE PATCHES FOUND\n"
                    "(All residues are buried or structure has no named residues)")
                return

            from collections import Counter
            counts = Counter(patches.values())

            lines = [
                f"SURFACE PATCHES  ({len(patches)} exposed residues)\n",
                f"HYDROPHOBIC : {counts.get('hydrophobic', 0):>4} residues",
                f"POSITIVE    : {counts.get('positive', 0):>4} residues",
                f"NEGATIVE    : {counts.get('negative', 0):>4} residues",
                f"POLAR       : {counts.get('polar', 0):>4} residues",
                f"OTHER       : {counts.get('other', 0):>4} residues",
                "",
                "HYDROPHOBIC PATCHES:",
            ]

            hydro_res = [rid for rid, pt in patches.items() if pt == "hydrophobic"]
            if hydro_res:
                lines.append("  residues: " + ", ".join(str(r) for r in sorted(hydro_res)[:20]))
                if len(hydro_res) > 20:
                    lines.append(f"  ... {len(hydro_res)-20} more")
            else:
                lines.append("  none found")

            lines.append("\nCHARGED PATCHES:")
            pos_res = [rid for rid, pt in patches.items() if pt == "positive"]
            neg_res = [rid for rid, pt in patches.items() if pt == "negative"]
            if pos_res:
                lines.append("  (+): " + ", ".join(str(r) for r in sorted(pos_res)[:15]))
            if neg_res:
                lines.append("  (-): " + ", ".join(str(r) for r in sorted(neg_res)[:15]))

            self._patch_result.setText("\n".join(lines))

        except Exception as e:
            self._patch_result.setText(f"ERROR: {e}")

# ── Ligand / MCS tab (Phase 5 — Feature 5) ────────────────────────────

    def _build_ligand_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("MAXIMUM COMMON SUBSTRUCTURE (MCS)"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Find the largest shared scaffold among a set of molecules. "
            "Enter one SMILES per line.",
            hint=True))

        layout.addWidget(_lbl("SMILES  (one per line):", dim=True))
        self._mcs_input = QTextEdit()
        self._mcs_input.setFixedHeight(100)
        self._mcs_input.setPlaceholderText(
            "CC(=O)Nc1ccc(O)cc1\nc1ccc(NC(=O)C)cc1\nCC(=O)Nc1cccc(O)c1")
        self._mcs_input.setStyleSheet(
            f"QTextEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; font-family:{MONO}; font-size:10px; padding:6px; }}")
        layout.addWidget(self._mcs_input)

        self._mcs_btn = _btn("FIND MCS")
        layout.addWidget(self._mcs_btn)
        self._mcs_result = _result_box(140)
        layout.addWidget(self._mcs_result)
        self._mcs_btn.clicked.connect(self._run_mcs)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("FINGERPRINT CLUSTERING", dim=True))
        nc_row = QHBoxLayout(); nc_row.setSpacing(8)
        nc_row.addWidget(_lbl("CLUSTERS:", dim=True))
        self._mcs_n_clusters = QSpinBox()
        self._mcs_n_clusters.setRange(2, 20); self._mcs_n_clusters.setValue(5)
        self._mcs_n_clusters.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._mcs_n_clusters.setFixedWidth(70); nc_row.addWidget(self._mcs_n_clusters)
        nc_row.addStretch(); layout.addLayout(nc_row)

        self._cluster_btn = _btn("CLUSTER MOLECULES")
        layout.addWidget(self._cluster_btn)
        self._cluster_result = _result_box(120)
        layout.addWidget(self._cluster_result)
        self._cluster_btn.clicked.connect(self._run_cluster)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── MMP tab (Phase 5 — Feature 6) ─────────────────────────────────────

    def _build_mmp_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("MATCHED MOLECULAR PAIR ANALYSIS"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Find molecule pairs differing by a single structural transformation. "
            "Identifies which chemical changes affect properties.",
            hint=True))

        layout.addWidget(_lbl("SMILES  (one per line):", dim=True))
        self._mmp_input = QTextEdit()
        self._mmp_input.setFixedHeight(100)
        self._mmp_input.setStyleSheet(
            f"QTextEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; font-family:{MONO}; font-size:10px; padding:6px; }}")
        layout.addWidget(self._mmp_input)

        hr_row = QHBoxLayout(); hr_row.setSpacing(8)
        hr_row.addWidget(_lbl("MAX FRAGMENT RATIO:", dim=True))
        self._mmp_ratio = QLineEdit("0.33")
        self._mmp_ratio.setFixedWidth(60)
        self._mmp_ratio.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        hr_row.addWidget(self._mmp_ratio); hr_row.addStretch()
        layout.addLayout(hr_row)

        self._mmp_btn = _btn("FIND MATCHED PAIRS")
        layout.addWidget(self._mmp_btn)
        self._mmp_result = _result_box(200)
        layout.addWidget(self._mmp_result)
        self._mmp_btn.clicked.connect(self._run_mmp)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Pharmacophore tab (Phase 5 — Feature 8) ───────────────────────────

    def _build_pharma_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("PHARMACOPHORE ANALYSIS"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Extract 3D pharmacophoric features from loaded structure: "
            "H-bond donors/acceptors, hydrophobic regions, charges, aromatic rings.",
            hint=True))

        layout.addWidget(_lbl("ATOM SELECTION  (index range or 'all'):", dim=True))
        self._pharma_sel = _input_line("all  or  0-499")
        layout.addWidget(self._pharma_sel)

        self._pharma_btn = _btn("EXTRACT PHARMACOPHORE")
        layout.addWidget(self._pharma_btn)
        self._pharma_result = _result_box(220)
        layout.addWidget(self._pharma_result)
        self._pharma_btn.clicked.connect(self._run_pharmacophore)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── QSAR tab (Phase 5 — Feature 15) ───────────────────────────────────

    def _build_qsar_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("QSAR ANALYSIS"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Compute molecular descriptors and build QSAR models. "
            "Enter SMILES and comma-separated activity values.",
            hint=True))

        layout.addWidget(_lbl("SMILES  (one per line):", dim=True))
        self._qsar_smiles = QTextEdit()
        self._qsar_smiles.setFixedHeight(80)
        self._qsar_smiles.setStyleSheet(
            f"QTextEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; font-family:{MONO}; font-size:10px; padding:6px; }}")
        layout.addWidget(self._qsar_smiles)

        layout.addWidget(_lbl("ACTIVITY VALUES  (comma separated, same order):", dim=True))
        self._qsar_activities = _input_line("6.2, 7.1, 5.8, 8.3, ...")
        layout.addWidget(self._qsar_activities)

        mg = QHBoxLayout(); mg.setSpacing(8)
        mg.addWidget(_lbl("MODEL:", dim=True))
        self._qsar_model_type = QComboBox()
        self._qsar_model_type.addItems(["RANDOM FOREST", "LINEAR (RIDGE)", "SVM"])
        self._qsar_model_type.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        mg.addWidget(self._qsar_model_type, stretch=1); layout.addLayout(mg)

        self._qsar_btn = _btn("COMPUTE DESCRIPTORS + BUILD MODEL")
        layout.addWidget(self._qsar_btn)
        self._qsar_result = _result_box(160)
        layout.addWidget(self._qsar_result)
        self._qsar_btn.clicked.connect(self._run_qsar)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("PREDICT ACTIVITY FOR NEW SMILES:", dim=True))
        self._qsar_predict_input = _input_line("Enter SMILES to predict")
        layout.addWidget(self._qsar_predict_input)
        self._qsar_predict_btn = _btn("PREDICT")
        layout.addWidget(self._qsar_predict_btn)
        self._qsar_predict_result = _result_box(80)
        layout.addWidget(self._qsar_predict_result)
        self._qsar_predict_btn.clicked.connect(self._run_qsar_predict)
        self._qsar_model_result = None  # stores last QSARResult

        layout.addStretch()
        w.setWidget(inner); return w

    # ── pKa tab (Phase 5 — Feature 16) ────────────────────────────────────

    def _build_pka_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("pKa ANALYSIS"))
        layout.addWidget(_divider())

        ph_row = QHBoxLayout(); ph_row.setSpacing(8)
        ph_row.addWidget(_lbl("REFERENCE pH:", dim=True))
        self._pka_ph = QLineEdit("7.4")
        self._pka_ph.setFixedWidth(60)
        self._pka_ph.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        ph_row.addWidget(self._pka_ph); ph_row.addStretch()
        layout.addLayout(ph_row)

        layout.addWidget(_lbl("PROPKA3 (if installed — requires PDB file):", dim=True))
        pdb_row = QHBoxLayout(); pdb_row.setSpacing(8)
        self._pka_pdb = _input_line("path/to/structure.pdb")
        self._pka_browse = _btn("BROWSE")
        pdb_row.addWidget(self._pka_pdb, stretch=1)
        pdb_row.addWidget(self._pka_browse)
        layout.addLayout(pdb_row)

        self._pka_propka_btn = _btn("RUN PROPKA3")
        layout.addWidget(self._pka_propka_btn)
        self._pka_propka_result = _result_box(160)
        layout.addWidget(self._pka_propka_result)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("FALLBACK ESTIMATE (from loaded structure):", dim=True))
        self._pka_estimate_btn = _btn("ESTIMATE FROM RESIDUES")
        layout.addWidget(self._pka_estimate_btn)
        self._pka_estimate_result = _result_box(160)
        layout.addWidget(self._pka_estimate_result)

        self._pka_browse.clicked.connect(self._browse_pka_pdb)
        self._pka_propka_btn.clicked.connect(self._run_propka)
        self._pka_estimate_btn.clicked.connect(self._run_pka_estimate)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Conformational search tab (Phase 5 — Feature 23) ──────────────────

    def _build_conform_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("CONFORMATIONAL SEARCH"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Generate a diverse ensemble of low-energy 3D conformations "
            "for a small molecule using ETKDG + MMFF94 minimisation.",
            hint=True))

        layout.addWidget(_lbl("SMILES:", dim=True))
        self._conf_smiles = _input_line("e.g. CC(=O)Nc1ccc(O)cc1")
        layout.addWidget(self._conf_smiles)

        cg = QGridLayout(); cg.setSpacing(8)
        cg.addWidget(_lbl("Conformers to generate:", dim=True), 0, 0)
        self._conf_n = QSpinBox()
        self._conf_n.setRange(1, 500); self._conf_n.setValue(100)
        self._conf_n.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._conf_n.setFixedWidth(80); cg.addWidget(self._conf_n, 0, 1)

        cg.addWidget(_lbl("Energy window (kcal/mol):", dim=True), 1, 0)
        self._conf_ew = QLineEdit("10.0")
        self._conf_ew.setFixedWidth(70)
        self._conf_ew.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        cg.addWidget(self._conf_ew, 1, 1)

        cg.addWidget(_lbl("Min RMSD (Å):", dim=True), 2, 0)
        self._conf_rmsd = QLineEdit("0.5")
        self._conf_rmsd.setFixedWidth(70)
        self._conf_rmsd.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        cg.addWidget(self._conf_rmsd, 2, 1)
        layout.addLayout(cg)

        self._conf_btn = _btn("GENERATE CONFORMERS")
        layout.addWidget(self._conf_btn)
        self._conf_result = _result_box(200)
        layout.addWidget(self._conf_result)
        self._conf_btn.clicked.connect(self._run_conformers)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Phase 5 compute slots ──────────────────────────────────────────────

    @Slot()
    def _run_mcs(self) -> None:
        try:
            from PSVAP.analysis.clustering import find_mcs
            text = self._mcs_input.toPlainText().strip()
            smiles = [s.strip() for s in text.splitlines() if s.strip()]
            if len(smiles) < 2:
                self._mcs_result.setText("Enter at least 2 SMILES."); return
            result = find_mcs(smiles)
            lines = [
                f"MCS RESULT  ({result.n_molecules} molecules)\n",
                f"SMARTS  : {result.smarts or '(none found)'}",
                f"ATOMS   : {result.n_atoms}",
                f"BONDS   : {result.n_bonds}",
            ]
            if result.timed_out:
                lines.append("\nWARNING: Search timed out — result may be suboptimal.")
            if result.atom_map:
                lines.append("\nATOM MAPPINGS (first 5 molecules):")
                for i, mapping in enumerate(result.atom_map[:5]):
                    lines.append(f"  mol {i+1}: {mapping[:10]}"
                                 + ("..." if len(mapping) > 10 else ""))
            self._mcs_result.setText("\n".join(lines))
        except ImportError:
            self._mcs_result.setText(
                "RDKit required for MCS.\n"
                "Install: conda install -c conda-forge rdkit")
        except Exception as e:
            self._mcs_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_cluster(self) -> None:
        try:
            from PSVAP.analysis.clustering import cluster_by_fingerprint
            text   = self._mcs_input.toPlainText().strip()
            smiles = [s.strip() for s in text.splitlines() if s.strip()]
            if len(smiles) < 2:
                self._cluster_result.setText(
                    "Enter SMILES in the LIGAND tab input box above."); return
            n_cl   = self._mcs_n_clusters.value()
            result = cluster_by_fingerprint(smiles, n_clusters=n_cl)
            lines  = [f"FINGERPRINT CLUSTERS  ({len(result)} clusters)\n"]
            for cid, idx_list in sorted(result.items()):
                lines.append(f"  Cluster {cid+1}  ({len(idx_list)} molecules):")
                for i in idx_list[:5]:
                    lines.append(f"    {smiles[i][:50]}")
                if len(idx_list) > 5:
                    lines.append(f"    ... {len(idx_list)-5} more")
            self._cluster_result.setText("\n".join(lines))
        except ImportError:
            self._cluster_result.setText(
                "RDKit + scikit-learn required for clustering.")
        except Exception as e:
            self._cluster_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_mmp(self) -> None:
        try:
            from PSVAP.analysis.qsar import find_matched_pairs
            text   = self._mmp_input.toPlainText().strip()
            smiles = [s.strip() for s in text.splitlines() if s.strip()]
            if len(smiles) < 2:
                self._mmp_result.setText("Enter at least 2 SMILES."); return
            try:
                ratio = float(self._mmp_ratio.text().strip())
            except ValueError:
                ratio = 0.33
            pairs = find_matched_pairs(smiles, max_heavy_ratio=ratio)
            if not pairs:
                self._mmp_result.setText(
                    f"NO MATCHED PAIRS FOUND\n"
                    f"({len(smiles)} molecules, ratio cutoff {ratio:.2f})\n\n"
                    "Try increasing the max fragment ratio or add more molecules.")
                return
            lines = [f"MATCHED MOLECULAR PAIRS  ({len(pairs)} pairs)\n",
                     f"{'MOL A':>5}  {'MOL B':>5}  {'RATIO':>6}  TRANSFORM"]
            for p in pairs[:30]:
                lines.append(
                    f"{p.index_a+1:>5}  {p.index_b+1:>5}  "
                    f"{p.heavy_ratio:>6.3f}  "
                    f"{p.transform_a[:20]} → {p.transform_b[:20]}")
            if len(pairs) > 30:
                lines.append(f"... {len(pairs)-30} more pairs")
            self._mmp_result.setText("\n".join(lines))
        except ImportError:
            self._mmp_result.setText("RDKit required for MMP analysis.")
        except Exception as e:
            self._mmp_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_pharmacophore(self) -> None:
        try:
            from PSVAP.analysis.pharmacophore import (
                extract_pharmacophore, summarise_pharmacophore
            )
            atoms = self._get_atoms()
            pos   = self._get_positions()
            n     = self._get_atom_count()
            if not atoms or pos is None:
                self._pharma_result.setText("NO DATA LOADED"); return

            sel_text = self._pharma_sel.text().strip()
            if not sel_text or sel_text.lower() == "all":
                indices = None
            else:
                indices = self._parse_index_range(sel_text, n)

            features = extract_pharmacophore(atoms, pos, indices=indices)
            self._pharma_result.setText(summarise_pharmacophore(features))
        except Exception as e:
            self._pharma_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_qsar(self) -> None:
        try:
            from PSVAP.analysis.qsar import compute_descriptors, build_qsar_model
            smiles_text = self._qsar_smiles.toPlainText().strip()
            smiles = [s.strip() for s in smiles_text.splitlines() if s.strip()]
            if not smiles:
                self._qsar_result.setText("Enter SMILES first."); return

            # Descriptor computation only (no model if no activities)
            act_text = self._qsar_activities.text().strip()
            if not act_text:
                descs = compute_descriptors(smiles)
                lines = [f"DESCRIPTORS  ({len(smiles)} molecules)\n"]
                for smi, d in list(descs.items())[:5]:
                    lines.append(f"  {smi[:30]}")
                    for k, v in list(d.items())[:6]:
                        lines.append(f"    {k}: {v:.4f}")
                self._qsar_result.setText("\n".join(lines)); return

            # Parse activities
            try:
                activities = [float(x.strip()) for x in act_text.split(",") if x.strip()]
            except ValueError:
                self._qsar_result.setText(
                    "Could not parse activity values.\n"
                    "Use comma-separated numbers: 6.2, 7.1, 5.8"); return

            if len(activities) != len(smiles):
                self._qsar_result.setText(
                    f"Mismatch: {len(smiles)} SMILES but {len(activities)} activities."); return

            mt_map = {
                "RANDOM FOREST": "random_forest",
                "LINEAR (RIDGE)": "linear",
                "SVM": "svm",
            }
            mt = mt_map.get(self._qsar_model_type.currentText(), "random_forest")
            result = build_qsar_model(smiles, activities, model_type=mt)
            self._qsar_model_result = result

            lines = [
                f"QSAR MODEL  ({result.model_type})\n",
                f"  Training molecules : {result.n_train}",
                f"  R² (train)         : {result.r2_train:.4f}",
                f"  R² (CV)            : {result.r2_cv:.4f}",
                f"  RMSE (CV)          : {result.rmse_cv:.4f}",
            ]
            self._qsar_result.setText("\n".join(lines))
        except ImportError:
            self._qsar_result.setText(
                "RDKit + scikit-learn required for QSAR.\n"
                "Install: conda install -c conda-forge rdkit\n"
                "         pip install scikit-learn")
        except Exception as e:
            self._qsar_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_qsar_predict(self) -> None:
        if self._qsar_model_result is None:
            self._qsar_predict_result.setText(
                "Build a QSAR model first."); return
        try:
            from PSVAP.analysis.qsar import predict_activity
            smi = self._qsar_predict_input.text().strip()
            if not smi:
                self._qsar_predict_result.setText(
                    "Enter a SMILES string."); return
            preds = predict_activity(self._qsar_model_result, [smi])
            pred  = preds[0]
            if pred is None:
                self._qsar_predict_result.setText(
                    f"Could not compute descriptors for:\n{smi}")
            else:
                self._qsar_predict_result.setText(
                    f"PREDICTED ACTIVITY\n\n"
                    f"  SMILES     : {smi[:60]}\n"
                    f"  Predicted  : {pred:.4f}")
        except Exception as e:
            self._qsar_predict_result.setText(f"ERROR: {e}")

    @Slot()
    def _browse_pka_pdb(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PDB File", "",
            "PDB Files (*.pdb);;All Files (*)"
        )
        if path:
            self._pka_pdb.setText(path)

    @Slot()
    def _run_propka(self) -> None:
        try:
            from PSVAP.analysis.pka import run_propka, format_pka_results
            pdb  = self._pka_pdb.text().strip()
            if not pdb:
                self._pka_propka_result.setText(
                    "Enter or browse to a PDB file path."); return
            try:
                ph = float(self._pka_ph.text().strip())
            except ValueError:
                ph = 7.4
            results = run_propka(pdb, ph=ph)
            self._pka_propka_result.setText(format_pka_results(results, ph=ph))
        except FileNotFoundError as e:
            self._pka_propka_result.setText(
                f"PROPKA NOT FOUND\n\n{e}\n\n"
                "Use ESTIMATE FROM RESIDUES as a fallback.")
        except Exception as e:
            self._pka_propka_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_pka_estimate(self) -> None:
        try:
            from PSVAP.analysis.pka import estimate_pka_from_residues, format_pka_results
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._pka_estimate_result.setText("NO DATA LOADED"); return
            has_resnames = any(getattr(a, "resname", None) for a in atoms[:20])
            if not has_resnames:
                self._pka_estimate_result.setText(
                    "NO RESIDUE NAMES FOUND\n\n"
                    "pKa estimation requires named residues.\n"
                    "Load a PDB or GRO file."); return
            try:
                ph = float(self._pka_ph.text().strip())
            except ValueError:
                ph = 7.4
            results = estimate_pka_from_residues(atoms, pos)
            if not results:
                self._pka_estimate_result.setText(
                    "No ionisable residues found.\n"
                    "(ARG, LYS, ASP, GLU, HIS, CYS, TYR)"); return
            self._pka_estimate_result.setText(format_pka_results(results, ph=ph))
        except Exception as e:
            self._pka_estimate_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_conformers(self) -> None:
        try:
            from PSVAP.analysis.conformational_search import (
                generate_conformers, format_conformer_result
            )
            smi = self._conf_smiles.text().strip()
            if not smi:
                self._conf_result.setText("Enter a SMILES string."); return
            n_conf = self._conf_n.value()
            try:
                ew = float(self._conf_ew.text().strip())
            except ValueError:
                ew = 10.0
            try:
                min_rmsd = float(self._conf_rmsd.text().strip())
            except ValueError:
                min_rmsd = 0.5
            result = generate_conformers(
                smi, n_conformers=n_conf,
                energy_window=ew, min_rmsd=min_rmsd,
            )
            self._conf_result.setText(format_conformer_result(result))
        except ImportError:
            self._conf_result.setText(
                "RDKit required for conformational search.\n"
                "Install: conda install -c conda-forge rdkit")
        except Exception as e:
            self._conf_result.setText(f"ERROR: {e}")

# ── Site Finder tab (Phase 6 — Feature 19) ────────────────────────────

    def _build_sites_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("BINDING SITE DETECTION"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Detects potential ligand-binding pockets using a grid-based "
            "algorithm. Use fpocket for research-quality results.",
            hint=True))

        sg = QGridLayout(); sg.setSpacing(8)
        sg.addWidget(_lbl("GRID SPACING (Å):", dim=True), 0, 0)
        self._sites_spacing = _input_line("1.0")
        self._sites_spacing.setMaximumWidth(70)
        sg.addWidget(self._sites_spacing, 0, 1)

        sg.addWidget(_lbl("MIN VOLUME (Å³):", dim=True), 1, 0)
        self._sites_min_vol = _input_line("100.0")
        self._sites_min_vol.setMaximumWidth(70)
        sg.addWidget(self._sites_min_vol, 1, 1)
        layout.addLayout(sg)

        self._sites_btn = _btn("DETECT SITES (GRID)")
        layout.addWidget(self._sites_btn)
        self._sites_result = _result_box(220)
        layout.addWidget(self._sites_result)
        self._sites_btn.clicked.connect(self._run_site_finder)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("FPOCKET INTEGRATION (requires fpocket in PATH)", dim=True))
        pdb_row = QHBoxLayout(); pdb_row.setSpacing(8)
        self._sites_pdb = _input_line("path/to/structure.pdb")
        self._sites_browse = _btn("BROWSE")
        pdb_row.addWidget(self._sites_pdb, stretch=1)
        pdb_row.addWidget(self._sites_browse)
        layout.addLayout(pdb_row)
        self._sites_fpocket_btn = _btn("RUN FPOCKET")
        layout.addWidget(self._sites_fpocket_btn)
        self._sites_fpocket_result = _result_box(120)
        layout.addWidget(self._sites_fpocket_result)

        self._sites_browse.clicked.connect(self._browse_sites_pdb)
        self._sites_fpocket_btn.clicked.connect(self._run_fpocket)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Water Map tab (Phase 6 — Feature 20) ──────────────────────────────

    def _build_watermap_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("WATER MAP ANALYSIS"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Computes spatial density of water molecules across the trajectory. "
            "High-density sites may be important for binding.",
            hint=True))

        wg = QGridLayout(); wg.setSpacing(8)
        wg.addWidget(_lbl("GRID SPACING (Å):", dim=True), 0, 0)
        self._water_spacing = _input_line("0.5")
        self._water_spacing.setMaximumWidth(70)
        wg.addWidget(self._water_spacing, 0, 1)

        wg.addWidget(_lbl("CONTOUR LEVEL (σ):", dim=True), 1, 0)
        self._water_contour = _input_line("2.0")
        self._water_contour.setMaximumWidth(70)
        wg.addWidget(self._water_contour, 1, 1)
        layout.addLayout(wg)

        self._water_btn = _btn("COMPUTE WATER DENSITY")
        layout.addWidget(self._water_btn)
        self._water_result = _result_box(200)
        layout.addWidget(self._water_result)
        self._water_btn.clicked.connect(self._run_water_map)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Trajectory Clustering tab (Phase 6 — Feature 22) ──────────────────

    def _build_cluster_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner); layout.setContentsMargins(16,16,16,16); layout.setSpacing(14)

        layout.addWidget(_lbl("TRAJECTORY CLUSTERING"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Groups trajectory frames into conformational clusters. "
            "Returns representative (medoid) frame per cluster.",
            hint=True))

        cg = QGridLayout(); cg.setSpacing(8)
        cg.addWidget(_lbl("CLUSTERS:", dim=True), 0, 0)
        self._clust_n = QSpinBox()
        self._clust_n.setRange(2, 20); self._clust_n.setValue(5)
        self._clust_n.setStyleSheet(
            f"QSpinBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        self._clust_n.setFixedWidth(70); cg.addWidget(self._clust_n, 0, 1)

        cg.addWidget(_lbl("METHOD:", dim=True), 1, 0)
        self._clust_method = QComboBox()
        self._clust_method.addItems(["K-MEANS", "HIERARCHICAL"])
        self._clust_method.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        cg.addWidget(self._clust_method, 1, 1)

        cg.addWidget(_lbl("ATOM SUBSET:", dim=True), 2, 0)
        self._clust_atoms = _input_line("all  or  0-99  (CA only)")
        cg.addWidget(self._clust_atoms, 2, 1)
        layout.addLayout(cg)

        self._clust_btn = _btn("RUN CLUSTERING")
        layout.addWidget(self._clust_btn)
        self._clust_result = _result_box(220)
        layout.addWidget(self._clust_result)
        self._clust_btn.clicked.connect(self._run_trajectory_clustering)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Phase 6 compute slots ──────────────────────────────────────────────

    @Slot()
    def _run_site_finder(self) -> None:
        try:
            from PSVAP.analysis.site_finder import find_sites_grid, format_sites
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._sites_result.setText("NO DATA LOADED"); return
            try:
                spacing = float(self._sites_spacing.text().strip())
            except ValueError:
                spacing = 1.0
            try:
                min_vol = float(self._sites_min_vol.text().strip())
            except ValueError:
                min_vol = 100.0
            if len(atoms) > 3000:
                self._sites_result.setText(
                    f"WARNING: {len(atoms)} atoms — grid detection may be slow.\n"
                    "Computing on first 2000 atoms...\n")
                atoms = atoms[:2000]
                pos   = pos[:2000]
            sites = find_sites_grid(
                atoms, pos,
                grid_spacing=spacing,
                min_pocket_volume=min_vol,
            )
            self._sites_result.setText(format_sites(sites))
        except Exception as e:
            self._sites_result.setText(f"ERROR: {e}")

    @Slot()
    def _browse_sites_pdb(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PDB File", "", "PDB Files (*.pdb);;All Files (*)")
        if path:
            self._sites_pdb.setText(path)

    @Slot()
    def _run_fpocket(self) -> None:
        try:
            from PSVAP.analysis.site_finder import (
                find_sites_fpocket, format_sites, check_fpocket_available
            )
            pdb = self._sites_pdb.text().strip()
            if not pdb:
                self._sites_fpocket_result.setText(
                    "Browse to a PDB file first."); return
            if not check_fpocket_available():
                self._sites_fpocket_result.setText(
                    "FPOCKET NOT FOUND IN PATH\n\n"
                    "Install: apt install fpocket\n"
                    "         brew install fpocket\n"
                    "Use GRID method as fallback.")
                return
            sites = find_sites_fpocket(pdb)
            self._sites_fpocket_result.setText(format_sites(sites))
        except Exception as e:
            self._sites_fpocket_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_water_map(self) -> None:
        try:
            from PSVAP.analysis.surface import compute_water_density, format_water_map
            atoms = self._get_atoms()
            traj  = self._get_trajectory()
            if not atoms or not traj:
                self._water_result.setText("NO DATA LOADED"); return
            has_water = any(
                (getattr(a, "resname", None) or "").upper()
                in {"HOH", "WAT", "TIP3", "SOL"}
                for a in atoms
            )
            if not has_water:
                self._water_result.setText(
                    "NO WATER MOLECULES FOUND\n\n"
                    "Water map analysis requires a solvated structure.\n"
                    "Use the SOLVATE tab or load a pre-solvated file.")
                return
            try:
                spacing = float(self._water_spacing.text().strip())
            except ValueError:
                spacing = 0.5
            try:
                contour = float(self._water_contour.text().strip())
            except ValueError:
                contour = 2.0
            data = compute_water_density(
                atoms, traj,
                grid_spacing=spacing,
                contour_level=contour,
            )
            self._water_result.setText(format_water_map(data))
        except Exception as e:
            self._water_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_trajectory_clustering(self) -> None:
        try:
            from PSVAP.analysis.clustering import (
                cluster_trajectory, format_cluster_result
            )
            traj  = self._get_trajectory()
            atoms = self._get_atoms()
            n     = self._get_atom_count()
            if not traj:
                self._clust_result.setText("NO DATA LOADED"); return
            if len(traj) < 2:
                self._clust_result.setText(
                    "Need at least 2 trajectory frames."); return
            n_clusters = self._clust_n.value()
            method = (
                "hierarchical"
                if "HIER" in self._clust_method.currentText().upper()
                else "kmeans"
            )
            sel_text = self._clust_atoms.text().strip()
            if not sel_text or sel_text.lower() in {"all", ""}:
                atom_indices = None
            else:
                atom_indices = self._parse_index_range(sel_text, n)
            result = cluster_trajectory(
                traj,
                atom_indices=atom_indices,
                n_clusters=n_clusters,
                method=method,
            )
            self._clust_result.setText(format_cluster_result(result))
        except ImportError:
            self._clust_result.setText(
                "scikit-learn required for clustering.\n"
                "Install: pip install scikit-learn")
        except Exception as e:
            self._clust_result.setText(f"ERROR: {e}")