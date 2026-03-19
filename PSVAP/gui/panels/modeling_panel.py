"""
gui/panels/modeling_panel.py
-----------------------------
Phase 4 Modeling Panel — Structure Preparation, Mutation, Alanine Scan,
and Solvation tools.

Features implemented
--------------------
  Feature 7  : Amino acid mutations (MUTATE tab)
  Feature 11 : Alanine scanning (ALA SCAN tab)
  Feature 17 : Structure preparation / QC (PREP tab)
  Feature 10 : Solvent box construction (SOLVATE tab)
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget, QCheckBox,
    QFileDialog,
)

from PSVAP.app.controller import ApplicationController

# ── Theme (matches rest of UI) ─────────────────────────────────────────────
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
            f"QPushButton:disabled {{ color:{TEXT_HINT}; border-color:{TEXT_HINT}; }}"
        )
    return b


def _safe_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return None


class ModelingPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self._last_mutation_atoms    = None
        self._last_mutation_positions = None
        self._build()

    # ── Safe accessors ─────────────────────────────────────────────────────

    def _get_atoms(self) -> list:
        return getattr(self.controller.model, 'atoms', [])

    def _get_positions(self) -> "np.ndarray | None":
        try:
            model = self.controller.model
            frame = model.get_frame(getattr(model, '_current_frame', 0))
            return np.asarray(frame, dtype=np.float64) if frame is not None else None
        except Exception:
            return None

    def _get_atom_count(self) -> int:
        return len(self._get_atoms())

    def _parse_index_range(self, text: str, n_atoms: int) -> list[int] | None:
        text = text.strip()
        if not text or text.lower() == "all":
            return list(range(n_atoms))
        if "-" in text and "," not in text:
            parts = text.split("-")
            if len(parts) == 2:
                try:
                    s, e = int(parts[0].strip()), int(parts[1].strip())
                    return [i for i in range(s, e + 1) if i < n_atoms]
                except ValueError:
                    return None
        if "," in text:
            try:
                return [int(x.strip()) for x in text.split(",")
                        if x.strip() and int(x.strip()) < n_atoms]
            except ValueError:
                return None
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
            f"QTabWidget::pane {{ border:none; border-top:1px solid {BORDER}; "
            f"background:{BG}; }}"
            f"QTabBar {{ background:{BG}; }}"
            f"QTabBar::tab {{ background:{BG}; color:{TEXT_HINT}; font-size:8px; "
            f"letter-spacing:2px; padding:8px 12px; border:none; "
            f"border-bottom:2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; "
            f"background:{PANEL_ALT}; }}"
            f"QTabBar::tab:hover:!selected {{ color:{TEXT_DIM}; background:{PANEL_ALT}; }}"
        )
        tabs.addTab(self._build_prep_tab(),    "PREP")
        tabs.addTab(self._build_mutate_tab(),  "MUTATE")
        tabs.addTab(self._build_scan_tab(),    "ALA SCAN")
        tabs.addTab(self._build_solvate_tab(), "SOLVATE")
        tabs.addTab(self._build_mdsetup_tab(), "MD SETUP")
        tabs.addTab(self._build_cg_tab(),      "CG")
        layout.addWidget(tabs)

    # ── PREP tab ───────────────────────────────────────────────────────────

    def _build_prep_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("STRUCTURE PREPARATION"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl("STRUCTURE CHECK & REPORT", dim=True))
        self._prep_check_btn = _btn("RUN STRUCTURE CHECK")
        layout.addWidget(self._prep_check_btn)
        self._prep_report = _result_box(200)
        layout.addWidget(self._prep_report)
        self._prep_check_btn.clicked.connect(self._run_structure_check)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("CLEAN STRUCTURE", dim=True))

        cb_row = QHBoxLayout(); cb_row.setSpacing(12)
        self._prep_rm_water  = self._make_checkbox("REMOVE WATERS",  checked=True)
        self._prep_rm_hetatm = self._make_checkbox("REMOVE HETATM",  checked=False)
        self._prep_cap       = self._make_checkbox("CAP TERMINI",    checked=False)
        cb_row.addWidget(self._prep_rm_water)
        cb_row.addWidget(self._prep_rm_hetatm)
        cb_row.addWidget(self._prep_cap)
        cb_row.addStretch()
        layout.addLayout(cb_row)

        renumber_row = QHBoxLayout(); renumber_row.setSpacing(8)
        self._prep_renumber = self._make_checkbox("RENUMBER RESIDUES", checked=False)
        renumber_row.addWidget(self._prep_renumber)
        renumber_row.addStretch()
        layout.addLayout(renumber_row)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self._prep_apply_btn = _btn("APPLY CLEANING")
        self._prep_save_btn  = _btn("SAVE AS PDB")
        btn_row.addWidget(self._prep_apply_btn)
        btn_row.addWidget(self._prep_save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._prep_result = _result_box(100)
        layout.addWidget(self._prep_result)
        self._prep_apply_btn.clicked.connect(self._run_structure_clean)
        self._prep_save_btn.clicked.connect(self._run_save_prep_pdb)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── MUTATE tab ─────────────────────────────────────────────────────────

    def _build_mutate_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("POINT MUTATION"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl(
            "Mutates a residue in the loaded structure. "
            "Backbone is preserved; side chains are removed (backbone-only mode). "
            "For full rotamer placement, export as PDB and use SCWRL4.",
            hint=True))

        layout.addWidget(_lbl("LIST ALL RESIDUES", dim=True))
        self._mut_list_btn = _btn("LIST RESIDUES")
        layout.addWidget(self._mut_list_btn)
        self._mut_list_result = _result_box(120)
        layout.addWidget(self._mut_list_result)
        self._mut_list_btn.clicked.connect(self._run_list_residues)

        layout.addWidget(_divider())

        layout.addWidget(_lbl("SINGLE MUTATION", dim=True))
        mg = QGridLayout(); mg.setSpacing(8)
        mg.addWidget(_lbl("RESIDUE ID:", dim=True), 0, 0)
        self._mut_res_id = _input_line("e.g. 42")
        mg.addWidget(self._mut_res_id, 0, 1)

        mg.addWidget(_lbl("CHAIN:", dim=True), 0, 2)
        self._mut_chain = _input_line("A")
        self._mut_chain.setMaximumWidth(50)
        mg.addWidget(self._mut_chain, 0, 3)

        mg.addWidget(_lbl("TARGET:", dim=True), 1, 0)
        self._mut_target = QComboBox()
        self._mut_target.addItems([
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL",
        ])
        self._mut_target.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        mg.addWidget(self._mut_target, 1, 1, 1, 3)
        layout.addLayout(mg)

        self._mut_apply_btn = _btn("APPLY MUTATION", accent=True)
        layout.addWidget(self._mut_apply_btn)
        self._mut_result = _result_box(100)
        layout.addWidget(self._mut_result)
        self._mut_apply_btn.clicked.connect(self._run_single_mutation)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("SAVE MUTATED STRUCTURE", dim=True))
        self._mut_save_btn = _btn("SAVE AS PDB")
        layout.addWidget(self._mut_save_btn)
        self._mut_save_btn.clicked.connect(self._run_save_mutant_pdb)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── ALA SCAN tab ───────────────────────────────────────────────────────

    def _build_scan_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("ALANINE SCANNING"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl(
            "Mutates each residue in a range to ALA and estimates the change "
            "in interaction count as a binding energy proxy. "
            "Hot-spot residues are those with the highest energy penalty.",
            hint=True))

        layout.addWidget(_lbl("RESIDUES TO SCAN  (range: 1-50, or comma list)", dim=True))
        self._scan_residues = _input_line("e.g. 1-50 or 10,15,20")
        layout.addWidget(self._scan_residues)

        layout.addWidget(_lbl("CHAIN (optional):", dim=True))
        self._scan_chain = _input_line("A  (leave blank for all chains)")
        layout.addWidget(self._scan_chain)

        layout.addWidget(_lbl("INTERACTION GROUPS  (atom index ranges)", dim=True))
        gg = QGridLayout(); gg.setSpacing(8)
        gg.addWidget(_lbl("GROUP A:", dim=True), 0, 0)
        self._scan_group_a = _input_line("e.g. 0-399 or all")
        gg.addWidget(self._scan_group_a, 0, 1)
        gg.addWidget(_lbl("GROUP B:", dim=True), 1, 0)
        self._scan_group_b = _input_line("e.g. 400-499")
        gg.addWidget(self._scan_group_b, 1, 1)
        layout.addLayout(gg)

        self._scan_btn = _btn("RUN ALANINE SCAN", accent=True)
        layout.addWidget(self._scan_btn)
        self._scan_result = _result_box(220)
        layout.addWidget(self._scan_result)
        self._scan_btn.clicked.connect(self._run_alanine_scan)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── SOLVATE tab ────────────────────────────────────────────────────────

    def _build_solvate_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("SOLVENT BOX CONSTRUCTION"))
        layout.addWidget(_divider())

        layout.addWidget(_lbl(
            "Builds a water box around the loaded structure. "
            "Water molecules clashing with the solute are automatically removed.",
            hint=True))

        sg = QGridLayout(); sg.setSpacing(8)
        sg.addWidget(_lbl("BUFFER DISTANCE (Å):", dim=True), 0, 0)
        self._solv_buffer = QLineEdit("10.0")
        self._solv_buffer.setMaximumWidth(70)
        self._solv_buffer.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        sg.addWidget(self._solv_buffer, 0, 1)

        sg.addWidget(_lbl("WATER MODEL:", dim=True), 1, 0)
        self._solv_model = QComboBox()
        self._solv_model.addItems(["TIP3P", "SPC/E"])
        self._solv_model.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        sg.addWidget(self._solv_model, 1, 1)

        sg.addWidget(_lbl("MAX WATERS:", dim=True), 2, 0)
        self._solv_max = QLineEdit("5000")
        self._solv_max.setMaximumWidth(70)
        self._solv_max.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        sg.addWidget(self._solv_max, 2, 1)
        layout.addLayout(sg)

        self._solv_btn = _btn("BUILD WATER BOX", accent=True)
        layout.addWidget(self._solv_btn)
        self._solv_result = _result_box(120)
        layout.addWidget(self._solv_result)
        self._solv_btn.clicked.connect(self._run_solvation)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("ESTIMATE ION COUNT (for 0.15 M NaCl)", dim=True))
        self._ion_btn = _btn("ESTIMATE IONS")
        layout.addWidget(self._ion_btn)
        self._ion_result = _result_box(80)
        layout.addWidget(self._ion_result)
        self._ion_btn.clicked.connect(self._run_estimate_ions)

        layout.addWidget(_lbl("SAVE SOLVATED STRUCTURE", dim=True))
        self._solv_save_btn = _btn("SAVE AS PDB")
        layout.addWidget(self._solv_save_btn)
        self._solv_save_btn.clicked.connect(self._run_save_solvated_pdb)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_checkbox(self, text: str, checked: bool = False) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setStyleSheet(
            f"QCheckBox {{ color:{TEXT_DIM}; font-size:9px; letter-spacing:1px; "
            f"background:transparent; spacing:6px; }}"
            f"QCheckBox::indicator {{ width:12px; height:12px; "
            f"border:1px solid {BORDER}; background:transparent; }}"
            f"QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}"
        )
        return cb

    def _parse_residue_range(self, text: str) -> list[int] | None:
        """Parse residue ID range string (same logic as atom range but for residue IDs)."""
        text = text.strip()
        if not text:
            return None
        if "-" in text and "," not in text:
            parts = text.split("-")
            if len(parts) == 2:
                try:
                    return list(range(int(parts[0].strip()), int(parts[1].strip()) + 1))
                except ValueError:
                    return None
        if "," in text:
            try:
                return [int(x.strip()) for x in text.split(",") if x.strip()]
            except ValueError:
                return None
        try:
            return [int(text)]
        except ValueError:
            return None

    # ── Compute slots ──────────────────────────────────────────────────────

    @Slot()
    def _run_structure_check(self) -> None:
        try:
            from PSVAP.modeling.structure_prep import check_structure
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._prep_report.setText("NO DATA LOADED"); return
            report = check_structure(atoms, pos)
            self._prep_report.setText(report.summary())
        except Exception as e:
            self._prep_report.setText(f"ERROR: {e}")

    @Slot()
    def _run_structure_clean(self) -> None:
        try:
            from PSVAP.modeling.structure_prep import (
                remove_waters, remove_hetatm, cap_termini, renumber_residues
            )
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._prep_result.setText("NO DATA LOADED"); return

            n_before = len(atoms)
            current_atoms = list(atoms)
            current_pos   = pos.copy()

            steps: list[str] = []

            if self._prep_rm_water.isChecked():
                current_atoms, current_pos = remove_waters(current_atoms, current_pos)
                n_removed = n_before - len(current_atoms)
                steps.append(f"Removed {n_removed} water atoms")

            if self._prep_rm_hetatm.isChecked():
                before = len(current_atoms)
                current_atoms, current_pos = remove_hetatm(current_atoms, current_pos)
                steps.append(f"Removed {before - len(current_atoms)} HETATM atoms")

            if self._prep_cap.isChecked():
                before = len(current_atoms)
                current_atoms, current_pos = cap_termini(current_atoms, current_pos)
                steps.append(f"Added {len(current_atoms) - before} cap atoms")

            if self._prep_renumber.isChecked():
                current_atoms = renumber_residues(current_atoms, start=1)
                steps.append("Residues renumbered from 1")

            # Store result for saving
            self._last_mutation_atoms     = current_atoms
            self._last_mutation_positions = current_pos

            lines = [
                f"CLEANING COMPLETE\n",
                f"  Before: {n_before} atoms",
                f"  After:  {len(current_atoms)} atoms",
                "",
            ] + steps

            self._prep_result.setText("\n".join(lines))

        except Exception as e:
            self._prep_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_save_prep_pdb(self) -> None:
        self._save_pdb(
            self._last_mutation_atoms,
            self._last_mutation_positions,
            self._prep_result,
        )

    @Slot()
    def _run_list_residues(self) -> None:
        try:
            from PSVAP.modeling.mutation_engine import list_residues
            atoms = self._get_atoms()
            if not atoms:
                self._mut_list_result.setText("NO DATA LOADED"); return
            residues = list_residues(atoms)
            if not residues:
                self._mut_list_result.setText("No residues found."); return
            lines = [
                f"{'RESID':>6}  {'RES':>4}  {'CHAIN':>5}  {'ATOMS':>6}",
                "-" * 30,
            ]
            for r in residues[:60]:
                lines.append(
                    f"{r['residue_id']:>6}  {r['resname']:>4}  "
                    f"{r['chain_id'] or '-':>5}  {r['n_atoms']:>6}"
                )
            if len(residues) > 60:
                lines.append(f"  ... {len(residues)-60} more residues")
            self._mut_list_result.setText("\n".join(lines))
        except Exception as e:
            self._mut_list_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_single_mutation(self) -> None:
        try:
            from PSVAP.modeling.mutation_engine import mutate_residue

            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._mut_result.setText("NO DATA LOADED"); return

            res_id_text = self._mut_res_id.text().strip()
            res_id = _safe_int(res_id_text)
            if res_id is None:
                self._mut_result.setText("Enter a valid residue ID."); return

            target  = self._mut_target.currentText()
            chain   = self._mut_chain.text().strip() or None

            # Find original resname for display
            from PSVAP.modeling.mutation_engine import get_residue_atoms
            res_idx = get_residue_atoms(atoms, res_id, chain)
            if not res_idx:
                self._mut_result.setText(
                    f"Residue {res_id} "
                    f"(chain={chain or 'any'}) not found.\n"
                    f"Use LIST RESIDUES to see valid IDs."); return

            orig_resname = (
                getattr(atoms[res_idx[0]], "resname", None) or "UNK"
            ).upper()

            new_atoms, new_pos = mutate_residue(
                atoms, pos, res_id, target, chain_id=chain
            )

            # Store for saving
            self._last_mutation_atoms     = new_atoms
            self._last_mutation_positions = new_pos

            self._mut_result.setText(
                f"MUTATION APPLIED\n\n"
                f"  Residue  : {res_id}  (chain {chain or 'auto'})\n"
                f"  Original : {orig_resname}\n"
                f"  Target   : {target}\n\n"
                f"  Atoms before : {len(atoms)}\n"
                f"  Atoms after  : {len(new_atoms)}\n\n"
                f"Use SAVE AS PDB to export the mutated structure."
            )

        except Exception as e:
            self._mut_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_save_mutant_pdb(self) -> None:
        self._save_pdb(
            self._last_mutation_atoms,
            self._last_mutation_positions,
            self._mut_result,
        )

    @Slot()
    def _run_alanine_scan(self) -> None:
        try:
            from PSVAP.modeling.alanine_scan import alanine_scan, format_scan_results

            atoms = self._get_atoms()
            pos   = self._get_positions()
            n     = self._get_atom_count()

            if not atoms or pos is None:
                self._scan_result.setText("NO DATA LOADED"); return

            # Parse residue range
            res_ids = self._parse_residue_range(self._scan_residues.text())
            if not res_ids:
                self._scan_result.setText(
                    "Enter a residue range.\n"
                    "Format: '1-50'  or  '10,15,20,25'"); return

            # Parse groups
            group_a = self._parse_index_range(self._scan_group_a.text(), n)
            group_b = self._parse_index_range(self._scan_group_b.text(), n)
            if group_a is None or group_b is None:
                self._scan_result.setText(
                    "Set valid GROUP A and GROUP B atom ranges."); return

            if not group_a or not group_b:
                self._scan_result.setText(
                    "One or both groups are empty. Check index ranges."); return

            chain = self._scan_chain.text().strip() or None

            # Warn if large scan
            if len(res_ids) > 30:
                self._scan_result.setText(
                    f"Running scan on {len(res_ids)} residues — this may take a moment...")

            results = alanine_scan(
                atoms, pos, res_ids, group_a, group_b,
                chain_id=chain,
            )
            self._scan_result.setText(format_scan_results(results))

        except Exception as e:
            self._scan_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_solvation(self) -> None:
        try:
            from PSVAP.modeling.solvation import build_water_box

            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._solv_result.setText("NO DATA LOADED"); return

            try:
                buffer = float(self._solv_buffer.text().strip())
            except ValueError:
                buffer = 10.0

            try:
                max_w = int(self._solv_max.text().strip())
            except ValueError:
                max_w = 5000

            model_str = self._solv_model.currentText().lower().replace("/", "")
            if "spce" in model_str or "spc" in model_str:
                model_str = "spce"
            else:
                model_str = "tip3p"

            result = build_water_box(
                atoms, pos,
                buffer=buffer,
                water_model=model_str,
                max_waters=max_w,
            )

            # Store for saving
            self._last_mutation_atoms     = result.atoms
            self._last_mutation_positions = result.positions
            self._last_solv_result        = result

            self._solv_result.setText(result.summary())

        except Exception as e:
            self._solv_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_estimate_ions(self) -> None:
        try:
            from PSVAP.modeling.solvation import estimate_ion_count

            atoms = self._get_atoms()
            if not atoms:
                self._ion_result.setText("NO DATA LOADED"); return

            # Estimate n_waters from solv result if available
            n_waters = getattr(
                getattr(self, '_last_solv_result', None),
                'n_waters_added', 3000
            )

            n_na, n_cl = estimate_ion_count(atoms, n_waters)
            self._ion_result.setText(
                f"ION ESTIMATE  (0.15 M NaCl)\n\n"
                f"  Na⁺ ions : {n_na}\n"
                f"  Cl⁻ ions : {n_cl}\n"
                f"  Based on {n_waters} water molecules"
            )

        except Exception as e:
            self._ion_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_save_solvated_pdb(self) -> None:
        solv = getattr(self, '_last_solv_result', None)
        if solv is None:
            return
        self._save_pdb(solv.atoms, solv.positions, self._solv_result)

    # ── PDB saving helper ──────────────────────────────────────────────────

    def _save_pdb(
        self,
        atoms,
        positions,
        result_box: QTextEdit,
    ) -> None:
        if atoms is None or positions is None:
            result_box.setText(
                "No modified structure available.\n"
                "Run a preparation/mutation/solvation step first.")
            return
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Structure as PDB", "", "PDB Files (*.pdb);;All Files (*)"
            )
            if not path:
                return
            from PSVAP.modeling.mutation_engine import write_pdb
            write_pdb(atoms, positions, path)
            result_box.setText(
                f"SAVED\n  {path}\n  {len(atoms):,} atoms"
            )
        except Exception as e:
            result_box.setText(f"SAVE ERROR: {e}")

# ── MD Setup tab (Phase 6 — Feature 14) ───────────────────────────────

    def _build_mdsetup_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("MD SIMULATION SETUP"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Generates input files for GROMACS or AMBER. "
            "PSVAP does not run simulations — it prepares the files.",
            hint=True))

        eg = QGridLayout(); eg.setSpacing(8)
        eg.addWidget(_lbl("ENGINE:", dim=True), 0, 0)
        self._md_engine = QComboBox()
        self._md_engine.addItems(["GROMACS", "AMBER"])
        self._md_engine.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        eg.addWidget(self._md_engine, 0, 1)

        eg.addWidget(_lbl("FORCE FIELD:", dim=True), 1, 0)
        self._md_ff = QComboBox()
        self._md_ff.addItems([
            "AMBER99SB-ILDN", "CHARMM36", "OPLS-AA", "GROMOS96",
            "FF14SB", "FF19SB",
        ])
        self._md_ff.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        eg.addWidget(self._md_ff, 1, 1)

        eg.addWidget(_lbl("WATER MODEL:", dim=True), 2, 0)
        self._md_water = QComboBox()
        self._md_water.addItems(["TIP3P", "SPC/E", "TIP4P"])
        self._md_water.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        eg.addWidget(self._md_water, 2, 1)

        eg.addWidget(_lbl("ENSEMBLE:", dim=True), 3, 0)
        self._md_ensemble = QComboBox()
        self._md_ensemble.addItems(["NPT", "NVT"])
        self._md_ensemble.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        eg.addWidget(self._md_ensemble, 3, 1)
        layout.addLayout(eg)

        layout.addWidget(_lbl("OUTPUT DIRECTORY:", dim=True))
        dir_row = QHBoxLayout(); dir_row.setSpacing(8)
        self._md_outdir = _input_line("md_setup")
        self._md_browse = _btn("BROWSE")
        dir_row.addWidget(self._md_outdir, stretch=1)
        dir_row.addWidget(self._md_browse)
        layout.addLayout(dir_row)

        self._md_gen_btn = _btn("GENERATE INPUT FILES", accent=True)
        layout.addWidget(self._md_gen_btn)
        self._md_result = _result_box(220)
        layout.addWidget(self._md_result)

        self._md_browse.clicked.connect(self._browse_md_outdir)
        self._md_gen_btn.clicked.connect(self._run_md_setup)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── CG Setup tab (Phase 6 — Feature 24) ───────────────────────────────

    def _build_cg_tab(self) -> QWidget:
        w = QScrollArea(); w.setWidgetResizable(True); w.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget(); inner.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(14)

        layout.addWidget(_lbl("COARSE-GRAINED SETUP (MARTINI)"))
        layout.addWidget(_divider())
        layout.addWidget(_lbl(
            "Converts all-atom structure to MARTINI CG representation. "
            "Uses martinize2 if installed, otherwise built-in bead mapping.",
            hint=True))

        vg = QGridLayout(); vg.setSpacing(8)
        vg.addWidget(_lbl("MARTINI VERSION:", dim=True), 0, 0)
        self._cg_version = QComboBox()
        self._cg_version.addItems(["MARTINI 3", "MARTINI 2"])
        self._cg_version.setStyleSheet(
            f"QComboBox {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:4px 8px; }}")
        vg.addWidget(self._cg_version, 0, 1)

        vg.addWidget(_lbl("OUTPUT DIR:", dim=True), 1, 0)
        self._cg_outdir = _input_line("cg_output")
        vg.addWidget(self._cg_outdir, 1, 1)
        layout.addLayout(vg)

        layout.addWidget(_lbl("USE MARTINIZE2 (if installed)", dim=True))
        pdb_row = QHBoxLayout(); pdb_row.setSpacing(8)
        self._cg_pdb = _input_line("path/to/structure.pdb")
        self._cg_browse = _btn("BROWSE")
        pdb_row.addWidget(self._cg_pdb, stretch=1)
        pdb_row.addWidget(self._cg_browse)
        layout.addLayout(pdb_row)
        self._cg_martinize_btn = _btn("RUN MARTINIZE2")
        layout.addWidget(self._cg_martinize_btn)
        self._cg_martinize_result = _result_box(100)
        layout.addWidget(self._cg_martinize_result)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("BUILT-IN BEAD MAPPING (no external tools)", dim=True))
        self._cg_builtin_btn = _btn("APPLY CG MAPPING")
        layout.addWidget(self._cg_builtin_btn)
        self._cg_builtin_result = _result_box(180)
        layout.addWidget(self._cg_builtin_result)

        self._cg_browse.clicked.connect(self._browse_cg_pdb)
        self._cg_martinize_btn.clicked.connect(self._run_martinize2)
        self._cg_builtin_btn.clicked.connect(self._run_cg_builtin)

        layout.addStretch()
        w.setWidget(inner); return w

    # ── Phase 6 modeling slots ─────────────────────────────────────────────

    @Slot()
    def _browse_md_outdir(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._md_outdir.setText(path)

    @Slot()
    def _run_md_setup(self) -> None:
        try:
            engine = self._md_engine.currentText().lower()
            ff     = self._md_ff.currentText()
            water  = self._md_water.currentText().replace("/", "").lower()
            ens    = self._md_ensemble.currentText()
            outdir = self._md_outdir.text().strip() or "md_setup"

            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._md_result.setText("NO DATA LOADED"); return

            model  = self.controller.model
            bb     = getattr(getattr(model, "metadata", None), "box_bounds", None)

            if engine == "gromacs":
                from PSVAP.modeling.md_setup import generate_gromacs_inputs
                result = generate_gromacs_inputs(
                    atoms, pos, bb,
                    force_field=ff,
                    water_model=water,
                    output_dir=outdir,
                    ensemble=ens,
                )
            else:
                from PSVAP.modeling.md_setup import generate_amber_inputs
                # Map GROMACS FF names to AMBER FF names
                amber_ff_map = {
                    "AMBER99SB-ILDN": "FF14SB",
                    "FF14SB": "FF14SB",
                    "FF19SB": "FF19SB",
                    "CHARMM36": "FF14SB",
                    "OPLS-AA": "FF14SB",
                    "GROMOS96": "FF14SB",
                }
                result = generate_amber_inputs(
                    atoms, pos, bb,
                    force_field=amber_ff_map.get(ff, "FF14SB"),
                    water_model=water.upper(),
                    output_dir=outdir,
                )

            self._md_result.setText(result.summary())

        except Exception as e:
            self._md_result.setText(f"ERROR: {e}")

    @Slot()
    def _browse_cg_pdb(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PDB File", "", "PDB Files (*.pdb);;All Files (*)")
        if path:
            self._cg_pdb.setText(path)

    @Slot()
    def _run_martinize2(self) -> None:
        try:
            from PSVAP.modeling.coarse_grain import (
                run_martinize2, check_martinize2_available
            )
            pdb    = self._cg_pdb.text().strip()
            outdir = self._cg_outdir.text().strip() or "cg_output"
            ver    = 3 if "3" in self._cg_version.currentText() else 2

            if not pdb:
                self._cg_martinize_result.setText(
                    "Browse to a PDB file first."); return
            if not check_martinize2_available():
                self._cg_martinize_result.setText(
                    "MARTINIZE2 NOT FOUND\n\n"
                    "Install: pip install martinize2 vermouth\n"
                    "Use BUILT-IN BEAD MAPPING as fallback.")
                return

            result = run_martinize2(pdb, outdir, martini_version=ver)
            self._cg_martinize_result.setText(result.summary())
        except Exception as e:
            self._cg_martinize_result.setText(f"ERROR: {e}")

    @Slot()
    def _run_cg_builtin(self) -> None:
        try:
            from PSVAP.modeling.coarse_grain import (
                build_cg_beads, format_bead_map, CGResult
            )
            atoms = self._get_atoms()
            pos   = self._get_positions()
            if not atoms or pos is None:
                self._cg_builtin_result.setText("NO DATA LOADED"); return

            has_resnames = any(
                getattr(a, "resname", None) for a in atoms[:20]
            )
            if not has_resnames:
                self._cg_builtin_result.setText(
                    "RESIDUE NAMES REQUIRED\n\n"
                    "CG bead mapping needs named residues.\n"
                    "Load a PDB or GRO file.")
                return

            ver = 3 if "3" in self._cg_version.currentText() else 2
            bead_map, cg_pos = build_cg_beads(atoms, pos, martini_version=ver)

            result = CGResult(
                n_atoms=len(atoms),
                n_beads=len(bead_map),
                bead_map=bead_map,
                method="built-in",
            )
            self._cg_builtin_result.setText(
                result.summary() + "\n\n" + format_bead_map(bead_map)
            )
        except Exception as e:
            self._cg_builtin_result.setText(f"ERROR: {e}")