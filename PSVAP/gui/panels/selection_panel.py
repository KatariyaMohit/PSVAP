"""
gui/panels/selection_panel.py — FIXED
--------------------------------------
Selection panel that actually filters the viewport.

The fix: after parsing the selection query, we build a boolean mask
and pass it to both the SystemModel AND the VisualizationEngine.

Additional fix (this version)
------------------------------
- "SELECTED X / Y ATOMS" counter is now cleared when a new file loads,
  so the count from a previous file doesn't persist into the next one.
- Counter is also cleared by the CLEAR button.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
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


def _divider() -> QFrame:
    f = QFrame(); f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORDER};"); return f


def _lbl(text: str, dim: bool = False) -> QLabel:
    lbl = QLabel(text)
    color = TEXT_DIM if dim else TEXT
    lbl.setStyleSheet(
        f"color:{color}; font-size:9px; letter-spacing:2px; background:transparent;")
    return lbl


class SelectionPanel(QWidget):
    def __init__(self, *, controller: ApplicationController) -> None:
        super().__init__()
        self.controller = controller
        self._build()
        self._connect_model_signals()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(_lbl("ATOM SELECTION"))
        layout.addWidget(_divider())

        # Query input
        self._query = QLineEdit()
        self._query.setPlaceholderText("e.g.  type==1  AND  z > 10")
        self._query.setStyleSheet(
            f"QLineEdit {{ background:{PANEL_ALT}; border:1px solid {BORDER}; "
            f"color:{TEXT}; padding:8px 12px; font-size:11px; "
            f"font-family:{MONO}; }}"
            f"QLineEdit:focus {{ border-color:{TEXT_DIM}; }}"
        )
        self._query.returnPressed.connect(self._apply)
        layout.addWidget(self._query)

        # Buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self._apply_btn = QPushButton("APPLY")
        self._apply_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
            f"border:1px solid {BORDER}; padding:7px 16px; font-size:9px; letter-spacing:2px; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}"
            f"QPushButton:pressed {{ background:{ACCENT}; color:#0A0A0A; border-color:{ACCENT}; }}"
        )
        self._apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(self._apply_btn)

        self._clear_btn = QPushButton("CLEAR")
        self._clear_btn.setStyleSheet(
            f"QPushButton {{ background:transparent; color:{TEXT_DIM}; "
            f"border:1px solid {BORDER}; padding:7px 16px; font-size:9px; letter-spacing:2px; }}"
            f"QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}"
        )
        self._clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        layout.addWidget(_divider())
        layout.addWidget(_lbl("SYNTAX REFERENCE", dim=True))

        for example in [
            "type==1",
            "type==2",
            "element==C",
            "z > 10.0",
            "x > 5 AND y < 20",
            "type==0 OR type==1",
            "NOT type==2",
            "mol==3",
        ]:
            eg = QLabel(example)
            eg.setStyleSheet(
                f"color:#444444; font-size:10px; font-family:{MONO}; "
                f"background:transparent; padding:1px 0;"
            )
            layout.addWidget(eg)

        layout.addStretch()

        # Status — shows "SELECTED N / TOTAL ATOMS" or error
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"color:{ACCENT}; font-size:9px; letter-spacing:1px; background:transparent;")
        layout.addWidget(self._status)

    # ── Connect to model so we can clear the counter on new file load ──────

    def _connect_model_signals(self) -> None:
        """
        Connect to SystemModel.data_loaded so we can reset the selection
        counter when a new file is opened. Without this, the "SELECTED N/M"
        counter from a previous file persists into the next one.
        """
        try:
            model = self.controller.model
            if hasattr(model, 'data_loaded'):
                model.data_loaded.connect(self._on_new_file_loaded)
        except Exception:
            pass

    @Slot()
    def _on_new_file_loaded(self) -> None:
        """Clear selection state whenever a new file is loaded."""
        self._query.clear()
        self._status.setText("")

    # ──────────────────────────────────────────────────────────────────────

    @Slot()
    def _apply(self) -> None:
        query = self._query.text().strip()
        if not query:
            self._clear()
            return

        try:
            mask = self._evaluate_query(query)
            n_sel = int(np.sum(mask))
            n_tot = len(mask)

            # Update model selection mask
            model = self.controller.model
            if hasattr(model, 'apply_selection'):
                model.apply_selection(mask)
            else:
                model._selection_mask = mask
                if hasattr(model, 'selection_changed'):
                    model.selection_changed.emit()

            # Update engine directly as well
            try:
                engine = self.controller._engine
                if engine and hasattr(engine, 'apply_selection'):
                    engine.apply_selection(mask)
            except Exception:
                pass

            self._status.setText(f"SELECTED  {n_sel} / {n_tot}  ATOMS")

        except Exception as exc:
            self._status.setText(f"SYNTAX ERROR:\n{exc}")

    @Slot()
    def _clear(self) -> None:
        self._query.clear()
        self._status.setText("")

        model = self.controller.model
        if hasattr(model, 'clear_selection'):
            model.clear_selection()
        else:
            model._selection_mask = None
            if hasattr(model, 'selection_changed'):
                model.selection_changed.emit()

        try:
            engine = self.controller._engine
            if engine and hasattr(engine, 'apply_selection'):
                engine.apply_selection(None)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    #  Query evaluator
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_query(self, query: str) -> np.ndarray:
        """
        Parse and evaluate an atom selection query.
        Returns boolean np.ndarray of length n_atoms.

        Supported syntax:
          type==N        → atoms with type_id == N (0-indexed)
          element==SYM   → atoms with element == SYM
          mol==N         → atoms with molecule/chain ID == N
          x > V, y < V, z >= V, z <= V
          NOT <expr>
          <expr> AND <expr>
          <expr> OR  <expr>
        """
        model = self.controller.model
        atoms = getattr(model, 'atoms', getattr(model, '_atoms', []))

        if not atoms:
            raise ValueError("No atoms loaded")

        n = len(atoms)
        mask = self._parse_expr(query.strip(), atoms, n)
        return mask

    def _parse_expr(self, expr: str, atoms: list, n: int) -> np.ndarray:
        """Recursive expression parser with AND / OR / NOT."""
        expr = expr.strip()

        # Remove outer parentheses
        if expr.startswith("(") and expr.endswith(")"):
            inner = expr[1:-1].strip()
            # Only strip if the parens are balanced around the whole expression
            depth = 0
            valid = True
            for i, c in enumerate(inner):
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                if depth < 0:
                    valid = False; break
            if valid and depth == 0:
                expr = inner

        # OR (lowest precedence)
        for sep in [" OR ", " or "]:
            idx = self._find_top_level(expr, sep)
            if idx >= 0:
                left  = self._parse_expr(expr[:idx], atoms, n)
                right = self._parse_expr(expr[idx+len(sep):], atoms, n)
                return left | right

        # AND
        for sep in [" AND ", " and "]:
            idx = self._find_top_level(expr, sep)
            if idx >= 0:
                left  = self._parse_expr(expr[:idx], atoms, n)
                right = self._parse_expr(expr[idx+len(sep):], atoms, n)
                return left & right

        # NOT
        if expr.upper().startswith("NOT "):
            inner = self._parse_expr(expr[4:].strip(), atoms, n)
            return ~inner

        # Leaf condition
        return self._eval_leaf(expr, atoms, n)

    def _find_top_level(self, expr: str, sep: str) -> int:
        """Find sep at depth 0 (not inside parentheses)."""
        depth = 0
        sep_len = len(sep)
        for i in range(len(expr) - sep_len + 1):
            if expr[i] == '(': depth += 1
            elif expr[i] == ')': depth -= 1
            elif depth == 0 and expr[i:i+sep_len].upper() == sep.upper():
                return i
        return -1

    def _eval_leaf(self, expr: str, atoms: list, n: int) -> np.ndarray:
        """Evaluate a single atom property condition."""
        mask = np.zeros(n, dtype=bool)
        expr = expr.strip()

        # Coordinate comparisons: x > V, y <= V, z == V
        for op_str, op_fn in [
            (">=", lambda a, b: a >= b),
            ("<=", lambda a, b: a <= b),
            (">",  lambda a, b: a > b),
            ("<",  lambda a, b: a < b),
            ("==", lambda a, b: abs(a - b) < 1e-9),
            ("!=", lambda a, b: abs(a - b) >= 1e-9),
        ]:
            for prop in ["x", "y", "z"]:
                pattern = f"{prop}{op_str}"
                if expr.lower().startswith(pattern):
                    val = float(expr[len(pattern):].strip())
                    for idx, atom in enumerate(atoms):
                        mask[idx] = op_fn(float(getattr(atom, prop, 0.0)), val)
                    return mask

        # type==N  (0-indexed type IDs as stored in Atom.type_id)
        if expr.lower().startswith("type==") or expr.lower().startswith("type =="):
            val_str = expr.split("==", 1)[1].strip()
            val = int(val_str)
            for idx, atom in enumerate(atoms):
                tid = getattr(atom, 'type_id', 0)
                if tid is None:
                    tid = 0
                mask[idx] = (tid == val)
            return mask

        # element==SYM
        if expr.lower().startswith("element=="):
            sym = expr.split("==", 1)[1].strip().upper()
            for idx, atom in enumerate(atoms):
                elem = (getattr(atom, 'element', None) or "").upper()
                mask[idx] = elem == sym
            return mask

        # mol==N  (molecule / chain ID)
        if expr.lower().startswith("mol=="):
            val = int(expr.split("==", 1)[1].strip())
            for idx, atom in enumerate(atoms):
                rid = getattr(atom, 'residue_id', None)
                mask[idx] = rid == val
            return mask

        raise ValueError(f"Unrecognised expression: {expr!r}\n"
                         "Supported: type==N, element==X, mol==N, x>V, y<V, z>=V, AND, OR, NOT")