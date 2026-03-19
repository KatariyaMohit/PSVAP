"""Atom property table widget — displays atom data in a tabular view."""
from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from PSVAP.core.atom import Atom


class AtomTableWidget(QTableWidget):
    """Displays a list of Atom objects in a sortable table.

    Columns: ID, Type, Element, X, Y, Z, Charge, Residue, Chain, Name.
    """

    _COLUMNS = ("ID", "Type", "Element", "X", "Y", "Z", "Charge", "Residue", "Chain", "Name")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setColumnCount(len(self._COLUMNS))
        self.setHorizontalHeaderLabels(list(self._COLUMNS))
        self.setSortingEnabled(True)

    def set_atoms(self, atoms: list[Atom]) -> None:
        """Populate the table with the given atoms."""
        self.setRowCount(len(atoms))
        for row, a in enumerate(atoms):
            self.setItem(row, 0, QTableWidgetItem(str(a.id)))
            self.setItem(row, 1, QTableWidgetItem(str(a.type_id or "")))
            self.setItem(row, 2, QTableWidgetItem(a.element or ""))
            self.setItem(row, 3, QTableWidgetItem(f"{a.x:.4f}"))
            self.setItem(row, 4, QTableWidgetItem(f"{a.y:.4f}"))
            self.setItem(row, 5, QTableWidgetItem(f"{a.z:.4f}"))
            self.setItem(row, 6, QTableWidgetItem(str(a.charge or "")))
            self.setItem(row, 7, QTableWidgetItem(str(a.residue_id or "")))
            self.setItem(row, 8, QTableWidgetItem(a.chain_id or ""))
            self.setItem(row, 9, QTableWidgetItem(a.name or ""))
