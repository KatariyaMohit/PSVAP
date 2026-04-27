"""
gui/dialogs/binary_config_dialog.py
-----------------------------------
Configuration dialog for raw binary trajectory loading.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from PSVAP.io.binary_parser import RawBinaryConfig

BG = "#111111"
PANEL = "#1A1A1A"
BORDER = "#2A2A2A"
TEXT = "#CCCCCC"
TEXT_DIM = "#888888"
TEXT_HINT = "#555555"
ACCENT = "#E8FF00"


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{TEXT_DIM}; font-size:9px; letter-spacing:2px; background:transparent;"
    )
    return lbl


class BinaryConfigDialog(QDialog):
    """Collect raw binary layout settings before loading a `.bin` file."""

    def __init__(self, path: str | Path, parent=None) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self._config: RawBinaryConfig | None = None

        self.setWindowTitle("RAW BINARY CONFIGURATION")
        self.setMinimumWidth(500)
        self.setModal(True)
        self.setStyleSheet(
            f"""
            QDialog {{ background:{BG}; color:{TEXT}; }}
            QWidget {{ background:{PANEL}; color:{TEXT}; }}
            QLabel {{ background:transparent; color:{TEXT}; }}
            QLineEdit, QSpinBox, QComboBox {{
                background:{BG};
                border:1px solid {BORDER};
                color:{TEXT};
                padding:6px 8px;
                font-size:11px;
            }}
            QPushButton {{
                background:transparent;
                color:{TEXT_DIM};
                border:1px solid {BORDER};
                padding:6px 14px;
                font-size:9px;
                letter-spacing:2px;
            }}
            QPushButton:hover {{ color:{TEXT}; border-color:{TEXT_DIM}; }}
            QPushButton:pressed {{ background:{ACCENT}; color:{BG}; border-color:{ACCENT}; }}
            QDialogButtonBox {{
                background:{BG};
                border-top:1px solid {BORDER};
                padding:8px 16px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 18, 20, 18)
        body_layout.setSpacing(14)

        intro = QLabel(
            "No known binary signature was detected in this `.bin` file.\n"
            "Provide the raw coordinate layout so PSVAP can map frames lazily from disk."
        )
        intro.setStyleSheet(
            f"color:{TEXT}; font-size:11px; line-height:1.4; background:transparent;"
        )
        body_layout.addWidget(intro)

        file_lbl = QLabel(f"FILE: {self._path.name}")
        file_lbl.setStyleSheet(
            f"color:{TEXT_HINT}; font-size:10px; letter-spacing:1px; background:transparent;"
        )
        body_layout.addWidget(file_lbl)

        form = QFormLayout()
        form.setSpacing(12)

        self._atoms_spin = QSpinBox()
        self._atoms_spin.setRange(1, 100_000_000)
        self._atoms_spin.setValue(1)
        form.addRow(_label("ATOMS PER FRAME:"), self._atoms_spin)

        self._precision_combo = QComboBox()
        self._precision_combo.addItem("Float32", "float32")
        self._precision_combo.addItem("Float64", "float64")
        self._precision_combo.setCurrentIndex(1)
        form.addRow(_label("DATA PRECISION:"), self._precision_combo)

        self._endianness_combo = QComboBox()
        self._endianness_combo.addItem("Little-endian", "little")
        self._endianness_combo.addItem("Big-endian", "big")
        form.addRow(_label("ENDIANNESS:"), self._endianness_combo)

        self._offset_edit = QLineEdit("0")
        self._offset_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d+"), self)
        )
        form.addRow(_label("BYTE OFFSET:"), self._offset_edit)

        body_layout.addLayout(form)

        hint = QLabel(
            "A separate PDB or GRO topology file will be required for atom names and bonds."
        )
        hint.setStyleSheet(
            f"color:{TEXT_HINT}; font-size:10px; line-height:1.3; background:transparent;"
        )
        body_layout.addWidget(hint)
        layout.addWidget(body)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> RawBinaryConfig | None:
        return self._config

    @Slot()
    def _on_accept(self) -> None:
        try:
            offset_text = self._offset_edit.text().strip() or "0"
            config = RawBinaryConfig(
                atoms_per_frame=int(self._atoms_spin.value()),
                data_precision=str(self._precision_combo.currentData()),
                endianness=str(self._endianness_combo.currentData()),
                byte_offset=int(offset_text),
            )
            config.validate()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "INVALID BINARY CONFIGURATION",
                str(exc),
            )
            return

        self._config = config
        self.accept()
