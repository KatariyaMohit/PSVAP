"""
gui/dialogs/about_dialog.py
----------------------------
About dialog — application version and team info.
Updated in Phase 7 to show complete version history.
"""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

BG     = "#111111"
PANEL  = "#1A1A1A"
BORDER = "#2A2A2A"
TEXT   = "#CCCCCC"
DIM    = "#888888"
HINT   = "#555555"
ACCENT = "#E8FF00"
WHITE  = "#FFFFFF"


class AboutDialog(QDialog):
    """About dialog showing PSVAP version, features, and team."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ABOUT PSVAP")
        self.setMinimumWidth(460)
        self.setStyleSheet(f"""
            QDialog {{ background:{BG}; color:{TEXT}; }}
            QLabel  {{ background:transparent; }}
            QPushButton {{
                background:transparent; color:{DIM};
                border:1px solid {BORDER}; padding:6px 20px;
                font-size:9px; letter-spacing:2px;
            }}
            QPushButton:hover {{ color:{TEXT}; border-color:{DIM}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 24)
        layout.setSpacing(0)

        title = QLabel("PSVAP")
        title.setStyleSheet(
            f"color:{WHITE}; font-size:22px; font-weight:700; letter-spacing:8px;")
        layout.addWidget(title)

        sub = QLabel("PARTICLE SIMULATION VISUALIZATION & ANALYSIS PACKAGE")
        sub.setStyleSheet(f"color:{DIM}; font-size:8px; letter-spacing:3px; margin-top:4px;")
        layout.addWidget(sub)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{BORDER}; margin:16px 0;")
        layout.addWidget(sep)

        version = QLabel("Version 1.0.0  ·  All Phases Complete")
        version.setStyleSheet(f"color:{DIM}; font-size:11px;")
        layout.addWidget(version)

        team = QLabel(
            "\nGroup F  ·  CS 310 Software Engineering  ·  Spring 2026\n\n"
            "Mohit Katariya (230001055)  ·  Sai Sashank (230001080)\n"
            "Sameer Choudhary (230001070)  ·  Srikanth (230001018)\n"
            "Annamareddi Suhitha (230001008)"
        )
        team.setStyleSheet(f"color:{HINT}; font-size:10px; line-height:1.6;")
        layout.addWidget(team)

        sep2 = QLabel()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background:{BORDER}; margin:16px 0;")
        layout.addWidget(sep2)

        features = QLabel(
            "Phases 0–7 Complete  ·  24 Analysis Features  ·  256 Tests\n\n"
            "Formats: LAMMPS · GROMACS · PDB · mmCIF · AMBER · DCD · XYZ · MOL2 · SDF\n"
            "Analysis: Geometry · RMSD · Alignment · Interactions · SASA · QSAR\n"
            "          Pharmacophore · Docking · pKa · Conformers · Site Finder\n"
            "Modeling: Mutations · Alanine Scan · Structure Prep · Solvation · MD Setup\n"
            "          Coarse-Graining (MARTINI 3)"
        )
        features.setStyleSheet(f"color:{HINT}; font-size:9px; letter-spacing:1px;")
        layout.addWidget(features)

        layout.addSpacing(20)
        ok_btn = QPushButton("CLOSE")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)