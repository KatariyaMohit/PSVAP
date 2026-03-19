"""
gui/dialogs/file_open_dialog.py
--------------------------------
File open dialog helpers.

Phase 7: Updated with complete format list matching all parsers.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog

# Complete filter list matching all parsers added through Phase 1
_FILTERS = ";;".join([
    "All Supported ("
    "*.lammpstrj *.traj *.data *.lammps "
    "*.gro *.xtc *.trr "
    "*.pdb *.cif *.mmcif "
    "*.nc *.ncdf *.mdcrd *.crd *.rst7 *.rst "
    "*.dcd *.xyz *.mol2 *.sdf *.mol"
    ")",
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


def get_open_file_path(parent=None) -> Path | None:
    """Show a native file open dialog. Returns selected Path or None."""
    path_str, _ = QFileDialog.getOpenFileName(
        parent, "Open Molecular Data File", "", _FILTERS
    )
    return Path(path_str) if path_str else None


def get_topology_path(parent=None) -> Path | None:
    """Show topology file open dialog."""
    path_str, _ = QFileDialog.getOpenFileName(
        parent, "Select Topology File — Step 1 of 2", "", _TOPO_FILTERS
    )
    return Path(path_str) if path_str else None


def get_trajectory_path(parent=None) -> Path | None:
    """Show trajectory file open dialog."""
    path_str, _ = QFileDialog.getOpenFileName(
        parent, "Select Trajectory File — Step 2 of 2", "", _TRAJ_FILTERS
    )
    return Path(path_str) if path_str else None