"""
io/gromacs_binary_parser.py
---------------------------
Parser for GROMACS binary topology/checkpoint files:
  - .tpr  : GROMACS run input file (contains atoms, topology, parameters)
  - .cpt  : GROMACS checkpoint file (contains full system state + coordinates)

Both are XDR binary format files. We use MDAnalysis which has native
support for reading these formats.

For .tpr: Extracts topology only (no trajectory data)
For .cpt: Can be used as single-frame coordinate file OR as topology

Rule compliance
---------------
  Rule 4  : One parser — GROMACS binary formats only.
  Rule 2  : No GUI imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class GromacsRuntimeFormatError(ValueError):
    """Raised when a GROMACS .tpr or .cpt file cannot be parsed."""


class GromacsRuntimeParser(BaseParser):
    """
    Parses GROMACS binary files (.tpr, .cpt) using MDAnalysis.

    These files contain complete topology and coordinate information.
    They can be used as standalone structure files or paired with
    trajectory files.
    """

    def __init__(self, progress_callback: Callable[[int], None] | None = None):
        self._progress = progress_callback or (lambda x: None)

    def parse(
        self,
        path: Path,
        topology_path: Path | None = None,
    ) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        """
        Parse a GROMACS .tpr or .cpt file.

        Parameters
        ----------
        path : Path
            Path to .tpr or .cpt file
        topology_path : Path, optional
            Ignored for this parser; .tpr/.cpt already contain topology

        Returns
        -------
        atoms : list[Atom]
            Atoms from the file
        trajectory : list[np.ndarray]
            Coordinate frame(s)
        metadata : SystemMetadata
            Box bounds, bonds, etc.
        """
        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise GromacsRuntimeFormatError(
                "MDAnalysis is required for .tpr/.cpt support. "
                "Install it with: pip install MDAnalysis"
            ) from exc

        ext = path.suffix.lower()
        if ext == ".tpr":
            return self._parse_tpr(path, mda)
        elif ext == ".cpt":
            return self._parse_cpt(path, mda)
        else:
            raise GromacsRuntimeFormatError(f"Unsupported extension: {ext}")

    def _parse_tpr(self, path: Path, mda) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        """Parse .tpr file as topology + single frame."""
        self._progress(5)
        try:
            u = mda.Universe(str(path))
        except Exception as exc:
            raise GromacsRuntimeFormatError(f"Cannot read .tpr file '{path.name}': {exc}") from exc

        self._progress(60)
        atoms = self._build_atoms(u)
        positions = u.atoms.positions.astype(np.float64).copy()
        self._progress(90)

        box_bounds = self._extract_box(u)
        bonds = self._build_bonds(atoms, positions)
        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=[0],
            bonds=bonds,
        )
        self._progress(100)

        return atoms, [positions], metadata

    def _parse_cpt(self, path: Path, mda) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        """Parse .cpt checkpoint file as topology + single frame."""
        self._progress(5)
        try:
            u = mda.Universe(str(path))
        except Exception as exc:
            raise GromacsRuntimeFormatError(f"Cannot read .cpt file '{path.name}': {exc}") from exc

        self._progress(60)
        atoms = self._build_atoms(u)
        positions = u.atoms.positions.astype(np.float64).copy()
        self._progress(90)

        box_bounds = self._extract_box(u)
        bonds = self._build_bonds(atoms, positions)
        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=[0],
            bonds=bonds,
        )
        self._progress(100)

        return atoms, [positions], metadata

    @staticmethod
    def _build_atoms(u) -> list[Atom]:
        """Build Atom objects from MDAnalysis Universe."""
        atoms: list[Atom] = []
        for mda_atom in u.atoms:
            element = (getattr(mda_atom, "element", "") or "").strip().upper()
            resid = int(getattr(mda_atom.residue, "resid", 0))
            atoms.append(Atom(
                id=int(mda_atom.index),
                type_id=None,
                element=element if element else None,
                x=float(mda_atom.position[0]),
                y=float(mda_atom.position[1]),
                z=float(mda_atom.position[2]),
                mass=_safe_float(getattr(mda_atom, "mass", None)),
                residue_id=resid,
                chain_id=_safe_str(getattr(mda_atom, "segid", None)),
                name=_safe_str(getattr(mda_atom, "name", None)),
            ))
        return atoms

    @staticmethod
    def _extract_box(u) -> np.ndarray | None:
        """Extract box dimensions from MDAnalysis Universe."""
        try:
            if hasattr(u, "trajectory") and len(u.trajectory) > 0:
                ts = u.trajectory[0]
                dimensions = getattr(ts, "dimensions", None)
                if dimensions is not None:
                    return np.array(dimensions[:3], dtype=np.float64)
            # Fallback: try from current timestep
            if hasattr(u, "ts"):
                dimensions = getattr(u.ts, "dimensions", None)
                if dimensions is not None:
                    return np.array(dimensions[:3], dtype=np.float64)
        except Exception:
            pass
        return None

    @staticmethod
    def _build_bonds(atoms: list[Atom], positions: np.ndarray) -> np.ndarray | None:
        """Infer bonds from atomic distances using van der Waals radii."""
        if not atoms or len(atoms) < 2:
            return None

        try:
            bonds_set: set[tuple[int, int]] = set()
            vdw_radii = {
                "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
                "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
            }

            for i in range(len(atoms) - 1):
                for j in range(i + 1, len(atoms)):
                    pos_i = positions[i]
                    pos_j = positions[j]
                    dist = np.linalg.norm(pos_j - pos_i)

                    elem_i = atoms[i].element or "C"
                    elem_j = atoms[j].element or "C"
                    r_i = vdw_radii.get(elem_i, 1.70)
                    r_j = vdw_radii.get(elem_j, 1.70)
                    cutoff = 0.6 * (r_i + r_j)

                    if dist < cutoff:
                        bond = tuple(sorted([i, j]))
                        bonds_set.add(bond)

            if not bonds_set:
                return None

            line_data = []
            for i, j in sorted(bonds_set):
                line_data.extend([2, i, j])

            return np.array(line_data, dtype=np.int32)
        except Exception:
            return None


def _safe_float(value) -> float | None:
    """Safely convert value to float."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _safe_str(value) -> str | None:
    """Safely convert value to string."""
    try:
        return str(value).strip() if value is not None else None
    except (ValueError, TypeError):
        return None
