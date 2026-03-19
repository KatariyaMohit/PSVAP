"""
io/dcd_parser.py
----------------
Parser for CHARMM / NAMD DCD binary trajectory files (.dcd).

DCD is the native trajectory format for CHARMM and NAMD simulations.
Like AMBER trajectories, DCD files contain only coordinates — atom
names and residue information come from a separate topology file
(.psf for CHARMM/NAMD, or .pdb as a minimal topology).

This parser uses MDAnalysis which handles both CHARMM-style and
NAMD-style DCD files transparently.

Rule compliance
---------------
  Rule 4  : One parser — DCD format only.
  Rule 2  : No GUI imports.
  Rule 7  : Tested in tests/test_parsers_phase1.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class DCDFormatError(ValueError):
    """Raised when a DCD file cannot be parsed."""


class DCDParser(BaseParser):
    """
    Parses CHARMM/NAMD DCD binary trajectory files (.dcd).

    Parameters
    ----------
    topology_path : optional
        Path to a topology file (.psf, .pdb, or .gro).  Strongly recommended;
        without it atom names and residue data are unavailable.
    """

    def __init__(self, topology_path: Path | None = None) -> None:
        self._topology_path = topology_path

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise DCDFormatError(
                "MDAnalysis is required for DCD support. "
                "Install it with: pip install MDAnalysis"
            ) from exc

        if path.suffix.lower() != ".dcd":
            raise DCDFormatError(f"Expected .dcd file, got: {path.suffix}")

        try:
            if self._topology_path is not None:
                u = mda.Universe(str(self._topology_path), str(path))
            else:
                u = mda.Universe(str(path))
        except Exception as exc:
            raise DCDFormatError(f"Cannot read DCD file '{path.name}': {exc}") from exc

        atoms = self._build_atoms(u)
        trajectory: list[np.ndarray] = []
        timesteps: list[int] = []

        for ts in u.trajectory:
            trajectory.append(u.atoms.positions.astype(np.float64).copy())
            timesteps.append(int(ts.frame))

        if not trajectory:
            raise DCDFormatError(f"No frames found in DCD file: {path.name}")

        box_bounds = self._extract_box(u)
        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=timesteps,
        )
        return atoms, trajectory, metadata

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_atoms(u) -> list[Atom]:
        """Build PSVAP Atom list from MDAnalysis Universe."""
        _ELEMENT_TO_TYPE: dict[str, int] = {
            "C": 1, "N": 2, "O": 3, "S": 4, "H": 5, "P": 6,
        }
        atoms: list[Atom] = []
        for mda_atom in u.atoms:
            element = (getattr(mda_atom, "element", "") or "").strip().upper()
            type_id = _ELEMENT_TO_TYPE.get(element, 0)
            resid = int(getattr(mda_atom.residue, "resid", 0))
            atoms.append(
                Atom(
                    id=int(mda_atom.index),
                    type_id=type_id,
                    element=element if element else None,
                    x=float(mda_atom.position[0]),
                    y=float(mda_atom.position[1]),
                    z=float(mda_atom.position[2]),
                    mass=_safe_float(getattr(mda_atom, "mass", None)),
                    residue_id=resid,
                    chain_id=_safe_str(getattr(mda_atom, "segid", None)),
                    name=_safe_str(getattr(mda_atom, "name", None)),
                )
            )
        return atoms

    @staticmethod
    def _extract_box(u) -> np.ndarray | None:
        try:
            dims = u.trajectory.ts.dimensions
            if dims is None:
                return None
            lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
            return np.array([[0.0, lx], [0.0, ly], [0.0, lz]], dtype=np.float64)
        except Exception:
            return None


def _safe_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s not in {"nan", "None", ""} else None


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None