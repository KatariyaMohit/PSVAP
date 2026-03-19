"""
io/amber_parser.py
------------------
Parser for AMBER trajectory formats using MDAnalysis:
  - .nc / .ncdf  : NetCDF binary trajectory (AMBER NetCDF Convention)
  - .mdcrd       : ASCII AMBER trajectory (AMBER CRD)
  - .rst7 / .rst : AMBER restart file (single frame)

AMBER workflows always pair a trajectory with a topology file (.prmtop /
.parm7).  If no topology is supplied, atoms are created with minimal
metadata.

Note on units
-------------
AMBER trajectories store coordinates in Angstroms — same as PSVAP's
internal representation.  No conversion needed.

Rule compliance
---------------
  Rule 4  : One parser — AMBER formats only.
  Rule 2  : No GUI imports.
  Rule 7  : Tested in tests/test_parsers_phase1.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class AmberFormatError(ValueError):
    """Raised when an AMBER file cannot be parsed."""


class AmberParser(BaseParser):
    """
    Parses AMBER trajectory and restart files (.nc, .ncdf, .mdcrd, .rst7).

    Parameters
    ----------
    topology_path : optional
        Path to the AMBER topology (.prmtop / .parm7).  When provided, atom
        names, residue information, and masses are populated from it.
    """

    def __init__(self, topology_path: Path | None = None) -> None:
        self._topology_path = topology_path

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise AmberFormatError(
                "MDAnalysis is required for AMBER support. "
                "Install it with: pip install MDAnalysis"
            ) from exc

        ext = path.suffix.lower()
        if ext not in {".nc", ".ncdf", ".mdcrd", ".crd", ".rst7", ".rst", ".restrt"}:
            raise AmberFormatError(f"Unsupported AMBER extension: {ext}")

        try:
            if self._topology_path is not None:
                u = mda.Universe(str(self._topology_path), str(path))
            else:
                u = mda.Universe(str(path))
        except Exception as exc:
            raise AmberFormatError(f"Cannot read AMBER file '{path.name}': {exc}") from exc

        atoms = self._build_atoms(u)
        trajectory: list[np.ndarray] = []
        timesteps: list[int] = []

        for ts in u.trajectory:
            trajectory.append(u.atoms.positions.astype(np.float64).copy())
            timesteps.append(int(ts.frame))

        if not trajectory:
            raise AmberFormatError(f"No frames found in AMBER file: {path.name}")

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
        """Build PSVAP Atom list from an MDAnalysis Universe."""
        atoms: list[Atom] = []
        for mda_atom in u.atoms:
            element = (getattr(mda_atom, "element", "") or "").strip().upper()
            # AMBER type_id: use mass-based heuristic if element not available
            type_id = _element_to_type(element)
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
                    charge=_safe_float(getattr(mda_atom, "charge", None)),
                    residue_id=resid,
                    chain_id=_safe_str(getattr(mda_atom, "segid", None)),
                    name=_safe_str(getattr(mda_atom, "name", None)),
                )
            )
        return atoms

    @staticmethod
    def _extract_box(u) -> np.ndarray | None:
        """Extract box dimensions from MDAnalysis trajectory."""
        try:
            dims = u.trajectory.ts.dimensions
            if dims is None:
                return None
            lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
            return np.array([[0.0, lx], [0.0, ly], [0.0, lz]], dtype=np.float64)
        except Exception:
            return None


# ── Utilities ──────────────────────────────────────────────────────────────

_ELEMENT_TO_TYPE: dict[str, int] = {
    "C": 1, "N": 2, "O": 3, "S": 4, "H": 5,
    "P": 6, "F": 7, "CL": 8, "BR": 9, "I": 10,
}


def _element_to_type(element: str) -> int:
    return _ELEMENT_TO_TYPE.get(element.upper(), 0)


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