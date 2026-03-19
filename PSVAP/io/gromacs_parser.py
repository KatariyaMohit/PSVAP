"""
io/gromacs_parser.py
--------------------
Parser for GROMACS file formats:
  - .gro  : GROMACS structure file (single frame)
  - .xtc  : Compressed trajectory
  - .trr  : Full-precision trajectory

Windows freeze fix
------------------
On Windows, MDAnalysis GROReader can block the OS message pump when reading
large files because it holds the GIL during file I/O and numpy operations.
Even on a QThread this can stall WM_PAINT messages, making the window
appear frozen and unresponsive to minimize/restore.

Fix strategy:
1. We release the Python GIL on the background thread by calling
   `QCoreApplication.processEvents()` every N frames via a progress
   callback injected through the constructor.
2. For .gro files (single frame) the parse is fast so no special handling.
3. For trajectory formats we yield frames and optionally call the callback.

The LoaderWorker passes its `progress` signal emit as the callback.

Performance note
----------------
MDAnalysis positions are ALWAYS in Angstroms regardless of source format.
.gro stores nm → MDAnalysis converts to Å automatically.
Box dimensions from ts.dimensions are also in Angstroms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class GromacsFormatError(ValueError):
    """Raised when a GROMACS file cannot be parsed."""


class GromacsParser(BaseParser):
    """
    Parses GROMACS .gro, .xtc, and .trr files using MDAnalysis.

    Parameters
    ----------
    progress_callback : optional
        A callable(int) that receives a 0-100 progress percentage.
        Used by LoaderWorker to keep the Qt event loop breathing.
    """

    def __init__(self, progress_callback: Callable[[int], None] | None = None):
        self._progress = progress_callback or (lambda x: None)

    def parse(
        self,
        path: Path,
        topology_path: Path | None = None,
    ) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        try:
            import MDAnalysis as mda
        except ImportError as exc:
            raise GromacsFormatError(
                "MDAnalysis is required for GROMACS support. "
                "Install it with: pip install MDAnalysis"
            ) from exc

        ext = path.suffix.lower()
        if ext == ".gro":
            return self._parse_gro(path, mda)
        elif ext in {".xtc", ".trr"}:
            return self._parse_trajectory(path, topology_path, mda)
        else:
            raise GromacsFormatError(f"Unsupported GROMACS extension: {ext}")

    # ------------------------------------------------------------------ #
    #  .gro — single frame                                                 #
    # ------------------------------------------------------------------ #

    def _parse_gro(self, path: Path, mda) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        self._progress(5)
        try:
            u = mda.Universe(str(path))
        except Exception as exc:
            raise GromacsFormatError(f"Cannot read .gro file '{path.name}': {exc}") from exc

        self._progress(60)
        atoms = self._build_atoms(u)
        positions = u.atoms.positions.astype(np.float64).copy()
        self._progress(90)

        box_bounds = self._extract_box(u)
        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=[0],
        )
        return atoms, [positions], metadata

    # ------------------------------------------------------------------ #
    #  .xtc / .trr — trajectory                                           #
    # ------------------------------------------------------------------ #

    def _parse_trajectory(
        self, traj_path: Path, topo_path: Path | None, mda
    ) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        self._progress(5)
        try:
            if topo_path is not None:
                u = mda.Universe(str(topo_path), str(traj_path))
            else:
                u = mda.Universe(str(traj_path))
        except Exception as exc:
            raise GromacsFormatError(
                f"Cannot read trajectory '{traj_path.name}': {exc}"
            ) from exc

        atoms = self._build_atoms(u)
        trajectory: list[np.ndarray] = []
        timesteps: list[int] = []
        box_bounds = None
        total = u.trajectory.n_frames

        self._progress(10)
        for ts in u.trajectory:
            # .copy() is critical — without it all frames share the same buffer
            trajectory.append(u.atoms.positions.astype(np.float64).copy())
            timesteps.append(int(ts.frame))
            if box_bounds is None:
                box_bounds = self._extract_box(u)
            # Report progress to keep Qt event loop alive
            pct = 10 + int(80 * ts.frame / max(total, 1))
            self._progress(pct)

        if not trajectory:
            raise GromacsFormatError(f"Trajectory '{traj_path.name}' contains no frames.")

        metadata = SystemMetadata(
            source_path=traj_path,
            box_bounds=box_bounds,
            timesteps=timesteps,
        )
        return atoms, trajectory, metadata

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_atoms(u) -> list[Atom]:
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
        try:
            dims = u.trajectory.ts.dimensions
            if dims is None:
                return None
            lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
            if lx <= 0 or ly <= 0 or lz <= 0:
                return None
            return np.array([[0.0, lx], [0.0, ly], [0.0, lz]], dtype=np.float64)
        except Exception:
            return None


def _safe_str(value) -> str | None:
    if value is None: return None
    s = str(value).strip()
    return s if s and s not in {"nan","None",""} else None

def _safe_float(value) -> float | None:
    try: return float(value)
    except (TypeError, ValueError): return None