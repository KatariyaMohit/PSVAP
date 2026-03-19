"""
io/base_parser.py
-----------------
Abstract base class for all parsers and the central format-detection router.

Adding a new format
-------------------
1. Create io/<format>_parser.py implementing BaseParser.parse().
2. Add the new extension(s) to detect_parser() below.
3. Add tests in tests/test_parsers_phase1.py.
4. Update CHANGELOG.md and requirements.txt if a new library is needed.

Rule compliance
---------------
  Rule 4  : Each format has exactly one parser file.
  Rule 5  : No eval/exec.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata


class BaseParser(ABC):
    """
    Contract that every parser must fulfil.

    parse() receives a Path and returns a 3-tuple:
      - list[Atom]         : atom identity/metadata (constant across frames)
      - list[np.ndarray]   : trajectory frames, each shape (N, 3) float64
      - SystemMetadata     : box bounds, timesteps, source path
    """

    @abstractmethod
    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        raise NotImplementedError


# ── Format registry ────────────────────────────────────────────────────────

def detect_parser(path: Path) -> BaseParser:
    """
    Inspect the file extension and return the appropriate parser instance.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    ext = path.suffix.lower()

    # ── LAMMPS ────────────────────────────────────────────────────────
    if ext in {".lammpstrj", ".traj", ".data", ".lammps"}:
        from PSVAP.io.lammps_parser import LammpsParser
        return LammpsParser()

    # ── GROMACS ───────────────────────────────────────────────────────
    if ext == ".gro":
        from PSVAP.io.gromacs_parser import GromacsParser
        return GromacsParser()

    if ext in {".xtc", ".trr"}:
        from PSVAP.io.gromacs_parser import GromacsParser
        return GromacsParser()

    # ── PDB ───────────────────────────────────────────────────────────
    if ext == ".pdb":
        from PSVAP.io.pdb_parser import PDBParser
        return PDBParser()

    # ── mmCIF / PDBx ──────────────────────────────────────────────────
    if ext in {".cif", ".mmcif"}:
        from PSVAP.io.mmcif_parser import MMCIFParser
        return MMCIFParser()

    # ── AMBER ─────────────────────────────────────────────────────────
    if ext in {".nc", ".ncdf", ".mdcrd", ".crd", ".rst7", ".rst", ".restrt"}:
        from PSVAP.io.amber_parser import AmberParser
        return AmberParser()

    # ── CHARMM / NAMD DCD ─────────────────────────────────────────────
    if ext == ".dcd":
        from PSVAP.io.dcd_parser import DCDParser
        return DCDParser()

    # ── XYZ ───────────────────────────────────────────────────────────
    if ext == ".xyz":
        from PSVAP.io.xyz_parser import XYZParser
        return XYZParser()

    # ── Small molecules ───────────────────────────────────────────────
    if ext in {".mol2", ".sdf", ".mol"}:
        from PSVAP.io.mol_parser import MolParser
        return MolParser()

    raise ValueError(
        f"Unsupported file format: '{path.name}'\n"
        f"Supported extensions: "
        ".lammpstrj .traj .data .lammps "
        ".gro .xtc .trr "
        ".pdb .cif .mmcif "
        ".nc .ncdf .mdcrd .crd .rst7 .rst "
        ".dcd .xyz .mol2 .sdf .mol"
    )