"""
io/gromacs_itp_parser.py
------------------------
Parser for GROMACS include/parameter files (.itp).

GROMACS .itp files define molecular building blocks:
  - Atom types and parameters
  - Bond, angle, dihedral definitions
  - Virtual sites and constraints
  - Nonbonded parameters

These files are typically included in .top files. This parser
extracts the atomic structure and bonding information.

Rule compliance
---------------
  Rule 4  : One parser — GROMACS .itp format only.
  Rule 2  : No GUI imports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class GromacsItpFormatError(ValueError):
    """Raised when a GROMACS .itp file cannot be parsed."""


class GromacsItpParser(BaseParser):
    """
    Parses GROMACS include/parameter files (.itp).

    Since .itp files define molecular structure without explicit coordinates,
    we create atoms with zero positions. For actual coordinates, pair with
    a .gro or .xtc file.
    """

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        """
        Parse a GROMACS .itp file.

        Returns
        -------
        atoms : list[Atom]
            Atoms defined in the file
        trajectory : list[np.ndarray]
            Single frame at origin
        metadata : SystemMetadata
            Topology metadata including bonds
        """
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as exc:
            raise GromacsItpFormatError(f"Cannot read .itp file '{path.name}': {exc}") from exc

        atoms, bonds = self._parse_content(content, path)

        if not atoms:
            raise GromacsItpFormatError(f"No atoms found in '{path.name}'")

        # Create zero positions for all atoms
        positions = np.zeros((len(atoms), 3), dtype=np.float64)

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=None,
            timesteps=[0],
            bonds=bonds,
        )

        return atoms, [positions], metadata

    def _parse_content(
        self, content: str, path: Path
    ) -> tuple[list[Atom], np.ndarray | None]:
        """
        Parse GROMACS .itp file content.

        Extracts atoms and bonds sections.
        """
        atoms: list[Atom] = []
        bonds_list: list[tuple[int, int]] = []

        lines = content.split("\n")
        current_section = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            # Skip empty lines and comments
            if not line or line.startswith(";"):
                continue

            # Section headers
            if line.startswith("["):
                section = line.strip("[]").strip().lower()
                if section == "atoms":
                    current_section = "atoms"
                elif section == "bonds":
                    current_section = "bonds"
                else:
                    current_section = None
                continue

            # Parse atoms section
            if current_section == "atoms" and not line.startswith(";"):
                try:
                    parts = line.split()
                    if len(parts) >= 8:
                        atom_nr = int(parts[0])
                        atom_type = parts[1]
                        res_nr = int(parts[2])
                        res_name = parts[3]
                        atom_name = parts[4]
                        cg_nr = int(parts[5])
                        charge = float(parts[6])
                        mass = float(parts[7]) if len(parts) > 7 else None

                        atom = Atom(
                            id=atom_nr - 1,  # 0-indexed
                            type_id=None,
                            element=self._guess_element(atom_name),
                            x=0.0,
                            y=0.0,
                            z=0.0,
                            mass=mass,
                            residue_id=res_nr,
                            chain_id=None,
                            name=atom_name,
                        )
                        atoms.append(atom)
                except (ValueError, IndexError):
                    continue

            # Parse bonds section
            elif current_section == "bonds" and not line.startswith(";"):
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        atom_i = int(parts[0]) - 1  # 0-indexed
                        atom_j = int(parts[1]) - 1
                        if 0 <= atom_i < len(atoms) and 0 <= atom_j < len(atoms):
                            bonds_list.append((atom_i, atom_j))
                except (ValueError, IndexError):
                    continue

        # Convert bonds to PyVista line array format
        bonds = self._bonds_to_pyvista_format(bonds_list) if bonds_list else None

        return atoms, bonds

    @staticmethod
    def _guess_element(atom_name: str) -> str | None:
        """Guess element symbol from atom name."""
        if not atom_name:
            return None

        element_str = "".join(c for c in atom_name if c.isalpha()).upper()
        if element_str in {"H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I"}:
            return element_str

        first_char = atom_name[0].upper()
        if first_char in {"H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I"}:
            return first_char

        return None

    @staticmethod
    def _bonds_to_pyvista_format(bonds_list: list[tuple[int, int]]) -> np.ndarray | None:
        """Convert bond list to PyVista line array format."""
        if not bonds_list:
            return None

        line_data = []
        for i, j in bonds_list:
            line_data.extend([2, i, j])

        return np.array(line_data, dtype=np.int32) if line_data else None
