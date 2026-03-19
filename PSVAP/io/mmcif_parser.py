"""
io/mmcif_parser.py
------------------
Parser for mmCIF / PDBx files (.cif, .mmcif) using the Gemmi library.

mmCIF is the modern replacement for PDB format and is the primary format
for structures deposited in the wwPDB.  It is a structured data format
(not line-oriented like PDB) and supports:
  - Multiple assemblies
  - Proper handling of alternate conformations
  - Full symmetry information
  - Very large structures (no 99,999-atom limit like PDB)

This parser reads atom_site records from the mmCIF file, which contain
the 3D coordinates of all atoms.

Rule compliance
---------------
  Rule 4  : One parser — mmCIF only.
  Rule 2  : No GUI imports.
  Rule 7  : Tested in tests/test_parsers_phase1.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser

# Reuse element → type_id mapping from pdb_parser
_ELEMENT_TO_TYPE: dict[str, int] = {
    "C": 1, "N": 2, "O": 3, "S": 4, "H": 5,
    "P": 6, "F": 7, "CL": 8, "BR": 9, "I": 10,
    "FE": 11, "ZN": 12, "MG": 13, "CA": 14, "MN": 15,
    "CU": 16, "NA": 17, "K": 18,
}


class MMCIFFormatError(ValueError):
    """Raised when an mmCIF file cannot be parsed."""


class MMCIFParser(BaseParser):
    """
    Parses mmCIF / PDBx files (.cif, .mmcif) using Gemmi.

    Produces a single-frame trajectory (mmCIF is a static structure format).
    """

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        try:
            import gemmi
        except ImportError as exc:
            raise MMCIFFormatError(
                "Gemmi is required for mmCIF support. "
                "Install it with: pip install gemmi"
            ) from exc

        try:
            doc = gemmi.cif.read(str(path))
        except Exception as exc:
            raise MMCIFFormatError(f"Cannot read mmCIF file '{path.name}': {exc}") from exc

        # mmCIF files can have multiple blocks; the first data block is the
        # structure block.
        if len(doc) == 0:
            raise MMCIFFormatError(f"Empty mmCIF file: {path.name}")

        block = doc[0]
        atoms, positions = self._parse_atom_site(block, path.name)

        if not atoms:
            raise MMCIFFormatError(f"No atom_site records found in: {path.name}")

        box_bounds = self._extract_cell(block)
        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=[0],
        )
        return atoms, [positions], metadata

    # ------------------------------------------------------------------ #
    #  atom_site parsing                                                   #
    # ------------------------------------------------------------------ #

    def _parse_atom_site(
        self, block, filename: str
    ) -> tuple[list[Atom], np.ndarray]:
        """
        Read the _atom_site loop from the mmCIF block.

        We look for these columns (using standard mmCIF tags):
          _atom_site.id               → atom serial (integer)
          _atom_site.type_symbol      → element
          _atom_site.label_atom_id    → atom name
          _atom_site.label_comp_id    → residue name
          _atom_site.label_asym_id    → chain ID
          _atom_site.label_seq_id     → residue sequence number
          _atom_site.Cartn_x/y/z      → Cartesian coordinates (Angstrom)
          _atom_site.occupancy        → occupancy (for alt conf filtering)
          _atom_site.label_alt_id     → alternate conformation indicator
        """
        # Find the atom_site table
        try:
            table = block.find(
                "_atom_site.",
                [
                    "id", "type_symbol",
                    "label_atom_id", "label_comp_id",
                    "label_asym_id", "label_seq_id",
                    "Cartn_x", "Cartn_y", "Cartn_z",
                    "label_alt_id",
                ],
            )
        except Exception as exc:
            raise MMCIFFormatError(
                f"Cannot find atom_site table in {filename}: {exc}"
            ) from exc

        atoms: list[Atom] = []
        coords: list[tuple[float, float, float]] = []
        seen_ids: set[int] = set()  # for deduplication of alt conformations

        for row in table:
            # Skip alternate conformations — keep only "" or "A"
            alt = _cif_str(row, 9)
            if alt not in {"", ".", "A", "?"}:
                continue

            try:
                atom_id = int(_cif_str(row, 0) or "0")
            except ValueError:
                atom_id = len(atoms)

            # Skip duplicate atom IDs from alt conformations
            if atom_id in seen_ids:
                continue
            seen_ids.add(atom_id)

            element = (_cif_str(row, 1) or "").strip().upper()
            atom_name = _cif_str(row, 2)
            res_name = _cif_str(row, 3)
            chain_id = _cif_str(row, 4)

            try:
                res_id = int(_cif_str(row, 5) or "0")
            except ValueError:
                res_id = 0

            try:
                x = float(_cif_str(row, 6) or "0")
                y = float(_cif_str(row, 7) or "0")
                z = float(_cif_str(row, 8) or "0")
            except ValueError as exc:
                raise MMCIFFormatError(
                    f"Bad coordinate in {filename}: {exc}"
                ) from exc

            type_id = _ELEMENT_TO_TYPE.get(element, 0)

            atoms.append(
                Atom(
                    id=atom_id,
                    type_id=type_id,
                    element=element if element else None,
                    x=x,
                    y=y,
                    z=z,
                    residue_id=res_id,
                    chain_id=chain_id if chain_id not in {".", "?"} else None,
                    name=atom_name if atom_name not in {".", "?"} else None,
                )
            )
            coords.append((x, y, z))

        positions = np.array(coords, dtype=np.float64)
        return atoms, positions

    # ------------------------------------------------------------------ #
    #  Unit cell extraction                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_cell(block) -> np.ndarray | None:
        """
        Read unit cell dimensions from _cell.length_a/b/c if present.
        Returns (3, 2) array [[0, a], [0, b], [0, c]] in Angstroms.
        """
        try:
            a = float(block.find_value("_cell.length_a").replace("'", ""))
            b = float(block.find_value("_cell.length_b").replace("'", ""))
            c = float(block.find_value("_cell.length_c").replace("'", ""))
            return np.array([[0.0, a], [0.0, b], [0.0, c]], dtype=np.float64)
        except Exception:
            return None


# ── Utility ────────────────────────────────────────────────────────────────

def _cif_str(row, col: int) -> str:
    """Safely read a column value from a Gemmi CIF row."""
    try:
        val = str(row[col]).strip()
        return val if val not in {".", "?", ""} else ""
    except (IndexError, TypeError):
        return ""