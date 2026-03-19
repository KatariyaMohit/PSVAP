"""
io/pdb_parser.py
----------------
Parser for Protein Data Bank (.pdb) files using Biopython.

Biopython element detection fix
--------------------------------
Biopython's PDB parser DOES read the element column (cols 76-78) but it
only sets atom.element if the column is non-blank.  Many real-world PDB
files have blank element columns — in that case Biopython leaves element
as an empty string or None.

Our fix: if Biopython gives us an empty element, we derive it from the
atom NAME using a reliable heuristic (strip leading digits and position
numbers, take the first 1-2 uppercase letters).

Phase 4 addition
----------------
_build_atoms now sets atom.resname from the Biopython residue object.
This enables mutation_engine, ramachandran, and sequence extraction to
work correctly with PDB files.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class PDBFormatError(ValueError):
    """Raised when a PDB file cannot be parsed."""


_ELEMENT_TO_TYPE: dict[str, int] = {
    "C": 1, "N": 2, "O": 3, "S": 4, "H": 5,
    "P": 6, "F": 7, "CL": 8, "BR": 9, "I": 10,
    "FE": 11, "ZN": 12, "MG": 13, "CA": 14, "MN": 15,
    "CU": 16, "NA": 17, "K": 18,
}

_ATOM_NAME_TO_ELEMENT: dict[str, str] = {
    "N": "N", "CA": "C", "CB": "C", "C": "C", "O": "O",
    "OXT": "O", "H": "H", "HA": "H", "HB": "H",
    "SD": "S", "SG": "S", "FE": "FE", "ZN": "ZN",
    "MG": "MG", "CA2": "CA", "MN": "MN", "P": "P",
    "OP1": "O", "OP2": "O", "O3'": "O", "O5'": "O",
}


def _derive_element(atom_name: str) -> str:
    """
    Derive element symbol from atom name when the element column is blank.
    """
    name = atom_name.strip().upper()
    if name in _ATOM_NAME_TO_ELEMENT:
        return _ATOM_NAME_TO_ELEMENT[name]
    stripped = name.lstrip("0123456789")
    if len(stripped) >= 2 and stripped[:2] in _ELEMENT_TO_TYPE:
        return stripped[:2]
    if stripped and stripped[0].isalpha():
        return stripped[0]
    return "C"


class PDBParser(BaseParser):
    """
    Parses PDB files (.pdb) using Biopython's PDBParser.

    Multi-model PDB files (NMR ensembles) are read as trajectory frames.
    Single-model files produce a one-frame trajectory.
    """

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        try:
            from Bio.PDB import PDBParser as BioPDBParser
            from Bio.PDB.PDBExceptions import PDBConstructionWarning
        except ImportError as exc:
            raise PDBFormatError(
                "Biopython is required for PDB support. "
                "Install it with: pip install biopython"
            ) from exc

        bio_parser = BioPDBParser(QUIET=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
            try:
                structure = bio_parser.get_structure("mol", str(path))
            except Exception as exc:
                raise PDBFormatError(
                    f"Cannot read PDB file '{path.name}': {exc}"
                ) from exc

        models = list(structure.get_models())
        if not models:
            raise PDBFormatError(f"No models found in PDB file: {path.name}")

        atoms = self._build_atoms(models[0])
        if not atoms:
            raise PDBFormatError(
                f"No ATOM/HETATM records found in: {path.name}\n"
                "The file exists but contains no 3D coordinate records."
            )

        trajectory: list[np.ndarray] = []
        timesteps: list[int] = []
        for i, model in enumerate(models):
            positions = self._extract_positions(model, len(atoms))
            trajectory.append(positions)
            timesteps.append(i)

        box_bounds = self._extract_cryst1_box(path)

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=timesteps,
        )
        return atoms, trajectory, metadata

    # ------------------------------------------------------------------ #
    #  Atom building                                                       #
    # ------------------------------------------------------------------ #

    def _build_atoms(self, model) -> list[Atom]:
        atoms: list[Atom] = []
        atom_serial = 0

        for chain in model.get_chains():
            chain_id = chain.get_id().strip() or None
            for residue in chain.get_residues():
                res_id = residue.get_id()[1]
                # PHASE 4 FIX: read residue name from Biopython residue object.
                # residue.get_resname() returns e.g. "ALA", "GLY", "HOH".
                # This populates atom.resname so that mutation_engine,
                # ramachandran, and sequence extraction work correctly.
                resname = residue.get_resname().strip().upper() or None

                for bio_atom in residue.get_atoms():
                    if bio_atom.is_disordered():
                        bio_atom = bio_atom.disordered_get_list()[0]

                    element = (bio_atom.element or "").strip().upper()
                    atom_name_str = bio_atom.get_name().strip()
                    if not element or element in {"", "X"}:
                        element = _derive_element(atom_name_str)

                    type_id = _ELEMENT_TO_TYPE.get(element, 0)
                    coord = bio_atom.get_vector().get_array()

                    atoms.append(
                        Atom(
                            id=atom_serial,
                            type_id=type_id,
                            element=element if element else None,
                            x=float(coord[0]),
                            y=float(coord[1]),
                            z=float(coord[2]),
                            charge=None,
                            mass=None,
                            residue_id=int(res_id),
                            chain_id=chain_id,
                            name=atom_name_str,
                            resname=resname,   # Phase 4 addition
                        )
                    )
                    atom_serial += 1
        return atoms

    # ------------------------------------------------------------------ #
    #  Position extraction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_positions(model, expected_n: int) -> np.ndarray:
        coords: list[tuple[float, float, float]] = []
        for chain in model.get_chains():
            for residue in chain.get_residues():
                for bio_atom in residue.get_atoms():
                    if bio_atom.is_disordered():
                        bio_atom = bio_atom.disordered_get_list()[0]
                    c = bio_atom.get_vector().get_array()
                    coords.append((float(c[0]), float(c[1]), float(c[2])))

        positions = np.array(coords, dtype=np.float64)
        if positions.shape[0] != expected_n:
            padded = np.zeros((expected_n, 3), dtype=np.float64)
            n = min(positions.shape[0], expected_n)
            padded[:n] = positions[:n]
            return padded
        return positions

    # ------------------------------------------------------------------ #
    #  CRYST1 box                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_cryst1_box(path: Path) -> np.ndarray | None:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
                for line in fh:
                    if line.startswith("CRYST1"):
                        parts = line.split()
                        if len(parts) >= 4:
                            a = float(parts[1])
                            b = float(parts[2])
                            c = float(parts[3])
                            return np.array(
                                [[0.0, a], [0.0, b], [0.0, c]], dtype=np.float64
                            )
        except Exception:
            pass
        return None