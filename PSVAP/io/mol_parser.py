"""
io/mol_parser.py
----------------
Parser for small molecule file formats (.mol2, .sdf, .mol).

RDKit DLL fix strategy
----------------------
On Windows, RDKit DLL errors occur when the conda environment has a
mismatched Visual C++ runtime.  The ONLY permanent fix is:
    conda install -c conda-forge rdkit --force-reinstall

However, we also provide a pure-Python fallback that can parse V2000
SDF/MOL files without RDKit for basic coordinate loading, so the app
doesn't crash while the user fixes their environment.

Bond extraction fix (this version)
------------------------------------
Previously metadata.bonds was never set → no bonds shown for SDF/MOL files.
Now:
- RDKit path: bonds extracted via mol.GetBonds() → stored in metadata.bonds
- Pure-Python path: V2000 bond block is now parsed and stored in metadata.bonds
Both paths convert to PyVista [2,i,j,...] format used by the viz engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata
from PSVAP.io.base_parser import BaseParser


class MolFormatError(ValueError):
    """Raised when a .mol2 / .sdf file cannot be parsed."""


_ELEMENT_TO_TYPE: dict[str, int] = {
    "C": 1, "N": 2, "O": 3, "S": 4, "H": 5, "P": 6,
    "F": 7, "CL": 8, "BR": 9, "I": 10,
    "FE": 11, "ZN": 12, "MG": 13,
}


def _try_import_rdkit():
    """Return rdkit.Chem or None if unavailable."""
    try:
        from rdkit import Chem
        return Chem
    except Exception:
        return None


def _bonds_to_pyvista(bonds: list[tuple[int, int]]) -> np.ndarray | None:
    """Convert (i,j) bond pair list to PyVista line array [2,i,j,...]."""
    if not bonds:
        return None
    arr: list[int] = []
    for i, j in bonds:
        arr.extend([2, int(i), int(j)])
    return np.array(arr, dtype=np.int64)


class MolParser(BaseParser):
    """
    Parses .mol2 and .sdf / .mol files.

    Tries RDKit first. Falls back to pure-Python V2000 parser if RDKit
    has a DLL error.  The fallback handles standard V2000 SDF/MOL files
    correctly for coordinates, elements, and bonds.
    """

    def parse(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        ext = path.suffix.lower()
        if ext not in {".mol2", ".sdf", ".mol"}:
            raise MolFormatError(f"Unsupported small molecule extension: {ext}")

        Chem = _try_import_rdkit()
        if Chem is not None:
            return self._parse_with_rdkit(path, ext, Chem)
        else:
            # RDKit unavailable — use pure-Python fallback for SDF/MOL
            if ext == ".mol2":
                raise MolFormatError(
                    "RDKit is required for .mol2 files.\n"
                    "Fix: conda install -c conda-forge rdkit --force-reinstall"
                )
            return self._parse_sdf_pure(path)

    # ------------------------------------------------------------------ #
    #  RDKit path                                                          #
    # ------------------------------------------------------------------ #

    def _parse_with_rdkit(self, path: Path, ext: str, Chem) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        if ext == ".mol2":
            mol = Chem.MolFromMol2File(str(path), removeHs=False)
            if mol is None:
                raise MolFormatError(f"RDKit could not parse MOL2 file: {path.name}")
            atoms, positions = self._rdkit_mol_to_atoms(mol)
            if not atoms:
                raise MolFormatError(f"No atoms found in MOL2 file: {path.name}")
            bond_array = _bonds_to_pyvista(self._rdkit_mol_to_bonds(mol))
            return atoms, [positions], SystemMetadata(
                source_path=path, timesteps=[0], bonds=bond_array)

        # SDF / MOL
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        if supplier is None:
            raise MolFormatError(f"RDKit cannot open SDF file: {path.name}")
        mols = [m for m in supplier if m is not None]
        if not mols:
            raise MolFormatError(f"No valid molecules found in: {path.name}")

        atoms, first_pos = self._rdkit_mol_to_atoms(mols[0])
        bond_array = _bonds_to_pyvista(self._rdkit_mol_to_bonds(mols[0]))
        trajectory = [first_pos]
        timesteps = [0]
        for i, mol in enumerate(mols[1:], 1):
            try:
                _, pos = self._rdkit_mol_to_atoms(mol)
                if pos.shape[0] == len(atoms):
                    trajectory.append(pos)
                    timesteps.append(i)
            except Exception:
                continue
        return atoms, trajectory, SystemMetadata(
            source_path=path, timesteps=timesteps, bonds=bond_array)

    @staticmethod
    def _rdkit_mol_to_atoms(mol) -> tuple[list[Atom], np.ndarray]:
        if mol.GetNumConformers() == 0:
            raise MolFormatError("Molecule has no 3D conformer.")
        conf = mol.GetConformer(0)
        atoms: list[Atom] = []
        coords: list[tuple] = []
        for i, rdatom in enumerate(mol.GetAtoms()):
            elem = rdatom.GetSymbol().upper()
            pos = conf.GetAtomPosition(i)
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
            mi = rdatom.GetMonomerInfo()
            atoms.append(Atom(
                id=i, type_id=_ELEMENT_TO_TYPE.get(elem, 0),
                element=elem or None, x=x, y=y, z=z,
                residue_id=int(mi.GetResidueNumber()) if mi else None,
                chain_id=mi.GetChainId() if mi else None,
                name=(mi.GetName().strip() if mi else rdatom.GetSymbol()) or None,
            ))
            coords.append((x, y, z))
        return atoms, np.array(coords, dtype=np.float64)

    @staticmethod
    def _rdkit_mol_to_bonds(mol) -> list[tuple[int, int]]:
        """Extract explicit bond pairs (0-indexed) from an RDKit molecule."""
        bonds: list[tuple[int, int]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bonds.append((min(i, j), max(i, j)))
        return bonds

    # ------------------------------------------------------------------ #
    #  Pure-Python V2000 fallback                                          #
    # ------------------------------------------------------------------ #

    def _parse_sdf_pure(self, path: Path) -> tuple[list[Atom], list[np.ndarray], SystemMetadata]:
        """
        Parse V2000 SDF/MOL without RDKit.
        Handles multi-molecule SDF (each molecule → one frame).
        Also reads the V2000 bond block so bonds render correctly.
        """
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise MolFormatError(f"Cannot read '{path.name}': {exc}") from exc

        records = text.split("$$$$")
        all_frames: list[np.ndarray] = []
        first_atoms: list[Atom] | None = None
        first_bonds: list[tuple[int, int]] | None = None
        timesteps: list[int] = []

        for rec_idx, record in enumerate(records):
            record = record.strip()
            if not record:
                continue
            try:
                atoms, positions, bonds = self._parse_v2000_record(record)
                if first_atoms is None:
                    first_atoms = atoms
                    first_bonds = bonds
                if positions.shape[0] == len(first_atoms or []):
                    all_frames.append(positions)
                    timesteps.append(rec_idx)
            except MolFormatError:
                continue

        if not first_atoms or not all_frames:
            raise MolFormatError(
                f"No valid V2000 molecule records found in: {path.name}\n"
                "Install RDKit for full SDF support:\n"
                "  conda install -c conda-forge rdkit --force-reinstall"
            )

        bond_array = _bonds_to_pyvista(first_bonds) if first_bonds else None
        return first_atoms, all_frames, SystemMetadata(
            source_path=path, timesteps=timesteps, bonds=bond_array)

    @staticmethod
    def _parse_v2000_record(record: str) -> tuple[list[Atom], np.ndarray, list[tuple[int, int]]]:
        """
        Parse a single V2000 MOL record.
        Returns (atoms, positions, bonds) where bonds is a list of (i, j) pairs.

        V2000 format:
          Lines 0-2: header (name, program, comment)
          Line 3:    counts — aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
                     aaa = n_atoms, bbb = n_bonds
          Lines 4..4+n_atoms-1:    atom block (x y z symbol ...)
          Lines 4+n_atoms onwards: bond block (atom1 atom2 type ...)
        """
        lines = record.splitlines()
        # Skip blank leading lines
        while lines and not lines[0].strip():
            lines.pop(0)
        if len(lines) < 4:
            raise MolFormatError("Record too short for V2000 format.")

        # Lines 0-2: header (name, program, comment) — skip
        # Line 3: counts line
        counts_line = lines[3] if len(lines) > 3 else ""
        try:
            n_atoms = int(counts_line[:3].strip())
            n_bonds_v = int(counts_line[3:6].strip()) if len(counts_line) >= 6 else 0
        except (ValueError, IndexError) as exc:
            raise MolFormatError(f"Cannot parse counts line: {counts_line!r}") from exc

        if len(lines) < 4 + n_atoms:
            raise MolFormatError(f"Record has {len(lines)} lines but needs {4 + n_atoms}.")

        atoms: list[Atom] = []
        coords: list[tuple] = []

        for i in range(n_atoms):
            atom_line = lines[4 + i]
            # V2000 atom line: 10-char x, 10-char y, 10-char z, space, 3-char symbol
            parts = atom_line.split()
            if len(parts) < 4:
                raise MolFormatError(f"Bad atom line: {atom_line!r}")
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as exc:
                raise MolFormatError(f"Bad coordinate: {atom_line!r}") from exc

            elem = parts[3].upper().strip()
            atoms.append(Atom(
                id=i, type_id=_ELEMENT_TO_TYPE.get(elem, 0),
                element=elem or None, x=x, y=y, z=z,
            ))
            coords.append((x, y, z))

        # Read bond block (lines after atom block)
        bonds: list[tuple[int, int]] = []
        bond_start = 4 + n_atoms
        for k in range(n_bonds_v):
            if bond_start + k >= len(lines):
                break
            bond_line = lines[bond_start + k]
            # V2000 bond line: atom1 atom2 type ...  (1-indexed)
            bp = bond_line.split()
            if len(bp) >= 2:
                try:
                    ai = int(bp[0]) - 1   # 1-indexed → 0-indexed
                    aj = int(bp[1]) - 1
                    bonds.append((min(ai, aj), max(ai, aj)))
                except (ValueError, IndexError):
                    continue

        return atoms, np.array(coords, dtype=np.float64), bonds