"""
io/lammps_parser.py
--------------------
LAMMPS .lammpstrj and .data parser.

Key behaviour
-------------
amyloid_topo.lammpstrj  →  actually a LAMMPS DATA file (starts with "LAMMPS Description")
amyloid_trj.lammpstrj   →  LAMMPS DUMP file (starts with "ITEM: TIMESTEP")

Detection is done by reading file CONTENT, NOT the extension.

Bugs fixed in this version
--------------------------
1. "Atoms # bond" section header was excluded by `not sl.startswith("atoms #")`
   — now ANY line starting with "atoms" (except "atom type") enters the Atoms block.
2. Atom parsing loop read exactly n_atoms lines blindly; if a section keyword like
   "Bonds" appeared before that many lines the parser crashed.
   — now the loop breaks when it hits any section keyword.
3. Type IDs in amyloid file are 0-indexed (types 0,1,2 in Masses + Atoms).
   — id_offset auto-detected from first atom ID; types also offset the same way.
4. Bond array format is PyVista lines format: [2, i, j,  2, i2, j2, ...]
   Bonds are built ONCE from the Bonds section and stored in metadata.bonds.
   The viz engine updates only mesh.points per frame — no per-frame bond rebuild.
5. DUMP file type_id: do NOT subtract 1 from type column.
   The amyloid simulation uses 0-indexed types (0,1,2) in both data and dump files.
   Subtracting 1 produced -1,0,1 instead of 0,1,2, causing wrong legend colours
   and broken type==0 selections (matched wrong atoms).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PSVAP.core.atom import Atom
from PSVAP.io.base_parser import BaseParser

# Section header keywords that mark the END of a data block
_SECTION_ENDS = {
    "bonds", "angles", "dihedrals", "impropers", "velocities",
    "masses", "pair coeffs", "bond coeffs", "angle coeffs",
    "dihedral coeffs", "improper coeffs",
}


class LammpsFormatError(ValueError):
    """Raised when a LAMMPS file cannot be parsed."""


class LammpsParser(BaseParser):
    """
    Parses LAMMPS .lammpstrj dump files and LAMMPS data files.

    amyloid_topo.lammpstrj  →  data file format  →  _parse_data()
    amyloid_trj.lammpstrj   →  dump file format  →  _parse_dump()
    """

    def parse(self, path: Path):
        """
        Route to data or dump parser by reading file content (not extension).
        """
        if self._is_data_file(path):
            return self._parse_data(path)
        return self._parse_dump(path)

    # ──────────────────────────────────────────────────────────────────────
    #  Format detection
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_data_file(path: Path) -> bool:
        """
        True  → LAMMPS data file  (topology: atoms + bonds + box)
        False → LAMMPS dump file  (trajectory: ITEM: TIMESTEP blocks)
        """
        try:
            with path.open(encoding="utf-8-sig", errors="replace") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    # Dump file signature — unambiguous
                    if stripped.startswith("ITEM:"):
                        return False
                    # Data file signatures
                    if stripped.lower().startswith("lammps description"):
                        return True
                    sl = stripped.lower()
                    # count lines like "8000 atoms", "3 atom types"
                    if (sl.endswith("atoms") or sl.endswith("bonds") or
                            sl.endswith("atom types") or sl.endswith("bond types")):
                        return True
        except Exception:
            pass
        return False

    # ──────────────────────────────────────────────────────────────────────
    #  DATA file parser  (amyloid_topo.lammpstrj)
    # ──────────────────────────────────────────────────────────────────────

    def _parse_data(self, path: Path):
        """
        Parse a LAMMPS data file.

        Handles:
        - "Atoms # bond" style section headers (the '# bond' annotation is
          the atom_style specifier, NOT a comment that excludes the section)
        - 0-indexed atom IDs (amyloid file starts atoms at 0, not 1)
        - 1-indexed atom IDs (standard LAMMPS, starts at 1)
        - Auto-detects indexing from first atom ID seen

        Data file Atoms line format for atom_style=bond:
            atom_id  mol_id  type  x  y  z
        """
        from PSVAP.core.system_model import SystemMetadata

        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise LammpsFormatError(f"Cannot read '{path.name}': {exc}") from exc

        lines = text.splitlines()

        # ── Pass 1: read header counts and box ────────────────────────────
        n_atoms    = 0
        n_bonds    = 0
        box_rows: list[list[float]] = []
        atom_style = "bond"   # default for amyloid-type files

        for raw in lines:
            s  = raw.strip()
            sl = s.lower()

            # Count lines: "8000 atoms", "15000 bonds"
            if (sl.endswith("atoms")
                    and not any(k in sl for k in ("bond", "angle", "dihedral", "improp"))):
                try:
                    n_atoms = int(s.split()[0])
                except (ValueError, IndexError):
                    pass
            elif sl.endswith("bonds") and "bond type" not in sl:
                try:
                    n_bonds = int(s.split()[0])
                except (ValueError, IndexError):
                    pass
            elif "xlo xhi" in sl:
                parts = s.split()
                try:
                    box_rows.append([float(parts[0]), float(parts[1])])
                except Exception:
                    pass
            elif "ylo yhi" in sl:
                parts = s.split()
                try:
                    box_rows.append([float(parts[0]), float(parts[1])])
                except Exception:
                    pass
            elif "zlo zhi" in sl:
                parts = s.split()
                try:
                    box_rows.append([float(parts[0]), float(parts[1])])
                except Exception:
                    pass
            # atom_style from "Atoms # bond" header
            elif sl.startswith("atoms #"):
                rest = s.split("#")
                if len(rest) > 1:
                    atom_style = rest[1].strip().lower()

        box_np = np.array(box_rows) if len(box_rows) == 3 else None

        # ── Pass 2: parse Atoms and Bonds sections ─────────────────────────
        atoms: list[Atom]           = []
        bond_pairs: list[tuple[int, int]] = []
        id_offset: int | None       = None
        i = 0

        while i < len(lines):
            s  = lines[i].strip()
            sl = s.lower()

            # ── Atoms section ─────────────────────────────────────────────
            # Matches: "Atoms", "Atoms # bond", "Atoms # molecular", etc.
            # Does NOT match: "Atom Types", "atom type"
            if (sl.startswith("atoms") and
                    not sl.startswith("atom type") and
                    "angle" not in sl and "dihedral" not in sl):

                # Grab atom_style from inline annotation
                if "#" in s:
                    atom_style = s.split("#")[1].strip().lower()

                i += 1
                # Skip blank lines
                while i < len(lines) and not lines[i].strip():
                    i += 1

                parsed = 0
                while parsed < n_atoms and i < len(lines):
                    raw = lines[i].strip()
                    i += 1

                    if not raw:
                        continue

                    # Stop if we hit another section keyword
                    raw_l = raw.lower().split()[0] if raw.split() else ""
                    if raw.lower() in _SECTION_ENDS or raw_l in _SECTION_ENDS:
                        # Back up one so outer loop sees this line
                        i -= 1
                        break

                    parts = raw.split()
                    try:
                        raw_id = int(parts[0])

                        # Auto-detect 0- vs 1-indexed
                        if id_offset is None:
                            id_offset = 0 if raw_id == 0 else 1

                        aid = raw_id - id_offset   # always 0-indexed internally

                        # Parse by atom_style
                        if atom_style in {"bond", "molecular"}:
                            # atom_id  mol_id  type  x  y  z
                            mol_id = int(parts[1])
                            atyp   = int(parts[2])
                            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        elif atom_style == "full":
                            # atom_id  mol_id  type  charge  x  y  z
                            mol_id = int(parts[1])
                            atyp   = int(parts[2])
                            x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                        elif atom_style == "charge":
                            # atom_id  type  charge  x  y  z
                            mol_id = None
                            atyp   = int(parts[1])
                            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        else:
                            # atomic: atom_id  type  x  y  z
                            mol_id = None
                            atyp   = int(parts[1])
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

                        # type_id: same offset as atom_id
                        type_id = atyp - id_offset

                        atoms.append(Atom(
                            id=aid,
                            type_id=type_id,
                            element=None,
                            x=x, y=y, z=z,
                            residue_id=mol_id,
                        ))
                        parsed += 1

                    except (IndexError, ValueError):
                        continue

                continue   # back to outer while

            # ── Bonds section ─────────────────────────────────────────────
            if (sl == "bonds" or
                    (sl.startswith("bonds") and "bond type" not in sl)):
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1

                off = id_offset if id_offset is not None else 1
                parsed = 0

                while parsed < n_bonds and i < len(lines):
                    raw = lines[i].strip()
                    i += 1

                    if not raw:
                        continue

                    # Stop on next section keyword
                    raw_l = raw.lower().split()[0] if raw.split() else ""
                    if raw.lower() in _SECTION_ENDS or raw_l in _SECTION_ENDS:
                        i -= 1
                        break

                    parts = raw.split()
                    if len(parts) < 4:
                        continue
                    try:
                        # bond_id  bond_type  atom_i  atom_j
                        ai = int(parts[2]) - off
                        aj = int(parts[3]) - off
                        bond_pairs.append((min(ai, aj), max(ai, aj)))
                        parsed += 1
                    except (IndexError, ValueError):
                        continue

                continue

            i += 1

        if not atoms:
            raise LammpsFormatError(
                f"No atoms found in '{path.name}'.\n"
                f"File appears to be a LAMMPS data file but no Atoms section was parsed.\n"
                f"Detected atom_style='{atom_style}', n_atoms={n_atoms}"
            )

        positions = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)

        # Build PyVista bond line array
        if bond_pairs:
            bond_array = self._pairs_to_pyvista(bond_pairs)
        else:
            bond_array = self._detect_bonds_pyvista(positions)

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_np,
            timesteps=[0],
            bonds=bond_array,
        )
        return atoms, [positions], metadata

    # ──────────────────────────────────────────────────────────────────────
    #  DUMP file parser  (amyloid_trj.lammpstrj)
    # ──────────────────────────────────────────────────────────────────────

    def _parse_dump(self, path: Path):
        """
        Parse LAMMPS dump file (ITEM: TIMESTEP blocks).

        Works for single-frame (topology use) or multi-frame (trajectory).
        On frame 0: builds atom list from id/type/mol columns.
        On subsequent frames: records positions only.

        Type ID note: types are stored AS-IS from the dump file.
        The amyloid simulation uses 0-indexed types (0,1,2) matching the
        data file. We do NOT subtract 1 from the type column — that would
        produce -1,0,1 instead of 0,1,2 for this file.
        """
        from PSVAP.core.system_model import SystemMetadata

        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise LammpsFormatError(f"Cannot read '{path.name}': {exc}") from exc

        lines     = text.splitlines()
        n         = len(lines)
        atoms: list[Atom] = []
        trajectory: list[np.ndarray] = []
        timesteps:  list[int]        = []
        box_bounds: np.ndarray | None = None
        n_atoms    = 0
        frame_count = 0
        i = 0

        while i < n:
            line = lines[i].strip()

            if line == "ITEM: TIMESTEP":
                i += 1
                try:
                    ts = int(lines[i].strip())
                except (ValueError, IndexError):
                    ts = frame_count
                timesteps.append(ts)
                i += 1
                continue

            if line == "ITEM: NUMBER OF ATOMS":
                i += 1
                try:
                    n_atoms = int(lines[i].strip())
                except (ValueError, IndexError):
                    n_atoms = 0
                i += 1
                continue

            if line.startswith("ITEM: BOX BOUNDS"):
                rows = []
                i += 1
                for _ in range(3):
                    if i < n:
                        parts = lines[i].split()
                        try:
                            rows.append([float(parts[0]), float(parts[1])])
                        except (ValueError, IndexError):
                            rows.append([0.0, 0.0])
                        i += 1
                if len(rows) == 3:
                    box_bounds = np.array(rows)
                continue

            if line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]
                col = {h: idx for idx, h in enumerate(headers)}

                id_col   = col.get("id",   0)
                type_col = col.get("type", 1)
                mol_col  = col.get("mol",  None)
                x_col = col.get("x",  col.get("xu", col.get("xs", 2)))
                y_col = col.get("y",  col.get("yu", col.get("ys", 3)))
                z_col = col.get("z",  col.get("zu", col.get("zs", 4)))

                i += 1
                positions: list[list[float]] = []
                frame_atoms: list[Atom] = []

                for _ in range(n_atoms):
                    if i >= n:
                        break
                    parts = lines[i].split()
                    i += 1
                    if not parts:
                        continue
                    try:
                        # Atom ID is 1-indexed in LAMMPS → 0-indexed internally
                        aid  = int(parts[id_col]) - 1
                        # Type ID: stored as-is (0-indexed in this simulation).
                        # Do NOT subtract 1 — would produce -1 for type=0.
                        atyp = int(parts[type_col])
                        x = float(parts[x_col])
                        y = float(parts[y_col])
                        z = float(parts[z_col])
                    except (IndexError, ValueError):
                        positions.append([0.0, 0.0, 0.0])
                        continue

                    mol_id = None
                    if mol_col is not None:
                        try:
                            mol_id = int(parts[mol_col])
                        except (IndexError, ValueError):
                            pass

                    positions.append([x, y, z])

                    if frame_count == 0:
                        frame_atoms.append(Atom(
                            id=aid,
                            type_id=atyp,
                            element=None,
                            x=x, y=y, z=z,
                            residue_id=mol_id,
                        ))

                if frame_count == 0:
                    atoms = frame_atoms

                trajectory.append(np.array(positions, dtype=np.float64))
                frame_count += 1
                continue

            i += 1

        if not atoms:
            raise LammpsFormatError(
                f"No atoms found in '{path.name}'.\n"
                f"Expected LAMMPS dump format with 'ITEM: ATOMS' section."
            )
        if not trajectory:
            raise LammpsFormatError(
                f"No valid frames found in '{path.name}'."
            )

        # KDTree bonds from first frame (dump files have no explicit bonds)
        bond_array = self._detect_bonds_pyvista(trajectory[0])

        metadata = SystemMetadata(
            source_path=path,
            box_bounds=box_bounds,
            timesteps=timesteps,
            bonds=bond_array,
        )
        return atoms, trajectory, metadata

    # ──────────────────────────────────────────────────────────────────────
    #  Bond helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_bonds_pyvista(
        positions: np.ndarray,
        cutoff: float = 1.8,
    ) -> np.ndarray | None:
        """
        KDTree bond detection → PyVista line array [2,i,j, 2,i2,j2,...].

        Same approach as the original plotter.py — fast, built once.
        cutoff=1.8 Å works for coarse-grained LAMMPS beads.
        For amyloid simulations with large bead spacing, use a larger cutoff.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return None

        if len(positions) < 2:
            return None

        try:
            tree  = cKDTree(positions)
            pairs = list(tree.query_pairs(cutoff))
            if not pairs:
                return None
            bond_data: list[int] = []
            for p in pairs:
                bond_data.extend([2, int(p[0]), int(p[1])])
            return np.array(bond_data, dtype=np.int64)
        except Exception:
            return None

    @staticmethod
    def _pairs_to_pyvista(pairs: list[tuple[int, int]]) -> np.ndarray:
        """Convert (i,j) pair list to PyVista line array [2,i,j, ...]."""
        bond_data: list[int] = []
        for i, j in pairs:
            bond_data.extend([2, int(i), int(j)])
        return np.array(bond_data, dtype=np.int64)