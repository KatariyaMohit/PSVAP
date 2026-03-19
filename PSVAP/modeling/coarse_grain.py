"""
modeling/coarse_grain.py
-------------------------
Feature 24: Coarse-Grained Setup (MARTINI force field).

Converts all-atom structures to MARTINI 3 coarse-grained representation
using martinize2 as a subprocess. Also provides a pure-Python bead
mapping for visualisation when martinize2 is not installed.

Public API
----------
  run_martinize2(pdb_path, output_dir, martini_version=3,
                 martinize_executable='martinize2')
      → CGResult

  build_cg_beads(atoms, positions)
      → (cg_atoms, cg_positions)   (simple centre-of-mass mapping)

  CGResult (dataclass)
      cg_structure_path, topology_path, bead_map, n_beads, n_atoms
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BeadMap:
    """Mapping from one CG bead to its constituent all-atom atoms."""
    bead_name:   str
    bead_type:   str       # MARTINI bead type e.g. 'P5', 'C1', 'Na'
    residue_id:  int
    resname:     str
    atom_indices: list[int]  # indices into original all-atom list
    center:      np.ndarray  # (3,) centre-of-mass position


@dataclass
class CGResult:
    """Result of coarse-graining."""
    n_atoms:            int
    n_beads:            int
    bead_map:           list[BeadMap]
    cg_structure_path:  Path | None = None
    topology_path:      Path | None = None
    itp_path:           Path | None = None
    method:             str = "martinize2"
    warnings:           list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"COARSE-GRAINING COMPLETE\n",
            f"  Method        : {self.method}",
            f"  All-atom atoms: {self.n_atoms}",
            f"  CG beads      : {self.n_beads}",
            f"  Reduction     : {self.n_atoms/max(self.n_beads,1):.1f}× fewer particles",
        ]
        if self.cg_structure_path:
            lines.append(f"  Structure     : {self.cg_structure_path}")
        if self.topology_path:
            lines.append(f"  Topology      : {self.topology_path}")
        if self.itp_path:
            lines.append(f"  ITP file      : {self.itp_path}")
        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ── MARTINI residue-to-bead mapping (simplified) ──────────────────────────
# Maps amino acid three-letter code to list of (bead_name, bead_type, atom_names)
# Based on MARTINI 3 protein mapping
_MARTINI3_PROTEIN: dict[str, list[tuple[str, str, list[str]]]] = {
    "ALA": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC1", ["CB"])],
    "GLY": [("BB", "P2", ["N", "CA", "C", "O"])],
    "VAL": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC2", ["CB", "CG1", "CG2"])],
    "LEU": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC2", ["CB", "CG", "CD1", "CD2"])],
    "ILE": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC2", ["CB", "CG1", "CG2", "CD1"])],
    "PHE": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC4", ["CB", "CG"]),
            ("SC2", "SC4", ["CD1", "CE1"]),
            ("SC3", "SC4", ["CD2", "CE2", "CZ"])],
    "TRP": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC4", ["CB", "CG", "CD1"]),
            ("SC2", "SC4", ["CD2", "CE2", "NE1"]),
            ("SC3", "SC4", ["CE3", "CZ3", "CH2"]),
            ("SC4", "SC4", ["CZ2"])],
    "MET": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "C5",  ["CB", "CG", "SD", "CE"])],
    "SER": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "P5",  ["CB", "OG"])],
    "THR": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "P5",  ["CB", "OG1", "CG2"])],
    "CYS": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "C5",  ["CB", "SG"])],
    "TYR": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "SC4", ["CB", "CG"]),
            ("SC2", "SC4", ["CD1", "CE1"]),
            ("SC3", "P5",  ["CD2", "CE2", "CZ", "OH"])],
    "ASN": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "P5",  ["CB", "CG", "OD1", "ND2"])],
    "GLN": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "P5",  ["CB", "CG", "CD", "OE1", "NE2"])],
    "ASP": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "Qa",  ["CB", "CG", "OD1", "OD2"])],
    "GLU": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "Qa",  ["CB", "CG", "CD", "OE1", "OE2"])],
    "LYS": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "C3",  ["CB", "CG", "CD"]),
            ("SC2", "Qd",  ["CE", "NZ"])],
    "ARG": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "N5",  ["CB", "CG", "CD"]),
            ("SC2", "Qd",  ["NE", "CZ", "NH1", "NH2"])],
    "HIS": [("BB", "P2", ["N", "CA", "C", "O"]),
            ("SC1", "TC4", ["CB", "CG"]),
            ("SC2", "TN6d",["ND1", "CE1"]),
            ("SC3", "TN5a",["NE2", "CD2"])],
    "PRO": [("BB", "N4",  ["N", "CA", "C", "O"]),
            ("SC1", "C3",  ["CB", "CG", "CD"])],
}
# Add common HIS variants
for _v in ("HSD", "HSE", "HSP", "HIE", "HID", "HIP"):
    _MARTINI3_PROTEIN[_v] = _MARTINI3_PROTEIN["HIS"]


# ── Public API ────────────────────────────────────────────────────────────

def check_martinize2_available(executable: str = "martinize2") -> bool:
    """Return True if martinize2 is found in PATH."""
    try:
        subprocess.run(
            [executable, "--help"],
            capture_output=True, timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_martinize2(
    pdb_path: str | Path,
    output_dir: str | Path = "cg_output",
    martini_version: int = 3,
    martinize_executable: str = "martinize2",
    dssp_executable: str = "dssp",
) -> CGResult:
    """
    Run martinize2 to convert all-atom PDB to MARTINI CG topology.

    Parameters
    ----------
    pdb_path             : path to all-atom PDB file
    output_dir           : directory to write CG files into
    martini_version      : 2 or 3 (default 3 for MARTINI 3)
    martinize_executable : name or path of martinize2 executable
    dssp_executable      : name or path of DSSP (for secondary structure)

    Returns
    -------
    CGResult

    Raises
    ------
    FileNotFoundError if martinize2 not found
    FileNotFoundError if pdb_path does not exist
    RuntimeError if martinize2 fails
    """
    pdb_path = Path(pdb_path)
    out      = Path(output_dir)

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    out.mkdir(parents=True, exist_ok=True)

    if not check_martinize2_available(martinize_executable):
        raise FileNotFoundError(
            f"martinize2 not found at '{martinize_executable}'.\n"
            "Install with: pip install martinize2 vermouth\n"
            "Ensure 'martinize2' is in your PATH."
        )

    out_gro = out / "cg_structure.gro"
    out_itp = out / "molecule.itp"

    ff_flag = f"-ff martini{martini_version}"

    cmd = [
        martinize_executable,
        "-f",   str(pdb_path),
        "-o",   str(out / "topol.top"),
        "-x",   str(out_gro),
        "-ff",  f"martini{martini_version}",
    ]

    # Add DSSP if available
    try:
        subprocess.run([dssp_executable, "--version"],
                       capture_output=True, timeout=5)
        cmd += ["-dssp", dssp_executable]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("martinize2 timed out after 120 seconds.")

    if proc.returncode != 0:
        raise RuntimeError(
            f"martinize2 failed (exit {proc.returncode}):\n{proc.stderr[:600]}"
        )

    # Count beads from output GRO file
    n_beads = 0
    if out_gro.exists():
        lines = out_gro.read_text().splitlines()
        if len(lines) >= 2:
            try:
                n_beads = int(lines[1].strip())
            except ValueError:
                pass

    # Find generated ITP files
    itp_files = list(out.glob("*.itp"))
    itp_path = itp_files[0] if itp_files else None

    return CGResult(
        n_atoms=0,   # unknown without reading original PDB
        n_beads=n_beads,
        bead_map=[],
        cg_structure_path=out_gro if out_gro.exists() else None,
        topology_path=out / "topol.top" if (out / "topol.top").exists() else None,
        itp_path=itp_path,
        method="martinize2",
    )


def build_cg_beads(
    atoms: list,
    positions: np.ndarray,
    martini_version: int = 3,
) -> tuple[list[BeadMap], np.ndarray]:
    """
    Build CG bead mapping without martinize2 (pure Python fallback).

    Uses the residue-to-bead mapping table to group atoms into beads
    and computes the centre-of-mass position for each bead.

    Parameters
    ----------
    atoms           : all-atom list
    positions       : (N, 3) positions in Å
    martini_version : only version 3 mapping is built-in

    Returns
    -------
    (bead_map, cg_positions)
      bead_map     : list[BeadMap]
      cg_positions : (N_beads, 3) array of bead centres
    """
    from collections import defaultdict

    pos = np.asarray(positions, dtype=float)

    # Group atoms by (chain_id, residue_id)
    res_atoms: dict[tuple, list[int]] = defaultdict(list)
    for i, atom in enumerate(atoms):
        chain = getattr(atom, "chain_id",  None) or ""
        rid   = getattr(atom, "residue_id", None)
        if rid is not None:
            res_atoms[(chain, rid)].append(i)

    bead_map:     list[BeadMap]    = []
    cg_positions: list[np.ndarray] = []

    for (chain, rid), atom_indices in sorted(res_atoms.items()):
        if not atom_indices:
            continue

        resname = (
            getattr(atoms[atom_indices[0]], "resname", None) or "UNK"
        ).upper()

        bead_defs = _MARTINI3_PROTEIN.get(resname)

        if bead_defs is None:
            # Unknown residue: single bead at centre of mass
            res_pos  = pos[atom_indices]
            center   = res_pos.mean(axis=0)
            bead_map.append(BeadMap(
                bead_name="BB",
                bead_type="P2",
                residue_id=rid,
                resname=resname,
                atom_indices=atom_indices,
                center=center,
            ))
            cg_positions.append(center)
            continue

        # Build name → index map for this residue
        name_to_idx: dict[str, int] = {}
        for i in atom_indices:
            name = (getattr(atoms[i], "name", None) or "").strip().upper()
            if name:
                name_to_idx[name] = i

        for bead_name, bead_type, atom_names in bead_defs:
            bead_indices = [
                name_to_idx[n] for n in atom_names
                if n in name_to_idx
            ]
            if not bead_indices:
                # Bead has no matching atoms — use CA as fallback
                if "CA" in name_to_idx:
                    bead_indices = [name_to_idx["CA"]]
                else:
                    bead_indices = atom_indices[:1]

            bead_pos = pos[bead_indices].mean(axis=0)
            bead_map.append(BeadMap(
                bead_name=bead_name,
                bead_type=bead_type,
                residue_id=rid,
                resname=resname,
                atom_indices=bead_indices,
                center=bead_pos,
            ))
            cg_positions.append(bead_pos)

    if not cg_positions:
        return [], np.zeros((0, 3))

    return bead_map, np.array(cg_positions, dtype=float)


def format_bead_map(bead_map: list[BeadMap]) -> str:
    """Format bead mapping table for GUI display."""
    if not bead_map:
        return "NO BEAD MAPPING AVAILABLE"

    from collections import Counter
    type_counts = Counter(b.bead_type for b in bead_map)

    lines = [
        f"MARTINI BEAD MAPPING  ({len(bead_map)} beads)\n",
        "BEAD TYPE DISTRIBUTION:",
    ]
    for btype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {btype:<6} : {count}")

    lines.extend(["",
        f"{'RESID':>6}  {'RES':>4}  {'BEAD':>5}  "
        f"{'TYPE':>6}  {'ATOMS':>6}",
        "-" * 34,
    ])

    prev_rid = None
    for b in bead_map[:50]:
        sep = "" if b.residue_id == prev_rid else ""
        lines.append(
            f"{b.residue_id:>6}  {b.resname:>4}  {b.bead_name:>5}  "
            f"{b.bead_type:>6}  {len(b.atom_indices):>6}"
        )
        prev_rid = b.residue_id

    if len(bead_map) > 50:
        lines.append(f"... {len(bead_map)-50} more beads")

    return "\n".join(lines)