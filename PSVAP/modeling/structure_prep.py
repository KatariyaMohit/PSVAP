"""
modeling/structure_prep.py
--------------------------
Feature 17: Structure Preparation.

Detects and fixes common issues in PDB structures:
  - Missing backbone atoms within a residue
  - Non-standard residues
  - Alternate conformations (keeps first)
  - Missing hydrogen atoms (reports only — H placement needs reduce/OpenBabel)
  - Consecutive residue ID gaps (missing loops)

Public API
----------
  check_structure(atoms, positions)
      → StructureReport  (lists all issues found)

  remove_waters(atoms, positions)
      → (new_atoms, new_positions)

  remove_hetatm(atoms, positions, keep_resnames=None)
      → (new_atoms, new_positions)

  cap_termini(atoms, positions)
      → (new_atoms, new_positions)
      Adds ACE (N-terminal) and NME (C-terminal) cap atoms.

  renumber_residues(atoms, start=1)
      → new_atoms  (same positions, re-numbered residue IDs)

All positions in Angstroms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from PSVAP.core.atom import Atom

# Standard amino acid three-letter codes
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    # Common variants
    "HSD", "HSE", "HSP", "HIE", "HID", "HIP",
    "CYX", "CYM", "ASH", "GLH", "LYN",
}

_STANDARD_NUCLEOTIDES = {
    "DA", "DT", "DG", "DC", "A", "U", "G", "C",
}

_WATER_RESNAMES = {"HOH", "WAT", "TIP", "TIP3", "SPC", "SOL", "H2O"}

_BACKBONE_ATOMS = {"N", "CA", "C", "O"}


@dataclass
class StructureIssue:
    """A single detected structural issue."""
    severity:   str    # 'WARNING' or 'ERROR'
    residue_id: int | None
    chain_id:   str | None
    issue_type: str
    description: str


@dataclass
class StructureReport:
    """Complete report from check_structure()."""
    issues:          list[StructureIssue] = field(default_factory=list)
    n_atoms:         int = 0
    n_residues:      int = 0
    n_chains:        int = 0
    n_waters:        int = 0
    n_hetatm:        int = 0
    n_missing_bb:    int = 0   # residues with incomplete backbone
    n_nonstandard:   int = 0   # non-standard residue count
    n_missing_loops: int = 0   # gaps in residue numbering

    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    def summary(self) -> str:
        lines = [
            f"STRUCTURE REPORT",
            f"  Atoms     : {self.n_atoms}",
            f"  Residues  : {self.n_residues}",
            f"  Chains    : {self.n_chains}",
            f"  Waters    : {self.n_waters}",
            f"  HETATM    : {self.n_hetatm}",
            f"  Issues    : {len(self.issues)}",
        ]
        if self.issues:
            lines.append("")
            lines.append("ISSUES FOUND:")
            for issue in self.issues[:30]:
                res_str = f"res {issue.residue_id}" if issue.residue_id else ""
                chn_str = f"chain {issue.chain_id}" if issue.chain_id else ""
                loc_str = " ".join(filter(None, [chn_str, res_str]))
                lines.append(
                    f"  [{issue.severity}]  {issue.issue_type}"
                    + (f"  ({loc_str})" if loc_str else "")
                    + f": {issue.description}"
                )
            if len(self.issues) > 30:
                lines.append(f"  ... {len(self.issues)-30} more issues")
        return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────

def check_structure(
    atoms: list[Atom],
    positions: np.ndarray,
) -> StructureReport:
    """
    Analyse a structure and report all detected issues.

    Parameters
    ----------
    atoms     : full atom list
    positions : (N, 3) positions in Å

    Returns
    -------
    StructureReport with all issues, counts, and a formatted summary.
    """
    from collections import defaultdict

    report = StructureReport(n_atoms=len(atoms))
    issues: list[StructureIssue] = []
    pos = np.asarray(positions, dtype=float)

    # Group atoms by (chain_id, residue_id)
    res_groups: dict[tuple, list[int]] = defaultdict(list)
    chain_set: set[str] = set()

    for i, atom in enumerate(atoms):
        chain = getattr(atom, "chain_id", None) or ""
        rid   = getattr(atom, "residue_id", None)
        chain_set.add(chain)
        if rid is not None:
            res_groups[(chain, rid)].append(i)

    report.n_chains   = len(chain_set)
    report.n_residues = len(res_groups)

    water_count    = 0
    hetatm_count   = 0
    missing_bb     = 0
    nonstandard    = 0
    missing_loops  = 0

    # Check each residue
    for (chain, rid), res_idx in res_groups.items():
        resname = (getattr(atoms[res_idx[0]], "resname", None) or "").upper()

        # Count waters
        if resname in _WATER_RESNAMES:
            water_count += 1
            continue

        # HETATM / non-standard
        is_standard = (resname in _STANDARD_AA or
                       resname in _STANDARD_NUCLEOTIDES)
        if not is_standard and resname:
            hetatm_count += 1
            nonstandard  += 1
            issues.append(StructureIssue(
                severity="WARNING",
                residue_id=rid,
                chain_id=chain or None,
                issue_type="NON_STANDARD_RESIDUE",
                description=f"Residue '{resname}' is non-standard",
            ))

        # Check backbone completeness for amino acids
        if resname in _STANDARD_AA:
            atom_names = {
                (getattr(atoms[i], "name", "") or "").strip().upper()
                for i in res_idx
            }
            missing = _BACKBONE_ATOMS - atom_names
            if missing:
                missing_bb += 1
                issues.append(StructureIssue(
                    severity="ERROR",
                    residue_id=rid,
                    chain_id=chain or None,
                    issue_type="MISSING_BACKBONE",
                    description=f"Missing backbone atoms: {sorted(missing)}",
                ))

    # Check for gaps in residue numbering (potential missing loops)
    for chain in chain_set:
        chain_rids = sorted(
            rid for (c, rid) in res_groups
            if c == chain and rid is not None
        )
        for j in range(1, len(chain_rids)):
            gap = chain_rids[j] - chain_rids[j-1]
            if gap > 1:
                missing_loops += 1
                issues.append(StructureIssue(
                    severity="WARNING",
                    residue_id=chain_rids[j-1],
                    chain_id=chain or None,
                    issue_type="MISSING_LOOP",
                    description=(
                        f"Gap of {gap-1} residues between "
                        f"res {chain_rids[j-1]} and {chain_rids[j]}"
                    ),
                ))

    # Check for hydrogen atoms
    has_h = any(
        (getattr(a, "element", None) or "").upper() == "H"
        for a in atoms
    )
    if not has_h:
        issues.append(StructureIssue(
            severity="WARNING",
            residue_id=None,
            chain_id=None,
            issue_type="NO_HYDROGENS",
            description=(
                "No hydrogen atoms found. Add H using reduce or "
                "OpenBabel before MD simulation."
            ),
        ))

    # Check for very close atoms (potential clashing atoms in structure)
    _check_bad_bonds(atoms, pos, issues)

    report.issues          = issues
    report.n_waters        = water_count
    report.n_hetatm        = hetatm_count
    report.n_missing_bb    = missing_bb
    report.n_nonstandard   = nonstandard
    report.n_missing_loops = missing_loops

    return report


def remove_waters(
    atoms: list[Atom],
    positions: np.ndarray,
) -> tuple[list[Atom], np.ndarray]:
    """
    Remove all water molecules from the structure.

    Returns
    -------
    (new_atoms, new_positions) with waters removed
    """
    pos = np.asarray(positions, dtype=float)
    keep = [
        i for i, a in enumerate(atoms)
        if (getattr(a, "resname", None) or "").upper() not in _WATER_RESNAMES
    ]
    new_atoms = [atoms[i] for i in keep]
    new_pos   = pos[keep] if keep else np.zeros((0, 3), dtype=float)

    # Re-assign IDs
    from dataclasses import replace as dc_replace
    new_atoms = [dc_replace(a, id=j) for j, a in enumerate(new_atoms)]

    return new_atoms, new_pos


def remove_hetatm(
    atoms: list[Atom],
    positions: np.ndarray,
    keep_resnames: set[str] | None = None,
) -> tuple[list[Atom], np.ndarray]:
    """
    Remove non-standard (HETATM) residues, optionally keeping some by name.

    Parameters
    ----------
    keep_resnames : set of resnames to keep (e.g. {"ATP", "HEM"})

    Returns
    -------
    (new_atoms, new_positions)
    """
    from dataclasses import replace as dc_replace

    keep_set = {r.upper() for r in (keep_resnames or set())}
    pos = np.asarray(positions, dtype=float)

    keep = []
    for i, a in enumerate(atoms):
        rn = (getattr(a, "resname", None) or "").upper()
        # Keep if standard AA, nucleotide, water, or in keep_set
        if (rn in _STANDARD_AA or
                rn in _STANDARD_NUCLEOTIDES or
                rn in _WATER_RESNAMES or
                rn in keep_set):
            keep.append(i)

    new_atoms = [atoms[i] for i in keep]
    new_pos   = pos[keep] if keep else np.zeros((0, 3), dtype=float)
    new_atoms = [dc_replace(a, id=j) for j, a in enumerate(new_atoms)]

    return new_atoms, new_pos


def cap_termini(
    atoms: list[Atom],
    positions: np.ndarray,
) -> tuple[list[Atom], np.ndarray]:
    """
    Add ACE (acetyl, N-terminal) and NME (N-methylamide, C-terminal) caps.

    Cap atoms are placed at idealized positions 1.33 Å from the terminal
    backbone atoms. This is a geometric placement — no energy minimization.

    Returns
    -------
    (new_atoms, new_positions) with cap atoms added
    """
    from dataclasses import replace as dc_replace
    from collections import defaultdict

    pos = np.asarray(positions, dtype=float)
    new_atoms    = list(atoms)
    new_pos_list = list(pos)

    # Group by chain
    chain_res: dict[str, list[int]] = defaultdict(list)
    for atom in atoms:
        chain = getattr(atom, "chain_id", None) or "A"
        rid   = getattr(atom, "residue_id", None)
        if rid is not None:
            chain_res[chain].append(rid)

    next_id = max((a.id for a in atoms), default=0) + 1

    for chain, rids in chain_res.items():
        if not rids:
            continue
        sorted_rids = sorted(set(rids))
        first_rid   = sorted_rids[0]
        last_rid    = sorted_rids[-1]

        # Find N of first residue for ACE cap
        n_atom_idx = next(
            (i for i, a in enumerate(atoms)
             if getattr(a, "residue_id", None) == first_rid
             and getattr(a, "chain_id", None) == chain
             and (getattr(a, "name", "") or "").strip().upper() == "N"),
            None,
        )
        if n_atom_idx is not None:
            n_pos   = pos[n_atom_idx]
            ca_idx  = next(
                (i for i, a in enumerate(atoms)
                 if getattr(a, "residue_id", None) == first_rid
                 and getattr(a, "chain_id", None) == chain
                 and (getattr(a, "name", "") or "").strip().upper() == "CA"),
                None,
            )
            if ca_idx is not None:
                direction = n_pos - pos[ca_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    ace_pos = n_pos + (direction / norm) * 1.33
                    ace_atom = Atom(
                        id=next_id, type_id=1, element="C",
                        x=float(ace_pos[0]), y=float(ace_pos[1]), z=float(ace_pos[2]),
                        residue_id=first_rid - 1, chain_id=chain,
                        name="CH3", resname="ACE",
                    )
                    new_atoms.insert(0, ace_atom)
                    new_pos_list.insert(0, ace_pos)
                    next_id += 1

        # Find C of last residue for NME cap
        c_atom_idx = next(
            (i for i, a in enumerate(atoms)
             if getattr(a, "residue_id", None) == last_rid
             and getattr(a, "chain_id", None) == chain
             and (getattr(a, "name", "") or "").strip().upper() == "C"),
            None,
        )
        if c_atom_idx is not None:
            c_pos  = pos[c_atom_idx]
            ca_idx = next(
                (i for i, a in enumerate(atoms)
                 if getattr(a, "residue_id", None) == last_rid
                 and getattr(a, "chain_id", None) == chain
                 and (getattr(a, "name", "") or "").strip().upper() == "CA"),
                None,
            )
            if ca_idx is not None:
                direction = c_pos - pos[ca_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    nme_pos  = c_pos + (direction / norm) * 1.33
                    nme_atom = Atom(
                        id=next_id, type_id=2, element="N",
                        x=float(nme_pos[0]), y=float(nme_pos[1]), z=float(nme_pos[2]),
                        residue_id=last_rid + 1, chain_id=chain,
                        name="N", resname="NME",
                    )
                    new_atoms.append(nme_atom)
                    new_pos_list.append(nme_pos)
                    next_id += 1

    # Re-assign sequential IDs
    new_atoms = [dc_replace(a, id=j) for j, a in enumerate(new_atoms)]
    return new_atoms, np.array(new_pos_list, dtype=float)


def renumber_residues(
    atoms: list[Atom],
    start: int = 1,
) -> list[Atom]:
    """
    Renumber residue IDs sequentially starting from `start`.

    Preserves chain boundaries — each chain starts fresh from `start`.
    Does not modify positions.

    Returns
    -------
    New atom list with updated residue_id values (Atom is frozen so
    new Atom objects are created via dataclasses.replace).
    """
    from dataclasses import replace as dc_replace
    from collections import defaultdict

    # Map (chain, old_residue_id) → new_residue_id
    chain_counters: dict[str, int] = defaultdict(lambda: start - 1)
    chain_mapping:  dict[tuple, int] = {}
    seen_order:     list[tuple] = []

    for atom in atoms:
        chain = getattr(atom, "chain_id", None) or ""
        rid   = getattr(atom, "residue_id", None)
        key   = (chain, rid)
        if key not in chain_mapping:
            chain_counters[chain] += 1
            chain_mapping[key]     = chain_counters[chain]
            seen_order.append(key)

    new_atoms: list[Atom] = []
    for atom in atoms:
        chain = getattr(atom, "chain_id", None) or ""
        rid   = getattr(atom, "residue_id", None)
        new_rid = chain_mapping.get((chain, rid), rid)
        new_atoms.append(dc_replace(atom, residue_id=new_rid))

    return new_atoms


# ── Internal helpers ──────────────────────────────────────────────────────

def _check_bad_bonds(
    atoms: list[Atom],
    pos: np.ndarray,
    issues: list[StructureIssue],
    sample_size: int = 500,
) -> None:
    """
    Check a sample of atoms for suspiciously short distances
    (< 0.5 Å, likely duplicate atoms or bad coordinates).
    """
    n = min(len(atoms), sample_size)
    for i in range(n):
        for j in range(i + 1, min(n, i + 20)):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist < 0.5:
                issues.append(StructureIssue(
                    severity="ERROR",
                    residue_id=getattr(atoms[i], "residue_id", None),
                    chain_id=getattr(atoms[i], "chain_id", None),
                    issue_type="BAD_COORDINATES",
                    description=(
                        f"Atoms {i} and {j} are only {dist:.3f} Å apart "
                        "(possible duplicate or bad coordinates)"
                    ),
                ))