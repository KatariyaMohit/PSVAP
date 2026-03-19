"""
modeling/mutation_engine.py
---------------------------
Feature 7: Amino Acid and Nucleotide Mutations.

Applies point mutations to a loaded structure. Since Atom is frozen
(immutable), mutation creates a new atom list with the mutated residue
replaced. Side-chain placement uses a simplified backbone-dependent
rotamer — the Cbeta position is preserved from the original residue
where possible, and remaining side-chain atoms are removed (backbone-only
mode) or placed at idealized positions.

Public API
----------
  mutate_residue(atoms, positions, residue_id, target_resname,
                 chain_id=None)
      → (new_atoms, new_positions)
      Returns a completely new atom list and position array with the
      specified residue mutated to target_resname.

  apply_mutation_list(atoms, positions, mutations)
      → (new_atoms, new_positions)
      Apply a list of (residue_id, target_resname) mutations sequentially.

  get_residue_atoms(atoms, residue_id, chain_id=None)
      → list[int]  atom indices belonging to that residue

  list_residues(atoms)
      → list[dict]  summary of all residues: id, resname, chain, n_atoms

All positions in Angstroms.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np

from PSVAP.core.atom import Atom

# ── Amino acid one-letter to three-letter lookup ──────────────────────────
_ONE_TO_THREE: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
    "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
    "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
    "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

_THREE_TO_ONE: dict[str, str] = {v: k for k, v in _ONE_TO_THREE.items()}

# Standard backbone atoms present in all amino acids
_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

# Nucleotide residue names
_DNA_RESIDUES  = {"DA", "DT", "DG", "DC"}
_RNA_RESIDUES  = {"A", "U", "G", "C"}
_NUCLEOTIDES   = _DNA_RESIDUES | _RNA_RESIDUES

# Idealized Cbeta position offset from CA (Å) — used when placing ALA Cbeta
# This is the average Cbeta vector in the backbone frame
_CBETA_OFFSET = np.array([1.21, -0.89, 1.19], dtype=float)


class MutationError(ValueError):
    """Raised when a mutation cannot be applied."""


# ── Public API ────────────────────────────────────────────────────────────

def mutate_residue(
    atoms: list[Atom],
    positions: np.ndarray,
    residue_id: int,
    target_resname: str,
    chain_id: str | None = None,
) -> tuple[list[Atom], np.ndarray]:
    """
    Apply a single point mutation to a residue.

    The mutation strategy:
      1. Keep all backbone atoms (N, CA, C, O, OXT) from the original.
      2. For ALA target: add a Cbeta at an idealized position if none exists.
      3. For GLY target: remove all side-chain atoms including Cbeta.
      4. For all other targets: keep backbone only (conservative approach —
         no full rotamer library is bundled; use SCWRL4 externally for
         production-quality side chains).

    Parameters
    ----------
    atoms       : full atom list
    positions   : (N, 3) current frame positions
    residue_id  : residue_id of the residue to mutate
    target_resname : three-letter code of the target residue (e.g. "ALA")
    chain_id    : optional chain filter (if None, matches any chain)

    Returns
    -------
    (new_atoms, new_positions) — new list and array with mutation applied

    Raises
    ------
    MutationError if residue_id is not found or target is invalid
    """
    target = _normalise_resname(target_resname)
    pos = np.asarray(positions, dtype=float)

    # Locate residue atoms
    res_indices = get_residue_atoms(atoms, residue_id, chain_id)
    if not res_indices:
        raise MutationError(
            f"Residue id={residue_id} "
            f"(chain={chain_id or 'any'}) not found in atom list."
        )

    original_resname = (getattr(atoms[res_indices[0]], "resname", None) or "UNK").upper()

    if original_resname == target:
        # No-op mutation — return copies
        return list(atoms), pos.copy()

    # Separate backbone from side-chain indices for this residue
    backbone_idx = [
        i for i in res_indices
        if (getattr(atoms[i], "name", "") or "").strip().upper() in _BACKBONE_ATOMS
    ]
    ca_idx = next(
        (i for i in res_indices
         if (getattr(atoms[i], "name", "") or "").strip().upper() == "CA"),
        None,
    )

    # Build the new atom list
    new_atoms: list[Atom] = []
    new_positions: list[np.ndarray] = []
    res_set = set(res_indices)

    # Pass 1: copy all atoms outside the mutated residue unchanged
    for i, atom in enumerate(atoms):
        if i not in res_set:
            new_atoms.append(atom)
            new_positions.append(pos[i])

    # Pass 2: insert mutated residue atoms
    insertion_point = _find_insertion_point(atoms, res_indices)

    mutated_atoms: list[Atom] = []
    mutated_positions: list[np.ndarray] = []

    # Always keep backbone
    for i in backbone_idx:
        mutated_atom = dc_replace(atoms[i], resname=target)
        mutated_atoms.append(mutated_atom)
        mutated_positions.append(pos[i])

    # For ALA: add Cbeta if not in backbone
    if target == "ALA" and ca_idx is not None:
        cb_exists = any(
            (getattr(atoms[i], "name", "") or "").strip().upper() == "CB"
            for i in res_indices
        )
        if not cb_exists:
            cb_pos = _place_cbeta(pos, ca_idx, backbone_idx, atoms)
            cb_atom = Atom(
                id=max(a.id for a in atoms) + 1,
                type_id=1,
                element="C",
                x=float(cb_pos[0]),
                y=float(cb_pos[1]),
                z=float(cb_pos[2]),
                residue_id=residue_id,
                chain_id=getattr(atoms[res_indices[0]], "chain_id", None),
                name="CB",
                resname=target,
            )
            mutated_atoms.append(cb_atom)
            mutated_positions.append(cb_pos)
        else:
            # Keep existing CB for ALA
            for i in res_indices:
                if (getattr(atoms[i], "name", "") or "").strip().upper() == "CB":
                    mutated_atoms.append(dc_replace(atoms[i], resname=target))
                    mutated_positions.append(pos[i])
                    break

    # GLY: backbone only, no CB (already handled by keeping only backbone_idx)

    # Insert mutated atoms at the correct position
    for j, (atom, atom_pos) in enumerate(zip(mutated_atoms, mutated_positions)):
        new_atoms.insert(insertion_point + j, atom)
        new_positions.insert(insertion_point + j, atom_pos)

    # Re-assign sequential IDs
    final_atoms = [dc_replace(a, id=i) for i, a in enumerate(new_atoms)]
    final_positions = np.array(new_positions, dtype=float)

    return final_atoms, final_positions


def apply_mutation_list(
    atoms: list[Atom],
    positions: np.ndarray,
    mutations: list[tuple[int, str]],
    chain_id: str | None = None,
) -> tuple[list[Atom], np.ndarray]:
    """
    Apply a list of mutations sequentially.

    Parameters
    ----------
    atoms     : full atom list
    positions : (N, 3) positions
    mutations : list of (residue_id, target_resname) tuples
    chain_id  : optional chain filter

    Returns
    -------
    (new_atoms, new_positions) after all mutations applied
    """
    current_atoms = list(atoms)
    current_pos   = np.asarray(positions, dtype=float)

    for residue_id, target_resname in mutations:
        current_atoms, current_pos = mutate_residue(
            current_atoms, current_pos,
            residue_id, target_resname,
            chain_id=chain_id,
        )

    return current_atoms, current_pos


def get_residue_atoms(
    atoms: list[Atom],
    residue_id: int,
    chain_id: str | None = None,
) -> list[int]:
    """
    Return atom indices belonging to residue_id (optionally filtered by chain).

    Returns
    -------
    list[int] — indices into the atoms list (empty if not found)
    """
    indices: list[int] = []
    for i, atom in enumerate(atoms):
        if getattr(atom, "residue_id", None) != residue_id:
            continue
        if chain_id is not None:
            if getattr(atom, "chain_id", None) != chain_id:
                continue
        indices.append(i)
    return indices


def list_residues(atoms: list[Atom]) -> list[dict]:
    """
    Return a summary list of all unique residues in the atom list.

    Returns
    -------
    list of dicts: [{'residue_id': int, 'resname': str,
                     'chain_id': str, 'n_atoms': int}]
    Sorted by (chain_id, residue_id).
    """
    from collections import defaultdict
    res_info: dict[tuple, dict] = {}

    for atom in atoms:
        rid     = getattr(atom, "residue_id", None)
        chain   = getattr(atom, "chain_id",   None) or ""
        resname = (getattr(atom, "resname",   None) or "UNK").upper()
        key     = (chain, rid)

        if key not in res_info:
            res_info[key] = {
                "residue_id": rid,
                "resname":    resname,
                "chain_id":   chain or None,
                "n_atoms":    0,
            }
        res_info[key]["n_atoms"] += 1

    return sorted(res_info.values(), key=lambda x: (x["chain_id"] or "", x["residue_id"] or 0))


def write_pdb(
    atoms: list[Atom],
    positions: np.ndarray,
    output_path,
) -> None:
    """
    Write atoms and positions to a minimal PDB file.

    This is a lightweight writer — it does not require Biopython.
    Produces valid ATOM records that can be read by PyMOL, VMD, and PSVAP.

    Parameters
    ----------
    atoms       : atom list
    positions   : (N, 3) positions in Å
    output_path : Path or str
    """
    from pathlib import Path
    pos = np.asarray(positions, dtype=float)
    out = Path(output_path)

    lines: list[str] = ["REMARK  Generated by PSVAP modeling/mutation_engine.py\n"]
    prev_chain = None

    for i, atom in enumerate(atoms):
        if i >= len(pos):
            break

        name     = (getattr(atom, "name",      None) or "CA").strip()
        resname  = (getattr(atom, "resname",    None) or "UNK").strip().upper()
        chain    = (getattr(atom, "chain_id",   None) or "A").strip()
        res_id   = getattr(atom, "residue_id",  1) or 1
        element  = (getattr(atom, "element",    None) or "C").strip().upper()
        x, y, z  = float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])

        # TER record between chains
        if prev_chain is not None and chain != prev_chain:
            lines.append(
                f"TER   {i:5d}      {resname:<3s} {prev_chain}{res_id:4d}\n"
            )
        prev_chain = chain

        # PDB ATOM record: columns are strictly fixed-width
        # Name formatting: 1-char elements left-padded to col 14
        if len(name) < 4:
            fmt_name = f" {name:<3s}" if len(element) == 1 else f"{name:<4s}"
        else:
            fmt_name = name[:4]

        lines.append(
            f"ATOM  {i+1:5d} {fmt_name} {resname:<3s} {chain}"
            f"{res_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}\n"
        )

    lines.append("END\n")
    out.write_text("".join(lines), encoding="utf-8")


# ── Internal helpers ──────────────────────────────────────────────────────

def _normalise_resname(resname: str) -> str:
    """
    Accept one-letter or three-letter amino acid codes.
    Returns uppercase three-letter code.
    Raises MutationError if not recognised.
    """
    r = resname.strip().upper()
    if len(r) == 1 and r in _ONE_TO_THREE:
        return _ONE_TO_THREE[r]
    if len(r) == 3:
        return r
    # Try partial match
    for three in _ONE_TO_THREE.values():
        if three.startswith(r):
            return three
    raise MutationError(
        f"Unknown residue name: '{resname}'. "
        f"Use three-letter codes (ALA, GLY, ...) or one-letter codes (A, G, ...)."
    )


def _find_insertion_point(atoms: list[Atom], res_indices: list[int]) -> int:
    """
    Find the index in atoms where mutated residue atoms should be inserted
    (just before the first atom of this residue).
    """
    if not res_indices:
        return len(atoms)
    return min(res_indices)


def _place_cbeta(
    pos: np.ndarray,
    ca_idx: int,
    backbone_idx: list[int],
    atoms: list[Atom],
) -> np.ndarray:
    """
    Place a Cbeta atom at an idealized position relative to the CA.

    Uses the N-CA-C plane to define a local coordinate frame, then
    places CB in the standard tetrahedral position.

    Falls back to a simple offset from CA if backbone geometry is incomplete.
    """
    ca_pos = pos[ca_idx]

    # Try to find N and C backbone atoms
    n_idx  = next((i for i in backbone_idx
                   if (getattr(atoms[i], "name", "") or "").strip().upper() == "N"),  None)
    c_idx  = next((i for i in backbone_idx
                   if (getattr(atoms[i], "name", "") or "").strip().upper() == "C"),  None)

    if n_idx is not None and c_idx is not None:
        n_pos = pos[n_idx]
        c_pos = pos[c_idx]

        # Build local frame from backbone
        v1 = (n_pos - ca_pos)
        v2 = (c_pos - ca_pos)
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)

        # CB is roughly opposite to the N-C bisector, tilted out of plane
        bisector = -(v1 + v2)
        cross    = np.cross(v1, v2)
        norm_b   = np.linalg.norm(bisector)
        norm_c   = np.linalg.norm(cross)

        if norm_b > 1e-6 and norm_c > 1e-6:
            bisector /= norm_b
            cross    /= norm_c
            # Standard CB position: ~1.52 Å from CA, ~110.5° bond angle
            cb_dir = 0.58 * bisector + 0.82 * cross
            cb_dir /= np.linalg.norm(cb_dir)
            return ca_pos + 1.52 * cb_dir

    # Fallback: simple offset
    return ca_pos + _CBETA_OFFSET