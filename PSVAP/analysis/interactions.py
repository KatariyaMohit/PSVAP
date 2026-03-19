"""
analysis/interactions.py
------------------------
Feature 12 & 21: Non-covalent Interaction Detection and 2D Interaction Map.

Detects all non-covalent interactions between two atom groups:
  - Hydrogen bonds      (donor-H...acceptor, dist < 3.5 Å, angle > 120°)
  - Salt bridges        (oppositely charged groups, dist < 4.0 Å)
  - Halogen bonds       (C-X...O/N/S, dist < 3.5 Å)
  - Pi-stacking         (aromatic ring centroids < 5.5 Å)
  - Hydrophobic contacts (nonpolar atoms < 4.0 Å)
  - Steric clashes      (atoms closer than sum of vdW radii - 0.4 Å)

Public API
----------
  detect_hbonds(atoms, positions, donor_indices, acceptor_indices)
      → list[HBond]

  detect_salt_bridges(atoms, positions, pos_indices, neg_indices)
      → list[SaltBridge]

  detect_clashes(atoms, positions, group_a, group_b)
      → list[Clash]

  detect_hydrophobic(atoms, positions, group_a, group_b)
      → list[HydrophobicContact]

  detect_all_interactions(atoms, positions, group_a, group_b)
      → InteractionResult

  interactions_over_trajectory(atoms, trajectory, group_a, group_b)
      → dict  (persistence data across frames)

All positions in Angstroms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from PSVAP.core.constants import (
    CLASH_VDW_OVERLAP,
    HBOND_ANGLE_CUTOFF,
    HBOND_DISTANCE_CUTOFF,
    HALOGEN_BOND_CUTOFF,
    PI_STACKING_CUTOFF,
    SALT_BRIDGE_CUTOFF,
)


# ── VdW radii (Å) — used for clash detection ──────────────────────────────
_VDW_RADIUS: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "DEFAULT": 1.70,
}

# ── Element sets for interaction typing ───────────────────────────────────
_HBOND_DONORS    = {"N", "O", "S"}
_HBOND_ACCEPTORS = {"N", "O", "S", "F"}
_HALOGEN_DONORS  = {"CL", "BR", "I", "F"}
_HALOGEN_ACCEPTORS = {"N", "O", "S"}
_HYDROPHOBIC_ELEMENTS = {"C", "S"}
_POS_CHARGED_RESNAMES = {"ARG", "LYS", "HIS", "HSD", "HSE", "HSP"}
_NEG_CHARGED_RESNAMES = {"ASP", "GLU"}
_AROMATIC_RESNAMES = {"PHE", "TYR", "TRP", "HIS", "HSD", "HSE", "HSP"}


def _vdw(element: str | None) -> float:
    if element:
        return _VDW_RADIUS.get(element.upper(), _VDW_RADIUS["DEFAULT"])
    return _VDW_RADIUS["DEFAULT"]


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass(frozen=True)
class HBond:
    donor_idx:    int
    hydrogen_idx: int | None   # may be None if H not explicitly in structure
    acceptor_idx: int
    distance:     float        # donor–acceptor distance (Å)
    angle:        float        # D-H...A angle (degrees); NaN if no H

@dataclass(frozen=True)
class SaltBridge:
    pos_idx: int
    neg_idx: int
    distance: float

@dataclass(frozen=True)
class HalogenBond:
    donor_idx:    int   # the halogen-bearing carbon
    halogen_idx:  int
    acceptor_idx: int
    distance:     float

@dataclass(frozen=True)
class PiStack:
    ring_a_center: np.ndarray   # centroid of ring A
    ring_b_center: np.ndarray   # centroid of ring B
    distance:      float
    indices_a:     tuple[int, ...]
    indices_b:     tuple[int, ...]

@dataclass(frozen=True)
class HydrophobicContact:
    idx_a:    int
    idx_b:    int
    distance: float

@dataclass(frozen=True)
class Clash:
    idx_a:    int
    idx_b:    int
    distance: float
    overlap:  float   # amount of vdW overlap (Å)


@dataclass
class InteractionResult:
    """All interactions detected between group_a and group_b."""
    hbonds:       list[HBond]             = field(default_factory=list)
    salt_bridges: list[SaltBridge]        = field(default_factory=list)
    halogen_bonds: list[HalogenBond]      = field(default_factory=list)
    pi_stacks:    list[PiStack]           = field(default_factory=list)
    hydrophobic:  list[HydrophobicContact] = field(default_factory=list)
    clashes:      list[Clash]             = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"H-BONDS: {len(self.hbonds)}  "
            f"SALT BRIDGES: {len(self.salt_bridges)}  "
            f"HALOGEN BONDS: {len(self.halogen_bonds)}  "
            f"PI-STACKS: {len(self.pi_stacks)}  "
            f"HYDROPHOBIC: {len(self.hydrophobic)}  "
            f"CLASHES: {len(self.clashes)}"
        )

    def total(self) -> int:
        return (len(self.hbonds) + len(self.salt_bridges) +
                len(self.halogen_bonds) + len(self.pi_stacks) +
                len(self.hydrophobic) + len(self.clashes))


# ── Core detection functions ──────────────────────────────────────────────

def detect_hbonds(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
) -> list[HBond]:
    """
    Detect hydrogen bonds between atoms in group_a and group_b.

    Criteria (Baker & Hubbard 1984):
      - Donor element in {N, O, S}
      - Acceptor element in {N, O, S, F}
      - Donor–acceptor distance < HBOND_DISTANCE_CUTOFF (3.5 Å)
      - If H present: D-H...A angle > HBOND_ANGLE_CUTOFF (120°)

    Parameters
    ----------
    atoms     : full atom list
    positions : (N, 3) current frame positions
    group_a   : atom indices for first group (e.g. protein)
    group_b   : atom indices for second group (e.g. ligand)

    Returns
    -------
    list[HBond]
    """
    results: list[HBond] = []
    pos = np.asarray(positions, dtype=float)

    # Build donor and acceptor index sets from both groups
    donors:    list[int] = []
    acceptors: list[int] = []

    all_indices = list(set(group_a) | set(group_b))
    for idx in all_indices:
        elem = (getattr(atoms[idx], "element", None) or "").upper()
        if elem in _HBOND_DONORS:
            donors.append(idx)
        if elem in _HBOND_ACCEPTORS:
            acceptors.append(idx)

    # Cross-group pairs only (no intra-group)
    set_a = set(group_a)
    set_b = set(group_b)

    for d_idx in donors:
        for a_idx in acceptors:
            if d_idx == a_idx:
                continue
            # Must be cross-group
            if not ((d_idx in set_a and a_idx in set_b) or
                    (d_idx in set_b and a_idx in set_a)):
                continue

            dist = float(np.linalg.norm(pos[d_idx] - pos[a_idx]))
            if dist > HBOND_DISTANCE_CUTOFF:
                continue

            # Find nearest H attached to donor (within 1.2 Å)
            h_idx = _find_bonded_hydrogen(atoms, pos, d_idx)

            angle_val = float("nan")
            if h_idx is not None:
                # D-H...A angle
                v1 = pos[d_idx] - pos[h_idx]
                v2 = pos[a_idx] - pos[h_idx]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle_val = float(np.degrees(np.arccos(cos_a)))
                if angle_val < HBOND_ANGLE_CUTOFF:
                    continue  # angle criterion failed

            results.append(HBond(
                donor_idx=d_idx,
                hydrogen_idx=h_idx,
                acceptor_idx=a_idx,
                distance=dist,
                angle=angle_val,
            ))

    return results


def detect_salt_bridges(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
) -> list[SaltBridge]:
    """
    Detect salt bridges between oppositely charged residues.

    Criteria: charged groups within SALT_BRIDGE_CUTOFF (4.0 Å).
    Positive: ARG, LYS, HIS.  Negative: ASP, GLU.

    Returns
    -------
    list[SaltBridge]
    """
    results: list[SaltBridge] = []
    pos = np.asarray(positions, dtype=float)
    set_a = set(group_a)
    set_b = set(group_b)

    pos_indices: list[int] = []
    neg_indices: list[int] = []

    all_indices = list(set_a | set_b)
    for idx in all_indices:
        resname = (getattr(atoms[idx], "resname", None) or "").upper()
        if resname in _POS_CHARGED_RESNAMES:
            pos_indices.append(idx)
        elif resname in _NEG_CHARGED_RESNAMES:
            neg_indices.append(idx)

    for p_idx in pos_indices:
        for n_idx in neg_indices:
            if not ((p_idx in set_a and n_idx in set_b) or
                    (p_idx in set_b and n_idx in set_a)):
                continue
            dist = float(np.linalg.norm(pos[p_idx] - pos[n_idx]))
            if dist <= SALT_BRIDGE_CUTOFF:
                results.append(SaltBridge(
                    pos_idx=p_idx, neg_idx=n_idx, distance=dist
                ))

    return results


def detect_clashes(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
) -> list[Clash]:
    """
    Detect steric clashes between atoms.

    Clash criterion: distance < (vdW_i + vdW_j) - CLASH_VDW_OVERLAP (0.4 Å).
    Excludes H-H and C-H clashes (too common in low-resolution structures).

    Returns
    -------
    list[Clash]
    """
    results: list[Clash] = []
    pos = np.asarray(positions, dtype=float)
    set_a = set(group_a)
    set_b = set(group_b)

    for i in group_a:
        elem_i = (getattr(atoms[i], "element", None) or "").upper()
        if elem_i == "H":
            continue
        for j in group_b:
            if i == j:
                continue
            elem_j = (getattr(atoms[j], "element", None) or "").upper()
            if elem_j == "H":
                continue

            dist = float(np.linalg.norm(pos[i] - pos[j]))
            vdw_sum = _vdw(elem_i) + _vdw(elem_j)
            clash_threshold = vdw_sum - CLASH_VDW_OVERLAP

            if dist < clash_threshold:
                overlap = clash_threshold - dist
                results.append(Clash(
                    idx_a=i, idx_b=j,
                    distance=dist,
                    overlap=overlap,
                ))

    return results


def detect_hydrophobic(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
    cutoff: float = 4.0,
) -> list[HydrophobicContact]:
    """
    Detect hydrophobic contacts between nonpolar carbon atoms.

    Criterion: C–C or C–S distance < cutoff (4.0 Å).

    Returns
    -------
    list[HydrophobicContact]
    """
    results: list[HydrophobicContact] = []
    pos = np.asarray(positions, dtype=float)

    hydro_a = [i for i in group_a
               if (getattr(atoms[i], "element", None) or "").upper()
               in _HYDROPHOBIC_ELEMENTS]
    hydro_b = [j for j in group_b
               if (getattr(atoms[j], "element", None) or "").upper()
               in _HYDROPHOBIC_ELEMENTS]

    for i in hydro_a:
        for j in hydro_b:
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist <= cutoff:
                results.append(HydrophobicContact(
                    idx_a=i, idx_b=j, distance=dist
                ))

    return results


def detect_all_interactions(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
) -> InteractionResult:
    """
    Detect all non-covalent interactions between group_a and group_b.

    Parameters
    ----------
    atoms     : full atom list (Atom objects)
    positions : (N, 3) current frame positions
    group_a   : atom indices for group A (e.g. protein residues)
    group_b   : atom indices for group B (e.g. ligand or chain B)

    Returns
    -------
    InteractionResult with all detected interactions
    """
    return InteractionResult(
        hbonds       = detect_hbonds(atoms, positions, group_a, group_b),
        salt_bridges = detect_salt_bridges(atoms, positions, group_a, group_b),
        halogen_bonds= _detect_halogen_bonds(atoms, positions, group_a, group_b),
        pi_stacks    = _detect_pi_stacks(atoms, positions, group_a, group_b),
        hydrophobic  = detect_hydrophobic(atoms, positions, group_a, group_b),
        clashes      = detect_clashes(atoms, positions, group_a, group_b),
    )


def interactions_over_trajectory(
    atoms: list,
    trajectory: list[np.ndarray],
    group_a: list[int],
    group_b: list[int],
) -> dict:
    """
    Compute interaction counts for every frame in the trajectory.

    Returns
    -------
    dict with keys: 'frames', 'hbonds', 'salt_bridges', 'clashes',
                    'hydrophobic', 'total'
    Each value is a list of counts per frame.
    """
    frame_indices: list[int] = []
    hbond_counts:  list[int] = []
    sb_counts:     list[int] = []
    clash_counts:  list[int] = []
    hydro_counts:  list[int] = []
    total_counts:  list[int] = []

    for i, frame in enumerate(trajectory):
        result = detect_all_interactions(atoms, frame, group_a, group_b)
        frame_indices.append(i)
        hbond_counts.append(len(result.hbonds))
        sb_counts.append(len(result.salt_bridges))
        clash_counts.append(len(result.clashes))
        hydro_counts.append(len(result.hydrophobic))
        total_counts.append(result.total())

    return {
        "frames":       frame_indices,
        "hbonds":       hbond_counts,
        "salt_bridges": sb_counts,
        "clashes":      clash_counts,
        "hydrophobic":  hydro_counts,
        "total":        total_counts,
    }


# ── Internal helpers ──────────────────────────────────────────────────────

def _find_bonded_hydrogen(
    atoms: list,
    positions: np.ndarray,
    donor_idx: int,
    h_cutoff: float = 1.2,
) -> int | None:
    """Find the nearest hydrogen atom bonded to donor_idx (within h_cutoff Å)."""
    donor_pos = positions[donor_idx]
    best_idx  = None
    best_dist = h_cutoff + 1.0

    for idx, atom in enumerate(atoms):
        if idx == donor_idx:
            continue
        elem = (getattr(atom, "element", None) or "").upper()
        if elem != "H":
            continue
        dist = float(np.linalg.norm(positions[idx] - donor_pos))
        if dist < h_cutoff and dist < best_dist:
            best_dist = dist
            best_idx  = idx

    return best_idx


def _detect_halogen_bonds(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
) -> list[HalogenBond]:
    """
    Detect halogen bonds (C-X...O/N/S).
    Halogen (Cl, Br, I) in one group acts as donor to O/N/S acceptor
    in the other group.
    """
    results: list[HalogenBond] = []
    pos = np.asarray(positions, dtype=float)
    set_a = set(group_a)
    set_b = set(group_b)
    all_indices = list(set_a | set_b)

    # Find halogens and their bonded carbons
    halogens: list[tuple[int, int]] = []  # (halogen_idx, carbon_idx)
    for idx in all_indices:
        elem = (getattr(atoms[idx], "element", None) or "").upper()
        if elem in _HALOGEN_DONORS:
            c_idx = _find_bonded_carbon(atoms, pos, idx)
            if c_idx is not None:
                halogens.append((idx, c_idx))

    # Find acceptors
    acceptors = [idx for idx in all_indices
                 if (getattr(atoms[idx], "element", None) or "").upper()
                 in _HALOGEN_ACCEPTORS]

    for (hal_idx, c_idx) in halogens:
        for acc_idx in acceptors:
            if acc_idx == hal_idx:
                continue
            # Must be cross-group
            hal_in_a = hal_idx in set_a
            acc_in_a = acc_idx in set_a
            if hal_in_a == acc_in_a:
                continue

            dist = float(np.linalg.norm(pos[hal_idx] - pos[acc_idx]))
            if dist <= HALOGEN_BOND_CUTOFF:
                results.append(HalogenBond(
                    donor_idx=c_idx,
                    halogen_idx=hal_idx,
                    acceptor_idx=acc_idx,
                    distance=dist,
                ))

    return results


def _find_bonded_carbon(
    atoms: list,
    positions: np.ndarray,
    halogen_idx: int,
    c_cutoff: float = 2.1,
) -> int | None:
    """Find a carbon atom bonded to the halogen (within c_cutoff Å)."""
    hal_pos = positions[halogen_idx]
    for idx, atom in enumerate(atoms):
        if idx == halogen_idx:
            continue
        elem = (getattr(atom, "element", None) or "").upper()
        if elem != "C":
            continue
        dist = float(np.linalg.norm(positions[idx] - hal_pos))
        if dist < c_cutoff:
            return idx
    return None


def _detect_pi_stacks(
    atoms: list,
    positions: np.ndarray,
    group_a: list[int],
    group_b: list[int],
) -> list[PiStack]:
    """
    Detect pi-stacking between aromatic rings.
    Uses residue name to identify aromatic residues (PHE, TYR, TRP, HIS).
    Ring centroid distance < PI_STACKING_CUTOFF (5.5 Å).
    """
    results: list[PiStack] = []
    pos = np.asarray(positions, dtype=float)
    set_a = set(group_a)
    set_b = set(group_b)

    rings_a = _get_aromatic_ring_atoms(atoms, group_a)
    rings_b = _get_aromatic_ring_atoms(atoms, group_b)

    for ring_a_indices in rings_a:
        center_a = pos[list(ring_a_indices)].mean(axis=0)
        for ring_b_indices in rings_b:
            center_b = pos[list(ring_b_indices)].mean(axis=0)
            dist = float(np.linalg.norm(center_a - center_b))
            if dist <= PI_STACKING_CUTOFF:
                results.append(PiStack(
                    ring_a_center=center_a,
                    ring_b_center=center_b,
                    distance=dist,
                    indices_a=tuple(ring_a_indices),
                    indices_b=tuple(ring_b_indices),
                ))

    return results


def _get_aromatic_ring_atoms(
    atoms: list,
    indices: list[int],
) -> list[list[int]]:
    """Group atoms into aromatic rings by residue."""
    from collections import defaultdict
    res_atoms: dict[int, list[int]] = defaultdict(list)

    for idx in indices:
        resname = (getattr(atoms[idx], "resname", None) or "").upper()
        elem    = (getattr(atoms[idx], "element",  None) or "").upper()
        if resname in _AROMATIC_RESNAMES and elem in {"C", "N"}:
            rid = getattr(atoms[idx], "residue_id", idx)
            res_atoms[rid].append(idx)

    # Return groups with 5–6 atoms (typical aromatic ring size)
    return [v for v in res_atoms.values() if 5 <= len(v) <= 6]