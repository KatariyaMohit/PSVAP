"""
analysis/geometry.py
--------------------
Feature 3: Distance, Angle, and Torsion Calculations.

All functions are pure NumPy — no external dependencies.
All inputs are in Angstroms.  Angles returned in degrees.

Public API (called by ApplicationController)
--------------------------------------------
  distance(p1, p2)                  → float (Å)
  angle(p1, p2, p3)                 → float (degrees)
  torsion(p1, p2, p3, p4)           → float (degrees, -180 to +180)
  distance_trajectory(traj, i, j)   → np.ndarray  (per-frame distances)
  angle_trajectory(traj, i, j, k)   → np.ndarray  (per-frame angles)
  torsion_trajectory(traj,i,j,k,l)  → np.ndarray  (per-frame torsions)
  ramachandran(atoms, trajectory)    → dict {resid: (phi_arr, psi_arr)}
"""
from __future__ import annotations

import numpy as np


# ── Single-frame geometry ─────────────────────────────────────────────────

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points (Å)."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Bond angle at p2 between vectors p2→p1 and p2→p3.
    Returns degrees in [0, 180].
    """
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def torsion(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> float:
    """
    Dihedral angle defined by atoms p1–p2–p3–p4.
    Returns degrees in (-180, +180].
    Uses the atan2 formulation for numerical stability.
    """
    b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    b3 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-15)

    m1 = np.cross(n1, b2_norm)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.degrees(np.arctan2(y, x)))


# ── Trajectory variants ───────────────────────────────────────────────────

def distance_trajectory(
    trajectory: list[np.ndarray],
    i: int,
    j: int,
) -> np.ndarray:
    """Distance between atoms i and j for every frame. Shape: (n_frames,)"""
    return np.array([
        float(np.linalg.norm(frame[i] - frame[j]))
        for frame in trajectory
    ])


def angle_trajectory(
    trajectory: list[np.ndarray],
    i: int,
    j: int,
    k: int,
) -> np.ndarray:
    """Bond angle i–j–k for every frame. Shape: (n_frames,)"""
    return np.array([
        angle(frame[i], frame[j], frame[k])
        for frame in trajectory
    ])


def torsion_trajectory(
    trajectory: list[np.ndarray],
    i: int,
    j: int,
    k: int,
    l: int,
) -> np.ndarray:
    """Dihedral i–j–k–l for every frame. Shape: (n_frames,)"""
    return np.array([
        torsion(frame[i], frame[j], frame[k], frame[l])
        for frame in trajectory
    ])


# ── Ramachandran ──────────────────────────────────────────────────────────

def ramachandran(
    atoms: list,
    trajectory: list[np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    """
    Compute phi and psi backbone torsion angles for all protein residues
    across all trajectory frames.

    Backbone atom names required: N, CA, C  (standard PDB/GROMACS naming).

    Returns
    -------
    {
      residue_id: {
        "phi": np.ndarray(n_frames),   # degrees
        "psi": np.ndarray(n_frames),   # degrees
        "resname": str,
      }
    }
    Residues at chain termini (missing prev/next residue) are omitted.

    Notes
    -----
    Requires atom.name to be set ("N", "CA", "C") AND atom.resname to
    be set ("ALA", "GLY", etc.). LAMMPS files have neither — load a
    PDB or GRO file to use Ramachandran analysis.
    """
    from collections import defaultdict

    # Group atoms by residue_id, keep only backbone atoms N, CA, C
    by_res: dict[int, dict[str, int]] = defaultdict(dict)

    for idx, atom in enumerate(atoms):
        if atom.residue_id is None:
            continue
        name = (getattr(atom, "name", "") or "").strip().upper()
        if name in {"N", "CA", "C"}:
            by_res[atom.residue_id][name] = idx

    sorted_res = sorted(by_res.keys())
    result: dict[int, dict[str, np.ndarray]] = {}

    for k, res_id in enumerate(sorted_res):
        bb = by_res[res_id]
        if not all(n in bb for n in ("N", "CA", "C")):
            continue

        # phi: C(i-1) – N(i) – CA(i) – C(i)
        phi_vals: np.ndarray | None = None
        if k > 0:
            prev_id = sorted_res[k - 1]
            prev_bb = by_res[prev_id]
            if "C" in prev_bb:
                phi_vals = torsion_trajectory(
                    trajectory,
                    prev_bb["C"], bb["N"], bb["CA"], bb["C"],
                )

        # psi: N(i) – CA(i) – C(i) – N(i+1)
        psi_vals: np.ndarray | None = None
        if k < len(sorted_res) - 1:
            next_id = sorted_res[k + 1]
            next_bb = by_res[next_id]
            if "N" in next_bb:
                psi_vals = torsion_trajectory(
                    trajectory,
                    bb["N"], bb["CA"], bb["C"], next_bb["N"],
                )

        if phi_vals is None and psi_vals is None:
            continue

        # FIX: get residue name from atom.resname (the three-letter residue name
        # field added in core/atom.py), NOT from atom.name (which is "CA"/"N"/"C").
        # Fall back to "UNK" if resname is not set (e.g. for LAMMPS files).
        ca_idx = bb["CA"]
        res_name = (getattr(atoms[ca_idx], "resname", None) or "UNK").strip().upper()

        result[res_id] = {
            "phi": phi_vals if phi_vals is not None else np.full(len(trajectory), np.nan),
            "psi": psi_vals if psi_vals is not None else np.full(len(trajectory), np.nan),
            "resname": res_name,
        }

    return result