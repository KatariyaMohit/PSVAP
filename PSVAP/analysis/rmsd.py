"""
analysis/rmsd.py
----------------
Feature 4: RMSD and RMSF Computation.

Pure NumPy — no MDAnalysis dependency for the core math.
MDAnalysis is only optionally used for pre-superimposition.

Public API
----------
  rmsd(mobile, reference, atom_indices=None)
      → float (RMSD in Å for a single frame pair)

  rmsd_trajectory(trajectory, reference_frame=0, atom_indices=None)
      → np.ndarray  shape (n_frames,)  RMSD vs reference frame

  rmsf(trajectory, atom_indices=None)
      → np.ndarray  shape (n_atoms,)   per-atom RMSF in Å

  rmsd_after_superimpose(mobile, reference, atom_indices=None)
      → float  (RMSD after optimal Kabsch rotation, see alignment.py)

All positions in Angstroms.
"""
from __future__ import annotations

import numpy as np


# ── RMSD ──────────────────────────────────────────────────────────────────

def rmsd(
    mobile: np.ndarray,
    reference: np.ndarray,
    atom_indices: np.ndarray | list[int] | None = None,
) -> float:
    """
    Root Mean Square Deviation between two sets of atom positions.

    Parameters
    ----------
    mobile, reference : (N, 3) float64 arrays
    atom_indices      : optional subset of atom indices to use

    Returns
    -------
    RMSD in Angstroms (float)
    """
    mob = np.asarray(mobile, dtype=float)
    ref = np.asarray(reference, dtype=float)

    if atom_indices is not None:
        idx = np.asarray(atom_indices)
        mob = mob[idx]
        ref = ref[idx]

    if mob.shape != ref.shape:
        raise ValueError(
            f"rmsd: shape mismatch {mob.shape} vs {ref.shape}"
        )

    diff = mob - ref
    return float(np.sqrt((diff * diff).sum() / len(mob)))


def rmsd_trajectory(
    trajectory: list[np.ndarray],
    reference_frame: int = 0,
    atom_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """
    RMSD of each frame against a reference frame.

    Parameters
    ----------
    trajectory      : list of (N, 3) arrays
    reference_frame : index into trajectory (default 0)
    atom_indices    : optional subset

    Returns
    -------
    np.ndarray shape (n_frames,) — RMSD in Å
    """
    if not trajectory:
        return np.array([])

    ref = trajectory[reference_frame]
    result = np.empty(len(trajectory))
    for i, frame in enumerate(trajectory):
        result[i] = rmsd(frame, ref, atom_indices)
    return result


# ── RMSF ──────────────────────────────────────────────────────────────────

def rmsf(
    trajectory: list[np.ndarray],
    atom_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """
    Root Mean Square Fluctuation per atom across the trajectory.

    This measures how much each atom deviates from its mean position —
    higher RMSF = more flexible.

    Parameters
    ----------
    trajectory   : list of (N, 3) arrays
    atom_indices : optional subset

    Returns
    -------
    np.ndarray shape (N,) or shape (len(atom_indices),) — RMSF in Å
    """
    if not trajectory:
        return np.array([])

    frames = np.stack(trajectory, axis=0)   # shape (n_frames, N, 3)

    if atom_indices is not None:
        idx = np.asarray(atom_indices)
        frames = frames[:, idx, :]

    mean_pos = frames.mean(axis=0)                        # (N, 3)
    deviation = frames - mean_pos[np.newaxis, :, :]       # (F, N, 3)
    msd = (deviation ** 2).sum(axis=2).mean(axis=0)       # (N,)
    return np.sqrt(msd)


# ── Per-residue RMSF ──────────────────────────────────────────────────────

def rmsf_per_residue(
    trajectory: list[np.ndarray],
    atoms: list,
) -> dict[int, float]:
    """
    Per-residue RMSF (averaged over all atoms in each residue).

    Parameters
    ----------
    trajectory : list of (N, 3) frames
    atoms      : list of Atom objects (must have residue_id attribute)

    Returns
    -------
    dict {residue_id: rmsf_value_in_Angstrom}
    """
    from collections import defaultdict

    per_atom_rmsf = rmsf(trajectory)

    res_atoms: dict[int, list[int]] = defaultdict(list)
    for idx, atom in enumerate(atoms):
        rid = getattr(atom, "residue_id", None)
        if rid is not None:
            res_atoms[rid].append(idx)

    result: dict[int, float] = {}
    for rid, indices in sorted(res_atoms.items()):
        result[rid] = float(per_atom_rmsf[indices].mean())
    return result


# ── RMSD after superimposition ────────────────────────────────────────────

def rmsd_after_superimpose(
    mobile: np.ndarray,
    reference: np.ndarray,
    atom_indices: np.ndarray | list[int] | None = None,
) -> float:
    """
    RMSD after optimal superimposition (Kabsch rotation).
    Uses relative import to avoid fragile absolute package paths.
    """
    # Relative import — works regardless of how tests are invoked
    from .alignment import kabsch_rmsd
    return kabsch_rmsd(mobile, reference, atom_indices=atom_indices)