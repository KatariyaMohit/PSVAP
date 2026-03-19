"""
analysis/alignment.py
---------------------
Feature 2: Structure Alignment (Kabsch algorithm).

Pure NumPy implementation.  No external dependencies.

The Kabsch algorithm finds the optimal rotation matrix R that minimises
RMSD between two sets of points after translation to their centroids.

Public API
----------
  kabsch_rotation(mobile, reference)
      → R (3×3 rotation matrix)

  superimpose(mobile, reference, atom_indices=None)
      → (mobile_aligned, R, t, rmsd_val)

  kabsch_rmsd(mobile, reference, atom_indices=None)
      → float (RMSD after optimal rotation)

  superimpose_trajectory(trajectory, reference, atom_indices=None)
      → list of aligned frames

  align_trajectory(model, reference_frame=0, atom_indices=None)
      → int  (number of frames aligned)
      Aligns all frames in-place on the SystemModel trajectory.

  rmsd_matrix(trajectory, atom_indices=None, align_first=True)
      → (n_frames × n_frames) float64 array

All positions in Angstroms.
"""
from __future__ import annotations

import numpy as np


# ── Kabsch algorithm ──────────────────────────────────────────────────────

def kabsch_rotation(
    mobile: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Compute the optimal rotation matrix R (3×3) that minimises RMSD
    between mobile and reference (already centred at origin).

    Uses singular value decomposition.  Handles reflection correction.

    Parameters
    ----------
    mobile, reference : (N, 3) arrays centred at origin

    Returns
    -------
    R : (3, 3) rotation matrix  (apply as mobile @ R.T)
    """
    H = mobile.T @ reference                     # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Reflection correction (ensure proper rotation det(R) = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = (Vt.T @ D @ U.T)
    return R


def superimpose(
    mobile: np.ndarray,
    reference: np.ndarray,
    atom_indices: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Superimpose mobile onto reference using the Kabsch algorithm.

    Steps:
      1. Select subset of atoms if atom_indices given.
      2. Translate both to centroids.
      3. Compute optimal rotation via Kabsch.
      4. Apply rotation to the FULL mobile structure.

    Parameters
    ----------
    mobile, reference : (N, 3) arrays
    atom_indices      : optional subset used for alignment calculation

    Returns
    -------
    mobile_aligned : (N, 3)  full mobile structure after transformation
    R              : (3, 3)  rotation matrix
    t              : (3,)    translation vector applied
    rmsd_val       : float   RMSD over alignment atoms after superimposition
    """
    mob = np.asarray(mobile, dtype=float)
    ref = np.asarray(reference, dtype=float)

    if atom_indices is not None:
        idx = np.asarray(atom_indices)
        mob_sub = mob[idx]
        ref_sub = ref[idx]
    else:
        mob_sub = mob
        ref_sub = ref

    # Centroids
    c_mob = mob_sub.mean(axis=0)
    c_ref = ref_sub.mean(axis=0)

    # Centre
    mob_c = mob_sub - c_mob
    ref_c = ref_sub - c_ref

    # Rotation
    R = kabsch_rotation(mob_c, ref_c)

    # Apply to full structure
    t = c_ref - c_mob @ R.T
    mobile_aligned = (mob - c_mob) @ R.T + c_ref

    # RMSD over alignment atoms only
    diff = (mob_sub - c_mob) @ R.T - ref_c
    rmsd_val = float(np.sqrt((diff ** 2).sum() / len(mob_sub)))

    return mobile_aligned, R, t, rmsd_val


def kabsch_rmsd(
    mobile: np.ndarray,
    reference: np.ndarray,
    atom_indices: np.ndarray | list[int] | None = None,
) -> float:
    """
    Minimum RMSD between mobile and reference after optimal superimposition.
    """
    _, _, _, rmsd_val = superimpose(mobile, reference, atom_indices)
    return rmsd_val


def superimpose_trajectory(
    trajectory: list[np.ndarray],
    reference: np.ndarray,
    atom_indices: np.ndarray | list[int] | None = None,
) -> list[np.ndarray]:
    """
    Superimpose every frame in trajectory onto reference.

    Parameters
    ----------
    trajectory   : list of (N, 3) frames
    reference    : (N, 3) reference frame (usually frame 0)
    atom_indices : subset used for alignment

    Returns
    -------
    List of aligned (N, 3) frames (new arrays, originals unchanged)
    """
    aligned: list[np.ndarray] = []
    for frame in trajectory:
        frame_aligned, _, _, _ = superimpose(frame, reference, atom_indices)
        aligned.append(frame_aligned)
    return aligned


def align_trajectory(
    model,
    reference_frame: int = 0,
    atom_indices: np.ndarray | list[int] | None = None,
) -> int:
    """
    Superimpose every frame of a SystemModel's trajectory onto a reference
    frame IN-PLACE (replaces model.trajectory with aligned frames).

    Called by analysis_panel._run_align_trajectory().

    Parameters
    ----------
    model           : SystemModel instance
    reference_frame : index of the frame to align to (default 0)
    atom_indices    : optional atom subset used for alignment calculation;
                      the full structure is transformed

    Returns
    -------
    int — number of frames that were successfully aligned

    Raises
    ------
    ValueError  if model has no trajectory or reference_frame is out of range
    """
    trajectory = getattr(model, 'trajectory', [])
    if not trajectory:
        raise ValueError("Model has no trajectory loaded.")

    n_frames = len(trajectory)
    if not (0 <= reference_frame < n_frames):
        raise ValueError(
            f"reference_frame={reference_frame} is out of range "
            f"(trajectory has {n_frames} frames, indices 0–{n_frames-1})."
        )

    ref = np.asarray(trajectory[reference_frame], dtype=float)
    aligned_frames: list[np.ndarray] = []
    n_aligned = 0

    for i, frame in enumerate(trajectory):
        try:
            frame_arr = np.asarray(frame, dtype=float)
            if frame_arr.shape != ref.shape:
                # Shape mismatch — keep original frame unchanged
                aligned_frames.append(frame)
                continue
            aligned, _, _, _ = superimpose(frame_arr, ref, atom_indices)
            aligned_frames.append(aligned)
            n_aligned += 1
        except Exception:
            # On any per-frame error keep original
            aligned_frames.append(frame)

    # Replace trajectory in-place
    model.trajectory = aligned_frames
    return n_aligned


def rmsd_matrix(
    trajectory: list[np.ndarray],
    atom_indices: np.ndarray | list[int] | None = None,
    align_first: bool = True,
) -> np.ndarray:
    """
    Compute a symmetric pairwise RMSD matrix between all frames.
    Useful for trajectory clustering (Phase 3).

    Parameters
    ----------
    trajectory   : list of (N, 3) frames
    atom_indices : optional subset
    align_first  : if True, superimpose each pair before computing RMSD

    Returns
    -------
    (n_frames × n_frames) float64 array
    """
    # Use relative import to avoid package-path fragility
    from .rmsd import rmsd as _rmsd

    n = len(trajectory)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if align_first:
                val = kabsch_rmsd(trajectory[j], trajectory[i], atom_indices)
            else:
                val = _rmsd(trajectory[j], trajectory[i], atom_indices)
            matrix[i, j] = val
            matrix[j, i] = val

    return matrix