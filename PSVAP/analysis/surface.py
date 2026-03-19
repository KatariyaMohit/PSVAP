"""
analysis/surface.py
-------------------
Features 9, 18: Solvent Accessible Surface Area (SASA) and Protein
Patch Analysis.

Pure-Python/NumPy implementation of the Lee & Richards rolling-probe
algorithm (approximated via Shrake-Rupley point sampling).

Public API
----------
  compute_sasa(atoms, positions, probe_radius=1.4)
      → dict {atom_idx: sasa_value_Å²}

  sasa_per_residue(atoms, positions, probe_radius=1.4)
      → dict {residue_id: total_sasa_Å²}

  sasa_trajectory(atoms, trajectory, probe_radius=1.4)
      → np.ndarray  shape (n_frames, n_atoms)

  classify_surface_patches(atoms, positions)
      → dict {residue_id: patch_type}
      patch_type: 'hydrophobic', 'positive', 'negative', 'polar', 'other'

All positions in Angstroms.
"""
from __future__ import annotations

import numpy as np

from PSVAP.core.constants import WATER_PROBE_RADIUS

# ── VdW radii for SASA computation (Å) ────────────────────────────────────
_SASA_VDW: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.47, "ZN": 1.39, "MG": 1.73, "CA": 1.97, "NA": 2.27,
    "K": 2.75, "DEFAULT": 1.70,
}

# ── Hydrophobicity scales (Kyte-Doolittle) ────────────────────────────────
_HYDROPHOBICITY: dict[str, float] = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2,
    "HSD": -3.2, "HSE": -3.2,
    "GLU": -3.5, "GLN": -3.5, "ASP": -3.5, "ASN": -3.5,
    "LYS": -3.9, "ARG": -4.5,
    "DEFAULT": 0.0,
}

# Residue surface charge classification
_POS_RESIDUES = {"ARG", "LYS", "HIS", "HSD", "HSE", "HSP"}
_NEG_RESIDUES = {"ASP", "GLU"}
_POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
_HYDRO_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "GLY"}

# Fibonacci sphere point count for Shrake-Rupley sampling
_N_SPHERE_POINTS = 92


def _get_vdw(element: str | None) -> float:
    if element:
        return _SASA_VDW.get(element.upper(), _SASA_VDW["DEFAULT"])
    return _SASA_VDW["DEFAULT"]


def _fibonacci_sphere(n: int) -> np.ndarray:
    """
    Generate n evenly-distributed points on a unit sphere using the
    Fibonacci lattice (golden ratio method).
    Returns (n, 3) array.
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n)
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    phi   = 2.0 * np.pi * i / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


_SPHERE_POINTS = _fibonacci_sphere(_N_SPHERE_POINTS)


def compute_sasa(
    atoms: list,
    positions: np.ndarray,
    probe_radius: float = WATER_PROBE_RADIUS,
) -> dict[int, float]:
    """
    Compute Solvent Accessible Surface Area per atom using the
    Shrake-Rupley algorithm.

    For each atom i:
      1. Place _N_SPHERE_POINTS on a sphere of radius (vdW_i + probe).
      2. Count points not buried by any neighbouring atom j
         (point buried if dist(point, atom_j) < vdW_j + probe).
      3. SASA_i = (accessible_points / total_points) × 4π(vdW_i + probe)²

    Parameters
    ----------
    atoms        : list of Atom objects (need .element)
    positions    : (N, 3) positions in Å
    probe_radius : water probe radius (default 1.4 Å)

    Returns
    -------
    dict {atom_index: SASA_in_Å²}
    """
    pos = np.asarray(positions, dtype=float)
    n = len(atoms)
    radii = np.array([_get_vdw(getattr(a, "element", None)) for a in atoms])
    extended = radii + probe_radius   # radius of accessible sphere

    # Build neighbour lists using a simple distance cutoff
    # (max possible interaction = max_extended * 2)
    max_ext = extended.max() if len(extended) > 0 else 3.0

    result: dict[int, float] = {}

    for i in range(n):
        r_i  = extended[i]
        area = 4.0 * np.pi * r_i * r_i

        # Find neighbours within interaction range
        diffs = pos - pos[i]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        neighbour_mask = (dists < (r_i + max_ext)) & (dists > 1e-6)
        neighbour_idx  = np.where(neighbour_mask)[0]

        if len(neighbour_idx) == 0:
            result[i] = area
            continue

        # Test sphere points
        test_points = pos[i] + r_i * _SPHERE_POINTS   # (N_pts, 3)
        accessible  = np.ones(len(_SPHERE_POINTS), dtype=bool)

        for j in neighbour_idx:
            r_j = extended[j]
            diff_pts = test_points - pos[j]
            dist_pts = np.sqrt((diff_pts * diff_pts).sum(axis=1))
            accessible &= (dist_pts >= r_j)

        fraction = accessible.sum() / len(_SPHERE_POINTS)
        result[i] = fraction * area

    return result


def sasa_per_residue(
    atoms: list,
    positions: np.ndarray,
    probe_radius: float = WATER_PROBE_RADIUS,
) -> dict[int, float]:
    """
    Total SASA per residue (sum of per-atom SASA within each residue).

    Returns
    -------
    dict {residue_id: total_SASA_in_Å²}
    """
    from collections import defaultdict
    per_atom = compute_sasa(atoms, positions, probe_radius)

    residue_sasa: dict[int, float] = defaultdict(float)
    for idx, atom in enumerate(atoms):
        rid = getattr(atom, "residue_id", None)
        if rid is not None and idx in per_atom:
            residue_sasa[rid] += per_atom[idx]

    return dict(residue_sasa)


def sasa_trajectory(
    atoms: list,
    trajectory: list[np.ndarray],
    probe_radius: float = WATER_PROBE_RADIUS,
) -> np.ndarray:
    """
    Compute per-atom SASA for every frame in trajectory.

    Returns
    -------
    np.ndarray shape (n_frames, n_atoms) — SASA in Å² per atom per frame
    Warning: this is computationally expensive for large systems.
    Consider using a subset (atom_indices) for long trajectories.
    """
    n_atoms  = len(atoms)
    n_frames = len(trajectory)
    result   = np.zeros((n_frames, n_atoms), dtype=float)

    for fi, frame in enumerate(trajectory):
        per_atom = compute_sasa(atoms, frame, probe_radius)
        for idx, val in per_atom.items():
            if idx < n_atoms:
                result[fi, idx] = val

    return result


def classify_surface_patches(
    atoms: list,
    positions: np.ndarray,
    probe_radius: float = WATER_PROBE_RADIUS,
    sasa_threshold: float = 5.0,
) -> dict[int, str]:
    """
    Classify each surface-exposed residue by patch type.

    A residue is considered surface-exposed if its total SASA > sasa_threshold.
    Patch types: 'hydrophobic', 'positive', 'negative', 'polar', 'other'.

    Parameters
    ----------
    sasa_threshold : minimum SASA (Å²) for a residue to be surface-exposed

    Returns
    -------
    dict {residue_id: patch_type}
    """
    res_sasa = sasa_per_residue(atoms, positions, probe_radius)

    # Map residue_id → resname
    res_names: dict[int, str] = {}
    for atom in atoms:
        rid = getattr(atom, "residue_id", None)
        if rid is not None and rid not in res_names:
            rn = (getattr(atom, "resname", None) or "").upper()
            if rn:
                res_names[rid] = rn

    patch_type: dict[int, str] = {}
    for rid, sasa_val in res_sasa.items():
        if sasa_val < sasa_threshold:
            continue
        rn = res_names.get(rid, "")
        if rn in _HYDRO_RESIDUES:
            patch_type[rid] = "hydrophobic"
        elif rn in _POS_RESIDUES:
            patch_type[rid] = "positive"
        elif rn in _NEG_RESIDUES:
            patch_type[rid] = "negative"
        elif rn in _POLAR_RESIDUES:
            patch_type[rid] = "polar"
        else:
            patch_type[rid] = "other"

    return patch_type


def get_hydrophobicity(resname: str) -> float:
    """Return Kyte-Doolittle hydrophobicity score for a residue name."""
    return _HYDROPHOBICITY.get(resname.upper(), _HYDROPHOBICITY["DEFAULT"])

# ── Water Map Analysis (Feature 20) ──────────────────────────────────────

def compute_water_density(
    atoms: list,
    trajectory: list[np.ndarray],
    grid_spacing: float = 0.5,
    water_resnames: set[str] | None = None,
    contour_level: float = 2.0,
) -> dict:
    """
    Compute the spatial density of water molecules across a trajectory.

    Counts water oxygen occurrences on a 3D grid, normalized by
    the number of frames. Grid points with density > contour_level
    standard deviations above the mean are reported as hydration sites.

    Parameters
    ----------
    atoms          : full atom list (must include water atoms with resname HOH/WAT)
    trajectory     : list of (N, 3) position arrays
    grid_spacing   : grid resolution in Å (default 0.5)
    water_resnames : set of water residue names (default HOH/WAT/TIP3/SOL)
    contour_level  : sigma threshold for hydration site detection

    Returns
    -------
    dict with keys:
      'grid_origin'     : (3,) array — grid lower-left corner
      'grid_spacing'    : float
      'grid_dims'       : (3,) int array
      'density'         : 3D numpy array — counts per voxel
      'hydration_sites' : list of (center_xyz, density_value) tuples
      'n_frames'        : int
      'n_water_atoms'   : int
    """
    _WATER = water_resnames or {"HOH", "WAT", "TIP3", "TIP3P", "SOL", "H2O"}

    if not trajectory or not atoms:
        return {}

    # Identify water oxygen atom indices
    water_o_indices = [
        i for i, a in enumerate(atoms)
        if (getattr(a, "resname", None) or "").upper() in _WATER
        and (getattr(a, "name", None) or "").upper() in {"O", "OW", "OH2", "O1"}
    ]

    if not water_o_indices:
        return {
            "grid_origin": np.zeros(3),
            "grid_spacing": grid_spacing,
            "grid_dims": np.zeros(3, dtype=int),
            "density": np.zeros((1, 1, 1)),
            "hydration_sites": [],
            "n_frames": len(trajectory),
            "n_water_atoms": 0,
        }

    # Determine grid bounds from all trajectory frames
    all_water_pos = []
    for frame in trajectory:
        for i in water_o_indices:
            if i < len(frame):
                all_water_pos.append(frame[i])

    if not all_water_pos:
        return {}

    all_water_pos = np.array(all_water_pos, dtype=float)
    lo = all_water_pos.min(axis=0) - 1.0
    hi = all_water_pos.max(axis=0) + 1.0

    dims = np.ceil((hi - lo) / grid_spacing).astype(int) + 1
    # Cap grid for memory
    max_pts = 100
    if dims.max() > max_pts:
        grid_spacing = float((hi - lo).max() / (max_pts - 1))
        dims = np.ceil((hi - lo) / grid_spacing).astype(int) + 1

    density = np.zeros(dims, dtype=np.float32)

    # Accumulate water oxygen positions across all frames
    for frame in trajectory:
        for i in water_o_indices:
            if i >= len(frame):
                continue
            pt = frame[i]
            idx = np.floor((pt - lo) / grid_spacing).astype(int)
            # Bounds check
            if np.all(idx >= 0) and np.all(idx < dims):
                density[idx[0], idx[1], idx[2]] += 1.0

    # Normalize by number of frames
    if len(trajectory) > 0:
        density /= len(trajectory)

    # Find hydration sites: peaks above contour_level sigma
    mean_d = float(density[density > 0].mean()) if (density > 0).any() else 0.0
    std_d  = float(density[density > 0].std())  if (density > 0).any() else 1.0
    threshold = mean_d + contour_level * std_d

    hydration_sites: list[tuple[np.ndarray, float]] = []
    peak_indices = np.argwhere(density > threshold)
    for idx in peak_indices:
        center = lo + idx * grid_spacing
        hydration_sites.append((center, float(density[tuple(idx)])))

    # Sort by density descending
    hydration_sites.sort(key=lambda x: -x[1])

    return {
        "grid_origin":     lo,
        "grid_spacing":    grid_spacing,
        "grid_dims":       dims,
        "density":         density,
        "hydration_sites": hydration_sites,
        "n_frames":        len(trajectory),
        "n_water_atoms":   len(water_o_indices),
    }


def format_water_map(water_data: dict) -> str:
    """Format water map results as text for GUI display."""
    if not water_data:
        return "NO WATER MAP DATA"

    n_frames = water_data.get("n_frames", 0)
    n_waters = water_data.get("n_water_atoms", 0)
    sites    = water_data.get("hydration_sites", [])

    lines = [
        f"WATER MAP ANALYSIS\n",
        f"  Frames analysed  : {n_frames}",
        f"  Water O atoms    : {n_waters}",
        f"  Hydration sites  : {len(sites)}",
        "",
    ]

    if not sites:
        lines.append(
            "No high-density hydration sites found.\n"
            "Try loading a longer trajectory or reducing the contour level."
        )
        return "\n".join(lines)

    lines.extend([
        f"TOP HYDRATION SITES:",
        f"{'RANK':>5}  {'X':>8}  {'Y':>8}  {'Z':>8}  {'DENSITY':>10}",
        "-" * 46,
    ])

    for i, (center, density) in enumerate(sites[:20]):
        lines.append(
            f"{i+1:>5}  {center[0]:>8.3f}  {center[1]:>8.3f}  "
            f"{center[2]:>8.3f}  {density:>10.4f}"
        )

    if len(sites) > 20:
        lines.append(f"  ... {len(sites)-20} more sites")

    return "\n".join(lines)