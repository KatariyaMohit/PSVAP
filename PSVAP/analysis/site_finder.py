"""
analysis/site_finder.py
------------------------
Feature 19: Binding Site Detection.

Detects potential ligand-binding pockets and cavities on a protein
surface using a grid-based algorithm (no fpocket required for basic
detection). fpocket subprocess integration is provided for
research-quality results when fpocket is installed.

Public API
----------
  find_sites_grid(atoms, positions, probe_radius=1.4,
                  grid_spacing=1.0, min_pocket_volume=100.0)
      → list[BindingSite]

  find_sites_fpocket(pdb_path, fpocket_executable='fpocket')
      → list[BindingSite]

  BindingSite (dataclass)
      center, volume, score, residue_ids, atom_indices, rank

All positions in Angstroms.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BindingSite:
    """A detected binding pocket or cavity."""
    rank:          int
    center:        np.ndarray      # (3,) centroid in Å
    volume:        float           # approximate volume in Å³
    score:         float           # higher = more druggable
    residue_ids:   list[int]       # lining residue IDs
    atom_indices:  list[int]       # atoms lining the pocket
    method:        str = "grid"    # 'grid' or 'fpocket'
    description:   str = ""


def find_sites_grid(
    atoms: list,
    positions: np.ndarray,
    probe_radius: float = 1.4,
    grid_spacing: float = 1.0,
    min_pocket_volume: float = 100.0,
    max_sites: int = 10,
) -> list[BindingSite]:
    """
    Grid-based binding site detection.

    Algorithm (simplified Fpocket-inspired approach):
      1. Build a 3D grid over the protein bounding box.
      2. Mark grid points as 'protein' (within vdW radius of any atom),
         'solvent' (accessible to probe), or 'pocket' (buried but
         not protein — enclosed empty space).
      3. Cluster pocket grid points into discrete sites.
      4. Score each site by size and hydrophobicity of lining residues.
      5. Return top max_sites ranked by score.

    Parameters
    ----------
    atoms             : full atom list
    positions         : (N, 3) positions in Å
    probe_radius      : water probe radius (Å)
    grid_spacing      : grid resolution (Å) — smaller = more accurate, slower
    min_pocket_volume : discard pockets smaller than this (Å³)
    max_sites         : maximum number of sites to return

    Returns
    -------
    list[BindingSite] sorted by score descending
    """
    pos = np.asarray(positions, dtype=float)
    n   = len(atoms)

    if n == 0:
        return []

    # VdW radii for buriedness calculation
    _VDW = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8,
            "P": 1.8, "DEFAULT": 1.7}

    def vdw(atom) -> float:
        e = (getattr(atom, "element", None) or "").upper()
        return _VDW.get(e, _VDW["DEFAULT"])

    radii = np.array([vdw(a) for a in atoms])

    # Build grid
    pad = probe_radius + 2.0
    lo  = pos.min(axis=0) - pad
    hi  = pos.max(axis=0) + pad
    dims = np.ceil((hi - lo) / grid_spacing).astype(int) + 1

    # Cap grid size for performance
    max_dim = 60
    if dims.max() > max_dim:
        grid_spacing = float((hi - lo).max() / (max_dim - 1))
        dims = np.ceil((hi - lo) / grid_spacing).astype(int) + 1

    # Create grid point coordinates
    xs = lo[0] + np.arange(dims[0]) * grid_spacing
    ys = lo[1] + np.arange(dims[1]) * grid_spacing
    zs = lo[2] + np.arange(dims[2]) * grid_spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])  # (M, 3)

    # Mark protein-occupied points
    # A grid point is 'protein' if within (vdW + probe) of any atom
    protein_mask = np.zeros(len(grid_pts), dtype=bool)

    # Process in batches for memory efficiency
    batch = 5000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        # distances from each atom to all grid points
        diff = grid_pts[:, np.newaxis, :] - pos[start:end][np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))   # (M, batch)
        thresh = (radii[start:end] + probe_radius)[np.newaxis, :]
        protein_mask |= (dist < thresh).any(axis=1)

    # 'Pocket' points: not protein, but enclosed
    # Simple buriedness: count protein grid neighbours in a shell
    empty_idx = np.where(~protein_mask)[0]

    if len(empty_idx) == 0:
        return []

    empty_pts = grid_pts[empty_idx]

    # For each empty point, count protein-occupied neighbours in radius 5Å
    neighbor_radius = 5.0
    buried_counts = np.zeros(len(empty_pts), dtype=int)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        diff = empty_pts[:, np.newaxis, :] - pos[start:end][np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        thresh = (radii[start:end] + neighbor_radius)[np.newaxis, :]
        buried_counts += (dist < thresh).sum(axis=1)

    # Pocket threshold: points with enough protein neighbours
    min_neighbors = max(3, int(neighbor_radius / grid_spacing))
    pocket_mask = buried_counts >= min_neighbors
    pocket_pts  = empty_pts[pocket_mask]

    if len(pocket_pts) == 0:
        return []

    # Cluster pocket points using simple distance-based grouping
    clusters = _cluster_points(pocket_pts, cutoff=grid_spacing * 3.0)

    # Build BindingSite for each cluster
    sites: list[BindingSite] = []
    voxel_vol = grid_spacing ** 3

    for cluster_pts in clusters:
        vol = len(cluster_pts) * voxel_vol
        if vol < min_pocket_volume:
            continue

        center = cluster_pts.mean(axis=0)

        # Find lining atoms (within 5Å of pocket center)
        dists_to_center = np.linalg.norm(pos - center, axis=1)
        lining_idx = np.where(dists_to_center < 5.0 + neighbor_radius)[0].tolist()

        # Residue IDs of lining atoms
        res_ids = sorted({
            getattr(atoms[i], "residue_id", None)
            for i in lining_idx
            if getattr(atoms[i], "residue_id", None) is not None
        })

        # Score: combination of volume + hydrophobic lining fraction
        hydro_count = sum(
            1 for i in lining_idx
            if (getattr(atoms[i], "element", None) or "").upper() == "C"
        )
        hydro_fraction = hydro_count / max(len(lining_idx), 1)
        score = vol * 0.01 + hydro_fraction * 10.0

        sites.append(BindingSite(
            rank=0,
            center=center,
            volume=vol,
            score=score,
            residue_ids=res_ids,
            atom_indices=lining_idx,
            method="grid",
            description=f"{len(res_ids)} lining residues, "
                        f"{hydro_fraction*100:.0f}% hydrophobic",
        ))

    # Sort by score and assign ranks
    sites.sort(key=lambda s: -s.score)
    for i, s in enumerate(sites[:max_sites]):
        s.rank = i + 1

    return sites[:max_sites]


def find_sites_fpocket(
    pdb_path: str | Path,
    fpocket_executable: str = "fpocket",
) -> list[BindingSite]:
    """
    Run fpocket for binding site detection.

    Requires fpocket to be installed and in PATH.
    fpocket is available at: https://github.com/Discngine/fpocket

    Parameters
    ----------
    pdb_path           : path to PDB file
    fpocket_executable : name or path of fpocket executable

    Returns
    -------
    list[BindingSite] parsed from fpocket output

    Raises
    ------
    FileNotFoundError  if fpocket not found
    FileNotFoundError  if pdb_path does not exist
    RuntimeError       if fpocket fails
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Check fpocket availability
    try:
        subprocess.run(
            [fpocket_executable, "--help"],
            capture_output=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise FileNotFoundError(
            f"fpocket not found at '{fpocket_executable}'.\n"
            "Install: apt install fpocket  or  brew install fpocket\n"
            "Or download from: https://github.com/Discngine/fpocket"
        )

    try:
        proc = subprocess.run(
            [fpocket_executable, "-f", str(pdb_path)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("fpocket timed out after 120 seconds.")

    if proc.returncode != 0:
        raise RuntimeError(
            f"fpocket failed (exit {proc.returncode}):\n{proc.stderr[:500]}"
        )

    return _parse_fpocket_output(pdb_path, proc.stdout)


def format_sites(sites: list[BindingSite]) -> str:
    """Format binding site list as text for GUI display."""
    if not sites:
        return "NO BINDING SITES FOUND"

    lines = [
        f"BINDING SITES  ({len(sites)} found)\n",
        f"{'RANK':>5}  {'VOLUME (Å³)':>12}  {'SCORE':>7}  "
        f"{'RESIDUES':>8}  DESCRIPTION",
        "-" * 60,
    ]
    for s in sites:
        lines.append(
            f"{s.rank:>5}  {s.volume:>12.1f}  {s.score:>7.2f}  "
            f"{len(s.residue_ids):>8}  {s.description}"
        )
        c = s.center
        lines.append(
            f"       Center: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Å"
        )
        if s.residue_ids:
            res_str = ", ".join(str(r) for r in s.residue_ids[:10])
            if len(s.residue_ids) > 10:
                res_str += f" ... +{len(s.residue_ids)-10}"
            lines.append(f"       Residues: {res_str}")
        lines.append("")

    return "\n".join(lines)


def check_fpocket_available(executable: str = "fpocket") -> bool:
    """Return True if fpocket is found in PATH."""
    try:
        proc = subprocess.run(
            [executable, "--help"],
            capture_output=True, timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Internal helpers ──────────────────────────────────────────────────────

def _cluster_points(
    points: np.ndarray,
    cutoff: float,
) -> list[np.ndarray]:
    """
    Greedy distance-based clustering of 3D points.
    Points within cutoff of any cluster member are merged.
    Returns list of (N_cluster, 3) arrays.
    """
    if len(points) == 0:
        return []

    assigned = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []

    for seed_idx in range(len(points)):
        if assigned[seed_idx]:
            continue

        cluster_indices = [seed_idx]
        assigned[seed_idx] = True
        queue = [seed_idx]

        while queue:
            curr = queue.pop()
            dists = np.linalg.norm(points - points[curr], axis=1)
            neighbours = np.where((dists < cutoff) & ~assigned)[0]
            for nb in neighbours:
                assigned[nb] = True
                cluster_indices.append(nb)
                queue.append(nb)

        clusters.append(points[cluster_indices])

    return clusters


def _parse_fpocket_output(
    pdb_path: Path,
    stdout: str,
) -> list[BindingSite]:
    """
    Parse fpocket stdout for pocket summary.

    fpocket output format (varies by version):
      Pocket N :
        Score :           X.XX
        Druggability Score : X.XX
        Volume :          X.XX
        ...
    """
    sites: list[BindingSite] = []
    lines = stdout.splitlines()
    current_rank  = 0
    current_score = 0.0
    current_vol   = 0.0
    in_pocket     = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Pocket") and ":" in stripped:
            if in_pocket and current_rank > 0:
                sites.append(BindingSite(
                    rank=current_rank,
                    center=np.zeros(3),
                    volume=current_vol,
                    score=current_score,
                    residue_ids=[],
                    atom_indices=[],
                    method="fpocket",
                    description=f"fpocket pocket {current_rank}",
                ))
            try:
                current_rank = int(stripped.split()[1])
            except (IndexError, ValueError):
                current_rank += 1
            current_score = 0.0
            current_vol   = 0.0
            in_pocket     = True

        elif in_pocket and "Score" in stripped and "Drugg" not in stripped:
            try:
                current_score = float(stripped.split(":")[1].strip())
            except (IndexError, ValueError):
                pass

        elif in_pocket and "Volume" in stripped:
            try:
                current_vol = float(stripped.split(":")[1].strip())
            except (IndexError, ValueError):
                pass

    # Last pocket
    if in_pocket and current_rank > 0:
        sites.append(BindingSite(
            rank=current_rank,
            center=np.zeros(3),
            volume=current_vol,
            score=current_score,
            residue_ids=[],
            atom_indices=[],
            method="fpocket",
            description=f"fpocket pocket {current_rank}",
        ))

    return sorted(sites, key=lambda s: -s.score)