"""
visualization/structure_renderer.py
------------------------------------
Handles atom sphere and bond cylinder rendering via PyVista.

Render modes
------------
  'atoms'       — spheres only (fast, for large systems)
  'atoms_bonds' — spheres + cylinders (default)
  'bonds'       — cylinders only (wireframe feel)

Bond detection
--------------
1. If the parser provided an explicit bond list → use it.
2. If no bonds provided → compute by distance cutoff per element pair.
   Standard covalent radii sum + 0.4 Å tolerance.

This module is called by VisualizationEngine and has NO knowledge of
Qt or the GUI.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

# ── Element data ─────────────────────────────────────────────────────────
# Covalent radii (Å) — from Alvarez 2008
_COVALENT_RADIUS: dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "FE": 1.32, "ZN": 1.22, "MG": 1.41, "CA": 1.76, "NA": 1.66,
    "K": 2.03,  "CU": 1.32, "MN": 1.61, "SE": 1.20, "SI": 1.11,
    "DEFAULT": 0.77,
}

# CPK colours (R,G,B) 0-1
_ELEMENT_COLOR: dict[str, tuple[float, float, float]] = {
    "H":  (0.90, 0.90, 0.90),
    "C":  (0.40, 0.40, 0.40),
    "N":  (0.20, 0.40, 0.90),
    "O":  (0.90, 0.20, 0.20),
    "S":  (0.90, 0.80, 0.10),
    "P":  (0.90, 0.50, 0.10),
    "FE": (0.80, 0.30, 0.10),
    "ZN": (0.50, 0.50, 0.75),
    "MG": (0.20, 0.70, 0.20),
    "CA": (0.30, 0.80, 0.80),
    "CL": (0.10, 0.80, 0.10),
    "BR": (0.60, 0.10, 0.10),
    "NA": (0.70, 0.20, 0.80),
    "DEFAULT": (0.60, 0.60, 0.60),
}

# VDW radii for sphere display
_VDW_RADIUS: dict[str, float] = {
    "H": 0.25, "C": 0.40, "N": 0.38, "O": 0.36, "S": 0.50,
    "P": 0.50, "DEFAULT": 0.40,
}

BOND_TOLERANCE = 0.45   # Å added to covalent radius sum for bond detection
MAX_BOND_DIST  = 3.0    # Å absolute maximum (avoids detecting non-bonds)
BOND_RADIUS    = 0.06   # cylinder radius in Å


def get_element_color(element: str | None) -> tuple[float, float, float]:
    if element:
        return _ELEMENT_COLOR.get(element.upper(), _ELEMENT_COLOR["DEFAULT"])
    return _ELEMENT_COLOR["DEFAULT"]


def get_covalent_radius(element: str | None) -> float:
    if element:
        return _COVALENT_RADIUS.get(element.upper(), _COVALENT_RADIUS["DEFAULT"])
    return _COVALENT_RADIUS["DEFAULT"]


def get_vdw_radius(element: str | None) -> float:
    if element:
        return _VDW_RADIUS.get(element.upper(), _VDW_RADIUS["DEFAULT"])
    return _VDW_RADIUS["DEFAULT"]


# ── Bond detection ─────────────────────────────────────────────────────────

def detect_bonds(
    positions: np.ndarray,
    elements: list[str | None],
    explicit_bonds: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Return list of (i, j) bond pairs where i < j.

    Uses explicit bonds if provided, otherwise distance-based detection.
    Limited to MAX_BOND_DIST to prevent false positives.
    For large systems (>5000 atoms) uses a spatial grid for performance.
    """
    n = len(positions)
    if n < 2:
        return []

    if explicit_bonds:
        return [(min(a, b), max(a, b)) for a, b in explicit_bonds]

    radii = np.array([get_covalent_radius(e) for e in elements], dtype=np.float64)

    # For large systems use cell-list for O(N) performance
    if n > 5000:
        return _detect_bonds_cell_list(positions, radii, n)
    else:
        return _detect_bonds_brute(positions, radii, n)


def _detect_bonds_brute(
    positions: np.ndarray,
    radii: np.ndarray,
    n: int,
) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min(radii[i] + radii[j] + BOND_TOLERANCE, MAX_BOND_DIST):
                if dist > 0.4:   # skip atoms that are too close (artefact)
                    bonds.append((i, j))
    return bonds


def _detect_bonds_cell_list(
    positions: np.ndarray,
    radii: np.ndarray,
    n: int,
) -> list[tuple[int, int]]:
    """
    Spatial cell-list for O(N) bond detection in large systems.
    Cell size = MAX_BOND_DIST.
    """
    from collections import defaultdict

    cell_size = MAX_BOND_DIST
    cells: dict[tuple, list[int]] = defaultdict(list)

    for i in range(n):
        cx = int(positions[i, 0] // cell_size)
        cy = int(positions[i, 1] // cell_size)
        cz = int(positions[i, 2] // cell_size)
        cells[(cx, cy, cz)].append(i)

    bonds: list[tuple[int, int]] = []
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)]

    visited_pairs: set[tuple[int, int]] = set()
    for (cx, cy, cz), cell_atoms in cells.items():
        for dx, dy, dz in offsets:
            neighbor = (cx + dx, cy + dy, cz + dz)
            if neighbor not in cells:
                continue
            for i in cell_atoms:
                for j in cells[neighbor]:
                    if i >= j:
                        continue
                    pair = (i, j)
                    if pair in visited_pairs:
                        continue
                    visited_pairs.add(pair)
                    dist = np.linalg.norm(positions[i] - positions[j])
                    threshold = min(radii[i] + radii[j] + BOND_TOLERANCE, MAX_BOND_DIST)
                    if 0.4 < dist < threshold:
                        bonds.append(pair)
    return bonds


# ── Mesh builders ──────────────────────────────────────────────────────────

def build_atom_mesh(
    positions: np.ndarray,
    elements: list[str | None],
    selection_mask: np.ndarray | None = None,
    scale: float = 1.0,
) -> "pv.PolyData":
    """
    Build a PyVista PolyData sphere mesh for all atoms.
    Returns a point cloud with scalar arrays for color and size.
    """
    import pyvista as pv

    n = len(positions)
    cloud = pv.PolyData(positions.astype(np.float32))

    # Per-atom color as RGB array
    colors = np.array([get_element_color(e) for e in elements], dtype=np.float32)
    cloud.point_data["rgb"] = colors

    # Per-atom radius
    radii = np.array([get_vdw_radius(e) * scale for e in elements], dtype=np.float32)
    cloud.point_data["radius"] = radii

    # Selection mask (1=selected, 0=normal)
    if selection_mask is not None:
        cloud.point_data["selected"] = selection_mask.astype(np.uint8)
    else:
        cloud.point_data["selected"] = np.zeros(n, dtype=np.uint8)

    return cloud


def build_bond_mesh(
    positions: np.ndarray,
    elements: list[str | None],
    bonds: list[tuple[int, int]],
) -> "pv.MultiBlock | None":
    """
    Build bond cylinders as a PyVista MultiBlock.
    Each bond is a cylinder between atom i and atom j,
    half-coloured by each atom's element colour.
    Returns None if there are no bonds.
    """
    if not bonds:
        return None

    import pyvista as pv

    blocks = pv.MultiBlock()
    for i, j in bonds:
        p0 = positions[i].astype(float)
        p1 = positions[j].astype(float)
        mid = (p0 + p1) / 2.0
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue

        # First half: colour of atom i
        half_len = length / 2.0
        center1 = (p0 + mid) / 2.0
        cyl1 = pv.Cylinder(
            center=center1,
            direction=direction,
            radius=BOND_RADIUS,
            height=half_len,
            resolution=8,
        )
        c1 = get_element_color(elements[i])
        cyl1.cell_data["bond_color"] = np.tile(c1, (cyl1.n_cells, 1)).astype(np.float32)
        blocks.append(cyl1)

        # Second half: colour of atom j
        center2 = (mid + p1) / 2.0
        cyl2 = pv.Cylinder(
            center=center2,
            direction=direction,
            radius=BOND_RADIUS,
            height=half_len,
            resolution=8,
        )
        c2 = get_element_color(elements[j])
        cyl2.cell_data["bond_color"] = np.tile(c2, (cyl2.n_cells, 1)).astype(np.float32)
        blocks.append(cyl2)

    return blocks if len(blocks) > 0 else None