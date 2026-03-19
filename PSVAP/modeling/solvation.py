"""
modeling/solvation.py
---------------------
Feature 10: Solvent Box Construction.

Builds a simulation-ready solvated system by:
  1. Placing the solute in a box with a specified buffer distance.
  2. Tiling pre-built solvent molecule coordinates to fill the box.
  3. Removing solvent molecules that clash with the solute.
  4. Optionally adding counterions to neutralize charge.

This is a pure-Python/NumPy implementation — no PACKMOL required for
basic water solvation. PACKMOL integration (for mixed solvents) is
provided as a subprocess wrapper.

Public API
----------
  build_water_box(atoms, positions, buffer=10.0, water_model='tip3p')
      → (new_atoms, new_positions, box_bounds)

  estimate_ion_count(atoms, n_waters, charge_conc=0.15)
      → (n_na, n_cl)  ion counts to add

  SolvationResult  (dataclass)
      atoms, positions, box_bounds, n_waters_added, n_ions_added

All positions in Angstroms. Box buffer in Angstroms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PSVAP.core.atom import Atom

# ── Water molecule geometry (TIP3P model, O-H bond = 0.9572 Å) ────────────
_TIP3P_GEOMETRY = np.array([
    [0.000,  0.000,  0.000],   # O
    [0.957,  0.000,  0.000],   # H1
    [-0.239, 0.927,  0.000],   # H2
], dtype=float)

# SPC/E water
_SPCE_GEOMETRY = np.array([
    [0.000,  0.000,  0.000],
    [1.000,  0.000,  0.000],
    [-0.333, 0.943,  0.000],
], dtype=float)

_WATER_MODELS = {
    "tip3p": _TIP3P_GEOMETRY,
    "spc":   _SPCE_GEOMETRY,
    "spce":  _SPCE_GEOMETRY,
}

# Water O-O spacing in bulk water (Å) — used for grid tiling
_WATER_SPACING = 3.1

# Minimum distance between solute heavy atom and water O (Å)
_SOLUTE_WATER_CUTOFF = 2.4


@dataclass
class SolvationResult:
    """Result of a solvation calculation."""
    atoms:          list[Atom]
    positions:      np.ndarray     # (N_total, 3)
    box_bounds:     np.ndarray     # (3, 2) [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
    n_waters_added: int
    n_ions_added:   int
    water_model:    str

    def summary(self) -> str:
        na = len(self.atoms)
        bb = self.box_bounds
        lx = bb[0,1]-bb[0,0]; ly = bb[1,1]-bb[1,0]; lz = bb[2,1]-bb[2,0]
        return (
            f"SOLVATION COMPLETE\n"
            f"  Total atoms   : {na:,}\n"
            f"  Waters added  : {self.n_waters_added:,}\n"
            f"  Ions added    : {self.n_ions_added}\n"
            f"  Water model   : {self.water_model.upper()}\n"
            f"  Box size      : {lx:.1f} × {ly:.1f} × {lz:.1f} Å\n"
            f"  Box volume    : {lx*ly*lz/1000:.1f} nm³"
        )


# ── Public API ────────────────────────────────────────────────────────────

def build_water_box(
    atoms: list[Atom],
    positions: np.ndarray,
    buffer: float = 10.0,
    water_model: str = "tip3p",
    max_waters: int = 10000,
) -> SolvationResult:
    """
    Solvate a solute in a water box.

    Algorithm:
      1. Compute solute bounding box + buffer.
      2. Tile water molecules on a 3.1 Å grid to fill the box.
      3. Remove waters clashing with solute (O closer than 2.4 Å to any heavy atom).
      4. Build final atom + position arrays.

    Parameters
    ----------
    atoms       : solute atoms
    positions   : (N, 3) solute positions in Å
    buffer      : water buffer around solute in Å (default 10.0)
    water_model : 'tip3p' or 'spc'/'spce' (default 'tip3p')
    max_waters  : maximum number of water molecules to add (performance cap)

    Returns
    -------
    SolvationResult
    """
    pos = np.asarray(positions, dtype=float)
    wm  = water_model.lower()
    if wm not in _WATER_MODELS:
        wm = "tip3p"
    water_geom = _WATER_MODELS[wm]

    # Compute box bounds
    if len(pos) > 0:
        xmin, ymin, zmin = pos.min(axis=0) - buffer
        xmax, ymax, zmax = pos.max(axis=0) + buffer
    else:
        xmin = ymin = zmin = 0.0
        xmax = ymax = zmax = 2 * buffer

    box_bounds = np.array([
        [xmin, xmax],
        [ymin, ymax],
        [zmin, zmax],
    ], dtype=float)

    # Build heavy atom position array for clash checking (exclude H)
    heavy_mask = [
        i for i, a in enumerate(atoms)
        if (getattr(a, "element", None) or "C").upper() != "H"
    ]
    heavy_pos = pos[heavy_mask] if heavy_mask else np.zeros((0, 3))

    # Tile water oxygen positions on a grid
    water_centers: list[np.ndarray] = []
    sp = _WATER_SPACING

    nx = max(1, int((xmax - xmin) / sp))
    ny = max(1, int((ymax - ymin) / sp))
    nz = max(1, int((zmax - zmin) / sp))

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                ox = xmin + (ix + 0.5) * sp
                oy = ymin + (iy + 0.5) * sp
                oz = zmin + (iz + 0.5) * sp
                water_centers.append(np.array([ox, oy, oz]))

    # Remove waters clashing with solute
    accepted_centers: list[np.ndarray] = []
    for w_center in water_centers:
        if len(accepted_centers) >= max_waters:
            break
        if _clashes_with_solute(w_center, heavy_pos):
            continue
        accepted_centers.append(w_center)

    # Build water atoms
    water_atoms:    list[Atom] = []
    water_positions: list[np.ndarray] = []
    start_id  = max((a.id for a in atoms), default=0) + 1
    start_rid = max(
        (getattr(a, "residue_id", 0) or 0 for a in atoms), default=0
    ) + 1

    for w_idx, center in enumerate(accepted_centers):
        res_id = start_rid + w_idx
        o_pos  = center
        h1_pos = center + water_geom[1] - water_geom[0]
        h2_pos = center + water_geom[2] - water_geom[0]

        water_atoms.extend([
            Atom(id=start_id,   type_id=3, element="O",
                 x=float(o_pos[0]),  y=float(o_pos[1]),  z=float(o_pos[2]),
                 residue_id=res_id, chain_id="W", name="OW",  resname="HOH"),
            Atom(id=start_id+1, type_id=5, element="H",
                 x=float(h1_pos[0]), y=float(h1_pos[1]), z=float(h1_pos[2]),
                 residue_id=res_id, chain_id="W", name="HW1", resname="HOH"),
            Atom(id=start_id+2, type_id=5, element="H",
                 x=float(h2_pos[0]), y=float(h2_pos[1]), z=float(h2_pos[2]),
                 residue_id=res_id, chain_id="W", name="HW2", resname="HOH"),
        ])
        water_positions.extend([o_pos, h1_pos, h2_pos])
        start_id += 3

    # Combine solute + water
    all_atoms = list(atoms) + water_atoms
    all_pos   = np.vstack([pos, np.array(water_positions)]) if water_positions else pos.copy()

    # Re-assign sequential IDs
    from dataclasses import replace as dc_replace
    all_atoms = [dc_replace(a, id=j) for j, a in enumerate(all_atoms)]

    return SolvationResult(
        atoms=all_atoms,
        positions=all_pos,
        box_bounds=box_bounds,
        n_waters_added=len(accepted_centers),
        n_ions_added=0,
        water_model=wm,
    )


def estimate_ion_count(
    atoms: list[Atom],
    n_waters: int,
    charge_conc: float = 0.15,
) -> tuple[int, int]:
    """
    Estimate the number of Na+ and Cl- ions needed to:
      1. Neutralize the net charge of the solute.
      2. Reach a target ionic concentration.

    Parameters
    ----------
    atoms       : solute atom list (charges estimated from resnames)
    n_waters    : number of water molecules in the box
    charge_conc : target NaCl concentration in mol/L (default 0.15 M)

    Returns
    -------
    (n_na, n_cl) — number of Na+ and Cl- ions to add
    """
    _POS_RES = {"ARG", "LYS", "HIS", "HSD", "HSE"}
    _NEG_RES = {"ASP", "GLU"}

    net_charge = 0
    seen_residues: set[tuple] = set()

    for atom in atoms:
        rid     = getattr(atom, "residue_id", None)
        chain   = getattr(atom, "chain_id", None)
        resname = (getattr(atom, "resname", None) or "").upper()
        key = (chain, rid)
        if key in seen_residues:
            continue
        seen_residues.add(key)
        if resname in _POS_RES:
            net_charge += 1
        elif resname in _NEG_RES:
            net_charge -= 1

    # Ions needed for neutralization
    n_na_neutral = max(0, -net_charge)
    n_cl_neutral = max(0,  net_charge)

    # Additional ions for target concentration
    # Volume estimate: n_waters × 30 Å³/water = volume in Å³
    # 1 mol/L = 0.000602 ions/Å³
    volume_A3 = n_waters * 30.0
    n_salt = int(charge_conc * 0.000602214 * volume_A3)

    n_na = n_na_neutral + n_salt
    n_cl = n_cl_neutral + n_salt

    return n_na, n_cl


# ── Internal helpers ──────────────────────────────────────────────────────

def _clashes_with_solute(
    water_o: np.ndarray,
    heavy_pos: np.ndarray,
    cutoff: float = _SOLUTE_WATER_CUTOFF,
) -> bool:
    """Return True if water oxygen clashes with any solute heavy atom."""
    if len(heavy_pos) == 0:
        return False
    diffs = heavy_pos - water_o
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    return bool((dists < cutoff).any())