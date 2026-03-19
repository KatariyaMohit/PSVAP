"""
modeling/alanine_scan.py
------------------------
Feature 11: Systematic Alanine Scanning.

Mutates each residue in a selection to alanine one-by-one and estimates
the change in interaction count as a proxy for binding energy contribution.
Hot-spot residues are those that lose the most interactions when mutated
to alanine.

Public API
----------
  alanine_scan(atoms, positions, residue_ids, group_a, group_b,
               chain_id=None)
      → list[AlanineResult]
      One result per residue, sorted by estimated energy penalty.

  AlanineResult  (dataclass)
      residue_id, original_resname, delta_hbonds, delta_clashes,
      delta_interactions, is_hotspot

All positions in Angstroms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PSVAP.modeling.mutation_engine import (
    MutationError,
    get_residue_atoms,
    list_residues,
    mutate_residue,
)


@dataclass
class AlanineResult:
    """Result of mutating one residue to alanine."""
    residue_id:        int
    original_resname:  str
    chain_id:          str | None
    # Change in interaction counts (negative = fewer interactions after mutation)
    delta_hbonds:      int
    delta_clashes:     int
    delta_hydrophobic: int
    # Weighted score: higher = more important residue
    delta_score:       float
    is_hotspot:        bool   # True if delta_score > hotspot_threshold


def alanine_scan(
    atoms: list,
    positions: np.ndarray,
    residue_ids: list[int],
    group_a: list[int],
    group_b: list[int],
    chain_id: str | None = None,
    hotspot_threshold: float = 1.5,
) -> list[AlanineResult]:
    """
    Systematic alanine scanning over a list of residues.

    For each residue:
      1. Count interactions between group_a and group_b in original structure.
      2. Mutate residue to ALA.
      3. Count interactions in mutated structure.
      4. Record delta (original - mutated) as energy proxy.

    Parameters
    ----------
    atoms             : full atom list
    positions         : (N, 3) positions in Å
    residue_ids       : list of residue IDs to scan
    group_a, group_b  : atom index groups for interaction detection
    chain_id          : optional chain filter for mutation
    hotspot_threshold : delta_score above this = hot-spot residue

    Returns
    -------
    list[AlanineResult] sorted by delta_score descending (hottest first)
    """
    from PSVAP.analysis.interactions import detect_hbonds, detect_clashes, detect_hydrophobic

    pos = np.asarray(positions, dtype=float)
    results: list[AlanineResult] = []

    # Baseline interaction counts
    base_hb     = len(detect_hbonds(atoms, pos, group_a, group_b))
    base_clash  = len(detect_clashes(atoms, pos, group_a, group_b))
    base_hydro  = len(detect_hydrophobic(atoms, pos, group_a, group_b))

    for res_id in residue_ids:
        res_idx = get_residue_atoms(atoms, res_id, chain_id)
        if not res_idx:
            continue

        original_resname = (
            getattr(atoms[res_idx[0]], "resname", None) or "UNK"
        ).upper()

        # Skip if already ALA or GLY
        if original_resname in {"ALA", "GLY"}:
            results.append(AlanineResult(
                residue_id=res_id,
                original_resname=original_resname,
                chain_id=chain_id,
                delta_hbonds=0,
                delta_clashes=0,
                delta_hydrophobic=0,
                delta_score=0.0,
                is_hotspot=False,
            ))
            continue

        try:
            mut_atoms, mut_pos = mutate_residue(
                atoms, pos, res_id, "ALA", chain_id=chain_id
            )
        except MutationError:
            continue

        # Re-map group indices after mutation (atom count may change)
        # Use residue_id membership to rebuild groups
        def _rebuild_group(orig_group: list[int], orig_atoms: list, new_atoms: list) -> list[int]:
            """Re-map group indices by residue_id membership."""
            orig_res_ids = {getattr(orig_atoms[i], "residue_id", None) for i in orig_group}
            return [
                j for j, a in enumerate(new_atoms)
                if getattr(a, "residue_id", None) in orig_res_ids
            ]

        mut_group_a = _rebuild_group(group_a, atoms, mut_atoms)
        mut_group_b = _rebuild_group(group_b, atoms, mut_atoms)

        mut_hb    = len(detect_hbonds(mut_atoms, mut_pos, mut_group_a, mut_group_b))
        mut_clash = len(detect_clashes(mut_atoms, mut_pos, mut_group_a, mut_group_b))
        mut_hydro = len(detect_hydrophobic(mut_atoms, mut_pos, mut_group_a, mut_group_b))

        delta_hb    = base_hb    - mut_hb
        delta_clash = base_clash - mut_clash
        delta_hydro = base_hydro - mut_hydro

        # Score: H-bonds weight 2.0, hydrophobic 1.0, clash -0.5 (clashes are bad)
        delta_score = float(
            2.0 * delta_hb + 1.0 * delta_hydro - 0.5 * delta_clash
        )

        results.append(AlanineResult(
            residue_id=res_id,
            original_resname=original_resname,
            chain_id=chain_id,
            delta_hbonds=delta_hb,
            delta_clashes=delta_clash,
            delta_hydrophobic=delta_hydro,
            delta_score=delta_score,
            is_hotspot=delta_score >= hotspot_threshold,
        ))

    return sorted(results, key=lambda r: -r.delta_score)


def format_scan_results(results: list[AlanineResult]) -> str:
    """
    Format alanine scan results as a text table for display in the GUI.
    """
    if not results:
        return "NO RESULTS — check residue IDs and group definitions"

    lines = [
        f"ALANINE SCAN  ({len(results)} residues)\n",
        f"{'RESID':>6}  {'RES':>4}  {'CHAIN':>5}  "
        f"{'ΔHB':>5}  {'ΔHYDRO':>6}  {'ΔCLASH':>6}  "
        f"{'SCORE':>7}  {'HOTSPOT':>7}",
        "-" * 58,
    ]

    for r in results:
        hotspot_marker = "  ★" if r.is_hotspot else ""
        lines.append(
            f"{r.residue_id:>6}  {r.original_resname:>4}  "
            f"{r.chain_id or '-':>5}  "
            f"{r.delta_hbonds:>5}  {r.delta_hydrophobic:>6}  "
            f"{r.delta_clashes:>6}  {r.delta_score:>7.2f}"
            f"{hotspot_marker}"
        )

    n_hotspots = sum(1 for r in results if r.is_hotspot)
    lines.append(f"\nHOT-SPOT RESIDUES: {n_hotspots}")
    return "\n".join(lines)