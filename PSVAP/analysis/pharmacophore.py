"""
analysis/pharmacophore.py
--------------------------
Feature 8: Pharmacophore Feature Extraction.

Identifies the 3D pharmacophoric features of a ligand or protein-ligand
complex: H-bond donors, H-bond acceptors, hydrophobic regions, positive
charge, negative charge, and aromatic rings.

Public API
----------
  extract_pharmacophore(atoms, positions, indices=None)
      → list[PharmacophoreFeature]

  pharmacophore_to_dict(features)
      → dict  (JSON-serialisable representation)

  PharmacophoreFeature (dataclass)
      feature_type, center, radius, atom_indices

Feature types: 'donor', 'acceptor', 'hydrophobic', 'positive',
               'negative', 'aromatic'
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Feature type definitions ──────────────────────────────────────────────
_DONOR_ELEMENTS    = {"N","S"}
_ACCEPTOR_ELEMENTS = {"N", "O", "S", "F"}
_HYDRO_ELEMENTS    = {"C", "S"}
_POS_RESNAMES      = {"ARG", "LYS", "HIS", "HSD", "HSE", "HSP"}
_NEG_RESNAMES      = {"ASP", "GLU"}
_AROMATIC_RESNAMES = {"PHE", "TYR", "TRP", "HIS", "HSD", "HSE"}

# Feature display colours (RGB 0-1) — used by interaction_renderer
FEATURE_COLORS = {
    "donor":      (0.20, 0.85, 0.20),   # green
    "acceptor":   (0.90, 0.20, 0.20),   # red
    "hydrophobic":(1.00, 0.85, 0.10),   # yellow
    "positive":   (0.20, 0.40, 0.90),   # blue
    "negative":   (0.90, 0.40, 0.10),   # orange
    "aromatic":   (0.70, 0.20, 0.90),   # purple
}


@dataclass
class PharmacophoreFeature:
    """A single pharmacophoric feature."""
    feature_type:  str              # donor/acceptor/hydrophobic/positive/negative/aromatic
    center:        np.ndarray       # (3,) centroid in Å
    radius:        float            # sphere radius for visualization (Å)
    atom_indices:  list[int]        # atoms contributing to this feature
    resname:       str = ""         # residue name if applicable
    description:   str = ""        # human-readable label


def extract_pharmacophore(
    atoms: list,
    positions: np.ndarray,
    indices: list[int] | None = None,
) -> list[PharmacophoreFeature]:
    """
    Extract pharmacophoric features from a set of atoms.

    Works with both ligands (element-based detection) and protein
    binding-site residues (resname-based detection).

    Parameters
    ----------
    atoms     : full atom list
    positions : (N, 3) positions in Å
    indices   : subset of atom indices to analyse (None = all atoms)

    Returns
    -------
    list[PharmacophoreFeature]  — one feature per detected group
    """
    pos = np.asarray(positions, dtype=float)
    idx_list = indices if indices is not None else list(range(len(atoms)))

    features: list[PharmacophoreFeature] = []
# ── Donor / Acceptor ─────────────────────────────────────────────────
    # N and S can act as both donor and acceptor.
    # O is treated as acceptor-only (it acts as donor only when protonated,
    # which requires H position data we don't always have).
    donor_indices:    list[int] = []
    acceptor_indices: list[int] = []

    for i in idx_list:
        elem = (getattr(atoms[i], "element", None) or "").upper()
        if elem in _DONOR_ELEMENTS:
            donor_indices.append(i)
        if elem in _ACCEPTOR_ELEMENTS:
            acceptor_indices.append(i)

    for i in donor_indices:
        features.append(PharmacophoreFeature(
            feature_type="donor",
            center=pos[i].copy(),
            radius=1.0,
            atom_indices=[i],
            resname=(getattr(atoms[i], "resname", None) or "").upper(),
            description=f"H-bond donor at atom {i}",
        ))

    for i in acceptor_indices:
        # N and S that are already donors get a separate acceptor feature too
        # (they can do both). O only gets the acceptor feature.
        features.append(PharmacophoreFeature(
            feature_type="acceptor",
            center=pos[i].copy(),
            radius=1.0,
            atom_indices=[i],
            resname=(getattr(atoms[i], "resname", None) or "").upper(),
            description=f"H-bond acceptor at atom {i}",
        ))
        
    # ── Hydrophobic ──────────────────────────────────────────────────────
    # Group nearby hydrophobic C/S atoms into clusters
    hydro_indices = [
        i for i in idx_list
        if (getattr(atoms[i], "element", None) or "").upper() in _HYDRO_ELEMENTS
    ]
    hydro_clusters = _cluster_nearby(hydro_indices, pos, cutoff=4.0)
    for cluster in hydro_clusters:
        center = pos[cluster].mean(axis=0)
        features.append(PharmacophoreFeature(
            feature_type="hydrophobic",
            center=center,
            radius=max(1.5, len(cluster) * 0.3),
            atom_indices=cluster,
            description=f"Hydrophobic region ({len(cluster)} atoms)",
        ))

    # ── Charged groups (from residue names) ──────────────────────────────
    seen_residues: set[tuple] = set()
    for i in idx_list:
        rn    = (getattr(atoms[i], "resname",    None) or "").upper()
        rid   = getattr(atoms[i], "residue_id",  None)
        chain = getattr(atoms[i], "chain_id",    None) or ""
        key   = (chain, rid)
        if key in seen_residues or not rn:
            continue
        seen_residues.add(key)

        if rn in _POS_RESNAMES:
            res_idx = [
                j for j in idx_list
                if getattr(atoms[j], "residue_id", None) == rid
                and (getattr(atoms[j], "chain_id", None) or "") == chain
            ]
            center = pos[res_idx].mean(axis=0) if res_idx else pos[i]
            features.append(PharmacophoreFeature(
                feature_type="positive",
                center=center,
                radius=1.5,
                atom_indices=res_idx,
                resname=rn,
                description=f"Positive charge: {rn} res {rid}",
            ))

        elif rn in _NEG_RESNAMES:
            res_idx = [
                j for j in idx_list
                if getattr(atoms[j], "residue_id", None) == rid
                and (getattr(atoms[j], "chain_id", None) or "") == chain
            ]
            center = pos[res_idx].mean(axis=0) if res_idx else pos[i]
            features.append(PharmacophoreFeature(
                feature_type="negative",
                center=center,
                radius=1.5,
                atom_indices=res_idx,
                resname=rn,
                description=f"Negative charge: {rn} res {rid}",
            ))

    # ── Aromatic rings ────────────────────────────────────────────────────
    seen_aromatic: set[tuple] = set()
    for i in idx_list:
        rn    = (getattr(atoms[i], "resname",   None) or "").upper()
        rid   = getattr(atoms[i], "residue_id", None)
        chain = getattr(atoms[i], "chain_id",   None) or ""
        key   = (chain, rid)
        if key in seen_aromatic or rn not in _AROMATIC_RESNAMES:
            continue
        seen_aromatic.add(key)

        ring_idx = [
            j for j in idx_list
            if getattr(atoms[j], "residue_id", None) == rid
            and (getattr(atoms[j], "chain_id", None) or "") == chain
            and (getattr(atoms[j], "element",  None) or "").upper() in {"C", "N"}
        ]
        if ring_idx:
            center = pos[ring_idx].mean(axis=0)
            features.append(PharmacophoreFeature(
                feature_type="aromatic",
                center=center,
                radius=1.8,
                atom_indices=ring_idx,
                resname=rn,
                description=f"Aromatic ring: {rn} res {rid}",
            ))

    return features


def pharmacophore_to_dict(features: list[PharmacophoreFeature]) -> dict:
    """
    Convert a pharmacophore feature list to a JSON-serialisable dict.

    Returns
    -------
    {
      'n_features': int,
      'features': [{'type', 'center', 'radius', 'atom_indices', 'description'}]
    }
    """
    return {
        "n_features": len(features),
        "features": [
            {
                "type":         f.feature_type,
                "center":       f.center.tolist(),
                "radius":       f.radius,
                "atom_indices": f.atom_indices,
                "resname":      f.resname,
                "description":  f.description,
            }
            for f in features
        ],
    }


def summarise_pharmacophore(features: list[PharmacophoreFeature]) -> str:
    """Return a formatted text summary of pharmacophore features."""
    from collections import Counter
    counts = Counter(f.feature_type for f in features)
    lines = [
        f"PHARMACOPHORE FEATURES  ({len(features)} total)\n",
        f"  H-BOND DONORS    : {counts.get('donor', 0)}",
        f"  H-BOND ACCEPTORS : {counts.get('acceptor', 0)}",
        f"  HYDROPHOBIC      : {counts.get('hydrophobic', 0)}",
        f"  POSITIVE CHARGE  : {counts.get('positive', 0)}",
        f"  NEGATIVE CHARGE  : {counts.get('negative', 0)}",
        f"  AROMATIC RINGS   : {counts.get('aromatic', 0)}",
        "",
    ]
    for i, f in enumerate(features):
        c = f.center
        lines.append(
            f"  {i+1:>3}. {f.feature_type.upper():<12} "
            f"({c[0]:6.2f}, {c[1]:6.2f}, {c[2]:6.2f})  r={f.radius:.1f} Å"
            + (f"  {f.resname}" if f.resname else "")
        )
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────

def _cluster_nearby(
    indices: list[int],
    positions: np.ndarray,
    cutoff: float = 4.0,
) -> list[list[int]]:
    """
    Simple greedy clustering of atom indices by distance.
    Atoms within cutoff of any cluster member are merged into that cluster.
    """
    if not indices:
        return []

    clusters: list[list[int]] = []
    assigned = set()

    for i in indices:
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in indices:
            if j in assigned:
                continue
            for k in cluster:
                if np.linalg.norm(positions[j] - positions[k]) < cutoff:
                    cluster.append(j)
                    assigned.add(j)
                    break
        clusters.append(cluster)

    return clusters