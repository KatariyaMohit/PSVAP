"""
analysis/conformational_search.py
----------------------------------
Feature 23: Conformational Search for Small Molecules.

Generates an ensemble of low-energy 3D conformations using RDKit's
ETKDG distance-geometry method followed by MMFF94 force-field
energy minimisation and RMSD-based diversity selection.

Public API
----------
  generate_conformers(smiles, n_conformers=100, energy_window=10.0,
                      min_rmsd=0.5, random_seed=42)
      → ConformerResult

  ConformerResult (dataclass)
      smiles, conformers (list of (N,3) arrays), energies, n_generated,
      n_accepted

All positions in Angstroms. Energies in kcal/mol (MMFF94).
Requires RDKit — import is guarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformerResult:
    """Result of a conformational search."""
    smiles:        str
    conformers:    list[np.ndarray]   # each (N_atoms, 3) in Å
    energies:      list[float]        # MMFF94 energies in kcal/mol
    n_generated:   int                # total conformers tried
    n_accepted:    int                # conformers after filtering
    n_atoms:       int
    elements:      list[str]          # element symbol per atom
    atom_names:    list[str]          # atom name per atom


def generate_conformers(
    smiles: str,
    n_conformers: int = 100,
    energy_window: float = 10.0,
    min_rmsd: float = 0.5,
    random_seed: int = 42,
    force_field: str = "MMFF94",
) -> ConformerResult:
    """
    Generate an ensemble of diverse, low-energy conformations.

    Algorithm:
      1. Parse SMILES and add explicit hydrogens.
      2. Embed n_conformers using ETKDG (ETKDGv3 if available).
      3. Minimise each with MMFF94 or UFF force field.
      4. Filter: keep conformers within energy_window kcal/mol of the minimum.
      5. Diversity filter: remove conformers with RMSD < min_rmsd to any
         already-accepted conformer (greedy diversity selection).

    Parameters
    ----------
    smiles        : SMILES string of the molecule
    n_conformers  : number of initial conformers to generate
    energy_window : only keep conformers within this many kcal/mol of min
    min_rmsd      : minimum pairwise RMSD between accepted conformers (Å)
    random_seed   : random seed for reproducibility
    force_field   : 'MMFF94' or 'UFF' (default MMFF94)

    Returns
    -------
    ConformerResult

    Raises
    ------
    ImportError if RDKit not installed
    ValueError  if SMILES cannot be parsed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdForceFieldHelpers
    except ImportError:
        raise ImportError(
            "RDKit is required for conformational search.\n"
            "Install with: conda install -c conda-forge rdkit"
        )

    # Parse and prepare molecule
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: '{smiles}'")

    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    # Embed conformers
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.numThreads = 0   # use all available
    except AttributeError:
        params = AllChem.ETKDG()
        params.randomSeed = random_seed

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    n_generated = len(conf_ids)

    if n_generated == 0:
        raise ValueError(
            f"Could not embed any conformers for: '{smiles}'\n"
            "Try a simpler molecule or increase n_conformers."
        )

    # Minimise and collect energies
    energies: list[float] = []
    ff_upper = force_field.upper()

    for cid in conf_ids:
        try:
            if ff_upper == "MMFF94":
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid
                )
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)

            if ff is not None:
                ff.Minimize(maxIts=500)
                energy = ff.CalcEnergy()
            else:
                energy = 0.0
        except Exception:
            energy = 0.0
        energies.append(energy)

    # Filter by energy window
    valid_energies = [e for e in energies if e != 0.0]
    e_min = min(valid_energies) if valid_energies else 0.0
    e_cutoff = e_min + energy_window

    accepted_cids: list[int] = []
    accepted_energies: list[float] = []

    # Sort by energy ascending
    sorted_pairs = sorted(zip(energies, conf_ids), key=lambda x: x[0])
    for energy, cid in sorted_pairs:
        if energy > e_cutoff and energy != 0.0:
            continue

        # Extract positions for this conformer
        conf = mol.GetConformer(cid)
        pos = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(n_atoms)
        ], dtype=np.float64)

        # RMSD diversity filter
        too_close = False
        for prev_cid in accepted_cids:
            prev_conf = mol.GetConformer(prev_cid)
            prev_pos = np.array([
                [prev_conf.GetAtomPosition(i).x,
                 prev_conf.GetAtomPosition(i).y,
                 prev_conf.GetAtomPosition(i).z]
                for i in range(n_atoms)
            ], dtype=np.float64)
            rmsd_val = float(np.sqrt(
                ((pos - prev_pos) ** 2).sum() / n_atoms
            ))
            if rmsd_val < min_rmsd:
                too_close = True
                break

        if not too_close:
            accepted_cids.append(cid)
            accepted_energies.append(energy)

    # Extract final conformer arrays
    conformers: list[np.ndarray] = []
    for cid in accepted_cids:
        conf = mol.GetConformer(cid)
        pos = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(n_atoms)
        ], dtype=np.float64)
        conformers.append(pos)

    # Extract atom info
    elements  = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
    atom_names = [
        f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}"
        for i in range(n_atoms)
    ]

    return ConformerResult(
        smiles=smiles,
        conformers=conformers,
        energies=accepted_energies,
        n_generated=n_generated,
        n_accepted=len(conformers),
        n_atoms=n_atoms,
        elements=elements,
        atom_names=atom_names,
    )


def format_conformer_result(result: ConformerResult) -> str:
    """Format conformer result as text for GUI display."""
    if not result.conformers:
        return "NO CONFORMERS GENERATED"

    lines = [
        f"CONFORMATIONAL SEARCH RESULTS\n",
        f"  SMILES         : {result.smiles[:60]}{'...' if len(result.smiles)>60 else ''}",
        f"  Atoms          : {result.n_atoms}",
        f"  Generated      : {result.n_generated}",
        f"  Accepted       : {result.n_accepted}",
        "",
        f"{'CONF':>5}  {'ENERGY (kcal/mol)':>18}  {'ΔE':>8}",
        "-" * 36,
    ]

    e_min = min(result.energies) if result.energies else 0.0
    for i, (conf, energy) in enumerate(zip(result.conformers, result.energies)):
        delta = energy - e_min
        lines.append(f"{i+1:>5}  {energy:>18.4f}  {delta:>8.4f}")

    return "\n".join(lines)