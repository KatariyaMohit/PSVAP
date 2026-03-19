"""
analysis/qsar.py
----------------
Feature 6:  Matched Molecular Pair (MMP) Analysis.
Feature 15: QSAR Descriptor Computation and Model Building.

Public API
----------
MMP Analysis:
  find_matched_pairs(smiles_list, max_heavy_ratio=0.33)
      → list[MatchedPair]

QSAR:
  compute_descriptors(smiles_list)
      → dict {smiles: {descriptor_name: value}}

  build_qsar_model(smiles_list, activities, model_type='random_forest')
      → QSARResult

  predict_activity(model_result, smiles_list)
      → list[float]

All RDKit imports guarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── MMP ───────────────────────────────────────────────────────────────────

@dataclass
class MatchedPair:
    """A pair of molecules differing by a single structural transformation."""
    smiles_a:        str
    smiles_b:        str
    index_a:         int
    index_b:         int
    core:            str   # SMILES of the shared core
    transform_a:     str   # SMILES of the fragment in molecule A
    transform_b:     str   # SMILES of the fragment in molecule B
    heavy_ratio:     float # fraction of heavy atoms changed


@dataclass
class QSARResult:
    """Result of QSAR model training."""
    model_type:    str
    n_train:       int
    r2_train:      float
    r2_cv:         float          # cross-validation R²
    rmse_cv:       float
    feature_names: list[str]
    model:         Any = None     # sklearn model object
    scaler:        Any = None     # sklearn StandardScaler


def find_matched_pairs(
    smiles_list: list[str],
    max_heavy_ratio: float = 0.33,
) -> list[MatchedPair]:
    """
    Find Matched Molecular Pairs — molecule pairs differing by a single
    fragment transformation (one bond broken, one group replaced).

    Uses RDKit's RECAP fragmentation to identify core + R-group splits.

    Parameters
    ----------
    smiles_list     : list of SMILES strings
    max_heavy_ratio : maximum fraction of heavy atoms allowed in the
                      changing fragment (default 0.33 = max 1/3 of molecule)

    Returns
    -------
    list[MatchedPair]

    Raises
    ------
    ImportError if RDKit not installed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMMPA
    except ImportError:
        raise ImportError(
            "RDKit is required for MMP analysis.\n"
            "Install with: conda install -c conda-forge rdkit"
        )

    # Parse valid molecules
    mols: list[Any] = []
    valid_idx: list[int] = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    if len(mols) < 2:
        return []

    # Fragment each molecule into core + R-group using MMPA
    fragments: list[list[tuple[str, str]]] = []
    for mol in mols:
        frags: list[tuple[str, str]] = []
        try:
            cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
            for core_smi, rgroup_smi in cuts:
                if core_smi and rgroup_smi:
                    frags.append((core_smi, rgroup_smi))
        except Exception:
            pass
        fragments.append(frags)

    # Find pairs with same core, different R-group
    pairs: list[MatchedPair] = []

    for i in range(len(mols)):
        cores_i = {f[0]: f[1] for f in fragments[i]}
        n_heavy_i = mols[i].GetNumHeavyAtoms()

        for j in range(i + 1, len(mols)):
            cores_j = {f[0]: f[1] for f in fragments[j]}
            n_heavy_j = mols[j].GetNumHeavyAtoms()

            # Find shared cores
            shared_cores = set(cores_i.keys()) & set(cores_j.keys())
            for core in shared_cores:
                rg_a = cores_i[core]
                rg_b = cores_j[core]
                if rg_a == rg_b:
                    continue  # identical R-groups — not a pair

                # Estimate heavy atom ratio of the changing fragment
                rg_mol_a = Chem.MolFromSmiles(rg_a)
                if rg_mol_a is None:
                    continue
                n_rg = rg_mol_a.GetNumHeavyAtoms()
                ratio = n_rg / max(n_heavy_i, n_heavy_j, 1)

                if ratio <= max_heavy_ratio:
                    pairs.append(MatchedPair(
                        smiles_a=smiles_list[valid_idx[i]],
                        smiles_b=smiles_list[valid_idx[j]],
                        index_a=valid_idx[i],
                        index_b=valid_idx[j],
                        core=core,
                        transform_a=rg_a,
                        transform_b=rg_b,
                        heavy_ratio=ratio,
                    ))
                    break  # one pair per molecule pair

    return pairs


# ── QSAR ──────────────────────────────────────────────────────────────────

# Core RDKit 2D descriptor names (subset — fast and reliable)
_CORE_DESCRIPTORS = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumHeteroatoms",
    "TPSA", "FractionCSP3", "RingCount",
    "NumAliphaticRings", "NumSaturatedRings",
    "MaxPartialCharge", "MinPartialCharge",
    "NumRadicalElectrons", "NumValenceElectrons",
]


def compute_descriptors(
    smiles_list: list[str],
    descriptor_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute 2D molecular descriptors for a list of SMILES.

    Parameters
    ----------
    smiles_list      : list of SMILES strings
    descriptor_names : specific descriptors to compute; None = use defaults

    Returns
    -------
    dict {smiles: {descriptor_name: value}}
    Molecules that cannot be parsed return an empty dict entry.

    Raises
    ------
    ImportError if RDKit not installed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        raise ImportError(
            "RDKit is required for descriptor computation.\n"
            "Install with: conda install -c conda-forge rdkit"
        )

    desc_names = descriptor_names or _CORE_DESCRIPTORS

    # Build a lookup of available RDKit descriptors
    available = dict(Descriptors.descList)

    results: dict[str, dict[str, float]] = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            results[smi] = {}
            continue

        desc_vals: dict[str, float] = {}
        for name in desc_names:
            try:
                fn = available.get(name)
                if fn is not None:
                    val = fn(mol)
                    desc_vals[name] = float(val) if val is not None else float("nan")
                else:
                    desc_vals[name] = float("nan")
            except Exception:
                desc_vals[name] = float("nan")

        results[smi] = desc_vals

    return results


def build_qsar_model(
    smiles_list: list[str],
    activities: list[float],
    model_type: str = "random_forest",
    descriptor_names: list[str] | None = None,
    cv_folds: int = 5,
) -> QSARResult:
    """
    Build a QSAR model correlating molecular descriptors with activity.

    Parameters
    ----------
    smiles_list      : list of SMILES
    activities       : list of activity values (same length as smiles_list)
    model_type       : 'random_forest', 'linear', or 'svm'
    descriptor_names : descriptors to use (None = defaults)
    cv_folds         : number of cross-validation folds

    Returns
    -------
    QSARResult with trained model, R², RMSE, and feature importance

    Raises
    ------
    ImportError if RDKit or scikit-learn not installed
    ValueError  if fewer than 5 molecules provided
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError:
        raise ImportError(
            "scikit-learn is required for QSAR modelling.\n"
            "Install with: pip install scikit-learn"
        )

    if len(smiles_list) < 5:
        raise ValueError(
            f"QSAR requires at least 5 molecules. Got {len(smiles_list)}."
        )
    if len(smiles_list) != len(activities):
        raise ValueError(
            "smiles_list and activities must have the same length."
        )

    # Compute descriptors
    desc_dict = compute_descriptors(smiles_list, descriptor_names)
    desc_names = descriptor_names or _CORE_DESCRIPTORS

    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    valid_smiles: list[str] = []

    for smi, act in zip(smiles_list, activities):
        row = desc_dict.get(smi, {})
        if not row:
            continue
        vals = [row.get(d, float("nan")) for d in desc_names]
        if any(np.isnan(v) for v in vals):
            continue
        X_rows.append(vals)
        y_vals.append(float(act))
        valid_smiles.append(smi)

    if len(X_rows) < 5:
        raise ValueError(
            f"Only {len(X_rows)} molecules had valid descriptors. Need at least 5."
        )

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mt = model_type.lower().replace(" ", "_").replace("-", "_")
    if mt in {"random_forest", "rf"}:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif mt in {"linear", "ridge"}:
        model = Ridge(alpha=1.0)
    elif mt in {"svm", "svr"}:
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Cross-validation
    n_cv = min(cv_folds, len(X_rows))
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=n_cv,
        scoring="r2", error_score="raise"
    )
    cv_rmse_scores = np.sqrt(-cross_val_score(
        model, X_scaled, y, cv=n_cv,
        scoring="neg_mean_squared_error"
    ))

    # Fit on all data
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2_train = float(r2_score(y, y_pred))

    return QSARResult(
        model_type=model_type,
        n_train=len(X_rows),
        r2_train=r2_train,
        r2_cv=float(cv_scores.mean()),
        rmse_cv=float(cv_rmse_scores.mean()),
        feature_names=desc_names,
        model=model,
        scaler=scaler,
    )


def predict_activity(
    model_result: QSARResult,
    smiles_list: list[str],
    descriptor_names: list[str] | None = None,
) -> list[float | None]:
    """
    Predict activity for new molecules using a trained QSARResult.

    Returns
    -------
    list of predicted values (None for molecules that could not be parsed)
    """
    if model_result.model is None or model_result.scaler is None:
        raise ValueError("QSARResult has no trained model.")

    desc_names = descriptor_names or model_result.feature_names
    desc_dict  = compute_descriptors(smiles_list, desc_names)
    predictions: list[float | None] = []

    for smi in smiles_list:
        row = desc_dict.get(smi, {})
        if not row:
            predictions.append(None)
            continue
        vals = [row.get(d, float("nan")) for d in desc_names]
        if any(np.isnan(v) for v in vals):
            predictions.append(None)
            continue
        X = np.array([vals], dtype=np.float64)
        X_scaled = model_result.scaler.transform(X)
        pred = model_result.model.predict(X_scaled)
        predictions.append(float(pred[0]))

    return predictions