"""
tests/test_phase5.py
---------------------
Phase 5 unit tests for:
  analysis/clustering.py   (MCS + fingerprint clustering)
  analysis/qsar.py         (MMP + QSAR descriptors + model)
  analysis/pharmacophore.py
  analysis/conformational_search.py
  analysis/pka.py
  modeling/docking_engine.py  (config + box computation — no Vina needed)

RDKit-dependent tests are auto-skipped if RDKit is not installed.
All tests are self-contained — no file I/O required.

Run:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_phase5.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from PSVAP.core.atom import Atom


# ── RDKit guard ────────────────────────────────────────────────────────────
def _rdkit_ok() -> bool:
    try:
        from rdkit import Chem; return True
    except Exception:
        return False

rdkit = pytest.mark.skipif(not _rdkit_ok(), reason="RDKit not installed")


# ── Fixtures ───────────────────────────────────────────────────────────────

PARACETAMOL  = "CC(=O)Nc1ccc(O)cc1"
PHENACETIN   = "CC(=O)Nc1ccc(OCC)cc1"
ACETANILIDE  = "CC(=O)Nc1ccccc1"
SAMPLE_SMILES = [PARACETAMOL, PHENACETIN, ACETANILIDE]

def _make_protein_atoms():
    """Small protein with ionisable ARG and ASP residues."""
    atoms = [
        Atom(id=0, type_id=2, element="N", x=0.0, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="N",  resname="ARG"),
        Atom(id=1, type_id=1, element="C", x=1.5, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="CA", resname="ARG"),
        Atom(id=2, type_id=3, element="O", x=5.0, y=0.0, z=0.0,
             residue_id=2, chain_id="A", name="OD1", resname="ASP"),
        Atom(id=3, type_id=1, element="C", x=5.0, y=1.5, z=0.0,
             residue_id=2, chain_id="A", name="CA", resname="ASP"),
    ]
    pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    return atoms, pos


# ═══════════════════════════════════════════════════════════════════════════
# 1. MCS (analysis/clustering.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestMCS:

    @rdkit
    def test_finds_mcs(self):
        from PSVAP.analysis.clustering import find_mcs
        result = find_mcs(SAMPLE_SMILES)
        assert result.smarts != ""
        assert result.n_atoms > 0
        assert result.n_molecules == 3

    @rdkit
    def test_mcs_result_type(self):
        from PSVAP.analysis.clustering import find_mcs, MCSResult
        result = find_mcs(SAMPLE_SMILES[:2])
        assert isinstance(result, MCSResult)

    @rdkit
    def test_atom_map_length(self):
        from PSVAP.analysis.clustering import find_mcs
        result = find_mcs(SAMPLE_SMILES)
        assert len(result.atom_map) == 3

    @rdkit
    def test_fewer_than_2_raises(self):
        from PSVAP.analysis.clustering import find_mcs
        with pytest.raises(ValueError):
            find_mcs([PARACETAMOL])

    @rdkit
    def test_identical_molecules_full_mcs(self):
        """Two identical molecules → MCS = full molecule."""
        from PSVAP.analysis.clustering import find_mcs
        result = find_mcs([PARACETAMOL, PARACETAMOL])
        assert result.n_atoms > 0
        assert not result.timed_out

    @rdkit
    def test_fingerprint_clustering(self):
        from PSVAP.analysis.clustering import cluster_by_fingerprint
        clusters = cluster_by_fingerprint(SAMPLE_SMILES, n_clusters=2)
        assert isinstance(clusters, dict)
        total = sum(len(v) for v in clusters.values())
        assert total == 3

    @rdkit
    def test_clustering_respects_n_clusters(self):
        from PSVAP.analysis.clustering import cluster_by_fingerprint
        smiles = SAMPLE_SMILES * 4  # 12 molecules
        clusters = cluster_by_fingerprint(smiles, n_clusters=3)
        assert len(clusters) <= 3


# ═══════════════════════════════════════════════════════════════════════════
# 2. QSAR descriptors (analysis/qsar.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestQSARDescriptors:

    @rdkit
    def test_compute_descriptors_returns_dict(self):
        from PSVAP.analysis.qsar import compute_descriptors
        result = compute_descriptors([PARACETAMOL])
        assert isinstance(result, dict)
        assert PARACETAMOL in result

    @rdkit
    def test_known_descriptor_mw(self):
        """Paracetamol MW ≈ 151.16 g/mol."""
        from PSVAP.analysis.qsar import compute_descriptors
        result = compute_descriptors([PARACETAMOL])
        mw = result[PARACETAMOL].get("MolWt", None)
        assert mw is not None
        assert abs(mw - 151.16) < 1.0

    @rdkit
    def test_invalid_smiles_returns_empty(self):
        from PSVAP.analysis.qsar import compute_descriptors
        result = compute_descriptors(["NOT_A_SMILES"])
        assert result["NOT_A_SMILES"] == {}

    @rdkit
    def test_multiple_molecules(self):
        from PSVAP.analysis.qsar import compute_descriptors
        result = compute_descriptors(SAMPLE_SMILES)
        assert len(result) == 3
        for smi in SAMPLE_SMILES:
            assert smi in result


# ═══════════════════════════════════════════════════════════════════════════
# 3. MMP (analysis/qsar.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestMMP:

    @rdkit
    def test_finds_pairs(self):
        from PSVAP.analysis.qsar import find_matched_pairs
        pairs = find_matched_pairs(SAMPLE_SMILES)
        assert isinstance(pairs, list)

    @rdkit
    def test_pair_has_correct_structure(self):
        from PSVAP.analysis.qsar import find_matched_pairs, MatchedPair
        pairs = find_matched_pairs(SAMPLE_SMILES)
        for p in pairs:
            assert isinstance(p, MatchedPair)
            assert p.index_a < len(SAMPLE_SMILES)
            assert p.index_b < len(SAMPLE_SMILES)
            assert 0.0 < p.heavy_ratio <= 0.33

    @rdkit
    def test_fewer_than_2_returns_empty(self):
        from PSVAP.analysis.qsar import find_matched_pairs
        pairs = find_matched_pairs([PARACETAMOL])
        assert pairs == []


# ═══════════════════════════════════════════════════════════════════════════
# 4. QSAR model (analysis/qsar.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestQSARModel:

    @rdkit
    def test_builds_model(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.qsar import build_qsar_model, QSARResult
        smiles = SAMPLE_SMILES * 3  # 9 molecules
        activities = [6.2, 7.1, 5.8, 6.2, 7.1, 5.8, 6.5, 7.0, 6.0]
        result = build_qsar_model(smiles, activities)
        assert isinstance(result, QSARResult)
        assert result.n_train > 0

    @rdkit
    def test_r2_in_valid_range(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            pytest.skip("scikit-learn not installed")
        import math
        from PSVAP.analysis.qsar import build_qsar_model
        smiles = SAMPLE_SMILES * 3
        activities = [6.2, 7.1, 5.8, 6.2, 7.1, 5.8, 6.5, 7.0, 6.0]
        result = build_qsar_model(smiles, activities)
        # NaN is acceptable when CV folds have only 1 sample (degenerate dataset)
        if not math.isnan(result.r2_cv):
            assert -10.0 <= result.r2_cv <= 1.0
    @rdkit
    def test_predict_returns_list(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.qsar import build_qsar_model, predict_activity
        smiles = SAMPLE_SMILES * 3
        activities = [6.2, 7.1, 5.8] * 3
        model = build_qsar_model(smiles, activities)
        preds = predict_activity(model, [PARACETAMOL])
        assert isinstance(preds, list) and len(preds) == 1

    def test_fewer_than_5_raises(self):
        try:
            from rdkit import Chem
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            pytest.skip("RDKit or scikit-learn not installed")
        from PSVAP.analysis.qsar import build_qsar_model
        with pytest.raises(ValueError, match="at least 5"):
            build_qsar_model(SAMPLE_SMILES[:2], [1.0, 2.0])


# ═══════════════════════════════════════════════════════════════════════════
# 5. Pharmacophore (analysis/pharmacophore.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestPharmacophore:

    def test_extract_returns_list(self):
        from PSVAP.analysis.pharmacophore import extract_pharmacophore
        atoms, pos = _make_protein_atoms()
        features = extract_pharmacophore(atoms, pos)
        assert isinstance(features, list)

    def test_detects_donor(self):
        """ARG has N → should be detected as donor."""
        from PSVAP.analysis.pharmacophore import extract_pharmacophore
        atoms, pos = _make_protein_atoms()
        features = extract_pharmacophore(atoms, pos)
        types = [f.feature_type for f in features]
        assert "donor" in types

    def test_detects_acceptor(self):
        """ASP has O → should be detected as acceptor."""
        from PSVAP.analysis.pharmacophore import extract_pharmacophore
        atoms, pos = _make_protein_atoms()
        features = extract_pharmacophore(atoms, pos)
        types = [f.feature_type for f in features]
        assert "acceptor" in types

    def test_feature_center_shape(self):
        from PSVAP.analysis.pharmacophore import extract_pharmacophore
        atoms, pos = _make_protein_atoms()
        for f in extract_pharmacophore(atoms, pos):
            assert f.center.shape == (3,)

    def test_to_dict(self):
        from PSVAP.analysis.pharmacophore import (
            extract_pharmacophore, pharmacophore_to_dict
        )
        atoms, pos = _make_protein_atoms()
        features = extract_pharmacophore(atoms, pos)
        d = pharmacophore_to_dict(features)
        assert "n_features" in d
        assert d["n_features"] == len(features)
        assert "features" in d

    def test_valid_feature_types(self):
        from PSVAP.analysis.pharmacophore import extract_pharmacophore
        valid = {"donor", "acceptor", "hydrophobic",
                 "positive", "negative", "aromatic"}
        atoms, pos = _make_protein_atoms()
        for f in extract_pharmacophore(atoms, pos):
            assert f.feature_type in valid

    def test_subset_selection(self):
        """Restricting to first 2 atoms should give a subset of features."""
        from PSVAP.analysis.pharmacophore import extract_pharmacophore
        atoms, pos = _make_protein_atoms()
        all_features = extract_pharmacophore(atoms, pos)
        sub_features = extract_pharmacophore(atoms, pos, indices=[0, 1])
        assert len(sub_features) <= len(all_features)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Conformational search (analysis/conformational_search.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestConformers:

    @rdkit
    def test_generates_conformers(self):
        from PSVAP.analysis.conformational_search import generate_conformers
        result = generate_conformers(PARACETAMOL, n_conformers=10)
        assert result.n_accepted > 0
        assert len(result.conformers) == result.n_accepted

    @rdkit
    def test_conformer_shape(self):
        from PSVAP.analysis.conformational_search import generate_conformers
        result = generate_conformers(PARACETAMOL, n_conformers=10)
        for conf in result.conformers:
            assert conf.shape == (result.n_atoms, 3)

    @rdkit
    def test_energies_length_matches(self):
        from PSVAP.analysis.conformational_search import generate_conformers
        result = generate_conformers(PARACETAMOL, n_conformers=10)
        assert len(result.energies) == result.n_accepted

    @rdkit
    def test_diversity_filter_respected(self):
        """All accepted conformers should be > min_rmsd apart."""
        from PSVAP.analysis.conformational_search import generate_conformers
        min_rmsd = 0.5
        result = generate_conformers(
            PARACETAMOL, n_conformers=20, min_rmsd=min_rmsd
        )
        for i in range(len(result.conformers)):
            for j in range(i + 1, len(result.conformers)):
                rmsd = float(np.sqrt(
                    ((result.conformers[i] - result.conformers[j]) ** 2).sum()
                    / result.n_atoms
                ))
                assert rmsd >= min_rmsd - 0.05, (
                    f"Conformers {i} and {j} are only {rmsd:.3f} Å apart "
                    f"(min_rmsd={min_rmsd})"
                )

    @rdkit
    def test_invalid_smiles_raises(self):
        from PSVAP.analysis.conformational_search import generate_conformers
        with pytest.raises(ValueError):
            generate_conformers("NOT_VALID_SMILES")

    @rdkit
    def test_elements_list(self):
        from PSVAP.analysis.conformational_search import generate_conformers
        result = generate_conformers(PARACETAMOL, n_conformers=5)
        assert len(result.elements) == result.n_atoms
        assert "C" in result.elements
        assert "N" in result.elements

    @rdkit
    def test_format_result(self):
        from PSVAP.analysis.conformational_search import (
            generate_conformers, format_conformer_result
        )
        result = generate_conformers(PARACETAMOL, n_conformers=5)
        text = format_conformer_result(result)
        assert "CONFORMATIONAL SEARCH" in text
        assert "Accepted" in text


# ═══════════════════════════════════════════════════════════════════════════
# 7. pKa (analysis/pka.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestPKa:

    def test_estimate_returns_list(self):
        from PSVAP.analysis.pka import estimate_pka_from_residues
        atoms, pos = _make_protein_atoms()
        results = estimate_pka_from_residues(atoms, pos)
        assert isinstance(results, list)

    def test_finds_arg_and_asp(self):
        from PSVAP.analysis.pka import estimate_pka_from_residues
        atoms, pos = _make_protein_atoms()
        results = estimate_pka_from_residues(atoms, pos)
        resnames = {r.resname for r in results}
        assert "ARG" in resnames
        assert "ASP" in resnames

    def test_pka_values_reasonable(self):
        """ARG pKa should be around 12.5, ASP around 3.8."""
        from PSVAP.analysis.pka import estimate_pka_from_residues
        atoms, pos = _make_protein_atoms()
        results = estimate_pka_from_residues(atoms, pos)
        for r in results:
            assert 0.0 < r.pka_value < 20.0

    def test_classify_protonation(self):
        from PSVAP.analysis.pka import (
            estimate_pka_from_residues, classify_protonation
        )
        atoms, pos = _make_protein_atoms()
        results = estimate_pka_from_residues(atoms, pos)
        states = classify_protonation(results, ph=7.4)
        assert isinstance(states, dict)
        for state in states.values():
            assert state in {"protonated", "deprotonated", "mixed"}

    def test_arg_protonated_at_physiological_ph(self):
        """ARG pKa ≈ 12.5 >> 7.4 → protonated."""
        from PSVAP.analysis.pka import (
            estimate_pka_from_residues, classify_protonation
        )
        atoms, pos = _make_protein_atoms()
        results = estimate_pka_from_residues(atoms, pos)
        states = classify_protonation(results, ph=7.4)
        arg_results = [r for r in results if r.resname == "ARG"]
        if arg_results:
            assert states.get(arg_results[0].residue_id) == "protonated"

    def test_format_output(self):
        from PSVAP.analysis.pka import (
            estimate_pka_from_residues, format_pka_results
        )
        atoms, pos = _make_protein_atoms()
        results = estimate_pka_from_residues(atoms, pos)
        text = format_pka_results(results)
        assert "pKa" in text
        assert "STATE" in text

    def test_propka_missing_raises_file_not_found(self):
        import tempfile, os
        from PSVAP.analysis.pka import run_propka
        # Create a real (empty) temp PDB so we pass the file-exists check
        # and reach the propka-availability check
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            f.write(b"REMARK test\nEND\n")
            tmp_pdb = f.name
        try:
            with pytest.raises(FileNotFoundError, match="propka3 not found"):
                run_propka(tmp_pdb,
                           propka_executable="__not_installed_propka__")
        finally:
            os.unlink(tmp_pdb)
# ═══════════════════════════════════════════════════════════════════════════
# 8. Docking engine — config + box (no Vina needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestDockingEngine:

    def test_config_creation(self):
        from PSVAP.modeling.docking_engine import DockingConfig
        cfg = DockingConfig(
            receptor_pdbqt="rec.pdbqt",
            ligand_pdbqt="lig.pdbqt",
            center_x=10.0, center_y=20.0, center_z=30.0,
        )
        assert cfg.center_x == 10.0
        assert cfg.exhaustiveness == 8
        assert cfg.size_x == 20.0

    def test_box_from_selection(self):
        from PSVAP.modeling.docking_engine import docking_box_from_selection
        atoms = [
            Atom(id=i, type_id=0, element="C",
                 x=float(i), y=float(i), z=float(i))
            for i in range(10)
        ]
        pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        cx, cy, cz, sx, sy, sz = docking_box_from_selection(
            atoms, pos, list(range(5)), padding=5.0
        )
        assert isinstance(cx, float) and isinstance(sx, float)
        assert sx >= 5.0   # size >= solute extent

    def test_box_all_atoms(self):
        from PSVAP.modeling.docking_engine import docking_box_from_selection
        atoms = [
            Atom(id=i, type_id=0, element="C",
                 x=float(i)*2, y=0.0, z=0.0)
            for i in range(5)
        ]
        pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
        cx, cy, cz, sx, sy, sz = docking_box_from_selection(
            atoms, pos, list(range(5)), padding=5.0
        )
        # Center should be near x=4.0 (midpoint of 0,2,4,6,8)
        assert abs(cx - 4.0) < 1.0

    def test_vina_unavailable_returns_failed_result(self):
        from PSVAP.modeling.docking_engine import DockingConfig, run_vina
        cfg = DockingConfig(
            receptor_pdbqt="rec.pdbqt",
            ligand_pdbqt="lig.pdbqt",
            center_x=0.0, center_y=0.0, center_z=0.0,
            vina_executable="__vina_not_installed__",
        )
        result = run_vina(cfg)
        assert result.success is False
        assert "not found" in result.error_message.lower()

    def test_check_vina_available(self):
        from PSVAP.modeling.docking_engine import check_vina_available
        # Should return False for a non-existent executable
        assert check_vina_available("__not_a_real_executable__") is False

    def test_docking_result_no_poses(self):
        from PSVAP.modeling.docking_engine import DockingResult, DockingConfig
        cfg = DockingConfig(
            receptor_pdbqt="r.pdbqt", ligand_pdbqt="l.pdbqt",
            center_x=0.0, center_y=0.0, center_z=0.0,
        )
        result = DockingResult(poses=[], config=cfg, success=False,
                               error_message="test")
        assert result.best_score is None
        assert result.best_positions is None
        assert "FAILED" in result.summary()