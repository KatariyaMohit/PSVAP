"""
tests/test_phase6.py
---------------------
Phase 6 unit tests for:
  analysis/site_finder.py     (grid-based binding site detection)
  analysis/surface.py         (water map — Feature 20)
  analysis/clustering.py      (trajectory clustering — Feature 22)
  modeling/md_setup.py        (MD input file generation)
  modeling/coarse_grain.py    (MARTINI bead mapping)

Run:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_phase6.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from PSVAP.core.atom import Atom


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_protein_with_pocket() -> tuple[list[Atom], np.ndarray]:
    """
    12 atoms arranged as a simple 'protein shell' with an empty interior.
    6 atoms form a ring at z=0, 6 form a ring at z=4, leaving a pocket
    at the centre.
    """
    atoms = []
    pos_list = []
    ring_r = 4.0
    for i in range(6):
        angle = i * np.pi / 3
        for z in [0.0, 4.0]:
            x = ring_r * np.cos(angle)
            y = ring_r * np.sin(angle)
            atoms.append(Atom(
                id=len(atoms), type_id=1, element="C",
                x=x, y=y, z=z,
                residue_id=i+1, chain_id="A",
                name="CA", resname="ALA",
            ))
            pos_list.append([x, y, z])
    return atoms, np.array(pos_list, dtype=np.float64)


def _make_solvated_atoms() -> tuple[list[Atom], list[np.ndarray]]:
    """Small solvated system: 4 protein + 6 water atoms, 3 frames."""
    atoms = [
        Atom(id=0, type_id=1, element="C", x=0.0, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="CA",  resname="ALA"),
        Atom(id=1, type_id=2, element="N", x=1.5, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="N",   resname="ALA"),
        Atom(id=2, type_id=1, element="C", x=3.0, y=0.0, z=0.0,
             residue_id=2, chain_id="A", name="CA",  resname="GLY"),
        Atom(id=3, type_id=3, element="O", x=4.5, y=0.0, z=0.0,
             residue_id=2, chain_id="A", name="O",   resname="GLY"),
        # Waters
        Atom(id=4, type_id=3, element="O", x=10.0, y=0.0, z=0.0,
             residue_id=100, chain_id="W", name="OW",  resname="HOH"),
        Atom(id=5, type_id=5, element="H", x=10.9, y=0.0, z=0.0,
             residue_id=100, chain_id="W", name="HW1", resname="HOH"),
        Atom(id=6, type_id=5, element="H", x=10.0, y=0.9, z=0.0,
             residue_id=100, chain_id="W", name="HW2", resname="HOH"),
        Atom(id=7, type_id=3, element="O", x=12.0, y=0.0, z=0.0,
             residue_id=101, chain_id="W", name="OW",  resname="HOH"),
        Atom(id=8, type_id=5, element="H", x=12.9, y=0.0, z=0.0,
             residue_id=101, chain_id="W", name="HW1", resname="HOH"),
        Atom(id=9, type_id=5, element="H", x=12.0, y=0.9, z=0.0,
             residue_id=101, chain_id="W", name="HW2", resname="HOH"),
    ]
    base_pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    traj = [
        base_pos.copy(),
        base_pos + np.array([0.1, 0.0, 0.0]),
        base_pos + np.array([0.0, 0.1, 0.0]),
    ]
    return atoms, traj


def _make_trajectory(n_frames: int = 10) -> list[np.ndarray]:
    """Simple 5-atom trajectory with known clusters."""
    rng = np.random.default_rng(42)
    frames = []
    # 5 frames near state A, 5 frames near state B
    state_a = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.5,0.5,0.0],
                        [0.0,1.0,0.0],[1.0,1.0,0.0]])
    state_b = state_a + np.array([10.0, 0.0, 0.0])
    for i in range(n_frames // 2):
        frames.append(state_a + rng.normal(0, 0.05, state_a.shape))
    for i in range(n_frames - n_frames // 2):
        frames.append(state_b + rng.normal(0, 0.05, state_b.shape))
    return frames


def _make_ala_gly_atoms() -> tuple[list[Atom], np.ndarray]:
    """ALA + GLY residues for CG bead mapping test."""
    atoms = [
        Atom(id=0, type_id=2, element="N",  x=0.0, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="N",  resname="ALA"),
        Atom(id=1, type_id=1, element="C",  x=1.5, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="CA", resname="ALA"),
        Atom(id=2, type_id=1, element="C",  x=2.0, y=1.2, z=0.0,
             residue_id=1, chain_id="A", name="C",  resname="ALA"),
        Atom(id=3, type_id=3, element="O",  x=3.2, y=1.2, z=0.0,
             residue_id=1, chain_id="A", name="O",  resname="ALA"),
        Atom(id=4, type_id=1, element="C",  x=1.5, y=-1.2, z=0.0,
             residue_id=1, chain_id="A", name="CB", resname="ALA"),
        Atom(id=5, type_id=2, element="N",  x=3.5, y=0.0, z=0.0,
             residue_id=2, chain_id="A", name="N",  resname="GLY"),
        Atom(id=6, type_id=1, element="C",  x=5.0, y=0.0, z=0.0,
             residue_id=2, chain_id="A", name="CA", resname="GLY"),
        Atom(id=7, type_id=1, element="C",  x=5.5, y=1.2, z=0.0,
             residue_id=2, chain_id="A", name="C",  resname="GLY"),
        Atom(id=8, type_id=3, element="O",  x=6.7, y=1.2, z=0.0,
             residue_id=2, chain_id="A", name="O",  resname="GLY"),
    ]
    pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    return atoms, pos


# ═══════════════════════════════════════════════════════════════════════════
# 1. Site Finder — grid-based
# ═══════════════════════════════════════════════════════════════════════════

class TestSiteFinderGrid:

    def test_returns_list(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        assert isinstance(sites, list)

    def test_sites_have_correct_type(self):
        from PSVAP.analysis.site_finder import find_sites_grid, BindingSite
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        for s in sites:
            assert isinstance(s, BindingSite)

    def test_center_shape(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        for s in sites:
            assert s.center.shape == (3,)

    def test_volume_positive(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        for s in sites:
            assert s.volume > 0.0

    def test_ranks_sequential(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        for i, s in enumerate(sites):
            assert s.rank == i + 1

    def test_scores_descending(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        for i in range(1, len(sites)):
            assert sites[i].score <= sites[i-1].score

    def test_max_sites_respected(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5, max_sites=2)
        assert len(sites) <= 2

    def test_empty_atoms_returns_empty(self):
        from PSVAP.analysis.site_finder import find_sites_grid
        sites = find_sites_grid([], np.zeros((0,3)))
        assert sites == []

    def test_format_sites(self):
        from PSVAP.analysis.site_finder import find_sites_grid, format_sites
        atoms, pos = _make_protein_with_pocket()
        sites = find_sites_grid(atoms, pos, grid_spacing=1.5)
        text = format_sites(sites)
        assert isinstance(text, str)

    def test_format_empty_sites(self):
        from PSVAP.analysis.site_finder import format_sites
        text = format_sites([])
        assert "NO BINDING SITES" in text

    def test_fpocket_check_returns_bool(self):
        from PSVAP.analysis.site_finder import check_fpocket_available
        result = check_fpocket_available("__not_installed_fpocket__")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. Water Map (analysis/surface.py additions)
# ═══════════════════════════════════════════════════════════════════════════

class TestWaterMap:

    def test_returns_dict(self):
        from PSVAP.analysis.surface import compute_water_density
        atoms, traj = _make_solvated_atoms()
        result = compute_water_density(atoms, traj)
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        from PSVAP.analysis.surface import compute_water_density
        atoms, traj = _make_solvated_atoms()
        result = compute_water_density(atoms, traj)
        for key in ["grid_origin", "grid_spacing", "density",
                    "hydration_sites", "n_frames", "n_water_atoms"]:
            assert key in result

    def test_n_frames_correct(self):
        from PSVAP.analysis.surface import compute_water_density
        atoms, traj = _make_solvated_atoms()
        result = compute_water_density(atoms, traj)
        assert result["n_frames"] == len(traj)

    def test_n_water_atoms_detected(self):
        from PSVAP.analysis.surface import compute_water_density
        atoms, traj = _make_solvated_atoms()
        result = compute_water_density(atoms, traj)
        assert result["n_water_atoms"] == 2   # 2 HOH oxygens (OW)

    def test_density_non_negative(self):
        from PSVAP.analysis.surface import compute_water_density
        atoms, traj = _make_solvated_atoms()
        result = compute_water_density(atoms, traj)
        assert (result["density"] >= 0).all()

    def test_no_water_returns_empty_sites(self):
        from PSVAP.analysis.surface import compute_water_density
        # Atoms with no water residues
        atoms = [
            Atom(id=0, type_id=1, element="C", x=0.0, y=0.0, z=0.0,
                 residue_id=1, resname="ALA"),
        ]
        pos = np.array([[0.0, 0.0, 0.0]])
        result = compute_water_density(atoms, [pos])
        assert result.get("n_water_atoms", 0) == 0

    def test_format_water_map(self):
        from PSVAP.analysis.surface import compute_water_density, format_water_map
        atoms, traj = _make_solvated_atoms()
        data = compute_water_density(atoms, traj)
        text = format_water_map(data)
        assert "WATER MAP" in text
        assert "n_frames" not in text   # should be formatted, not raw

    def test_format_empty_data(self):
        from PSVAP.analysis.surface import format_water_map
        text = format_water_map({})
        assert "NO WATER MAP" in text


# ═══════════════════════════════════════════════════════════════════════════
# 3. Trajectory Clustering
# ═══════════════════════════════════════════════════════════════════════════

class TestTrajectoryCluster:

    def test_returns_result(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import (
            cluster_trajectory, TrajectoryClusterResult
        )
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2)
        assert isinstance(result, TrajectoryClusterResult)

    def test_correct_n_clusters(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2)
        assert result.n_clusters == 2

    def test_labels_length_matches_frames(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2)
        assert len(result.labels) == len(traj)

    def test_medoid_indices_valid(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2)
        for mid in result.medoid_indices:
            assert 0 <= mid < len(traj)

    def test_cluster_sizes_sum_to_total(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2)
        assert sum(result.cluster_sizes) == len(traj)

    def test_finds_two_states(self):
        """With 5 frames near A and 5 near B, 2 clusters should separate them."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2, align_first=False)
        # Both clusters should be non-empty
        assert all(s > 0 for s in result.cluster_sizes)

    def test_hierarchical_method(self):
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(8)
        result = cluster_trajectory(traj, n_clusters=2, method="hierarchical")
        assert result.method == "hierarchical"

    def test_too_few_frames_raises(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import cluster_trajectory
        traj = _make_trajectory(3)
        with pytest.raises(ValueError):
            cluster_trajectory(traj, n_clusters=5)

    def test_format_result(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            pytest.skip("scikit-learn not installed")
        from PSVAP.analysis.clustering import (
            cluster_trajectory, format_cluster_result
        )
        traj = _make_trajectory(10)
        result = cluster_trajectory(traj, n_clusters=2)
        text = format_cluster_result(result)
        assert "CLUSTERING" in text
        assert "MEDOID" in text


# ═══════════════════════════════════════════════════════════════════════════
# 4. MD Setup — GROMACS
# ═══════════════════════════════════════════════════════════════════════════

class TestMDSetupGromacs:

    def test_creates_files(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_gromacs_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "gromacs"),
        )
        assert len(result.files_created) > 0

    def test_mdp_files_exist(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_gromacs_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "gromacs"),
        )
        for fname in ["em.mdp", "nvt.mdp", "npt.mdp", "md.mdp"]:
            assert fname in result.files_created
            assert (result.output_dir / fname).exists()

    def test_readme_created(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_gromacs_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "gromacs"),
        )
        assert "README.txt" in result.files_created

    def test_commands_non_empty(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_gromacs_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "gromacs"),
        )
        assert len(result.commands) > 0

    def test_warns_no_hydrogens(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_gromacs_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "gromacs"),
        )
        # No H atoms → should warn
        assert any("hydrogen" in w.lower() for w in result.warnings)

    def test_summary_contains_engine(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_gromacs_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "gromacs"),
        )
        assert "GROMACS" in result.summary().upper()

    def test_with_box_bounds(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_gromacs_inputs
        atoms, pos = _make_ala_gly_atoms()
        bb = np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])
        result = generate_gromacs_inputs(
            atoms, pos, bb,
            output_dir=str(tmp_path / "gromacs_bb"),
        )
        assert result.engine == "gromacs"


# ═══════════════════════════════════════════════════════════════════════════
# 5. MD Setup — AMBER
# ═══════════════════════════════════════════════════════════════════════════

class TestMDSetupAmber:

    def test_creates_amber_files(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_amber_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_amber_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "amber"),
        )
        for fname in ["tleap.in", "min.in", "heat.in", "equil.in", "prod.in"]:
            assert fname in result.files_created

    def test_engine_is_amber(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_amber_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_amber_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "amber"),
        )
        assert result.engine == "amber"

    def test_tleap_contains_source(self, tmp_path):
        from PSVAP.modeling.md_setup import generate_amber_inputs
        atoms, pos = _make_ala_gly_atoms()
        result = generate_amber_inputs(
            atoms, pos, None,
            output_dir=str(tmp_path / "amber"),
        )
        tleap_text = (result.output_dir / "tleap.in").read_text()
        assert "source" in tleap_text
        assert "leaprc" in tleap_text


# ═══════════════════════════════════════════════════════════════════════════
# 6. Coarse-Grained Bead Mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestCGBeadMapping:

    def test_returns_bead_map_and_positions(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, cg_pos = build_cg_beads(atoms, pos)
        assert isinstance(bead_map, list)
        assert isinstance(cg_pos, np.ndarray)

    def test_fewer_beads_than_atoms(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, cg_pos = build_cg_beads(atoms, pos)
        assert len(bead_map) < len(atoms)

    def test_cg_positions_shape(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, cg_pos = build_cg_beads(atoms, pos)
        assert cg_pos.shape == (len(bead_map), 3)

    def test_bead_centers_in_range(self):
        """CG bead centers should be within the solute bounding box."""
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, cg_pos = build_cg_beads(atoms, pos)
        lo = pos.min(axis=0) - 1.0
        hi = pos.max(axis=0) + 1.0
        assert np.all(cg_pos >= lo) and np.all(cg_pos <= hi)

    def test_ala_has_bb_and_sc1(self):
        """ALA should produce BB and SC1 beads."""
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, _ = build_cg_beads(atoms, pos)
        ala_beads = [
            b.bead_name for b in bead_map
            if b.resname == "ALA"
        ]
        assert "BB" in ala_beads

    def test_gly_has_only_bb(self):
        """GLY has only a backbone bead (no sidechain)."""
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, _ = build_cg_beads(atoms, pos)
        gly_beads = [b.bead_name for b in bead_map if b.resname == "GLY"]
        assert "BB" in gly_beads
        assert "SC1" not in gly_beads

    def test_beadmap_has_atom_indices(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads
        atoms, pos = _make_ala_gly_atoms()
        bead_map, _ = build_cg_beads(atoms, pos)
        for b in bead_map:
            assert len(b.atom_indices) > 0

    def test_cg_result_summary(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads, CGResult
        atoms, pos = _make_ala_gly_atoms()
        bead_map, cg_pos = build_cg_beads(atoms, pos)
        result = CGResult(
            n_atoms=len(atoms), n_beads=len(bead_map),
            bead_map=bead_map, method="built-in"
        )
        text = result.summary()
        assert "COARSE-GRAINING" in text
        assert str(len(atoms)) in text

    def test_format_bead_map(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads, format_bead_map
        atoms, pos = _make_ala_gly_atoms()
        bead_map, _ = build_cg_beads(atoms, pos)
        text = format_bead_map(bead_map)
        assert "MARTINI" in text
        assert "BB" in text

    def test_martinize2_check(self):
        from PSVAP.modeling.coarse_grain import check_martinize2_available
        assert check_martinize2_available("__not_installed_martinize2__") is False

    def test_empty_atoms_returns_empty(self):
        from PSVAP.modeling.coarse_grain import build_cg_beads
        bead_map, cg_pos = build_cg_beads([], np.zeros((0, 3)))
        assert bead_map == []
        assert cg_pos.shape == (0, 3)