"""
tests/test_geometry.py
----------------------
Phase 2 unit tests for analysis/geometry.py and analysis/rmsd.py
and analysis/alignment.py.

All expected values are validated against published references or
hand-calculated values.

Run:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_geometry.py -v
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def two_atoms():
    """Two atoms 5 Å apart along x-axis."""
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([5.0, 0.0, 0.0])
    return p1, p2


@pytest.fixture
def three_collinear():
    """Collinear atoms — angle should be 180°."""
    return (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    )


@pytest.fixture
def right_angle():
    """90° angle at origin."""
    return (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )


@pytest.fixture
def traj_2frames():
    """Simple 3-atom trajectory with 2 frames."""
    frame0 = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
    frame1 = np.array([[0.0,0.0,0.0],[1.1,0.0,0.0],[2.2,0.0,0.0]])
    return [frame0, frame1]


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry
# ═══════════════════════════════════════════════════════════════════════════

class TestDistance:

    def test_simple_x_axis(self, two_atoms):
        from PSVAP.analysis.geometry import distance
        p1, p2 = two_atoms
        assert distance(p1, p2) == pytest.approx(5.0, abs=1e-10)

    def test_symmetric(self, two_atoms):
        from PSVAP.analysis.geometry import distance
        p1, p2 = two_atoms
        assert distance(p1, p2) == pytest.approx(distance(p2, p1))

    def test_3d_diagonal(self):
        from PSVAP.analysis.geometry import distance
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 1.0])
        assert distance(p1, p2) == pytest.approx(np.sqrt(3), abs=1e-10)

    def test_zero_distance(self):
        from PSVAP.analysis.geometry import distance
        p = np.array([1.5, 2.3, 4.1])
        assert distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_trajectory(self, traj_2frames):
        from PSVAP.analysis.geometry import distance_trajectory
        vals = distance_trajectory(traj_2frames, 0, 1)
        assert vals.shape == (2,)
        assert vals[0] == pytest.approx(1.0, abs=1e-6)
        assert vals[1] == pytest.approx(1.1, abs=1e-6)


class TestAngle:

    def test_right_angle(self, right_angle):
        from PSVAP.analysis.geometry import angle
        assert angle(*right_angle) == pytest.approx(90.0, abs=1e-6)

    def test_collinear_180(self, three_collinear):
        from PSVAP.analysis.geometry import angle
        assert angle(*three_collinear) == pytest.approx(180.0, abs=1e-6)

    def test_equilateral_60(self):
        from PSVAP.analysis.geometry import angle
        # Equilateral triangle
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([0.5, np.sqrt(3)/2, 0.0])
        assert angle(p1, p2, p3) == pytest.approx(60.0, abs=1e-4)

    def test_symmetric(self, right_angle):
        from PSVAP.analysis.geometry import angle
        p1, p2, p3 = right_angle
        assert angle(p1, p2, p3) == pytest.approx(angle(p3, p2, p1), abs=1e-10)

    def test_trajectory_angles(self, traj_2frames):
        from PSVAP.analysis.geometry import angle_trajectory
        vals = angle_trajectory(traj_2frames, 0, 1, 2)
        assert vals.shape == (2,)
        assert all(v == pytest.approx(180.0, abs=1e-4) for v in vals)


class TestTorsion:

    def test_zero_torsion(self):
        """Planar atoms → 0° torsion."""
        from PSVAP.analysis.geometry import torsion
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])
        assert torsion(p1, p2, p3, p4) == pytest.approx(0.0, abs=1e-5)

    def test_180_torsion(self):
        """Trans conformation → ±180°."""
        from PSVAP.analysis.geometry import torsion
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0,-1.0, 0.0])
        t = abs(torsion(p1, p2, p3, p4))
        assert t == pytest.approx(180.0, abs=1e-5)

    def test_90_torsion(self):
        from PSVAP.analysis.geometry import torsion
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 1.0])
        t = torsion(p1, p2, p3, p4)
        assert abs(t) == pytest.approx(90.0, abs=1e-4)

    def test_range(self):
        """Torsion must be in (-180, 180]."""
        from PSVAP.analysis.geometry import torsion
        rng = np.random.default_rng(42)
        for _ in range(100):
            pts = rng.uniform(-10, 10, (4, 3))
            t = torsion(*pts)
            assert -180.0 <= t <= 180.0


# ═══════════════════════════════════════════════════════════════════════════
#  RMSD
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSD:

    def test_identical_structures_zero_rmsd(self, traj_2frames):
        from PSVAP.analysis.rmsd import rmsd
        frame = traj_2frames[0]
        assert rmsd(frame, frame) == pytest.approx(0.0, abs=1e-10)

    def test_known_rmsd(self):
        """Hand-calculated: shift by (1,0,0) gives RMSD = 1.0 Å."""
        from PSVAP.analysis.rmsd import rmsd
        a = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0]])
        b = a + np.array([1.0, 0.0, 0.0])
        assert rmsd(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_trajectory_rmsd_shape(self, traj_2frames):
        from PSVAP.analysis.rmsd import rmsd_trajectory
        vals = rmsd_trajectory(traj_2frames, reference_frame=0)
        assert vals.shape == (2,)
        assert vals[0] == pytest.approx(0.0, abs=1e-10)

    def test_trajectory_rmsd_symmetry(self, traj_2frames):
        from PSVAP.analysis.rmsd import rmsd_trajectory, rmsd
        vals = rmsd_trajectory(traj_2frames, reference_frame=0)
        direct = rmsd(traj_2frames[1], traj_2frames[0])
        assert vals[1] == pytest.approx(direct, abs=1e-10)

    def test_subset_rmsd(self):
        from PSVAP.analysis.rmsd import rmsd
        a = np.zeros((5, 3)); b = np.ones((5, 3))
        # Subset of 2 atoms
        r_all = rmsd(a, b)
        r_sub = rmsd(a, b, atom_indices=[0, 1])
        assert r_all == pytest.approx(r_sub, abs=1e-10)

    def test_rmsf_shape(self, traj_2frames):
        from PSVAP.analysis.rmsd import rmsf
        vals = rmsf(traj_2frames)
        assert vals.shape == (3,)

    def test_rmsf_static_is_zero(self):
        from PSVAP.analysis.rmsd import rmsf
        frame = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
        traj = [frame, frame, frame]
        vals = rmsf(traj)
        np.testing.assert_allclose(vals, 0.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
#  Alignment
# ═══════════════════════════════════════════════════════════════════════════

class TestKabsch:

    def test_identical_zero_rmsd(self):
        from PSVAP.analysis.alignment import kabsch_rmsd
        a = np.random.default_rng(0).random((10, 3))
        assert kabsch_rmsd(a, a) == pytest.approx(0.0, abs=1e-8)

    def test_rotation_gives_lower_rmsd(self):
        from PSVAP.analysis.rmsd import rmsd
        from PSVAP.analysis.alignment import kabsch_rmsd

        rng = np.random.default_rng(7)
        a = rng.random((20, 3))
        # Rotate a by 45 degrees around z
        theta = np.radians(45)
        R = np.array([[np.cos(theta),-np.sin(theta),0],
                      [np.sin(theta), np.cos(theta),0],
                      [0,0,1]])
        b = a @ R.T

        rmsd_no_align  = rmsd(a, b)
        rmsd_with_align = kabsch_rmsd(a, b)

        assert rmsd_with_align < rmsd_no_align
        assert rmsd_with_align == pytest.approx(0.0, abs=1e-6)

    def test_superimpose_translation(self):
        from PSVAP.analysis.alignment import superimpose
        a = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
        b = a + np.array([5.0, 3.0, -2.0])  # pure translation
        aligned, R, t, rmsd_val = superimpose(b, a)
        assert rmsd_val == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(aligned, a, atol=1e-6)

    def test_rmsd_matrix_shape(self):
        from PSVAP.analysis.alignment import rmsd_matrix
        n = 5
        traj = [np.random.default_rng(i).random((8,3)) for i in range(n)]
        mat = rmsd_matrix(traj, align_first=True)
        assert mat.shape == (n, n)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)
        np.testing.assert_allclose(mat, mat.T, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
#  Sequence
# ═══════════════════════════════════════════════════════════════════════════

class TestSequence:

    def test_global_identical(self):
        from PSVAP.analysis.sequence import align_pairwise
        seq = "ACDEFGHIKLM"
        result = align_pairwise(seq, seq, mode="global")
        assert result.identity == pytest.approx(1.0, abs=1e-6)

    def test_global_different(self):
        from PSVAP.analysis.sequence import align_pairwise
        result = align_pairwise("AACDEF", "AACGEF", mode="global")
        assert 0.0 < result.identity < 1.0

    def test_identity_range(self):
        from PSVAP.analysis.sequence import align_pairwise
        result = align_pairwise("MKTLLILAVLVVTIVCLDLGAVV", "MKVLILAVLVVTIVCLDLGAVV", mode="global")
        assert 0.0 <= result.identity <= 1.0
        assert 0.0 <= result.similarity <= 1.0

    def test_similarity_gte_identity(self):
        from PSVAP.analysis.sequence import align_pairwise
        result = align_pairwise("ACDEF", "ACGEF", mode="global")
        assert result.similarity >= result.identity

    def test_local_mode(self):
        from PSVAP.analysis.sequence import align_pairwise
        result = align_pairwise("XXXACDEFXXX", "ACDEF", mode="local")
        assert result.mode == "local"
        assert result.identity > 0