"""
tests/test_rmsd.py
------------------
Phase 2 unit tests for analysis/rmsd.py and analysis/alignment.py.

Run:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_rmsd.py -v

Coverage
--------
  rmsd()                    — zero, known value, subset, shape mismatch
  rmsd_trajectory()         — shape, reference frame, values
  rmsf()                    — shape, static trajectory = 0, subset
  rmsf_per_residue()        — grouping, values
  rmsd_after_superimpose()  — delegates to kabsch_rmsd correctly
  kabsch_rmsd()             — zero, rotation, translation
  superimpose()             — identity, pure translation, pure rotation
  superimpose_trajectory()  — all frames aligned, shapes preserved
  align_trajectory()        — in-place mutation, return count, bad frame index
  rmsd_matrix()             — shape, symmetry, diagonal zeros
"""
from __future__ import annotations

import numpy as np
import pytest

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata, SystemModel


# ═══════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def three_atoms_traj():
    """
    3-atom trajectory, 3 frames.
    frame 0: atoms at (0,0,0), (1,0,0), (2,0,0)
    frame 1: uniform z-shift of +0.5 Å
    frame 2: uniform z-shift of +1.0 Å
    RMSD frame0→frame1 = 0.5 Å, frame0→frame2 = 1.0 Å
    """
    f0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    f1 = f0.copy(); f1[:, 2] += 0.5
    f2 = f0.copy(); f2[:, 2] += 1.0
    return [f0, f1, f2]


@pytest.fixture
def model_3atoms(three_atoms_traj):
    m = SystemModel()
    atoms = [
        Atom(id=0, type_id=0, residue_id=1),
        Atom(id=1, type_id=1, residue_id=1),
        Atom(id=2, type_id=2, residue_id=2),
    ]
    m.set_data(atoms=atoms, trajectory=three_atoms_traj, metadata=SystemMetadata())
    return m


@pytest.fixture
def rotated_pair():
    """
    Two identical structures where b = a rotated 90° around z.
    After Kabsch alignment, RMSD should be ~0.
    """
    rng = np.random.default_rng(42)
    a = rng.uniform(-5, 5, (15, 3))
    theta = np.radians(90)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    b = a @ R.T
    return a, b


# ═══════════════════════════════════════════════════════════════════════════
# 1. rmsd()
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSD:

    def test_identical_is_zero(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsd
        f = three_atoms_traj[0]
        assert rmsd(f, f) == pytest.approx(0.0, abs=1e-12)

    def test_known_uniform_shift(self):
        """Shift every atom by (0,0,0.5) → RMSD = 0.5 Å exactly."""
        from PSVAP.analysis.rmsd import rmsd
        a = np.zeros((4, 3))
        b = a.copy(); b[:, 2] = 0.5
        assert rmsd(a, b) == pytest.approx(0.5, abs=1e-10)

    def test_known_mixed_shift(self):
        """
        2 atoms: shift by (1,0,0) and (-1,0,0).
        Squared diffs: (1,0,0) and (1,0,0) → mean=1 → RMSD=1.0
        """
        from PSVAP.analysis.rmsd import rmsd
        a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert rmsd(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsd
        f0, f1 = three_atoms_traj[0], three_atoms_traj[1]
        assert rmsd(f0, f1) == pytest.approx(rmsd(f1, f0), abs=1e-12)

    def test_subset(self):
        """RMSD on a subset should only consider those atoms."""
        from PSVAP.analysis.rmsd import rmsd
        a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        b = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Only atoms 0 and 1 (both shift by 1 in x) → RMSD = 1.0
        r = rmsd(a, b, atom_indices=[0, 1])
        assert r == pytest.approx(1.0, abs=1e-10)
        # All atoms: atom 2 has a larger shift → overall RMSD > 1.0
        r_all = rmsd(a, b)
        assert r_all > r

    def test_shape_mismatch_raises(self):
        from PSVAP.analysis.rmsd import rmsd
        a = np.zeros((3, 3))
        b = np.zeros((4, 3))
        with pytest.raises(ValueError, match="shape mismatch"):
            rmsd(a, b)

    def test_single_atom(self):
        """Single atom displaced by sqrt(3) in xyz."""
        from PSVAP.analysis.rmsd import rmsd
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 1.0, 1.0]])
        assert rmsd(a, b) == pytest.approx(np.sqrt(3), abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 2. rmsd_trajectory()
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSDTrajectory:

    def test_shape(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsd_trajectory
        vals = rmsd_trajectory(three_atoms_traj, reference_frame=0)
        assert vals.shape == (3,)

    def test_first_frame_is_zero(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsd_trajectory
        vals = rmsd_trajectory(three_atoms_traj, reference_frame=0)
        assert vals[0] == pytest.approx(0.0, abs=1e-12)

    def test_known_values(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsd_trajectory
        vals = rmsd_trajectory(three_atoms_traj, reference_frame=0)
        assert vals[1] == pytest.approx(0.5, abs=1e-10)
        assert vals[2] == pytest.approx(1.0, abs=1e-10)

    def test_different_reference(self, three_atoms_traj):
        """Using frame 1 as reference: RMSD[1] = 0, RMSD[0] = 0.5."""
        from PSVAP.analysis.rmsd import rmsd_trajectory
        vals = rmsd_trajectory(three_atoms_traj, reference_frame=1)
        assert vals[1] == pytest.approx(0.0, abs=1e-12)
        assert vals[0] == pytest.approx(0.5, abs=1e-10)

    def test_empty_trajectory_returns_empty(self):
        from PSVAP.analysis.rmsd import rmsd_trajectory
        vals = rmsd_trajectory([])
        assert vals.shape == (0,)

    def test_with_atom_indices(self, three_atoms_traj):
        """Subset RMSD should differ from full when atoms diverge differently."""
        from PSVAP.analysis.rmsd import rmsd_trajectory
        # atoms 0 and 1 both shift +0.5 in z — subset RMSD = full RMSD here
        vals_full = rmsd_trajectory(three_atoms_traj, reference_frame=0)
        vals_sub  = rmsd_trajectory(three_atoms_traj, reference_frame=0,
                                    atom_indices=[0, 1])
        # Both should equal 0.5 for frame 1 since all atoms shift equally
        assert vals_sub[1] == pytest.approx(0.5, abs=1e-10)
        np.testing.assert_allclose(vals_full, vals_sub, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 3. rmsf()
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSF:

    def test_shape(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsf
        vals = rmsf(three_atoms_traj)
        assert vals.shape == (3,)   # 3 atoms

    def test_static_trajectory_is_zero(self):
        """All frames identical → RMSF = 0 for every atom."""
        from PSVAP.analysis.rmsd import rmsf
        frame = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        traj = [frame.copy(), frame.copy(), frame.copy()]
        vals = rmsf(traj)
        np.testing.assert_allclose(vals, 0.0, atol=1e-12)

    def test_known_single_atom_fluctuation(self):
        """
        1 atom oscillating between z=0 and z=2 in 2 frames.
        Mean z = 1.0.  Deviations = ±1.0.  MSD = 1.0.  RMSF = 1.0.
        """
        from PSVAP.analysis.rmsd import rmsf
        f0 = np.array([[0.0, 0.0, 0.0]])
        f1 = np.array([[0.0, 0.0, 2.0]])
        vals = rmsf([f0, f1])
        assert vals.shape == (1,)
        assert vals[0] == pytest.approx(1.0, abs=1e-10)

    def test_subset(self, three_atoms_traj):
        from PSVAP.analysis.rmsd import rmsf
        vals_all = rmsf(three_atoms_traj)
        vals_sub = rmsf(three_atoms_traj, atom_indices=[0, 1])
        assert vals_sub.shape == (2,)
        np.testing.assert_allclose(vals_sub, vals_all[:2], atol=1e-12)

    def test_empty_trajectory_returns_empty(self):
        from PSVAP.analysis.rmsd import rmsf
        vals = rmsf([])
        assert vals.shape == (0,)

    def test_uniform_fluctuation(self):
        """
        All atoms shift by same vector each frame → same RMSF for all.
        """
        from PSVAP.analysis.rmsd import rmsf
        base = np.zeros((5, 3))
        shifts = [0.0, 1.0, -1.0, 2.0, -2.0]
        traj = [base + np.array([0, 0, s]) for s in shifts]
        vals = rmsf(traj)
        # All atoms should have same RMSF
        np.testing.assert_allclose(vals, vals[0], atol=1e-10)
        # RMSF = std of [0, 1, -1, 2, -2] z-coordinates
        expected = float(np.std(shifts))
        np.testing.assert_allclose(vals, expected, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 4. rmsf_per_residue()
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSFPerResidue:

    def test_returns_dict(self, model_3atoms):
        from PSVAP.analysis.rmsd import rmsf_per_residue
        data = rmsf_per_residue(
            model_3atoms.trajectory,
            model_3atoms.atoms,
        )
        assert isinstance(data, dict)

    def test_residue_ids_match(self, model_3atoms):
        from PSVAP.analysis.rmsd import rmsf_per_residue
        data = rmsf_per_residue(
            model_3atoms.trajectory,
            model_3atoms.atoms,
        )
        # Model has residue_id 1 (atoms 0,1) and 2 (atom 2)
        assert 1 in data
        assert 2 in data

    def test_values_non_negative(self, model_3atoms):
        from PSVAP.analysis.rmsd import rmsf_per_residue
        data = rmsf_per_residue(
            model_3atoms.trajectory,
            model_3atoms.atoms,
        )
        for v in data.values():
            assert v >= 0.0

    def test_static_is_zero(self):
        from PSVAP.analysis.rmsd import rmsf_per_residue
        atoms = [Atom(id=0, type_id=0, residue_id=1),
                 Atom(id=1, type_id=0, residue_id=1)]
        frame = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        traj = [frame.copy(), frame.copy()]
        data = rmsf_per_residue(traj, atoms)
        assert data[1] == pytest.approx(0.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# 5. rmsd_after_superimpose()
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSDAfterSuperimpose:

    def test_identical_is_zero(self):
        from PSVAP.analysis.rmsd import rmsd_after_superimpose
        a = np.random.default_rng(0).random((10, 3))
        assert rmsd_after_superimpose(a, a) == pytest.approx(0.0, abs=1e-8)

    def test_rotation_gives_zero(self, rotated_pair):
        """After optimal alignment, RMSD of a pure rotation should be ~0."""
        from PSVAP.analysis.rmsd import rmsd_after_superimpose
        a, b = rotated_pair
        assert rmsd_after_superimpose(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_lower_than_raw_rmsd(self, rotated_pair):
        """Aligned RMSD ≤ raw (unaligned) RMSD."""
        from PSVAP.analysis.rmsd import rmsd, rmsd_after_superimpose
        a, b = rotated_pair
        assert rmsd_after_superimpose(a, b) <= rmsd(a, b)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Kabsch — kabsch_rmsd() and superimpose()
# ═══════════════════════════════════════════════════════════════════════════

class TestKabsch:

    def test_identical_zero(self):
        from PSVAP.analysis.alignment import kabsch_rmsd
        a = np.random.default_rng(1).random((8, 3))
        assert kabsch_rmsd(a, a) == pytest.approx(0.0, abs=1e-8)

    def test_pure_translation_zero_after_align(self):
        """Pure translation should be fully removed by alignment."""
        from PSVAP.analysis.alignment import kabsch_rmsd
        a = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.5, 1.0, 0.0]])
        b = a + np.array([10.0, -5.0, 3.0])
        assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-8)

    def test_pure_rotation_zero_after_align(self, rotated_pair):
        from PSVAP.analysis.alignment import kabsch_rmsd
        a, b = rotated_pair
        assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_rotation_matrix_is_proper(self):
        """det(R) must be +1 (not -1, which would be a reflection)."""
        from PSVAP.analysis.alignment import kabsch_rotation
        rng = np.random.default_rng(7)
        a = rng.random((20, 3)) - 0.5
        b = rng.random((20, 3)) - 0.5
        R = kabsch_rotation(a, b)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)

    def test_superimpose_returns_correct_shape(self):
        from PSVAP.analysis.alignment import superimpose
        a = np.random.default_rng(2).random((10, 3))
        b = np.random.default_rng(3).random((10, 3))
        aligned, R, t, rmsd_val = superimpose(a, b)
        assert aligned.shape == (10, 3)
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        assert isinstance(rmsd_val, float)

    def test_superimpose_translation_aligns_perfectly(self):
        from PSVAP.analysis.alignment import superimpose
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = a + np.array([3.0, -2.0, 7.0])
        aligned, R, t, rmsd_val = superimpose(b, a)
        assert rmsd_val == pytest.approx(0.0, abs=1e-8)
        np.testing.assert_allclose(aligned, a, atol=1e-8)

    def test_superimpose_with_atom_subset(self):
        """Alignment on a subset should still transform the full structure."""
        from PSVAP.analysis.alignment import superimpose
        rng = np.random.default_rng(5)
        a = rng.random((20, 3))
        # b = a rotated 45° around z
        theta = np.radians(45)
        R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta),  np.cos(theta), 0],
                           [0,              0,             1]])
        b = a @ R_true.T
        aligned, R, t, rmsd_val = superimpose(b, a, atom_indices=list(range(10)))
        # RMSD over all atoms should be near 0 (rotation is global)
        diff = aligned - a
        full_rmsd = float(np.sqrt((diff**2).sum() / len(a)))
        assert full_rmsd == pytest.approx(0.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 7. superimpose_trajectory()
# ═══════════════════════════════════════════════════════════════════════════

class TestSuperimposeTrajectory:

    def test_shapes_preserved(self, three_atoms_traj):
        from PSVAP.analysis.alignment import superimpose_trajectory
        ref = three_atoms_traj[0]
        aligned = superimpose_trajectory(three_atoms_traj, ref)
        assert len(aligned) == 3
        for frame in aligned:
            assert frame.shape == (3, 3)

    def test_reference_frame_unchanged(self, three_atoms_traj):
        """After alignment to frame 0, frame 0 should map to itself."""
        from PSVAP.analysis.alignment import superimpose_trajectory
        ref = three_atoms_traj[0]
        aligned = superimpose_trajectory(three_atoms_traj, ref)
        np.testing.assert_allclose(aligned[0], ref, atol=1e-8)

    def test_does_not_modify_original(self, three_atoms_traj):
        """superimpose_trajectory returns new arrays, leaves originals alone."""
        from PSVAP.analysis.alignment import superimpose_trajectory
        original_f1 = three_atoms_traj[1].copy()
        ref = three_atoms_traj[0]
        superimpose_trajectory(three_atoms_traj, ref)
        np.testing.assert_allclose(three_atoms_traj[1], original_f1, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# 8. align_trajectory() — in-place model alignment
# ═══════════════════════════════════════════════════════════════════════════

class TestAlignTrajectory:

    def test_returns_frame_count(self, model_3atoms):
        from PSVAP.analysis.alignment import align_trajectory
        n = align_trajectory(model_3atoms, reference_frame=0)
        assert n == 3   # all 3 frames aligned

    def test_modifies_model_in_place(self, model_3atoms):
        from PSVAP.analysis.alignment import align_trajectory
        original_f1 = model_3atoms.trajectory[1].copy()
        align_trajectory(model_3atoms, reference_frame=0)
        # frame 1 should have been modified (it was shifted +0.5 in z)
        # after alignment to frame 0 it should be closer to frame 0
        diff_before = np.linalg.norm(original_f1 - model_3atoms.trajectory[0])
        diff_after  = np.linalg.norm(model_3atoms.trajectory[1] - model_3atoms.trajectory[0])
        assert diff_after <= diff_before

    def test_reference_frame_unchanged(self, model_3atoms):
        from PSVAP.analysis.alignment import align_trajectory
        ref_before = model_3atoms.trajectory[0].copy()
        align_trajectory(model_3atoms, reference_frame=0)
        np.testing.assert_allclose(model_3atoms.trajectory[0], ref_before, atol=1e-8)

    def test_invalid_reference_raises(self, model_3atoms):
        from PSVAP.analysis.alignment import align_trajectory
        with pytest.raises(ValueError):
            align_trajectory(model_3atoms, reference_frame=99)

    def test_empty_model_raises(self):
        from PSVAP.analysis.alignment import align_trajectory
        m = SystemModel()   # no data loaded
        with pytest.raises(ValueError):
            align_trajectory(m, reference_frame=0)


# ═══════════════════════════════════════════════════════════════════════════
# 9. rmsd_matrix()
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSDMatrix:

    def test_shape(self, three_atoms_traj):
        from PSVAP.analysis.alignment import rmsd_matrix
        mat = rmsd_matrix(three_atoms_traj)
        assert mat.shape == (3, 3)

    def test_diagonal_zeros(self, three_atoms_traj):
        from PSVAP.analysis.alignment import rmsd_matrix
        mat = rmsd_matrix(three_atoms_traj)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-8)

    def test_symmetric(self, three_atoms_traj):
        from PSVAP.analysis.alignment import rmsd_matrix
        mat = rmsd_matrix(three_atoms_traj)
        np.testing.assert_allclose(mat, mat.T, atol=1e-8)

    def test_no_align_values(self):
        """Without alignment, a pure translation gives known RMSD values."""
        from PSVAP.analysis.alignment import rmsd_matrix
        f0 = np.zeros((3, 3))
        f1 = f0 + np.array([0, 0, 1.0])   # +1 Å in z for all atoms
        mat = rmsd_matrix([f0, f1], align_first=False)
        assert mat[0, 1] == pytest.approx(1.0, abs=1e-10)
        assert mat[1, 0] == pytest.approx(1.0, abs=1e-10)

    def test_with_align_removes_rotation(self, rotated_pair):
        """With alignment, pure rotation gives near-zero RMSD."""
        from PSVAP.analysis.alignment import rmsd_matrix
        a, b = rotated_pair
        mat = rmsd_matrix([a, b], align_first=True)
        assert mat[0, 1] == pytest.approx(0.0, abs=1e-6)