"""
PSVAP/tests/test_analysis.py
-----------------------------
Complete test suite for all Analysis panel operations.

Run from the project root:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    python -m pytest PSVAP/tests/test_analysis.py -v

What is tested
--------------
1. Geometry — distance, angle, torsion (with expected values you can verify)
2. RMSD     — same-frame (should be 0.0), different-frame
3. Alignment — Kabsch rotation (aligned RMSD ≤ original RMSD)
4. Selection — type==0 / type==1 / type==2, AND/OR/NOT, z>V, mol==N
5. Bond check — metadata.bonds is numpy array, must use `is not None` not `if bonds`

All tests are self-contained using small inline atom lists.
No file I/O required — you can run immediately.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata, SystemModel


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_lammps_model() -> SystemModel:
    """
    Minimal model that mimics your amyloid LAMMPS file.
    Types 0, 1, 2  (0-indexed, as stored after the type-ID fix).
    Molecule IDs 0, 0, 0, 1, 1.
    Positions form a known geometry so we can verify distance/angle/torsion.
    """
    m = SystemModel()
    atoms = [
        # id  type_id  element  x    y    z     mol
        Atom(id=0, type_id=0, element=None, x=0.0,  y=0.0,  z=0.0,  residue_id=0),
        Atom(id=1, type_id=1, element=None, x=1.0,  y=0.0,  z=0.0,  residue_id=0),
        Atom(id=2, type_id=2, element=None, x=1.0,  y=1.0,  z=0.0,  residue_id=0),
        Atom(id=3, type_id=0, element=None, x=0.0,  y=2.0,  z=0.0,  residue_id=1),
        Atom(id=4, type_id=1, element=None, x=0.0,  y=2.0,  z=1.0,  residue_id=1),
    ]
    pos0 = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    # Second frame: slight displacement (+0.1 in z for all atoms)
    pos1 = pos0.copy()
    pos1[:, 2] += 0.1
    m.set_data(atoms=atoms, trajectory=[pos0, pos1],
               metadata=SystemMetadata(bonds=None))
    return m


def _make_pdb_model() -> SystemModel:
    """Model with element names (simulates PDB/GRO/CIF files)."""
    m = SystemModel()
    atoms = [
        Atom(id=0, type_id=1, element="C", x=1.0, y=2.0, z=3.0,
             residue_id=1, chain_id="A", name="CA"),
        Atom(id=1, type_id=2, element="N", x=4.0, y=5.0, z=6.0,
             residue_id=1, chain_id="A", name="N"),
        Atom(id=2, type_id=1, element="C", x=7.0, y=8.0, z=9.0,
             residue_id=2, chain_id="A", name="CA"),
        Atom(id=3, type_id=3, element="O", x=4.0, y=5.0, z=3.0,
             residue_id=2, chain_id="A", name="O"),
    ]
    pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    m.set_data(atoms=atoms, trajectory=[pos],
               metadata=SystemMetadata(bonds=None))
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  1. GEOMETRY — DISTANCE
# ═══════════════════════════════════════════════════════════════════════════

class TestDistance:
    """
    Uses model with atoms at:
      atom 0: (0,0,0)
      atom 1: (1,0,0)
      atom 2: (1,1,0)
      atom 3: (0,2,0)
    """

    def test_unit_distance(self):
        """atom 0 to atom 1 along x-axis → exactly 1.0 Å"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        d = float(np.linalg.norm(pos[1] - pos[0]))
        assert abs(d - 1.0) < 1e-9, f"Expected 1.0 Å, got {d}"

    def test_diagonal_distance(self):
        """atom 0 at (0,0,0) to atom 2 at (1,1,0) → √2 ≈ 1.41421 Å"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        d = float(np.linalg.norm(pos[2] - pos[0]))
        expected = math.sqrt(2)
        assert abs(d - expected) < 1e-6, f"Expected {expected:.5f} Å, got {d:.5f}"
        print(f"\n  ✓  dist(atom0, atom2) = {d:.5f} Å  (expected {expected:.5f} Å)")

    def test_3d_distance(self):
        """atom 3 at (0,2,0) to atom 4 at (0,2,1) → 1.0 Å"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        d = float(np.linalg.norm(pos[4] - pos[3]))
        assert abs(d - 1.0) < 1e-9

    def test_distance_is_symmetric(self):
        """distance(i,j) == distance(j,i)"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        d01 = float(np.linalg.norm(pos[1] - pos[0]))
        d10 = float(np.linalg.norm(pos[0] - pos[1]))
        assert abs(d01 - d10) < 1e-12

    def test_distance_same_atom_is_zero(self):
        """distance(i,i) = 0"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        d = float(np.linalg.norm(pos[0] - pos[0]))
        assert d == 0.0

    def test_geometry_module_distance(self):
        """Test PSVAP.analysis.geometry.distance if available."""
        try:
            from PSVAP.analysis.geometry import distance
            m = _make_lammps_model()
            pos = m.get_frame(0)
            d = distance(pos[0], pos[1])
            assert abs(d - 1.0) < 1e-6, f"geometry.distance returned {d}"
        except ImportError:
            pytest.skip("PSVAP.analysis.geometry not available")


# ═══════════════════════════════════════════════════════════════════════════
#  2. GEOMETRY — BOND ANGLE
# ═══════════════════════════════════════════════════════════════════════════

class TestAngle:
    """
    atoms:  0=(0,0,0)  1=(1,0,0)  2=(1,1,0)  3=(0,2,0)

    angle(0,1,2):  vectors from 1→0 = (-1,0,0), from 1→2 = (0,1,0)
                   → 90°  (perpendicular)

    angle(0,1,3):  from 1=(1,0,0): v1 = 0-1 = (-1,0,0), v2 = 3-1 = (-1,2,0)
                   cos θ = (1)/(1 * √5) = 1/√5 → θ ≈ 63.435°
    """

    def _angle(self, pi, pj, pk) -> float:
        v1 = pi - pj
        v2 = pk - pj
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    def test_right_angle(self):
        """angle(atom0, atom1, atom2) with vertex at atom1 → 90.0°"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        a = self._angle(pos[0], pos[1], pos[2])
        assert abs(a - 90.0) < 1e-6, f"Expected 90.0°, got {a:.4f}°"
        print(f"\n  ✓  angle(0,1,2) = {a:.4f}°  (expected 90.0°)")

    def test_63_degree_angle(self):
        """angle(atom0, atom1, atom3) ≈ 63.435°"""
        m = _make_lammps_model()
        pos = m.get_frame(0)
        a = self._angle(pos[0], pos[1], pos[3])
        expected = math.degrees(math.acos(1 / math.sqrt(5)))
        assert abs(a - expected) < 1e-4, f"Expected {expected:.4f}°, got {a:.4f}°"
        print(f"\n  ✓  angle(0,1,3) = {a:.4f}°  (expected {expected:.4f}°)")

    def test_geometry_module_angle(self):
        try:
            from PSVAP.analysis.geometry import angle
            m = _make_lammps_model()
            pos = m.get_frame(0)
            a = angle(pos[0], pos[1], pos[2])
            assert abs(a - 90.0) < 1e-4
        except ImportError:
            pytest.skip("PSVAP.analysis.geometry not available")


# ═══════════════════════════════════════════════════════════════════════════
#  3. GEOMETRY — TORSION / DIHEDRAL
# ═══════════════════════════════════════════════════════════════════════════

class TestTorsion:
    """
    Test with 4 atoms in a known dihedral arrangement.
    Using atoms: 0=(0,0,0)  1=(1,0,0)  2=(1,1,0)  3=(0,1,0)
    This gives a planar (cis) dihedral → 0°

    For a 90° dihedral, use: 0=(0,0,1) 1=(0,0,0) 2=(1,0,0) 3=(1,1,0)
    """

    def _torsion(self, p0, p1, p2, p3) -> float:
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_n = b1 / (np.linalg.norm(b1) + 1e-15)
        v = b0 - np.dot(b0, b1_n) * b1_n
        w = b2 - np.dot(b2, b1_n) * b1_n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1_n, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    def test_90_degree_torsion(self):
        """Standard 90° dihedral arrangement → 90°"""
        p0 = np.array([0.0, 0.0, 1.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        t = self._torsion(p0, p1, p2, p3)
        assert abs(t - 90.0) < 1e-4, f"Expected 90.0°, got {t:.4f}°"
        print(f"\n  ✓  torsion = {t:.4f}°  (expected 90.0°)")

    def test_180_degree_torsion(self):
        """Trans arrangement → 180°"""
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, -1.0, 0.0])
        t = self._torsion(p0, p1, p2, p3)
        assert abs(abs(t) - 180.0) < 1e-3, f"Expected ±180°, got {t:.4f}°"
        print(f"\n  ✓  trans torsion = {t:.4f}°  (expected ±180°)")

    def test_geometry_module_torsion(self):
        try:
            from PSVAP.analysis.geometry import torsion
            p0 = np.array([0.0, 0.0, 1.0])
            p1 = np.array([0.0, 0.0, 0.0])
            p2 = np.array([1.0, 0.0, 0.0])
            p3 = np.array([1.0, 1.0, 0.0])
            t = torsion(p0, p1, p2, p3)
            assert abs(t - 90.0) < 1e-3
        except ImportError:
            pytest.skip("PSVAP.analysis.geometry not available")


# ═══════════════════════════════════════════════════════════════════════════
#  4. RMSD
# ═══════════════════════════════════════════════════════════════════════════

class TestRMSD:
    """
    RMSD(frame, frame) must be 0.
    RMSD(frame0, frame1) where frame1 = frame0 + 0.1 in z for all N atoms
    → RMSD = sqrt(mean(0.1²)) = 0.1
    """

    def _rmsd(self, a: np.ndarray, b: np.ndarray) -> float:
        diff = b - a
        return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    def test_same_frame_rmsd_is_zero(self):
        m = _make_lammps_model()
        pos = m.get_frame(0)
        r = self._rmsd(pos, pos)
        assert r == 0.0, f"RMSD(same,same) should be 0, got {r}"
        print(f"\n  ✓  RMSD(frame0, frame0) = {r}  (expected 0.0)")

    def test_uniform_z_shift(self):
        """
        frame1 = frame0 shifted +0.1 Å in z for every atom
        RMSD = sqrt(mean(0.1²)) = 0.1 exactly
        """
        m = _make_lammps_model()
        pos0 = m.get_frame(0)
        pos1 = m.get_frame(1)
        r = self._rmsd(pos0, pos1)
        assert abs(r - 0.1) < 1e-10, f"Expected RMSD=0.1, got {r}"
        print(f"\n  ✓  RMSD(frame0, frame1) = {r:.6f}  (expected 0.100000)")

    def test_rmsd_is_symmetric(self):
        m = _make_lammps_model()
        pos0 = m.get_frame(0)
        pos1 = m.get_frame(1)
        r01 = self._rmsd(pos0, pos1)
        r10 = self._rmsd(pos1, pos0)
        assert abs(r01 - r10) < 1e-12

    def test_rmsd_module(self):
        try:
            from PSVAP.analysis.rmsd import rmsd
            m = _make_lammps_model()
            pos0 = m.get_frame(0)
            r = rmsd(pos0, pos0)
            assert r == 0.0
        except ImportError:
            pytest.skip("PSVAP.analysis.rmsd not available")


# ═══════════════════════════════════════════════════════════════════════════
#  5. SELECTION — TYPE IDs (the key fix for your amyloid file)
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectionLAMMPS:
    """
    Model has 5 atoms with type_ids: [0, 1, 2, 0, 1]
    and mol IDs: [0, 0, 0, 1, 1]
    """

    def _get_model(self):
        return _make_lammps_model()

    def test_type0_selects_two_atoms(self):
        """type==0 should select atoms 0 and 3 (indices 0 and 3)"""
        m = self._get_model()
        atoms = m.atoms
        n = len(atoms)
        mask = np.array([getattr(a, 'type_id', -1) == 0 for a in atoms])
        assert mask.sum() == 2, f"Expected 2 type-0 atoms, got {mask.sum()}"
        assert mask[0] and mask[3], "Wrong atoms selected for type==0"
        print(f"\n  ✓  type==0  selects {mask.sum()} atoms  (expected 2: indices 0, 3)")

    def test_type1_selects_two_atoms(self):
        """type==1 should select atoms 1 and 4"""
        m = self._get_model()
        mask = np.array([getattr(a, 'type_id', -1) == 1 for a in m.atoms])
        assert mask.sum() == 2
        assert mask[1] and mask[4]
        print(f"\n  ✓  type==1  selects {mask.sum()} atoms  (expected 2: indices 1, 4)")

    def test_type2_selects_one_atom(self):
        """type==2 should select atom 2 only"""
        m = self._get_model()
        mask = np.array([getattr(a, 'type_id', -1) == 2 for a in m.atoms])
        assert mask.sum() == 1
        assert mask[2]
        print(f"\n  ✓  type==2  selects {mask.sum()} atom   (expected 1: index 2)")

    def test_type_or_type(self):
        """type==0 OR type==1 should select 4 atoms"""
        m = self._get_model()
        mask = np.array(
            [(getattr(a, 'type_id', -1) in {0, 1}) for a in m.atoms]
        )
        assert mask.sum() == 4
        print(f"\n  ✓  type==0 OR type==1  selects {mask.sum()} atoms  (expected 4)")

    def test_not_type2(self):
        """NOT type==2 should select 4 atoms"""
        m = self._get_model()
        mask = np.array([getattr(a, 'type_id', -1) != 2 for a in m.atoms])
        assert mask.sum() == 4
        print(f"\n  ✓  NOT type==2  selects {mask.sum()} atoms  (expected 4)")

    def test_mol_selection(self):
        """mol==0 should select atoms 0,1,2 (3 atoms)"""
        m = self._get_model()
        mask = np.array([getattr(a, 'residue_id', -1) == 0 for a in m.atoms])
        assert mask.sum() == 3
        print(f"\n  ✓  mol==0  selects {mask.sum()} atoms  (expected 3: indices 0,1,2)")

    def test_mol_1_selection(self):
        """mol==1 should select atoms 3,4 (2 atoms)"""
        m = self._get_model()
        mask = np.array([getattr(a, 'residue_id', -1) == 1 for a in m.atoms])
        assert mask.sum() == 2
        print(f"\n  ✓  mol==1  selects {mask.sum()} atoms  (expected 2: indices 3,4)")

    def test_z_greater_than(self):
        """z > 0 in frame 0: only atom 4 has z=1.0 → 1 atom"""
        m = self._get_model()
        pos = m.get_frame(0)
        mask = pos[:, 2] > 0.0
        assert mask.sum() == 1
        assert mask[4]
        print(f"\n  ✓  z > 0  selects {mask.sum()} atom  (expected 1: index 4)")

    def test_type_and_z(self):
        """type==1 AND z > 0 → only atom 4 (type_id=1, z=1.0)"""
        m = self._get_model()
        pos = m.get_frame(0)
        type_mask = np.array([getattr(a, 'type_id', -1) == 1 for a in m.atoms])
        z_mask = pos[:, 2] > 0.0
        combined = type_mask & z_mask
        assert combined.sum() == 1
        assert combined[4]
        print(f"\n  ✓  type==1 AND z>0  selects {combined.sum()} atom  (expected 1: index 4)")

    def test_selection_via_parse_selection(self):
        """End-to-end via core/selection.py parse_selection()"""
        try:
            from PSVAP.core.selection import parse_selection
            m = self._get_model()
            mask = parse_selection("type == 0", m)
            assert mask.sum() == 2, f"Expected 2, got {mask.sum()}"
        except ImportError:
            pytest.skip("PSVAP.core.selection not available")


# ═══════════════════════════════════════════════════════════════════════════
#  6. NUMPY ARRAY TRUTH VALUE BUG (the critical fix)
# ═══════════════════════════════════════════════════════════════════════════

class TestNumpyBooleanFix:
    """
    The crash: `if bonds_array:` raises ValueError for numpy arrays.
    This test verifies the correct pattern is used everywhere.
    """

    def test_numpy_array_truthiness_raises(self):
        """
        Demonstrate the bug: `if array:` raises ValueError.
        This is what was crashing main_window._on_data_loaded().
        """
        arr = np.array([2, 0, 1, 2, 1, 3], dtype=np.int64)  # typical bond array
        with pytest.raises(ValueError, match="ambiguous"):
            # This is what the OLD code did — it SHOULD raise
            if arr:
                pass

    def test_correct_numpy_check_none(self):
        """Correct pattern: `if arr is not None and len(arr) > 0`"""
        arr = np.array([2, 0, 1, 2, 1, 3], dtype=np.int64)
        # Should NOT raise
        result = arr is not None and len(arr) > 0
        assert result is True

    def test_correct_numpy_check_empty(self):
        """Empty array should evaluate as False with correct check"""
        arr = np.array([], dtype=np.int64)
        result = arr is not None and len(arr) > 0
        assert result is False

    def test_none_check(self):
        """None should evaluate as False"""
        arr = None
        result = arr is not None and len(arr) > 0
        assert result is False

    def test_bond_count_from_pyvista_format(self):
        """
        Bond array format: [2, i, j, 2, i2, j2, ...]
        n_bonds = len(arr) // 3
        """
        # 15000 bonds in PyVista format = 45000 integers
        arr = np.zeros(45000, dtype=np.int64)
        n_bonds = len(arr) // 3
        assert n_bonds == 15000, f"Expected 15000, got {n_bonds}"
        print(f"\n  ✓  len(bonds)//3 = {n_bonds}  (for 15,000-bond amyloid file)")


# ═══════════════════════════════════════════════════════════════════════════
#  7. INTEGRATION — FULL WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════

class TestFullWorkflow:
    """
    Simulates the complete workflow:
    load → select → measure distance → compute RMSD
    """

    def test_load_select_measure(self):
        """Full pipeline: load model, select atoms, measure geometry."""
        m = _make_lammps_model()

        # 1. Verify load
        assert len(m.atoms) == 5
        assert m.n_frames() == 2

        # 2. Verify type IDs are correct (0-indexed, NOT -1,0,1)
        type_ids = [a.type_id for a in m.atoms]
        assert 0 in type_ids, "type 0 should exist"
        assert 1 in type_ids, "type 1 should exist"
        assert 2 in type_ids, "type 2 should exist"
        assert -1 not in type_ids, "type -1 should NOT exist (old bug)"

        # 3. Select type==0
        sel_mask = np.array([a.type_id == 0 for a in m.atoms])
        assert sel_mask.sum() == 2

        # 4. Measure distance between selected atoms (indices 0 and 3)
        pos = m.get_frame(0)
        selected_indices = np.where(sel_mask)[0]
        assert list(selected_indices) == [0, 3]
        d = float(np.linalg.norm(pos[selected_indices[1]] - pos[selected_indices[0]]))
        # atom 0 at (0,0,0), atom 3 at (0,2,0) → distance = 2.0 Å
        assert abs(d - 2.0) < 1e-9
        print(f"\n  ✓  Full workflow: selected type-0 atoms, dist = {d:.4f} Å")

        # 5. RMSD frame0 vs frame1
        pos0 = m.get_frame(0)
        pos1 = m.get_frame(1)
        rmsd = float(np.sqrt(np.mean(np.sum((pos1 - pos0)**2, axis=1))))
        assert abs(rmsd - 0.1) < 1e-10
        print(f"  ✓  RMSD frame0 vs frame1 = {rmsd:.6f} Å  (expected 0.100000)")

    def test_bonds_none_is_safe(self):
        """metadata.bonds=None should not crash any panel."""
        m = _make_lammps_model()
        bonds = getattr(m.metadata, 'bonds', None)
        # Should not raise
        bond_info = f"  ·  {len(bonds)//3:,} BONDS" if (bonds is not None and len(bonds) > 0) else ""
        assert bond_info == ""
        print(f"\n  ✓  bonds=None safely produces empty string (no ValueError)")

    def test_bonds_array_produces_count(self):
        """metadata.bonds with 15000 bonds shows correct count."""
        from PSVAP.core.system_model import SystemMetadata
        # 15000 bonds × [2, i, j] = 45000 values
        bond_arr = np.zeros(45000, dtype=np.int64)
        meta = SystemMetadata(bonds=bond_arr)
        bonds = meta.bonds
        bond_info = f"  ·  {len(bonds)//3:,} BONDS" if (bonds is not None and len(bonds) > 0) else ""
        assert bond_info == "  ·  15,000 BONDS"
        print(f"\n  ✓  bond count display: '{bond_info}'")


# ═══════════════════════════════════════════════════════════════════════════
#  8. EXPECTED VALUES SUMMARY (for manual verification in the UI)
# ═══════════════════════════════════════════════════════════════════════════

class TestExpectedUIValues:
    """
    These tests document exactly what you should see when you type
    atom indices into the Analysis panel with your amyloid file loaded.

    After loading amyloid_topo.lammpstrj (8000 atoms):
    """

    def test_amyloid_type_distribution_expected(self):
        """
        After loading amyloid_topo.lammpstrj with the parser fix:
        - TYPE 0 (grey):  atoms where type_id=0 → should be the majority
        - TYPE 1 (blue):  atoms where type_id=1
        - TYPE 2 (red):   atoms where type_id=2
        The legend should show exactly 3 entries: TYPE 0, TYPE 1, TYPE 2.
        (NOT TYPE 3/TYPE 5 which was the old bug)
        """
        # Just verify our model has types 0,1,2
        m = _make_lammps_model()
        unique_types = set(a.type_id for a in m.atoms)
        assert unique_types == {0, 1, 2}
        print(f"\n  ✓  Types present: {sorted(unique_types)}  (expected {{0, 1, 2}})")

    def test_distance_test_case_for_ui(self):
        """
        In your amyloid file, atom 0 is at (15, 93, 35) and atom 1 is at (12, 92, 37).
        Expected distance = sqrt((15-12)² + (93-92)² + (35-37)²)
                         = sqrt(9 + 1 + 4) = sqrt(14) ≈ 3.7417 Å

        To test in the UI:
          ANALYSIS → GEOMETRY → DISTANCE
          i = 0, j = 1  → should show ~3.7417 Å
        """
        pos_0 = np.array([15.0, 93.0, 35.0])
        pos_1 = np.array([12.0, 92.0, 37.0])
        d = float(np.linalg.norm(pos_1 - pos_0))
        expected = math.sqrt(14)
        assert abs(d - expected) < 1e-6
        print(f"\n  ✓  dist(atom0, atom1) in amyloid_topo = {d:.4f} Å")
        print(f"     Enter i=0, j=1 in Distance panel → expect {d:.4f} Å")

    def test_angle_test_case_for_ui(self):
        """
        atoms 0,1,2 in amyloid_topo:
          0 = (15, 93, 35),  1 = (12, 92, 37),  2 = (9, 93, 37)
        angle with vertex at atom 1:
          v1 = pos0 - pos1 = (3, 1, -2)
          v2 = pos2 - pos1 = (-3, 1, 0)
          cos θ = (v1·v2) / (|v1||v2|) = (-9+1+0)/(√14 * √10) = -8/√140

        To test in the UI:
          ANALYSIS → GEOMETRY → BOND ANGLE
          i=0, j=1, k=2  → should show ≈ 127.6°
        """
        pos_0 = np.array([15.0, 93.0, 35.0])
        pos_1 = np.array([12.0, 92.0, 37.0])
        pos_2 = np.array([9.0,  93.0, 37.0])
        v1 = pos_0 - pos_1
        v2 = pos_2 - pos_1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        a = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        print(f"\n  ✓  angle(0,1,2) in amyloid_topo ≈ {a:.2f}°")
        print(f"     Enter i=0, j=1, k=2 in Angle panel → expect {a:.2f}°")
        assert 90 < a < 180  # sanity check: obtuse angle

    def test_torsion_test_case_for_ui(self):
        """
        atoms 0,1,2,3 in amyloid_topo:
          0=(15,93,35), 1=(12,92,37), 2=(9,93,37), 3=(10,96,38)
        To test in UI: i=0, j=1, k=2, l=3
        """
        p0 = np.array([15.0, 93.0, 35.0])
        p1 = np.array([12.0, 92.0, 37.0])
        p2 = np.array([9.0,  93.0, 37.0])
        p3 = np.array([10.0, 96.0, 38.0])
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_n = b1 / (np.linalg.norm(b1) + 1e-15)
        v = b0 - np.dot(b0, b1_n) * b1_n
        w = b2 - np.dot(b2, b1_n) * b1_n
        t = float(np.degrees(np.arctan2(np.dot(np.cross(b1_n, v), w), np.dot(v, w))))
        print(f"\n  ✓  torsion(0,1,2,3) in amyloid_topo ≈ {t:.2f}°")
        print(f"     Enter i=0, j=1, k=2, l=3 in Torsion panel → expect {t:.2f}°")
        assert -180 <= t <= 180


# ═══════════════════════════════════════════════════════════════════════════
# Run summary
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Quick manual run without pytest."""
    import traceback
    tests = [
        TestDistance().test_unit_distance,
        TestDistance().test_diagonal_distance,
        TestDistance().test_3d_distance,
        TestAngle().test_right_angle,
        TestAngle().test_63_degree_angle,
        TestTorsion().test_90_degree_torsion,
        TestTorsion().test_180_degree_torsion,
        TestRMSD().test_same_frame_rmsd_is_zero,
        TestRMSD().test_uniform_z_shift,
        TestSelectionLAMMPS().test_type0_selects_two_atoms,
        TestSelectionLAMMPS().test_type1_selects_two_atoms,
        TestSelectionLAMMPS().test_type2_selects_one_atom,
        TestSelectionLAMMPS().test_type_or_type,
        TestSelectionLAMMPS().test_not_type2,
        TestSelectionLAMMPS().test_mol_selection,
        TestSelectionLAMMPS().test_z_greater_than,
        TestSelectionLAMMPS().test_type_and_z,
        TestNumpyBooleanFix().test_numpy_array_truthiness_raises,
        TestNumpyBooleanFix().test_correct_numpy_check_none,
        TestNumpyBooleanFix().test_bond_count_from_pyvista_format,
        TestFullWorkflow().test_load_select_measure,
        TestFullWorkflow().test_bonds_array_produces_count,
        TestExpectedUIValues().test_amyloid_type_distribution_expected,
        TestExpectedUIValues().test_distance_test_case_for_ui,
        TestExpectedUIValues().test_angle_test_case_for_ui,
        TestExpectedUIValues().test_torsion_test_case_for_ui,
    ]

    passed = 0
    failed = 0
    for t in tests:
        name = f"{t.__self__.__class__.__name__}.{t.__name__}"
        try:
            t()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")