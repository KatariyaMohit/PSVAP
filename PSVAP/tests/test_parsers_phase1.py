"""
tests/test_parsers_phase1.py
----------------------------
Phase 1 parser tests — expectations match the actual fixture files:
  sample.pdb  : 150 atoms per model, 2 models, chains A+B
  sample.gro  : 120 atoms (30 SPC/E water molecules)
  sample.xyz  : 90 atoms (30 water molecules), 5 frames
  sample.cif  : 100 atoms (20 residues × 5 backbone)
  sample.sdf  : 24 atoms (caffeine molecule)
  sample.lammpstrj : existing LAMMPS fixture (variable)

Run with:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_parsers_phase1.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDB       = FIXTURES / "sample.pdb"
SAMPLE_GRO       = FIXTURES / "sample.gro"
SAMPLE_XYZ       = FIXTURES / "sample.xyz"
SAMPLE_SDF       = FIXTURES / "sample.sdf"
SAMPLE_CIF       = FIXTURES / "sample.cif"
SAMPLE_LAMMPSTRJ = FIXTURES / "sample.lammpstrj"


# ── RDKit guard ────────────────────────────────────────────────────────────
def _rdkit_ok() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False

rdkit_available = pytest.mark.skipif(
    not _rdkit_ok(),
    reason="RDKit unavailable — run: conda install -c conda-forge rdkit --force-reinstall"
)


# ═══════════════════════════════════════════════════════════════════════════
#  Contract helper
# ═══════════════════════════════════════════════════════════════════════════

def assert_parse_contract(atoms, trajectory, metadata, *, n_atoms: int, n_frames: int):
    assert isinstance(atoms, list)
    assert isinstance(trajectory, list)
    assert len(atoms) == n_atoms, f"Expected {n_atoms} atoms, got {len(atoms)}"
    assert len(trajectory) == n_frames, f"Expected {n_frames} frames, got {len(trajectory)}"
    for i, frame in enumerate(trajectory):
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.float64
        assert frame.shape == (n_atoms, 3), f"Frame {i}: expected ({n_atoms},3) got {frame.shape}"
    assert metadata is not None
    assert metadata.source_path is not None
    assert isinstance(metadata.timesteps, list)
    assert len(metadata.timesteps) == n_frames


# ═══════════════════════════════════════════════════════════════════════════
#  detect_parser routing
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectParser:

    @pytest.mark.parametrize("ext,expected_class", [
        (".lammpstrj", "LammpsParser"), (".traj", "LammpsParser"),
        (".data", "LammpsParser"),      (".lammps", "LammpsParser"),
        (".gro", "GromacsParser"),      (".xtc", "GromacsParser"),
        (".trr", "GromacsParser"),      (".pdb", "PDBParser"),
        (".cif", "MMCIFParser"),        (".mmcif", "MMCIFParser"),
        (".nc", "AmberParser"),         (".ncdf", "AmberParser"),
        (".mdcrd", "AmberParser"),      (".rst7", "AmberParser"),
        (".dcd", "DCDParser"),          (".xyz", "XYZParser"),
        (".mol2", "MolParser"),         (".sdf", "MolParser"),
        (".mol", "MolParser"),
    ])
    def test_correct_type(self, ext, expected_class, tmp_path):
        from PSVAP.io.base_parser import detect_parser
        parser = detect_parser(tmp_path / f"dummy{ext}")
        assert type(parser).__name__ == expected_class

    def test_unknown_raises(self, tmp_path):
        from PSVAP.io.base_parser import detect_parser
        with pytest.raises(ValueError, match="Unsupported file format"):
            detect_parser(tmp_path / "mystery.abc123")


# ═══════════════════════════════════════════════════════════════════════════
#  XYZ Parser  (90 atoms, 5 frames — sample.xyz)
# ═══════════════════════════════════════════════════════════════════════════

class TestXYZParser:
    """sample.xyz: 90 atoms (30×OHH water), 5 frames, timesteps 0,100,200,300,400"""

    def test_parse_counts(self):
        from PSVAP.io.xyz_parser import XYZParser
        atoms, traj, meta = XYZParser().parse(SAMPLE_XYZ)
        assert_parse_contract(atoms, traj, meta, n_atoms=90, n_frames=5)

    def test_first_atom_is_oxygen(self):
        from PSVAP.io.xyz_parser import XYZParser
        atoms, _, _ = XYZParser().parse(SAMPLE_XYZ)
        assert atoms[0].element == "O"
        assert atoms[1].element == "H"
        assert atoms[2].element == "H"

    def test_timesteps_are_multiples_of_100(self):
        from PSVAP.io.xyz_parser import XYZParser
        _, _, meta = XYZParser().parse(SAMPLE_XYZ)
        assert meta.timesteps == [0, 100, 200, 300, 400]

    def test_first_frame_first_atom_coords(self):
        """Frame 0 is noise-free — atom 0 (O) should be near (1.0, 1.0, 1.0)."""
        from PSVAP.io.xyz_parser import XYZParser
        _, traj, _ = XYZParser().parse(SAMPLE_XYZ)
        np.testing.assert_allclose(traj[0][0], [1.0, 1.0, 1.0], atol=0.01)

    def test_type_id_oxygen(self):
        from PSVAP.io.xyz_parser import XYZParser
        atoms, _, _ = XYZParser().parse(SAMPLE_XYZ)
        assert atoms[0].type_id == 3   # O
        assert atoms[1].type_id == 5   # H

    def test_trajectory_coords_are_float64(self):
        from PSVAP.io.xyz_parser import XYZParser
        _, traj, _ = XYZParser().parse(SAMPLE_XYZ)
        for frame in traj:
            assert frame.dtype == np.float64

    def test_single_frame(self, tmp_path):
        from PSVAP.io.xyz_parser import XYZParser
        p = tmp_path / "mol.xyz"
        p.write_text("2\nwater\nH  0.0 0.0 0.0\nO  1.0 0.0 0.0\n", encoding="utf-8")
        atoms, traj, meta = XYZParser().parse(p)
        assert_parse_contract(atoms, traj, meta, n_atoms=2, n_frames=1)

    def test_extended_xyz_lattice(self, tmp_path):
        from PSVAP.io.xyz_parser import XYZParser
        content = '2\nLattice="10.0 0.0 0.0 0.0 12.0 0.0 0.0 0.0 15.0"\nC 0 0 0\nN 1 1 1\n'
        p = tmp_path / "ext.xyz"
        p.write_text(content, encoding="utf-8")
        _, _, meta = XYZParser().parse(p)
        assert meta.box_bounds is not None
        assert meta.box_bounds[0, 1] == pytest.approx(10.0)

    def test_bad_coordinate_raises(self, tmp_path):
        from PSVAP.io.xyz_parser import XYZParser, XYZFormatError
        p = tmp_path / "bad.xyz"
        p.write_text("1\nc\nC  bad 0 0\n", encoding="utf-8")
        with pytest.raises(XYZFormatError, match="Bad coordinate"):
            XYZParser().parse(p)

    def test_empty_raises(self, tmp_path):
        from PSVAP.io.xyz_parser import XYZParser, XYZFormatError
        p = tmp_path / "empty.xyz"
        p.write_text("", encoding="utf-8")
        with pytest.raises(XYZFormatError, match="No frames"):
            XYZParser().parse(p)

    def test_inconsistent_atom_count_raises(self, tmp_path):
        from PSVAP.io.xyz_parser import XYZParser, XYZFormatError
        content = "2\nf1\nC 0 0 0\nN 1 1 1\n3\nf2\nC 0 0 0\nN 1 1 1\nO 2 2 2\n"
        p = tmp_path / "bad.xyz"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(XYZFormatError, match="atom count"):
            XYZParser().parse(p)


# ═══════════════════════════════════════════════════════════════════════════
#  PDB Parser  (150 atoms per model, 2 models — sample.pdb)
# ═══════════════════════════════════════════════════════════════════════════

class TestPDBParser:
    """sample.pdb: 150 atoms (chains A+B × 15 res × 5 backbone), 2 NMR models."""

    def test_parse_counts(self):
        from PSVAP.io.pdb_parser import PDBParser
        atoms, traj, meta = PDBParser().parse(SAMPLE_PDB)
        assert_parse_contract(atoms, traj, meta, n_atoms=150, n_frames=2)

    def test_first_atom_is_nitrogen(self):
        from PSVAP.io.pdb_parser import PDBParser
        atoms, _, _ = PDBParser().parse(SAMPLE_PDB)
        assert atoms[0].element == "N"
        assert atoms[0].name == "N"

    def test_chain_ids_present(self):
        from PSVAP.io.pdb_parser import PDBParser
        atoms, _, _ = PDBParser().parse(SAMPLE_PDB)
        chains = {a.chain_id for a in atoms}
        assert "A" in chains
        assert "B" in chains

    def test_cryst1_box_50A(self):
        from PSVAP.io.pdb_parser import PDBParser
        _, _, meta = PDBParser().parse(SAMPLE_PDB)
        assert meta.box_bounds is not None
        np.testing.assert_allclose(meta.box_bounds[:, 1], [50.0, 50.0, 50.0], atol=0.1)

    def test_two_models_different_coords(self):
        """Model 2 has z_shift=0.2 so coords differ between frames."""
        from PSVAP.io.pdb_parser import PDBParser
        _, traj, _ = PDBParser().parse(SAMPLE_PDB)
        assert traj[0].shape == traj[1].shape
        diff = np.abs(traj[1] - traj[0]).max()
        assert diff > 0.1  # model 2 shifted by 0.2 Å

    def test_type_id_n_is_2(self):
        from PSVAP.io.pdb_parser import PDBParser
        atoms, _, _ = PDBParser().parse(SAMPLE_PDB)
        assert atoms[0].type_id == 2  # N

    def test_residue_id_starts_at_1(self):
        from PSVAP.io.pdb_parser import PDBParser
        atoms, _, _ = PDBParser().parse(SAMPLE_PDB)
        assert atoms[0].residue_id == 1

    def test_single_model(self, tmp_path):
        from PSVAP.io.pdb_parser import PDBParser
        content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000"
            "  1.00  0.00           N\nEND\n"
        )
        p = tmp_path / "s.pdb"
        p.write_text(content)
        atoms, traj, meta = PDBParser().parse(p)
        assert len(atoms) == 1 and len(traj) == 1

    def test_empty_raises(self, tmp_path):
        from PSVAP.io.pdb_parser import PDBParser, PDBFormatError
        p = tmp_path / "e.pdb"
        p.write_text("REMARK nothing\nEND\n")
        with pytest.raises(PDBFormatError):
            PDBParser().parse(p)

    def test_source_path(self):
        from PSVAP.io.pdb_parser import PDBParser
        _, _, meta = PDBParser().parse(SAMPLE_PDB)
        assert meta.source_path == SAMPLE_PDB


# ═══════════════════════════════════════════════════════════════════════════
#  GROMACS Parser  (120 atoms — sample.gro)
# ═══════════════════════════════════════════════════════════════════════════

class TestGromacsParser:
    """sample.gro: 30 SPC/E water molecules × 4 atoms = 120 atoms."""

    def test_parse_counts(self):
        from PSVAP.io.gromacs_parser import GromacsParser
        atoms, traj, meta = GromacsParser().parse(SAMPLE_GRO)
        assert_parse_contract(atoms, traj, meta, n_atoms=120, n_frames=1)

    def test_coords_in_angstrom(self):
        """GRO stores nm; MDAnalysis converts to Å. First OW at 1.2 nm = 12.0 Å."""
        from PSVAP.io.gromacs_parser import GromacsParser
        _, traj, _ = GromacsParser().parse(SAMPLE_GRO)
        # First atom is at x=0.3+0*0.9=0.3 nm = 3.0 Å (residue 1, x=1.2 nm = 12.0 Å)
        # Just verify coords are in Angstrom range (not nm), i.e. > 1.0
        assert traj[0][0, 0] > 1.0

    def test_box_bounds_in_angstrom(self):
        """Box = 5.0 nm = 50.0 Å."""
        from PSVAP.io.gromacs_parser import GromacsParser
        _, _, meta = GromacsParser().parse(SAMPLE_GRO)
        assert meta.box_bounds is not None
        np.testing.assert_allclose(meta.box_bounds[:, 1], [50.0, 50.0, 50.0], atol=0.5)

    def test_atom_names_present(self):
        from PSVAP.io.gromacs_parser import GromacsParser
        atoms, _, _ = GromacsParser().parse(SAMPLE_GRO)
        names = {a.name for a in atoms if a.name}
        assert "OW" in names

    def test_wrong_ext_raises(self, tmp_path):
        from PSVAP.io.gromacs_parser import GromacsParser, GromacsFormatError
        p = tmp_path / "wrong.abc"
        p.write_text("nothing")
        with pytest.raises(GromacsFormatError, match="Unsupported GROMACS"):
            GromacsParser().parse(p)

    def test_source_path(self):
        from PSVAP.io.gromacs_parser import GromacsParser
        _, _, meta = GromacsParser().parse(SAMPLE_GRO)
        assert meta.source_path == SAMPLE_GRO


# ═══════════════════════════════════════════════════════════════════════════
#  mmCIF Parser  (100 atoms — sample.cif)
# ═══════════════════════════════════════════════════════════════════════════

class TestMMCIFParser:
    """sample.cif: 100 atoms (20 residues × 5 backbone atoms)."""

    def test_parse_counts(self):
        from PSVAP.io.mmcif_parser import MMCIFParser
        atoms, traj, meta = MMCIFParser().parse(SAMPLE_CIF)
        assert_parse_contract(atoms, traj, meta, n_atoms=100, n_frames=1)

    def test_first_atom_nitrogen(self):
        from PSVAP.io.mmcif_parser import MMCIFParser
        atoms, _, _ = MMCIFParser().parse(SAMPLE_CIF)
        assert atoms[0].element == "N"

    def test_box_bounds_50A(self):
        from PSVAP.io.mmcif_parser import MMCIFParser
        _, _, meta = MMCIFParser().parse(SAMPLE_CIF)
        assert meta.box_bounds is not None
        np.testing.assert_allclose(meta.box_bounds[:, 1], [50.0, 50.0, 50.0], atol=1e-3)

    def test_chain_A(self):
        from PSVAP.io.mmcif_parser import MMCIFParser
        atoms, _, _ = MMCIFParser().parse(SAMPLE_CIF)
        assert all(a.chain_id == "A" for a in atoms)

    def test_inline_fixture(self, tmp_path):
        """Small inline CIF still works."""
        content = """\
data_TEST
_cell.length_a   20.000
_cell.length_b   20.000
_cell.length_c   20.000
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.label_alt_id
1 N N ALA A 1 1.000 1.000 1.000 .
2 C CA ALA A 1 2.000 1.500 1.000 .
3 C C  ALA A 1 3.000 1.000 1.500 .
"""
        p = tmp_path / "mini.cif"
        p.write_text(content)
        from PSVAP.io.mmcif_parser import MMCIFParser
        atoms, traj, meta = MMCIFParser().parse(p)
        assert_parse_contract(atoms, traj, meta, n_atoms=3, n_frames=1)
        np.testing.assert_allclose(traj[0][0], [1.0, 1.0, 1.0], atol=1e-6)

    def test_empty_raises(self, tmp_path):
        from PSVAP.io.mmcif_parser import MMCIFParser, MMCIFFormatError
        p = tmp_path / "e.cif"
        p.write_text("data_EMPTY\n")
        with pytest.raises(MMCIFFormatError):
            MMCIFParser().parse(p)


# ═══════════════════════════════════════════════════════════════════════════
#  SDF Parser  (24 atoms — caffeine)
# ═══════════════════════════════════════════════════════════════════════════

class TestMolParser:
    """sample.sdf: caffeine molecule, 24 atoms (14 heavy + 10 H)."""

    @rdkit_available
    def test_parse_counts(self):
        from PSVAP.io.mol_parser import MolParser
        atoms, traj, meta = MolParser().parse(SAMPLE_SDF)
        assert_parse_contract(atoms, traj, meta, n_atoms=24, n_frames=1)

    @rdkit_available
    def test_caffeine_elements(self):
        from PSVAP.io.mol_parser import MolParser
        atoms, _, _ = MolParser().parse(SAMPLE_SDF)
        elements = [a.element for a in atoms]
        assert "N" in elements
        assert "C" in elements
        assert "O" in elements
        assert "H" in elements

    @rdkit_available
    def test_first_atom_coords_reasonable(self):
        """Caffeine atom 0 (N) should be around (1.2, 0.76, 0.0)."""
        from PSVAP.io.mol_parser import MolParser
        _, traj, _ = MolParser().parse(SAMPLE_SDF)
        np.testing.assert_allclose(traj[0][0], [1.2124, 0.7629, 0.0], atol=1e-3)

    @rdkit_available
    def test_type_ids_assigned(self):
        from PSVAP.io.mol_parser import MolParser
        atoms, _, _ = MolParser().parse(SAMPLE_SDF)
        type_map = {a.element: a.type_id for a in atoms}
        assert type_map.get("N") == 2
        assert type_map.get("C") == 1
        assert type_map.get("O") == 3

    @rdkit_available
    def test_multi_molecule_sdf(self, tmp_path):
        """Two copies of caffeine → 2 frames, 24 atoms each."""
        from PSVAP.io.mol_parser import MolParser
        two = SAMPLE_SDF.read_text() + SAMPLE_SDF.read_text()
        p = tmp_path / "double.sdf"
        p.write_text(two)
        atoms, traj, meta = MolParser().parse(p)
        assert len(atoms) == 24
        assert len(traj) == 2

    def test_wrong_extension_raises(self, tmp_path):
        from PSVAP.io.mol_parser import MolParser, MolFormatError
        p = tmp_path / "wrong.txt"
        p.write_text("nothing")
        with pytest.raises(MolFormatError, match="Unsupported small molecule"):
            MolParser().parse(p)


# ═══════════════════════════════════════════════════════════════════════════
#  LAMMPS regression
# ═══════════════════════════════════════════════════════════════════════════

class TestLammpsRegression:

    def test_sample_loads(self):
        if not SAMPLE_LAMMPSTRJ.exists():
            pytest.skip("sample.lammpstrj not present")
        from PSVAP.io.lammps_parser import LammpsParser
        atoms, traj, meta = LammpsParser().parse(SAMPLE_LAMMPSTRJ)
        assert len(atoms) > 0 and len(traj) > 0

    def test_detect_returns_lammps(self, tmp_path):
        from PSVAP.io.base_parser import detect_parser
        from PSVAP.io.lammps_parser import LammpsParser
        assert isinstance(detect_parser(tmp_path / "r.lammpstrj"), LammpsParser)


# ═══════════════════════════════════════════════════════════════════════════
#  Atom contract — all parsers must return Atom with required fields
# ═══════════════════════════════════════════════════════════════════════════

class TestAtomContract:

    @pytest.mark.parametrize("parser_name,fixture_path", [
        ("xyz", str(SAMPLE_XYZ)),
        ("pdb", str(SAMPLE_PDB)),
        ("gro", str(SAMPLE_GRO)),
        ("cif", str(SAMPLE_CIF)),
    ])
    def test_required_fields(self, parser_name, fixture_path):
        fixture = Path(fixture_path)
        if parser_name == "xyz":
            from PSVAP.io.xyz_parser import XYZParser as P
        elif parser_name == "pdb":
            from PSVAP.io.pdb_parser import PDBParser as P
        elif parser_name == "gro":
            from PSVAP.io.gromacs_parser import GromacsParser as P
        else:
            from PSVAP.io.mmcif_parser import MMCIFParser as P

        atoms, traj, _ = P().parse(fixture)
        assert len(atoms) > 0
        assert len(traj) > 0
        for atom in atoms:
            assert isinstance(atom.id, int)
            assert isinstance(atom.x, float)
            assert isinstance(atom.y, float)
            assert isinstance(atom.z, float)

    @rdkit_available
    def test_sdf_required_fields(self):
        from PSVAP.io.mol_parser import MolParser
        atoms, _, _ = MolParser().parse(SAMPLE_SDF)
        for atom in atoms:
            assert isinstance(atom.id, int)
            assert isinstance(atom.x, float)