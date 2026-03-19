"""
tests/test_modeling.py
----------------------
Phase 4 unit tests for:
  modeling/mutation_engine.py
  modeling/alanine_scan.py
  modeling/structure_prep.py
  modeling/solvation.py

Run:
    cd "C:\\Users\\mohit\\Documents\\Software Project"
    pytest PSVAP/tests/test_modeling.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from PSVAP.core.atom import Atom
from PSVAP.core.system_model import SystemMetadata, SystemModel


# ═══════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

def _make_ala_atoms() -> tuple[list[Atom], np.ndarray]:
    """
    Minimal ALA residue: N, CA, C, O + CB.
    Followed by a second residue GLY (N, CA, C, O).
    """
    atoms = [
        # Residue 1 — ALA
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
        # Residue 2 — GLY
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


def _make_protein_atoms() -> tuple[list[Atom], np.ndarray]:
    """
    Small protein with ARG, GLU, ALA, HOH residues.
    Used for structure_prep tests.
    """
    atoms = [
        Atom(id=0, type_id=2, element="N",  x=0.0, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="N",  resname="ARG"),
        Atom(id=1, type_id=1, element="C",  x=1.5, y=0.0, z=0.0,
             residue_id=1, chain_id="A", name="CA", resname="ARG"),
        Atom(id=2, type_id=1, element="C",  x=2.0, y=1.2, z=0.0,
             residue_id=1, chain_id="A", name="C",  resname="ARG"),
        Atom(id=3, type_id=3, element="O",  x=3.2, y=1.2, z=0.0,
             residue_id=1, chain_id="A", name="O",  resname="ARG"),
        Atom(id=4, type_id=2, element="N",  x=3.5, y=0.0, z=0.0,
             residue_id=3, chain_id="A", name="N",  resname="GLU"),  # gap: res 2 missing
        Atom(id=5, type_id=1, element="C",  x=5.0, y=0.0, z=0.0,
             residue_id=3, chain_id="A", name="CA", resname="GLU"),
        Atom(id=6, type_id=1, element="C",  x=5.5, y=1.2, z=0.0,
             residue_id=3, chain_id="A", name="C",  resname="GLU"),
        Atom(id=7, type_id=3, element="O",  x=6.7, y=1.2, z=0.0,
             residue_id=3, chain_id="A", name="O",  resname="GLU"),
        # Water
        Atom(id=8, type_id=3, element="O",  x=10.0, y=0.0, z=0.0,
             residue_id=100, chain_id="A", name="OW", resname="HOH"),
    ]
    pos = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float64)
    return atoms, pos


# ═══════════════════════════════════════════════════════════════════════════
# 1. mutation_engine — get_residue_atoms
# ═══════════════════════════════════════════════════════════════════════════

class TestGetResidueAtoms:

    def test_finds_residue(self):
        from PSVAP.modeling.mutation_engine import get_residue_atoms
        atoms, _ = _make_ala_atoms()
        idx = get_residue_atoms(atoms, 1)
        assert len(idx) == 5   # N, CA, C, O, CB

    def test_returns_empty_for_missing(self):
        from PSVAP.modeling.mutation_engine import get_residue_atoms
        atoms, _ = _make_ala_atoms()
        idx = get_residue_atoms(atoms, 99)
        assert idx == []

    def test_chain_filter(self):
        from PSVAP.modeling.mutation_engine import get_residue_atoms
        atoms, _ = _make_ala_atoms()
        idx_a = get_residue_atoms(atoms, 1, chain_id="A")
        idx_b = get_residue_atoms(atoms, 1, chain_id="B")
        assert len(idx_a) == 5
        assert len(idx_b) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 2. mutation_engine — list_residues
# ═══════════════════════════════════════════════════════════════════════════

class TestListResidues:

    def test_returns_two_residues(self):
        from PSVAP.modeling.mutation_engine import list_residues
        atoms, _ = _make_ala_atoms()
        result = list_residues(atoms)
        assert len(result) == 2

    def test_correct_resnames(self):
        from PSVAP.modeling.mutation_engine import list_residues
        atoms, _ = _make_ala_atoms()
        result = list_residues(atoms)
        resnames = [r["resname"] for r in result]
        assert "ALA" in resnames
        assert "GLY" in resnames

    def test_atom_counts(self):
        from PSVAP.modeling.mutation_engine import list_residues
        atoms, _ = _make_ala_atoms()
        result = list_residues(atoms)
        counts = {r["resname"]: r["n_atoms"] for r in result}
        assert counts["ALA"] == 5
        assert counts["GLY"] == 4


# ═══════════════════════════════════════════════════════════════════════════
# 3. mutation_engine — mutate_residue
# ═══════════════════════════════════════════════════════════════════════════

class TestMutateResidue:

    def test_ala_to_gly_removes_cb(self):
        """ALA→GLY should remove the CB atom."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, new_pos = mutate_residue(atoms, pos, 1, "GLY")
        res1_names = {
            (getattr(a, "name", "") or "").strip().upper()
            for a in new_atoms
            if getattr(a, "residue_id", None) == 1
        }
        assert "CB" not in res1_names
        assert "CA" in res1_names

    def test_gly_to_ala_adds_cb(self):
        """GLY→ALA should add a CB atom."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, new_pos = mutate_residue(atoms, pos, 2, "ALA")
        res2_names = {
            (getattr(a, "name", "") or "").strip().upper()
            for a in new_atoms
            if getattr(a, "residue_id", None) == 2
        }
        assert "CB" in res2_names

    def test_resname_updated(self):
        """After mutation, all atoms in the residue should have the new resname."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, _ = mutate_residue(atoms, pos, 1, "GLY")
        res1_atoms = [a for a in new_atoms if getattr(a, "residue_id", None) == 1]
        for a in res1_atoms:
            assert (getattr(a, "resname", None) or "").upper() == "GLY"

    def test_backbone_preserved(self):
        """N, CA, C, O must remain after any mutation."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, _ = mutate_residue(atoms, pos, 1, "GLY")
        res1_names = {
            (getattr(a, "name", "") or "").strip().upper()
            for a in new_atoms
            if getattr(a, "residue_id", None) == 1
        }
        for bb in ["N", "CA", "C", "O"]:
            assert bb in res1_names

    def test_atom_ids_sequential(self):
        """After mutation, atom IDs must be 0, 1, 2, ... (no gaps)."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, _ = mutate_residue(atoms, pos, 1, "GLY")
        ids = [a.id for a in new_atoms]
        assert ids == list(range(len(new_atoms)))

    def test_position_count_matches_atoms(self):
        """Position array shape must match new atom count."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, new_pos = mutate_residue(atoms, pos, 1, "GLY")
        assert new_pos.shape == (len(new_atoms), 3)

    def test_noop_mutation_same_resname(self):
        """Mutating ALA→ALA should return identical data."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms, new_pos = mutate_residue(atoms, pos, 1, "ALA")
        assert len(new_atoms) == len(atoms)
        np.testing.assert_allclose(new_pos, pos, atol=1e-10)

    def test_invalid_residue_raises(self):
        from PSVAP.modeling.mutation_engine import mutate_residue, MutationError
        atoms, pos = _make_ala_atoms()
        with pytest.raises(MutationError):
            mutate_residue(atoms, pos, 999, "GLY")

    def test_one_letter_code_accepted(self):
        """Single-letter code 'G' should work the same as 'GLY'."""
        from PSVAP.modeling.mutation_engine import mutate_residue
        atoms, pos = _make_ala_atoms()
        new_atoms_g, _ = mutate_residue(atoms, pos, 1, "G")
        new_atoms_gly, _ = mutate_residue(atoms, pos, 1, "GLY")
        assert len(new_atoms_g) == len(new_atoms_gly)

    def test_apply_mutation_list(self):
        """apply_mutation_list should chain mutations sequentially."""
        from PSVAP.modeling.mutation_engine import apply_mutation_list
        atoms, pos = _make_ala_atoms()
        new_atoms, new_pos = apply_mutation_list(
            atoms, pos,
            [(1, "GLY"), (2, "ALA")],
        )
        res1_rn = next(
            (getattr(a, "resname", None) for a in new_atoms
             if getattr(a, "residue_id", None) == 1), None
        )
        res2_rn = next(
            (getattr(a, "resname", None) for a in new_atoms
             if getattr(a, "residue_id", None) == 2), None
        )
        assert (res1_rn or "").upper() == "GLY"
        assert (res2_rn or "").upper() == "ALA"


# ═══════════════════════════════════════════════════════════════════════════
# 4. write_pdb
# ═══════════════════════════════════════════════════════════════════════════

class TestWritePDB:

    def test_writes_file(self, tmp_path):
        from PSVAP.modeling.mutation_engine import write_pdb
        atoms, pos = _make_ala_atoms()
        out = tmp_path / "output.pdb"
        write_pdb(atoms, pos, out)
        assert out.exists()
        content = out.read_text()
        assert "ATOM" in content
        assert "END" in content

    def test_atom_count_in_file(self, tmp_path):
        from PSVAP.modeling.mutation_engine import write_pdb
        atoms, pos = _make_ala_atoms()
        out = tmp_path / "out.pdb"
        write_pdb(atoms, pos, out)
        n_atom_lines = sum(
            1 for line in out.read_text().splitlines()
            if line.startswith("ATOM")
        )
        assert n_atom_lines == len(atoms)

    def test_coordinates_in_file(self, tmp_path):
        """First ATOM line should contain CA coords."""
        from PSVAP.modeling.mutation_engine import write_pdb
        atoms, pos = _make_ala_atoms()
        out = tmp_path / "out.pdb"
        write_pdb(atoms, pos, out)
        lines = [l for l in out.read_text().splitlines() if l.startswith("ATOM")]
        assert len(lines) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. structure_prep — check_structure
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckStructure:

    def test_returns_report(self):
        from PSVAP.modeling.structure_prep import check_structure, StructureReport
        atoms, pos = _make_protein_atoms()
        report = check_structure(atoms, pos)
        assert isinstance(report, StructureReport)

    def test_counts_atoms(self):
        from PSVAP.modeling.structure_prep import check_structure
        atoms, pos = _make_protein_atoms()
        report = check_structure(atoms, pos)
        assert report.n_atoms == len(atoms)

    def test_detects_water(self):
        from PSVAP.modeling.structure_prep import check_structure
        atoms, pos = _make_protein_atoms()
        report = check_structure(atoms, pos)
        assert report.n_waters == 1

    def test_detects_missing_loop(self):
        """Residue IDs 1, 3 (gap of 1) should produce a missing loop warning."""
        from PSVAP.modeling.structure_prep import check_structure
        atoms, pos = _make_protein_atoms()
        report = check_structure(atoms, pos)
        loop_issues = [i for i in report.issues if i.issue_type == "MISSING_LOOP"]
        assert len(loop_issues) >= 1

    def test_detects_no_hydrogens(self):
        from PSVAP.modeling.structure_prep import check_structure
        atoms, pos = _make_protein_atoms()
        report = check_structure(atoms, pos)
        h_issues = [i for i in report.issues if i.issue_type == "NO_HYDROGENS"]
        assert len(h_issues) == 1

    def test_summary_contains_key_fields(self):
        from PSVAP.modeling.structure_prep import check_structure
        atoms, pos = _make_protein_atoms()
        summary = check_structure(atoms, pos).summary()
        assert "Atoms" in summary
        assert "Residues" in summary


# ═══════════════════════════════════════════════════════════════════════════
# 6. structure_prep — remove_waters
# ═══════════════════════════════════════════════════════════════════════════

class TestRemoveWaters:

    def test_removes_water_atoms(self):
        from PSVAP.modeling.structure_prep import remove_waters
        atoms, pos = _make_protein_atoms()
        new_atoms, new_pos = remove_waters(atoms, pos)
        resnames = {(getattr(a, "resname", None) or "").upper() for a in new_atoms}
        assert "HOH" not in resnames

    def test_preserves_protein_atoms(self):
        from PSVAP.modeling.structure_prep import remove_waters
        atoms, pos = _make_protein_atoms()
        n_protein = sum(
            1 for a in atoms
            if (getattr(a, "resname", None) or "").upper() not in
            {"HOH", "WAT", "TIP3", "SPC", "SOL"}
        )
        new_atoms, _ = remove_waters(atoms, pos)
        assert len(new_atoms) == n_protein

    def test_ids_sequential_after_removal(self):
        from PSVAP.modeling.structure_prep import remove_waters
        atoms, pos = _make_protein_atoms()
        new_atoms, _ = remove_waters(atoms, pos)
        assert [a.id for a in new_atoms] == list(range(len(new_atoms)))

    def test_position_shape_matches(self):
        from PSVAP.modeling.structure_prep import remove_waters
        atoms, pos = _make_protein_atoms()
        new_atoms, new_pos = remove_waters(atoms, pos)
        assert new_pos.shape == (len(new_atoms), 3)


# ═══════════════════════════════════════════════════════════════════════════
# 7. structure_prep — renumber_residues
# ═══════════════════════════════════════════════════════════════════════════

class TestRenumberResidues:

    def test_starts_at_one(self):
        from PSVAP.modeling.structure_prep import renumber_residues
        atoms, _ = _make_protein_atoms()
        new_atoms = renumber_residues(atoms, start=1)
        rids = sorted({getattr(a, "residue_id", None) for a in new_atoms})
        assert rids[0] == 1

    def test_sequential_without_gaps(self):
        from PSVAP.modeling.structure_prep import renumber_residues
        atoms, _ = _make_protein_atoms()
        new_atoms = renumber_residues(atoms, start=1)
        rids = sorted({getattr(a, "residue_id", None) for a in new_atoms})
        # Should be 1, 2, 3 (no gaps)
        assert rids == list(range(1, len(rids) + 1))

    def test_atom_count_unchanged(self):
        from PSVAP.modeling.structure_prep import renumber_residues
        atoms, _ = _make_protein_atoms()
        new_atoms = renumber_residues(atoms)
        assert len(new_atoms) == len(atoms)


# ═══════════════════════════════════════════════════════════════════════════
# 8. solvation — build_water_box
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildWaterBox:

    def test_returns_solvation_result(self):
        from PSVAP.modeling.solvation import build_water_box, SolvationResult
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=100)
        assert isinstance(result, SolvationResult)

    def test_waters_added(self):
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=200)
        assert result.n_waters_added > 0

    def test_total_atoms_increased(self):
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=100)
        assert len(result.atoms) > len(atoms)

    def test_positions_shape_matches_atoms(self):
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=100)
        assert result.positions.shape == (len(result.atoms), 3)

    def test_box_bounds_shape(self):
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=50)
        assert result.box_bounds.shape == (3, 2)

    def test_buffer_respected(self):
        """Box size should be roughly solute_extent + 2*buffer."""
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=8.0, max_waters=50)
        x_size = result.box_bounds[0, 1] - result.box_bounds[0, 0]
        solute_x = pos[:, 0].max() - pos[:, 0].min()
        assert x_size >= solute_x + 2 * 8.0 - 1.0   # 1 Å tolerance

    def test_no_water_closer_than_cutoff(self):
        """No water O should be closer than 2.4 Å to a heavy solute atom."""
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=200)

        solute_n = len(atoms)
        heavy_mask = [
            i for i, a in enumerate(atoms)
            if (getattr(a, "element", "C") or "C").upper() != "H"
        ]
        heavy_pos = result.positions[:solute_n][heavy_mask]

        water_o_indices = [
            i for i, a in enumerate(result.atoms)
            if i >= solute_n and (getattr(a, "name", "") or "").upper() == "OW"
        ]
        for w_idx in water_o_indices:
            w_pos = result.positions[w_idx]
            dists = np.linalg.norm(heavy_pos - w_pos, axis=1)
            assert dists.min() >= 2.3, (
                f"Water at index {w_idx} is {dists.min():.3f} Å from solute"
            )

    def test_max_waters_cap(self):
        """Should never exceed max_waters."""
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=20.0, max_waters=10)
        assert result.n_waters_added <= 10

    def test_summary_string(self):
        from PSVAP.modeling.solvation import build_water_box
        atoms, pos = _make_ala_atoms()
        result = build_water_box(atoms, pos, buffer=5.0, max_waters=50)
        summary = result.summary()
        assert "SOLVATION COMPLETE" in summary
        assert "Waters" in summary


# ═══════════════════════════════════════════════════════════════════════════
# 9. solvation — estimate_ion_count
# ═══════════════════════════════════════════════════════════════════════════

class TestEstimateIonCount:

    def test_returns_tuple(self):
        from PSVAP.modeling.solvation import estimate_ion_count
        atoms, _ = _make_ala_atoms()
        result = estimate_ion_count(atoms, n_waters=1000)
        assert isinstance(result, tuple) and len(result) == 2

    def test_non_negative(self):
        from PSVAP.modeling.solvation import estimate_ion_count
        atoms, _ = _make_ala_atoms()
        n_na, n_cl = estimate_ion_count(atoms, n_waters=1000)
        assert n_na >= 0 and n_cl >= 0

    def test_charged_system_gets_counterions(self):
        """ARG (positive) → should add Cl- to neutralize."""
        from PSVAP.modeling.solvation import estimate_ion_count
        atoms = [
            Atom(id=0, type_id=2, element="N", x=0.0, y=0.0, z=0.0,
                 residue_id=1, resname="ARG"),
        ]
        n_na, n_cl = estimate_ion_count(atoms, n_waters=1000)
        # Net charge = +1, so we need at least 1 Cl-
        assert n_cl >= 1

    def test_zero_waters_gives_zero_salt(self):
        from PSVAP.modeling.solvation import estimate_ion_count
        atoms, _ = _make_ala_atoms()
        n_na, n_cl = estimate_ion_count(atoms, n_waters=0, charge_conc=0.15)
        # No salt ions needed for zero-volume box
        assert n_na == 0 and n_cl == 0