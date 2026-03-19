"""
modeling/docking_engine.py
--------------------------
Feature 13: Molecular Docking via AutoDock Vina.

Handles the full docking workflow:
  1. Receptor preparation (remove waters, add charges — via Meeko or manual)
  2. Ligand preparation (from SMILES or SDF — via RDKit + Meeko)
  3. Docking box definition (center + size in Å)
  4. Vina subprocess execution
  5. Pose parsing and scoring
  6. Interaction analysis on best pose

Public API
----------
  DockingConfig (dataclass)  — all docking parameters

  prepare_receptor_pdbqt(pdb_path, output_path)
      → Path  (PDBQT file)

  prepare_ligand_pdbqt(smiles_or_sdf, output_path)
      → Path  (PDBQT file)

  run_vina(config)
      → DockingResult

  DockingResult (dataclass)
      poses, scores, best_pose_positions, config
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DockingConfig:
    """All parameters needed to run AutoDock Vina."""
    receptor_pdbqt:   Path | str
    ligand_pdbqt:     Path | str
    center_x:         float          # Å
    center_y:         float
    center_z:         float
    size_x:           float = 20.0   # box size in Å
    size_y:           float = 20.0
    size_z:           float = 20.0
    exhaustiveness:   int   = 8
    n_poses:          int   = 9
    energy_range:     float = 3.0    # kcal/mol window for poses
    seed:             int   = 42
    vina_executable:  str   = "vina"
    timeout_seconds:  int   = 300


@dataclass
class DockingPose:
    """A single docking pose."""
    rank:       int
    score:      float          # binding affinity in kcal/mol
    rmsd_lb:    float          # RMSD lower bound vs best pose
    rmsd_ub:    float          # RMSD upper bound vs best pose
    positions:  np.ndarray     # (N_ligand_atoms, 3)


@dataclass
class DockingResult:
    """Result of a docking run."""
    poses:              list[DockingPose]
    config:             DockingConfig
    output_pdbqt:       str = ""     # raw PDBQT output text
    error_message:      str = ""
    success:            bool = True

    @property
    def best_score(self) -> float | None:
        return self.poses[0].score if self.poses else None

    @property
    def best_positions(self) -> np.ndarray | None:
        return self.poses[0].positions if self.poses else None

    def summary(self) -> str:
        if not self.success:
            return f"DOCKING FAILED\n{self.error_message}"
        if not self.poses:
            return "DOCKING COMPLETED — NO POSES RETURNED"
        lines = [
            f"DOCKING RESULTS  ({len(self.poses)} poses)\n",
            f"{'RANK':>5}  {'SCORE (kcal/mol)':>18}  "
            f"{'RMSD LB':>8}  {'RMSD UB':>8}",
            "-" * 44,
        ]
        for p in self.poses:
            lines.append(
                f"{p.rank:>5}  {p.score:>18.2f}  "
                f"{p.rmsd_lb:>8.3f}  {p.rmsd_ub:>8.3f}"
            )
        lines.append(f"\nBEST SCORE: {self.poses[0].score:.2f} kcal/mol")
        return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────

def check_vina_available(executable: str = "vina") -> bool:
    """Return True if AutoDock Vina is found in PATH."""
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True, timeout=10,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def prepare_receptor_pdbqt(
    pdb_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Convert a PDB receptor file to PDBQT format for Vina.

    Strategy:
      1. Try Meeko (best quality, adds Gasteiger charges).
      2. Fall back to manual PDBQT conversion (strip waters,
         add TORSDOF 0 header, convert ATOM records).

    Parameters
    ----------
    pdb_path    : path to receptor PDB file
    output_path : where to write .pdbqt (None = same dir, .pdbqt extension)

    Returns
    -------
    Path to the .pdbqt file

    Raises
    ------
    FileNotFoundError if pdb_path does not exist
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {pdb_path}")

    out = Path(output_path) if output_path else pdb_path.with_suffix(".pdbqt")

    # Try Meeko first
    try:
        import meeko
        _prepare_receptor_meeko(pdb_path, out)
        return out
    except ImportError:
        pass

    # Manual PDBQT conversion
    _prepare_receptor_manual(pdb_path, out)
    return out


def prepare_ligand_pdbqt(
    smiles_or_sdf: str,
    output_path: str | Path | None = None,
    n_conformers: int = 1,
) -> Path:
    """
    Prepare a ligand in PDBQT format from a SMILES string or SDF file path.

    Strategy:
      1. If input is a file path ending in .sdf → read from SDF.
      2. Otherwise treat as SMILES.
      3. Generate 3D conformer with RDKit ETKDG.
      4. Try Meeko for PDBQT conversion; fall back to manual.

    Returns
    -------
    Path to the .pdbqt file

    Raises
    ------
    ImportError if RDKit not installed
    ValueError  if SMILES cannot be parsed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError(
            "RDKit is required for ligand preparation.\n"
            "Install with: conda install -c conda-forge rdkit"
        )

    # Determine if input is a file or SMILES
    if smiles_or_sdf.endswith(".sdf") and Path(smiles_or_sdf).exists():
        supplier = Chem.SDMolSupplier(smiles_or_sdf)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None:
            raise ValueError(f"No valid molecules in SDF: {smiles_or_sdf}")
        smiles = Chem.MolToSmiles(mol)
    else:
        smiles = smiles_or_sdf.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: '{smiles}'")

    # Generate 3D conformer
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
    except AttributeError:
        params = AllChem.ETKDG()
        params.randomSeed = 42

    if AllChem.EmbedMolecule(mol, params) < 0:
        raise ValueError(f"Could not embed 3D conformer for: '{smiles}'")

    AllChem.MMFFOptimizeMolecule(mol)

    # Write to temporary SDF
    out = Path(output_path) if output_path else Path(
        tempfile.mktemp(suffix=".pdbqt")
    )

    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        tmp_sdf = Path(tmp.name)
    writer = Chem.SDWriter(str(tmp_sdf))
    writer.write(mol)
    writer.close()

    # Try Meeko for PDBQT conversion
    try:
        import meeko
        _prepare_ligand_meeko(tmp_sdf, out)
    except ImportError:
        _prepare_ligand_manual(mol, out)

    try:
        tmp_sdf.unlink()
    except Exception:
        pass

    return out


def run_vina(config: DockingConfig) -> DockingResult:
    """
    Execute AutoDock Vina with the given configuration.

    Parameters
    ----------
    config : DockingConfig with all docking parameters

    Returns
    -------
    DockingResult with poses, scores, and raw output
    """
    if not check_vina_available(config.vina_executable):
        return DockingResult(
            poses=[],
            config=config,
            success=False,
            error_message=(
                f"AutoDock Vina not found at '{config.vina_executable}'.\n"
                "Download from: https://vina.scripps.edu/downloads/\n"
                "Ensure 'vina' is in your system PATH."
            ),
        )

    # Write output to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as tmp:
        output_pdbqt = Path(tmp.name)

    cmd = [
        config.vina_executable,
        "--receptor",     str(config.receptor_pdbqt),
        "--ligand",       str(config.ligand_pdbqt),
        "--center_x",     str(config.center_x),
        "--center_y",     str(config.center_y),
        "--center_z",     str(config.center_z),
        "--size_x",       str(config.size_x),
        "--size_y",       str(config.size_y),
        "--size_z",       str(config.size_z),
        "--exhaustiveness", str(config.exhaustiveness),
        "--num_modes",    str(config.n_poses),
        "--energy_range", str(config.energy_range),
        "--seed",         str(config.seed),
        "--out",          str(output_pdbqt),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=config.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return DockingResult(
            poses=[], config=config, success=False,
            error_message=f"Vina timed out after {config.timeout_seconds} seconds.",
        )
    except Exception as exc:
        return DockingResult(
            poses=[], config=config, success=False,
            error_message=str(exc),
        )

    if proc.returncode != 0:
        return DockingResult(
            poses=[], config=config, success=False,
            error_message=f"Vina failed (exit {proc.returncode}):\n{proc.stderr[:800]}",
        )

    # Parse output
    try:
        raw_output = output_pdbqt.read_text(errors="replace")
    except Exception:
        raw_output = ""

    poses = _parse_vina_output_pdbqt(raw_output)

    # Parse scores from stdout
    scores = _parse_vina_stdout_scores(proc.stdout)
    for i, pose in enumerate(poses):
        if i < len(scores):
            pose.score    = scores[i][0]
            pose.rmsd_lb  = scores[i][1]
            pose.rmsd_ub  = scores[i][2]

    try:
        output_pdbqt.unlink()
    except Exception:
        pass

    return DockingResult(
        poses=poses,
        config=config,
        output_pdbqt=raw_output,
        success=True,
    )


def docking_box_from_selection(
    atoms: list,
    positions: np.ndarray,
    selection_indices: list[int],
    padding: float = 5.0,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute docking box center and size from a selection of atoms.

    Parameters
    ----------
    atoms, positions   : loaded structure
    selection_indices  : atom indices defining the binding site
    padding            : extra space around selection in Å

    Returns
    -------
    (center_x, center_y, center_z, size_x, size_y, size_z)
    """
    pos = np.asarray(positions, dtype=float)
    if not selection_indices:
        center = pos.mean(axis=0)
        size   = np.array([20.0, 20.0, 20.0])
    else:
        sel_pos = pos[selection_indices]
        center  = sel_pos.mean(axis=0)
        extent  = sel_pos.max(axis=0) - sel_pos.min(axis=0)
        size    = extent + 2 * padding

    return (
        float(center[0]), float(center[1]), float(center[2]),
        float(size[0]),   float(size[1]),   float(size[2]),
    )


# ── Internal helpers ──────────────────────────────────────────────────────

def _prepare_receptor_manual(pdb_path: Path, out_path: Path) -> None:
    """
    Minimal manual PDBQT conversion: keep ATOM records, add AD4 atom types.
    This is a simplified conversion — use Meeko for production.
    """
    _ELEMENT_AD4: dict[str, str] = {
        "C": "C", "N": "NA", "O": "OA", "S": "SA",
        "H": "HD", "P": "P", "F": "F", "CL": "CL",
        "BR": "BR", "I": "I", "FE": "FE", "ZN": "ZN",
    }

    lines_out: list[str] = []
    with pdb_path.open("r", errors="replace") as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue
            resname = line[17:20].strip().upper()
            if resname in {"HOH", "WAT", "TIP3", "SOL"}:
                continue
            elem = line[76:78].strip().upper() if len(line) > 76 else ""
            if not elem:
                atom_name = line[12:16].strip().upper()
                elem = atom_name[0] if atom_name else "C"
            ad4_type = _ELEMENT_AD4.get(elem, "C")
            charge_str = "  0.000"
            pdbqt_line = line.rstrip()
            if len(pdbqt_line) < 80:
                pdbqt_line = pdbqt_line.ljust(79)
            pdbqt_line = pdbqt_line[:68] + charge_str + "  " + ad4_type
            lines_out.append(pdbqt_line + "\n")

    out_path.write_text("".join(lines_out), encoding="utf-8")


def _prepare_ligand_manual(mol, out_path: Path) -> None:
    """Write a minimal PDBQT for the ligand from an RDKit mol object."""
    try:
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit required for ligand PDBQT preparation.")

    conf = mol.GetConformer()
    lines: list[str] = ["REMARK  PSVAP ligand preparation\n", "TORSDOF 0\n"]

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        pos  = conf.GetAtomPosition(i)
        elem = atom.GetSymbol()
        lines.append(
            f"ATOM  {i+1:5d}  {elem:<3s} LIG     1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00     0.000  {elem}\n"
        )
    lines.append("ENDROOT\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _prepare_receptor_meeko(pdb_path: Path, out_path: Path) -> None:
    """Use Meeko to prepare receptor PDBQT."""
    import meeko
    proc = subprocess.run(
        ["mk_prepare_receptor.py", str(pdb_path), "-o", str(out_path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Meeko receptor prep failed: {proc.stderr[:300]}")


def _prepare_ligand_meeko(sdf_path: Path, out_path: Path) -> None:
    """Use Meeko to prepare ligand PDBQT from SDF."""
    import meeko
    proc = subprocess.run(
        ["mk_prepare_ligand.py", str(sdf_path), "-o", str(out_path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Meeko ligand prep failed: {proc.stderr[:300]}")


def _parse_vina_output_pdbqt(pdbqt_text: str) -> list[DockingPose]:
    """Parse multi-model PDBQT output from Vina into DockingPose objects."""
    poses: list[DockingPose] = []
    current_atoms: list[list[float]] = []
    model_rank = 0

    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            model_rank += 1
            current_atoms = []
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_atoms.append([x, y, z])
            except (ValueError, IndexError):
                pass
        elif line.startswith("ENDMDL"):
            if current_atoms:
                poses.append(DockingPose(
                    rank=model_rank,
                    score=0.0,
                    rmsd_lb=0.0,
                    rmsd_ub=0.0,
                    positions=np.array(current_atoms, dtype=np.float64),
                ))
    return poses


def _parse_vina_stdout_scores(
    stdout: str,
) -> list[tuple[float, float, float]]:
    """
    Parse the score table from Vina stdout.

    Format:
      -----+------------+----------+----------
      mode |  affinity  | dist from best mode
           | (kcal/mol) | rmsd l.b.| rmsd u.b.
    -----+------------+----------+----------
       1       -9.2          0          0
       2       -8.7        1.23       3.45
    """
    scores: list[tuple[float, float, float]] = []
    in_table = False

    for line in stdout.splitlines():
        stripped = line.strip()
        if "affinity" in stripped.lower() and "rmsd" in stripped.lower():
            in_table = True
            continue
        if in_table:
            if stripped.startswith("-") or stripped.startswith("="):
                continue
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    affinity = float(parts[1])
                    rmsd_lb  = float(parts[2])
                    rmsd_ub  = float(parts[3])
                    scores.append((affinity, rmsd_lb, rmsd_ub))
                except (ValueError, IndexError):
                    if scores:
                        break

    return scores