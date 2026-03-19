"""
modeling/md_setup.py
---------------------
Feature 14: MD Simulation Input File Preparation.

Generates input files for GROMACS or AMBER molecular dynamics simulations.
PSVAP does NOT run the simulation — it prepares the files and shows the
user the recommended command sequence.

Public API
----------
  generate_gromacs_inputs(atoms, positions, box_bounds,
                          force_field, output_dir)
      → MDSetupResult

  generate_amber_inputs(atoms, positions, box_bounds,
                        force_field, output_dir)
      → MDSetupResult

  MDSetupResult (dataclass)
      files_created, commands, force_field, engine, summary

Supported force fields: AMBER99SB-ILDN, CHARMM36, OPLS-AA, MARTINI 2.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MDSetupResult:
    """Result of MD input file generation."""
    engine:        str              # 'gromacs' or 'amber'
    force_field:   str
    output_dir:    Path
    files_created: list[str]        # filenames (relative to output_dir)
    commands:      list[str]        # recommended command sequence
    warnings:      list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"MD SETUP COMPLETE\n",
            f"  Engine       : {self.engine.upper()}",
            f"  Force field  : {self.force_field}",
            f"  Output dir   : {self.output_dir}",
            "",
            f"FILES CREATED:",
        ]
        for f in self.files_created:
            lines.append(f"  {f}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("\nRECOMMENDED COMMANDS:")
        for i, cmd in enumerate(self.commands, 1):
            lines.append(f"  {i}. {cmd}")

        return "\n".join(lines)


# ── Force field parameters ────────────────────────────────────────────────
_GROMACS_FF = {
    "AMBER99SB-ILDN": "amber99sb-ildn",
    "CHARMM36":       "charmm36-jul2021",
    "OPLS-AA":        "oplsaa",
    "GROMOS96":       "gromos96-53a6",
}

_AMBER_FF = {
    "FF14SB":   "leaprc.protein.ff14SB",
    "FF19SB":   "leaprc.protein.ff19SB",
    "GAFF2":    "leaprc.gaff2",
    "GLYCAM06": "leaprc.GLYCAM_06j-1",
}


# ── GROMACS ───────────────────────────────────────────────────────────────

def generate_gromacs_inputs(
    atoms: list,
    positions: np.ndarray,
    box_bounds: np.ndarray | None,
    force_field: str = "AMBER99SB-ILDN",
    water_model: str = "tip3p",
    output_dir: str | Path = "md_setup",
    ensemble: str = "NPT",
    n_steps_em: int = 50000,
    n_steps_equil: int = 100000,
    n_steps_prod: int = 5000000,
    dt: float = 0.002,
) -> MDSetupResult:
    """
    Generate GROMACS MD input files (.mdp files).

    Creates:
      - em.mdp       : energy minimisation
      - nvt.mdp      : NVT equilibration (constant volume)
      - npt.mdp      : NPT equilibration (constant pressure)
      - md.mdp       : production MD
      - README.txt   : step-by-step commands

    Parameters
    ----------
    atoms       : loaded atom list
    positions   : (N, 3) positions in Å
    box_bounds  : (3, 2) box dimensions; None = auto-compute
    force_field : force field name (see _GROMACS_FF)
    water_model : 'tip3p', 'spce', or 'tip4p'
    output_dir  : directory to write files into
    ensemble    : 'NVT' or 'NPT' for production
    n_steps_em  : energy minimisation steps
    n_steps_equil : equilibration steps
    n_steps_prod  : production MD steps
    dt          : integration timestep in ps (default 0.002 = 2 fs)

    Returns
    -------
    MDSetupResult
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ff_name = _GROMACS_FF.get(force_field, "amber99sb-ildn")
    warnings: list[str] = []
    files_created: list[str] = []

    # Estimate box size
    if box_bounds is not None:
        bb = np.asarray(box_bounds)
        box_x = float(bb[0, 1] - bb[0, 0]) / 10.0   # Å → nm
        box_y = float(bb[1, 1] - bb[1, 0]) / 10.0
        box_z = float(bb[2, 1] - bb[2, 0]) / 10.0
    else:
        pos_nm = np.asarray(positions) / 10.0   # Å → nm
        span = pos_nm.max(axis=0) - pos_nm.min(axis=0)
        box_x = float(span[0]) + 2.0
        box_y = float(span[1]) + 2.0
        box_z = float(span[2]) + 2.0

    # Check for hydrogen atoms
    has_h = any((getattr(a, "element", None) or "").upper() == "H" for a in atoms)
    if not has_h:
        warnings.append(
            "No hydrogen atoms found. Add hydrogens before running MD "
            "(use PREP tab → Add Hydrogens, or run 'gmx pdb2gmx')."
        )

    # ── em.mdp ─────────────────────────────────────────────────────────
    em_mdp = _gromacs_em_mdp(n_steps_em)
    (out / "em.mdp").write_text(em_mdp, encoding="utf-8")
    files_created.append("em.mdp")

    # ── nvt.mdp ────────────────────────────────────────────────────────
    nvt_mdp = _gromacs_nvt_mdp(n_steps_equil, dt)
    (out / "nvt.mdp").write_text(nvt_mdp, encoding="utf-8")
    files_created.append("nvt.mdp")

    # ── npt.mdp ────────────────────────────────────────────────────────
    npt_mdp = _gromacs_npt_mdp(n_steps_equil, dt)
    (out / "npt.mdp").write_text(npt_mdp, encoding="utf-8")
    files_created.append("npt.mdp")

    # ── md.mdp (production) ────────────────────────────────────────────
    md_mdp = _gromacs_prod_mdp(n_steps_prod, dt, ensemble)
    (out / "md.mdp").write_text(md_mdp, encoding="utf-8")
    files_created.append("md.mdp")

    # ── README ─────────────────────────────────────────────────────────
    commands = [
        f"gmx pdb2gmx -f structure.pdb -ff {ff_name} -water {water_model} -o processed.gro",
        f"gmx editconf -f processed.gro -o boxed.gro "
        f"-box {box_x:.3f} {box_y:.3f} {box_z:.3f}",
        "gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top",
        "gmx grompp -f em.mdp -c solvated.gro -p topol.top -o em.tpr",
        "gmx mdrun -v -deffnm em",
        "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr",
        "gmx mdrun -deffnm nvt",
        "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr",
        "gmx mdrun -deffnm npt",
        "gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr",
        "gmx mdrun -deffnm md",
    ]

    readme = _make_readme("GROMACS", commands, force_field, warnings)
    (out / "README.txt").write_text(readme, encoding="utf-8")
    files_created.append("README.txt")

    return MDSetupResult(
        engine="gromacs",
        force_field=force_field,
        output_dir=out,
        files_created=files_created,
        commands=commands,
        warnings=warnings,
    )


def generate_amber_inputs(
    atoms: list,
    positions: np.ndarray,
    box_bounds: np.ndarray | None,
    force_field: str = "FF14SB",
    water_model: str = "TIP3P",
    output_dir: str | Path = "md_setup_amber",
) -> MDSetupResult:
    """
    Generate AMBER MD input files (tleap script + sander/pmemd inputs).

    Creates:
      - tleap.in         : tleap setup script
      - min.in           : energy minimisation input
      - heat.in          : heating input (0K → 300K)
      - equil.in         : equilibration input
      - prod.in          : production MD input
      - README.txt       : step-by-step commands

    Returns
    -------
    MDSetupResult
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ff_leaprc = _AMBER_FF.get(force_field, "leaprc.protein.ff14SB")
    warnings: list[str] = []
    files_created: list[str] = []

    # tleap script
    tleap = _amber_tleap(ff_leaprc, water_model)
    (out / "tleap.in").write_text(tleap, encoding="utf-8")
    files_created.append("tleap.in")

    # min.in
    (out / "min.in").write_text(_amber_min_in(), encoding="utf-8")
    files_created.append("min.in")

    # heat.in
    (out / "heat.in").write_text(_amber_heat_in(), encoding="utf-8")
    files_created.append("heat.in")

    # equil.in
    (out / "equil.in").write_text(_amber_equil_in(), encoding="utf-8")
    files_created.append("equil.in")

    # prod.in
    (out / "prod.in").write_text(_amber_prod_in(), encoding="utf-8")
    files_created.append("prod.in")

    commands = [
        "tleap -f tleap.in",
        "sander -O -i min.in  -o min.out  -p system.prmtop -c system.inpcrd -r min.rst",
        "sander -O -i heat.in -o heat.out -p system.prmtop -c min.rst       -r heat.rst -x heat.nc",
        "sander -O -i equil.in -o equil.out -p system.prmtop -c heat.rst    -r equil.rst -x equil.nc",
        "sander -O -i prod.in  -o prod.out  -p system.prmtop -c equil.rst   -r prod.rst  -x prod.nc",
    ]

    readme = _make_readme("AMBER", commands, force_field, warnings)
    (out / "README.txt").write_text(readme, encoding="utf-8")
    files_created.append("README.txt")

    return MDSetupResult(
        engine="amber",
        force_field=force_field,
        output_dir=out,
        files_created=files_created,
        commands=commands,
        warnings=warnings,
    )


# ── MDP content generators ────────────────────────────────────────────────

def _gromacs_em_mdp(n_steps: int) -> str:
    return f"""; GROMACS Energy Minimisation Parameters
; Generated by PSVAP modeling/md_setup.py

integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = {n_steps}

nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""


def _gromacs_nvt_mdp(n_steps: int, dt: float) -> str:
    return f"""; GROMACS NVT Equilibration
; Generated by PSVAP modeling/md_setup.py

define      = -DPOSRES
integrator  = md
nsteps      = {n_steps}
dt          = {dt}

nstxout     = 500
nstvout     = 500
nstenergy   = 500
nstlog      = 500

cutoff-scheme   = Verlet
ns_type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pme_order       = 4
fourierspacing  = 0.16

tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300

pcoupl      = no
pbc         = xyz
"""


def _gromacs_npt_mdp(n_steps: int, dt: float) -> str:
    return f"""; GROMACS NPT Equilibration
; Generated by PSVAP modeling/md_setup.py

define      = -DPOSRES
integrator  = md
nsteps      = {n_steps}
dt          = {dt}

nstxout     = 500
nstvout     = 500
nstenergy   = 500
nstlog      = 500

cutoff-scheme   = Verlet
ns_type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pme_order       = 4
fourierspacing  = 0.16

tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300

pcoupl      = Parrinello-Rahman
pcoupltype  = isotropic
tau_p       = 2.0
ref_p       = 1.0
compressibility = 4.5e-5

pbc         = xyz
"""


def _gromacs_prod_mdp(n_steps: int, dt: float, ensemble: str) -> str:
    pcoupl = "Parrinello-Rahman" if ensemble.upper() == "NPT" else "no"
    return f"""; GROMACS Production MD
; Generated by PSVAP modeling/md_setup.py

integrator  = md
nsteps      = {n_steps}
dt          = {dt}

nstxout-compressed  = 5000
nstenergy           = 5000
nstlog              = 5000

cutoff-scheme   = Verlet
ns_type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0
coulombtype     = PME
pme_order       = 4
fourierspacing  = 0.16

tcoupl      = V-rescale
tc-grps     = Protein Non-Protein
tau_t       = 0.1 0.1
ref_t       = 300 300

pcoupl      = {pcoupl}
{"pcoupltype  = isotropic" if pcoupl != "no" else "; pcoupl = no (NVT ensemble)"}
{"tau_p       = 2.0" if pcoupl != "no" else ""}
{"ref_p       = 1.0" if pcoupl != "no" else ""}
{"compressibility = 4.5e-5" if pcoupl != "no" else ""}

pbc         = xyz
gen_vel     = no
"""


# ── AMBER content generators ──────────────────────────────────────────────

def _amber_tleap(ff_leaprc: str, water_model: str) -> str:
    wm = water_model.upper()
    water_leaprc = {
        "TIP3P": "leaprc.water.tip3p",
        "TIP4P": "leaprc.water.tip4pew",
        "SPCE":  "leaprc.water.spce",
    }.get(wm, "leaprc.water.tip3p")
    solvent = {
        "TIP3P": "TIP3PBOX",
        "TIP4P": "TIP4PEWBOX",
        "SPCE":  "SPCEBOX",
    }.get(wm, "TIP3PBOX")
    return f"""# tleap input — generated by PSVAP modeling/md_setup.py
source {ff_leaprc}
source {water_leaprc}

mol = loadpdb structure.pdb

# Solvate with 10 Å buffer
solvatebox mol {solvent} 10.0

# Add counterions
addions mol Na+ 0
addions mol Cl- 0

saveamberparm mol system.prmtop system.inpcrd
savepdb mol solvated.pdb
quit
"""


def _amber_min_in() -> str:
    return """Energy minimisation
 &cntrl
  imin   = 1,
  maxcyc = 5000,
  ncyc   = 2500,
  ntb    = 1,
  ntp    = 0,
  ntr    = 1,
  cut    = 10.0,
  restraintmask = '@CA',
  restraint_wt  = 50.0,
 /
"""


def _amber_heat_in() -> str:
    return """Heating 0K to 300K
 &cntrl
  imin   = 0,
  irest  = 0,
  ntx    = 1,
  ntb    = 1,
  cut    = 10.0,
  ntr    = 1,
  ntc    = 2,
  ntf    = 2,
  tempi  = 0.0,
  temp0  = 300.0,
  ntt    = 3,
  gamma_ln = 2.0,
  nstlim = 50000,
  dt     = 0.002,
  ntpr   = 500,
  ntwx   = 500,
  ntwr   = 1000,
  restraintmask = '@CA',
  restraint_wt  = 10.0,
 /
"""


def _amber_equil_in() -> str:
    return """NPT Equilibration
 &cntrl
  imin   = 0,
  irest  = 1,
  ntx    = 5,
  ntb    = 2,
  cut    = 10.0,
  ntr    = 0,
  ntc    = 2,
  ntf    = 2,
  temp0  = 300.0,
  ntt    = 3,
  gamma_ln = 2.0,
  ntp    = 1,
  taup   = 2.0,
  nstlim = 100000,
  dt     = 0.002,
  ntpr   = 1000,
  ntwx   = 1000,
  ntwr   = 5000,
 /
"""


def _amber_prod_in() -> str:
    return """Production MD
 &cntrl
  imin   = 0,
  irest  = 1,
  ntx    = 5,
  ntb    = 2,
  cut    = 10.0,
  ntr    = 0,
  ntc    = 2,
  ntf    = 2,
  temp0  = 300.0,
  ntt    = 3,
  gamma_ln = 2.0,
  ntp    = 1,
  taup   = 2.0,
  nstlim = 5000000,
  dt     = 0.002,
  ntpr   = 5000,
  ntwx   = 5000,
  ntwr   = 10000,
  ioutfm = 1,
 /
"""


def _make_readme(engine: str, commands: list[str],
                 force_field: str, warnings: list[str]) -> str:
    lines = [
        f"MD SETUP — {engine}",
        f"Generated by PSVAP modeling/md_setup.py",
        f"Force field: {force_field}",
        "",
    ]
    if warnings:
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  ! {w}")
        lines.append("")
    lines.append("COMMANDS (run in order):")
    for i, cmd in enumerate(commands, 1):
        lines.append(f"  {i}. {cmd}")
    return "\n".join(lines)